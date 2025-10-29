#!/usr/bin/env python3
"""
Manufactoria DELTA task driver for TPU-based ES fine-tuning.

This script mirrors the countdown TPU accelerator but swaps in the
Manufactoria adapter + vLLM TPU client so we can reuse the ES loop
without touching the RL Grok repo.

Usage (example):
    python es_fine-tuning_manufactoria_tpu.py \
        --config configs/delta_manufactoria_tpu.yaml \
        --dataset-limit 120 \
        --global-seed 42
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import ray
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

from engines import EngineRuntimeConfig, VllmTpuClient, launch_vllm_tpu_engines
from tasks.delta import ManufactoriaAdapter, ScoreMode

logger = logging.getLogger("manufactoria_es")


@dataclass
class ESConfig:
    sigma: float
    alpha: float
    population_size: int
    num_iterations: int
    global_seed: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manufactoria ES fine-tuning on TPU via vLLM.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/delta_manufactoria_tpu.yaml",
        help="Path to YAML config describing task/model/engine/ES parameters.",
    )
    parser.add_argument(
        "--dataset-limit",
        type=int,
        default=None,
        help="Optional cap on number of training examples to sample.",
    )
    parser.add_argument(
        "--global-seed",
        type=int,
        default=None,
        help="Override global seed in config (if provided).",
    )
    return parser.parse_args()


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def maybe_seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_logging_dir(experiment_dir: str, prefix: str) -> str:
    os.makedirs(experiment_dir, exist_ok=True)
    run_dir = os.path.join(
        experiment_dir,
        f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def build_adapter(task_cfg: Dict[str, Any], dataset_limit: Optional[int]) -> Tuple[ManufactoriaAdapter, List]:
    adapter = ManufactoriaAdapter(
        dataset_repo=task_cfg["dataset_repo"],
        split=task_cfg.get("split", "train"),
        score_mode=task_cfg.get("score_mode", ScoreMode.PASS_RATE),
        max_cases=task_cfg.get("max_cases", 40),
        max_runtime_seconds=task_cfg.get("max_runtime_seconds", 5.0),
        load_dataset_kwargs=task_cfg.get("load_dataset_kwargs"),
        custom_repo_path=task_cfg.get("custom_repo_path"),
    )
    examples = adapter.load_examples(limit=dataset_limit)
    if not examples:
        raise ValueError("Manufactoria dataset returned zero examples.")
    return adapter, examples


def format_prompts(
    adapter: ManufactoriaAdapter,
    examples: Sequence,
    assistant_prefix: str = "",
) -> List[str]:
    return [
        adapter.format_messages_as_prompt(ex.messages, assistant_prefix=assistant_prefix)
        for ex in examples
    ]


def setup_base_model(model_name: str, dtype: torch.dtype, output_dir: str) -> Tuple[str, AutoTokenizer]:
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer.save_pretrained(output_dir)
    base_model.save_pretrained(output_dir)
    del base_model
    gc.collect()

    return output_dir, tokenizer


def evaluate_population(
    clients: List[VllmTpuClient],
    prompts: Sequence[str],
    examples: Sequence,
    adapter: ManufactoriaAdapter,
    seeds: Sequence[int],
    sigma: float,
    sampling_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    results = []
    num_clients = len(clients)
    for idx, seed in enumerate(seeds):
        client = clients[idx % num_clients]
        client.perturb_self_weights(seed, sigma, negate=False)

        outputs = client.generate(prompts, seed=seed, **sampling_kwargs)
        rewards = []
        reward_infos = []

        for ex, output in zip(examples, outputs):
            text = output["outputs"][0]["text"] if output["outputs"] else ""
            score = adapter.score_response(text, ex)
            rewards.append(score["reward"])
            reward_infos.append(score["reward_info"])

        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        results.append(
            {
                "seed": seed,
                "avg_reward": avg_reward,
                "rewards": rewards,
                "reward_infos": reward_infos,
                "client_index": idx % num_clients,
            }
        )

        client.restore_self_weights(seed, sigma)
    return results


def apply_es_update(
    clients: List[VllmTpuClient],
    per_seed_coeffs: Sequence[Tuple[int, float]],
) -> None:
    if not clients:
        return
    primary = clients[0]
    for seed, coeff in per_seed_coeffs:
        primary.perturb_self_weights(seed, coeff, negate=False)

    state = primary.dump_state_dict()
    for client in clients[1:]:
        client.load_state_dict(state)


def maybe_switch_score_mode(
    adapter: ManufactoriaAdapter,
    task_cfg: Dict[str, Any],
    iteration: int,
) -> None:
    warmup = task_cfg.get("warmup_generations")
    phase2_mode = task_cfg.get("phase2_score_mode")
    if warmup is None or phase2_mode is None:
        return
    if iteration == warmup:
        adapter.set_score_mode(phase2_mode)
        logger.info("Switched Manufactoria score_mode to %s at iteration %d.", phase2_mode, iteration)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    # Basic logging configuration
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    task_cfg = config["task"]
    model_cfg = config["model"]
    engine_cfg = config["engine"]
    es_cfg_dict = config["es"]
    logging_cfg = config.get("logging", {})

    es_cfg = ESConfig(
        sigma=float(es_cfg_dict["sigma"]),
        alpha=float(es_cfg_dict["alpha"]),
        population_size=int(es_cfg_dict["population_size"]),
        num_iterations=int(es_cfg_dict["num_iterations"]),
        global_seed=args.global_seed if args.global_seed is not None else es_cfg_dict.get("global_seed"),
    )

    maybe_seed_everything(es_cfg.global_seed)

    adapter, examples = build_adapter(task_cfg, args.dataset_limit)
    prompts = format_prompts(adapter, examples, assistant_prefix=task_cfg.get("assistant_prefix", ""))

    experiment_dir = logging_cfg.get("experiment_dir", "es-ft-experiments")
    run_dir = prepare_logging_dir(experiment_dir, "manufactoria_tpu")
    model_saves_dir = os.path.join(run_dir, "model_saves")
    os.makedirs(model_saves_dir, exist_ok=True)

    tensorboard_enabled = logging_cfg.get("tensorboard", True)
    writer: Optional[SummaryWriter] = SummaryWriter(log_dir=run_dir) if tensorboard_enabled else None

    base_model_dir = os.path.join(model_saves_dir, "base_model")
    model_path, tokenizer = setup_base_model(
        model_name=model_cfg.get("name", "Qwen/Qwen2.5-3B-Instruct"),
        dtype=getattr(torch, model_cfg.get("dtype", "bfloat16")),
        output_dir=base_model_dir,
    )

    runtime_config = EngineRuntimeConfig(
        pip_packages=engine_cfg.get("pip_requirements", []),
        env_overrides=engine_cfg.get("env_overrides", {}),
    )

    sampling_defaults = engine_cfg.get("sampling_defaults", {"temperature": 0.0, "max_tokens": model_cfg.get("max_new_tokens", 1024)})

    work_dir = Path(__file__).resolve().parent
    ray.init(
        address="local",
        include_dashboard=False,
        ignore_reinit_error=True,
        runtime_env={
            "working_dir": str(work_dir),
            "excludes": [
                "*.safetensors",
                "*.bin",
                "*.ckpt",
                "*.pt",
                "*.pth",
                "*.log",
                "es-ft-experiments/",
                ".git/",
            ],
        },
    )

    try:
        clients = launch_vllm_tpu_engines(
            num_engines=int(engine_cfg.get("num_engines", 1)),
            model_dir=model_path,
            dtype=engine_cfg.get("dtype", "bfloat16"),
            tpu_chips_csv=engine_cfg.get("tpu_chips"),
            sampling_defaults=sampling_defaults,
            runtime_config=runtime_config,
        )

        sampling_kwargs = dict(sampling_defaults)
        population_size = es_cfg.population_size
        sigma = es_cfg.sigma
        alpha = es_cfg.alpha

        for iteration in range(es_cfg.num_iterations):
            maybe_switch_score_mode(adapter, task_cfg, iteration)
            logger.info("=== Generation %d ===", iteration)
            start_time = time.time()

            seeds = [random.randint(0, 1_000_000) for _ in range(population_size)]
            results = evaluate_population(
                clients=clients,
                prompts=prompts,
                examples=examples,
                adapter=adapter,
                seeds=seeds,
                sigma=sigma,
                sampling_kwargs=sampling_kwargs,
            )

            avg_rewards = [res["avg_reward"] for res in results]
            mean_r = float(np.mean(avg_rewards)) if avg_rewards else 0.0
            std_r = float(np.std(avg_rewards)) if avg_rewards else 0.0
            min_r = float(np.min(avg_rewards)) if avg_rewards else 0.0
            max_r = float(np.max(avg_rewards)) if avg_rewards else 0.0

            logger.info("Iteration %d stats -> mean: %.4f, std: %.4f, min: %.4f, max: %.4f", iteration, mean_r, std_r, min_r, max_r)

            if writer:
                writer.add_scalar("reward/mean", mean_r, iteration)
                writer.add_scalar("reward/std", std_r, iteration)
                writer.add_scalar("reward/min", min_r, iteration)
                writer.add_scalar("reward/max", max_r, iteration)

            denom = std_r if std_r > 0 else 1e-8
            per_seed_coeffs = []
            for res in results:
                norm_reward = (res["avg_reward"] - mean_r) / denom
                res["norm_reward"] = norm_reward
                coeff = (alpha / population_size) * float(norm_reward)
                per_seed_coeffs.append((res["seed"], coeff))
                logger.debug("Seed %d avg %.4f norm %.4f coeff %.6f", res["seed"], res["avg_reward"], norm_reward, coeff)

            apply_es_update(clients, per_seed_coeffs)

            step_time = time.time() - start_time
            if writer:
                writer.add_scalar("time/iteration", step_time, iteration)
            logger.info("Iteration %d completed in %.2fs", iteration, step_time)

        final_dir = os.path.join(model_saves_dir, f"final_model_iteration_{es_cfg.num_iterations}")
        os.makedirs(final_dir, exist_ok=True)
        clients[0].save_self_weights_to_disk(os.path.join(final_dir, "weights"))
        logger.info("Final model weights saved to %s", final_dir)
    finally:
        if writer:
            writer.flush()
            writer.close()
        ray.shutdown()


if __name__ == "__main__":
    main()
