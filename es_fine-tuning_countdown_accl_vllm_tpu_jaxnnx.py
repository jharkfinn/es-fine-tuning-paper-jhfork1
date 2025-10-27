#!/usr/bin/env python3
"""
TPU-only ES trainer using vLLM-TPU unified backend and JAX/NNX-native perturbation.

We register a worker extension 'nnx_es_worker.WorkerExtension' so ES ops happen
INSIDE the vLLM worker (where the NNX model lives). The trainer stays free of
JAX/torch_xla/vLLM; engines install their own deps via runtime_env.pip.

Determinism: each leaf's noise comes from PRNGKey(seed) folded with a stable
hash of the leaf's parameter path; order-independent and process-agnostic.
"""

import argparse
from datetime import datetime
import gc
import json
import os
import random
import shutil
import signal
import sys
import time

import numpy as np
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer

SIGMA = 0.001
ALPHA = 0.0005
POPULATION_SIZE = 30
NUM_ENGINES = 1
NUM_ITERATIONS = 1000
EXPERIMENT_DIR = "es-ft-experiment"

def parse_args():
    p = argparse.ArgumentParser(description="ES fine-tuning on TPU v6e with vLLM-TPU (NNX-native ES)")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--sigma", type=float, default=SIGMA)
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--population_size", type=int, default=POPULATION_SIZE)
    p.add_argument("--num_engines", type=int, default=NUM_ENGINES)
    p.add_argument("--num_iterations", type=int, default=NUM_ITERATIONS)
    p.add_argument("--experiment_dir", type=str, default=EXPERIMENT_DIR)
    p.add_argument("--global_seed", type=int, default=None)
    p.add_argument("--tpu_chips", type=str, default=None,
                   help="Comma-separated TPU chip ids per engine, e.g. '0,1,2,3'")
    # Per-actor pins (vllm-tpu bundles JAX/libtpu; keep this minimal)
    p.add_argument("--engine_vllm_tpu", type=str, default="vllm-tpu")
    p.add_argument("--engine_transformers", type=str, default="transformers>=4.30.0")
    p.add_argument("--engine_numpy", type=str, default="numpy>=1.21.0")
    p.add_argument("--engine_psutil", type=str, default="psutil>=5.8.0")
    return p.parse_args()

def _load_reward():
    from countdown.countdown_task import reward_function
    return reward_function

@ray.remote
class VllmTpuActor:
    def __init__(self, model_dir: str, dtype: str = "bfloat16"):
        # Import vLLM-TPU inside the actor
        from vllm import LLM, SamplingParams

        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_DEVICE", "tpu")
        os.environ.setdefault("PJRT_DEVICE", "TPU")

        self._SamplingParams = SamplingParams
        # Register our JAX/NNX ES worker extension here
        # Note: vllm-tpu doesn't accept 'device' parameter, uses env vars instead
        self._llm = LLM(
            model=model_dir,
            tensor_parallel_size=1,             # one chip per actor
            distributed_executor_backend="ray",
            worker_extension_cls="utils.nnx_es_worker.WorkerExtension",
            dtype=dtype,
            enable_prefix_caching=False,
            enforce_eager=False,
        )

    # Inference
    def generate(self, prompts, temperature: float = 0.0, seed: int = 42, max_tokens: int = 1024):
        params = self._SamplingParams(temperature=temperature, seed=seed, max_tokens=max_tokens)
        return self._llm.generate(prompts, params, use_tqdm=False)

    # ES ops - run inside vLLM worker via collective_rpc
    def perturb_self_weights(self, seed: int, sigma_or_scale: float, negate: bool = False):
        return self._llm.collective_rpc("perturb_self_weights",
                                        args=(int(seed), float(sigma_or_scale), bool(negate)))

    def restore_self_weights(self, seed: int, sigma: float):
        return self._llm.collective_rpc("restore_self_weights",
                                        args=(int(seed), float(sigma)))

    def dump_state_dict(self):
        return self._llm.collective_rpc("dump_state_dict", args=())

    def load_state_dict(self, state):
        return self._llm.collective_rpc("load_state_dict", args=(state,))

    def save_self_weights_to_disk(self, path: str):
        return self._llm.collective_rpc("save_self_weights_to_disk", args=(path,))

# -------------------- TPU launcher with isolated actor env ------------------
def launch_tpu_engines(num_engines: int, model_dir: str, tpu_chips_csv: str | None, pip_versions: dict):
    if not tpu_chips_csv or tpu_chips_csv.strip() == "":
        chips = [str(i) for i in range(max(1, num_engines))]
    else:
        chips = [c.strip() for c in tpu_chips_csv.split(",") if c.strip() != ""]
    if len(chips) < num_engines:
        raise ValueError(f"Need at least {num_engines} TPU chips, got {len(chips)}")

    pip_list = [
        pip_versions["engine_vllm_tpu"],
        pip_versions["engine_transformers"],
        pip_versions["engine_numpy"],
        pip_versions["engine_psutil"],
        "tensorboard>=2.20.0",
        "virtualenv>=20.0.0",
    ]

    engines = []
    for i in range(num_engines):
        chip_id = chips[i]
        runtime_env = {
            "env_vars": {
                "TPU_VISIBLE_CHIPS": chip_id,
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                "VLLM_DEVICE": "tpu",
                "PJRT_DEVICE": "TPU",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            },
            "pip": pip_list,
        }
        actor = VllmTpuActor.options(num_cpus=0, scheduling_strategy="DEFAULT",
                                     runtime_env=runtime_env).remote(
            model_dir=model_dir, dtype="bfloat16"
        )
        engines.append(actor)
    return engines

def evaluate_countdown_handle(actor, task_datas):
    prompts = [d["context"] for d in task_datas]
    return actor.generate.remote(prompts, temperature=0.0, seed=42, max_tokens=1024), time.time()

def _postprocess_outputs(outputs, task_datas, reward_fn):
    rewards, avg_rewards = [], []
    for output, data in zip(outputs, task_datas):
        response = output.outputs[0].text
        r = reward_fn(response, data["numbers"], data["target"])
        rewards.append(r)
        avg_rewards.append(r["reward"])
    return {"rewards": rewards, "avg_reward": float(np.mean(avg_rewards)) if avg_rewards else 0.0}

def main(args):
    if args.global_seed is not None:
        random.seed(args.global_seed)
        np.random.seed(args.global_seed)
        torch.manual_seed(args.global_seed)

    os.environ.pop("RAY_ADDRESS", None)
    os.environ.pop("RAY_HEAD_IP", None)
    os.environ.pop("RAY_GCS_SERVER_ADDRESS", None)

    # Set working_dir at job level to ship nnx_es_worker.py to actors
    work_dir = os.path.abspath(os.path.dirname(__file__))
    ray.init(address="local", include_dashboard=False, ignore_reinit_error=True,
             runtime_env={
                 "working_dir": work_dir,
                 "excludes": [
                     "*.safetensors",
                     "*.bin",
                     "*.ckpt",
                     "*.pt",
                     "*.pth",
                     "*.log",
                     "es-ft-experiment/",
                     ".git/",
                 ]
             })

    reward_fn = _load_reward()

    logging_dir = f"{args.experiment_dir}/countdown_vllm_tpu_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=logging_dir)

    # Prepare base HF checkpoint for vLLM to load (bf16)
    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    base_model_path = os.path.abspath(f"{model_saves_dir}/base_model")
    if os.path.exists(base_model_path):
        shutil.rmtree(base_model_path)
    os.makedirs(base_model_path, exist_ok=True)
    tokenizer.save_pretrained(base_model_path)
    base_model.save_pretrained(base_model_path)
    del base_model
    gc.collect()

    # Load task data
    data_path = "countdown/data/countdown.json"
    with open(data_path, "r") as f:
        task_datas = json.load(f)
    task_datas = task_datas[:200]

    # Launch TPU engines
    pip_versions = {
        "engine_vllm_tpu": args.engine_vllm_tpu,
        "engine_transformers": args.engine_transformers,
        "engine_numpy": args.engine_numpy,
        "engine_psutil": args.engine_psutil,
    }
    engines = launch_tpu_engines(args.num_engines, base_model_path, args.tpu_chips, pip_versions)

    # ES loop
    for i in range(args.num_iterations):
        print(f"\n=== Generation {i} ===")
        t0 = time.time()

        seeds = [random.randint(0, 1_000_000) for _ in range(args.population_size)]
        seeds_perf = {}

        seed_iter = iter(seeds)
        inflight = {}
        results_this_gen = []

        for eng_idx, actor in enumerate(engines):
            try:
                seed = next(seed_iter)
            except StopIteration:
                break
            ray.get(actor.perturb_self_weights.remote(seed, args.sigma, False))
            handle, st = evaluate_countdown_handle(actor, task_datas)
            inflight[handle] = {"actor": actor, "eng_idx": eng_idx, "seed": seed, "start_ts": st}

        while inflight:
            done, _ = ray.wait(list(inflight.keys()), num_returns=1)
            h = done[0]
            meta = inflight.pop(h)

            outputs = ray.get(h)
            metrics = _postprocess_outputs(outputs, task_datas, reward_fn)
            elapsed = time.time() - meta["start_ts"]

            seeds_perf[meta["seed"]] = metrics
            results_this_gen.append({"seed": meta["seed"], "avg_reward": metrics["avg_reward"], "time": elapsed})

            actor = meta["actor"]
            ray.get(actor.restore_self_weights.remote(meta["seed"], args.sigma))

            try:
                next_seed = next(seed_iter)
            except StopIteration:
                continue
            ray.get(actor.perturb_self_weights.remote(next_seed, args.sigma, False))
            handle, st = evaluate_countdown_handle(actor, task_datas)
            inflight[handle] = {"actor": actor, "eng_idx": meta["eng_idx"], "seed": next_seed, "start_ts": st}
            print(f"Scheduled seed {next_seed} on engine {meta['eng_idx']}")

        all_avg = [v["avg_reward"] for v in seeds_perf.values()]
        mean_r = float(np.mean(all_avg)) if all_avg else 0.0
        std_r = float(np.std(all_avg)) if all_avg else 0.0
        min_r = float(np.min(all_avg)) if all_avg else 0.0
        max_r = float(np.max(all_avg)) if all_avg else 0.0
        print(f"Mean reward: {mean_r}, std: {std_r}, min: {min_r}, max: {max_r}")

        for k in seeds_perf:
            seeds_perf[k]["norm_reward"] = (seeds_perf[k]["avg_reward"] - mean_r) / (std_r + 1e-8)
            print(f"Seed {k} normalized reward: {seeds_perf[k]['norm_reward']}")

        writer.add_scalar("reward/mean", mean_r, i)
        writer.add_scalar("reward/std", std_r, i)
        writer.add_scalar("reward/min", min_r, i)
        writer.add_scalar("reward/max", max_r, i)

        # ES update on engine 0
        per_seed_coeffs = [(seed, (args.alpha / args.population_size) * float(seeds_perf[seed]["norm_reward"]))
                           for seed in seeds]

        t1 = time.time()
        ray.get([engines[0].perturb_self_weights.remote(seed, coeff, False) for seed, coeff in per_seed_coeffs])
        writer.add_scalar("time/perturbation_application", time.time() - t1, i)

        # Driver-mediated broadcast (PyTree CPU numpy fan-out)
        t2 = time.time()
        cpu_state = ray.get(engines[0].dump_state_dict.remote())
        ray.get([e.load_state_dict.remote(cpu_state) for e in engines])
        writer.add_scalar("time/broadcast", time.time() - t2, i)

        for idx, res in enumerate(results_this_gen):
            print(f"IDX:{idx} Seed {res['seed']} avg_reward:{res['avg_reward']:.4f} time:{res['time']:.3f}s")
        writer.add_scalar("time/iteration", time.time() - t0, i)

    final_dir = f"{model_saves_dir}/final_model_iteration_{args.num_iterations}"
    os.makedirs(final_dir, exist_ok=True)
    ray.get(engines[0].save_self_weights_to_disk.remote(f"{final_dir}/weights"))
    print(f"Final model weights saved to {final_dir}.")
    ray.shutdown()

if __name__ == "__main__":
    args = parse_args()
    main(args)
