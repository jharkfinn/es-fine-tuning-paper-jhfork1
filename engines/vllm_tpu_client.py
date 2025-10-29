from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import ray


DEFAULT_ENGINE_REQUIREMENTS: Sequence[str] = (
    "vllm-tpu",
    "transformers>=4.30.0",
    "numpy>=1.21.0",
    "psutil>=5.8.0",
    "tensorboard>=2.20.0",
    "virtualenv>=20.0.0",
)


@dataclass
class EngineRuntimeConfig:
    """Runtime configuration shipped with each Ray actor."""

    pip_packages: Sequence[str] = field(default_factory=lambda: DEFAULT_ENGINE_REQUIREMENTS)
    env_overrides: Dict[str, str] = field(default_factory=dict)


def _resolve_chips(num_engines: int, chips_csv: Optional[str]) -> List[str]:
    if not chips_csv:
        return [str(i) for i in range(max(1, num_engines))]
    chips = [c.strip() for c in chips_csv.split(",") if c.strip()]
    if len(chips) < num_engines:
        raise ValueError(f"Need at least {num_engines} TPU chips, got {len(chips)}")
    return chips


@ray.remote
class _VllmTpuActor:
    def __init__(
        self,
        model_dir: str,
        dtype: str = "bfloat16",
        sampling_defaults: Optional[Dict[str, Any]] = None,
    ):
        from vllm import LLM, SamplingParams

        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_DEVICE", "tpu")
        os.environ.setdefault("PJRT_DEVICE", "TPU")

        self._SamplingParams = SamplingParams
        self._sampling_defaults = dict(sampling_defaults or {})
        self._llm = LLM(
            model=model_dir,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            worker_extension_cls="utils.nnx_es_worker.WorkerExtension",
            dtype=dtype,
            enable_prefix_caching=False,
            enforce_eager=False,
            kv_cache_dtype="fp8_e5m2",
            max_num_seqs=500,
            max_num_batched_tokens=81920,
        )

    # Inference ----------------------------------------------------------------
    def generate(self, prompts, **sampling_kwargs):
        params = {**self._sampling_defaults, **sampling_kwargs}
        sampling_params = self._SamplingParams(**params)
        outputs = self._llm.generate(prompts, sampling_params, use_tqdm=False)
        return [
            {
                "prompt": output.prompt,
                "outputs": [{"text": o.text, "token_ids": o.token_ids} for o in output.outputs],
                "finished": output.finished,
            }
            for output in outputs
        ]

    # ES hooks -----------------------------------------------------------------
    def perturb_self_weights(self, seed: int, sigma_or_scale: float, negate: bool = False):
        return self._llm.collective_rpc(
            "perturb_self_weights", args=(int(seed), float(sigma_or_scale), bool(negate))
        )

    def restore_self_weights(self, seed: int, sigma: float):
        return self._llm.collective_rpc("restore_self_weights", args=(int(seed), float(sigma)))

    def dump_state_dict(self):
        return self._llm.collective_rpc("dump_state_dict", args=())

    def load_state_dict(self, state):
        return self._llm.collective_rpc("load_state_dict", args=(state,))

    def save_self_weights_to_disk(self, path: str):
        return self._llm.collective_rpc("save_self_weights_to_disk", args=(path,))

    def collective_rpc(self, method: str, args: Sequence[Any] = (), kwargs: Optional[Dict[str, Any]] = None):
        return self._llm.collective_rpc(method, args=args, kwargs=kwargs or {})


@dataclass
class VllmTpuClient:
    """Thin wrapper around the Ray actor for ergonomic usage."""

    actor: ray.actor.ActorHandle
    sampling_defaults: Dict[str, Any] = field(default_factory=dict)

    def generate_async(self, prompts, **sampling_kwargs):
        params = {**self.sampling_defaults, **sampling_kwargs}
        return self.actor.generate.remote(prompts, **params)

    def generate(self, prompts, **sampling_kwargs):
        return ray.get(self.generate_async(prompts, **sampling_kwargs))

    def perturb_self_weights(self, seed: int, sigma_or_scale: float, negate: bool = False):
        return ray.get(self.actor.perturb_self_weights.remote(seed, sigma_or_scale, negate))

    def restore_self_weights(self, seed: int, sigma: float):
        return ray.get(self.actor.restore_self_weights.remote(seed, sigma))

    def dump_state_dict(self):
        return ray.get(self.actor.dump_state_dict.remote())

    def load_state_dict(self, state):
        return ray.get(self.actor.load_state_dict.remote(state))

    def save_self_weights_to_disk(self, path: str):
        return ray.get(self.actor.save_self_weights_to_disk.remote(path))

    def collective_rpc(self, method: str, args: Sequence[Any] = (), kwargs: Optional[Dict[str, Any]] = None):
        return ray.get(self.actor.collective_rpc.remote(method, args=args, kwargs=kwargs or {}))


def launch_vllm_tpu_engines(
    num_engines: int,
    model_dir: str,
    *,
    tpu_chips_csv: Optional[str] = None,
    dtype: str = "bfloat16",
    sampling_defaults: Optional[Dict[str, Any]] = None,
    runtime_config: Optional[EngineRuntimeConfig] = None,
    actor_options: Optional[Dict[str, Any]] = None,
) -> List[VllmTpuClient]:
    runtime_config = runtime_config or EngineRuntimeConfig()
    actor_options = actor_options or {}
    chips = _resolve_chips(num_engines, tpu_chips_csv)

    clients: List[VllmTpuClient] = []
    for engine_idx in range(num_engines):
        chip_id = chips[engine_idx]
        env_vars = {
            "TPU_VISIBLE_CHIPS": chip_id,
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_DEVICE": "tpu",
            "PJRT_DEVICE": "TPU",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
        env_vars.update(runtime_config.env_overrides)

        runtime_env = {
            "env_vars": env_vars,
            "pip": list(runtime_config.pip_packages),
        }

        actor = _VllmTpuActor.options(
            num_cpus=actor_options.get("num_cpus", 0),
            scheduling_strategy=actor_options.get("scheduling_strategy", "DEFAULT"),
            runtime_env=runtime_env,
            **{k: v for k, v in actor_options.items() if k not in {"num_cpus", "scheduling_strategy"}},
        ).remote(model_dir=model_dir, dtype=dtype, sampling_defaults=sampling_defaults)

        clients.append(VllmTpuClient(actor=actor, sampling_defaults=dict(sampling_defaults or {})))

    return clients


__all__ = [
    "EngineRuntimeConfig",
    "VllmTpuClient",
    "launch_vllm_tpu_engines",
]

