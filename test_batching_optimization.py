#!/usr/bin/env python3
"""
Test batching optimization to increase TPU utilization.
Compares serial vs batched inference throughput.
"""

import argparse
import os
import time
import ray
from transformers import AutoTokenizer

@ray.remote
class VllmTpuActorOptimized:
    def __init__(self, model_dir: str, dtype: str = "bfloat16",
                 max_num_batched_tokens: int = 8192, max_num_seqs: int = 64):
        from vllm import LLM, SamplingParams

        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        os.environ.setdefault("VLLM_DEVICE", "tpu")
        os.environ.setdefault("PJRT_DEVICE", "TPU")

        self._SamplingParams = SamplingParams

        print(f"Initializing vLLM with:")
        print(f"  max_num_batched_tokens={max_num_batched_tokens}")
        print(f"  max_num_seqs={max_num_seqs}")

        self._llm = LLM(
            model=model_dir,
            tensor_parallel_size=1,
            distributed_executor_backend="ray",
            dtype=dtype,
            enable_prefix_caching=False,
            enforce_eager=False,
            # Optimization: increase batching capacity
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=0.95,  # Use more HBM
        )

    def generate(self, prompts, temperature: float = 0.0, seed: int = 42, max_tokens: int = 1024):
        params = self._SamplingParams(temperature=temperature, seed=seed, max_tokens=max_tokens)
        t0 = time.time()
        outputs = self._llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.time() - t0

        # Return serializable output
        return {
            'outputs': [
                {
                    "prompt": output.prompt,
                    "outputs": [{"text": o.text, "token_ids": o.token_ids} for o in output.outputs],
                    "finished": output.finished,
                }
                for output in outputs
            ],
            'elapsed': elapsed,
            'num_prompts': len(prompts),
            'throughput': len(prompts) / elapsed if elapsed > 0 else 0,
        }

    def get_tpu_stats(self):
        return self._llm.collective_rpc("get_tpu_stats", args=())


def test_batching(model_path: str, num_problems: int = 50):
    """Test serial vs batched inference."""

    # Initialize Ray
    ray.init(address="local", ignore_reinit_error=True)

    # Create test prompts
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_prompt = "Solve this math problem: If you have 3 apples and get 2 more, how many do you have?"
    prompts = [test_prompt] * num_problems

    # Test configurations
    configs = [
        {"max_num_batched_tokens": 2048, "max_num_seqs": 8, "name": "Default (small batch)"},
        {"max_num_batched_tokens": 8192, "max_num_seqs": 32, "name": "Medium batch"},
        {"max_num_batched_tokens": 16384, "max_num_seqs": 64, "name": "Large batch"},
    ]

    results = []

    for config in configs:
        print(f"\n{'='*80}")
        print(f"Testing: {config['name']}")
        print(f"{'='*80}")

        # Create actor with this config
        actor = VllmTpuActorOptimized.options(
            runtime_env={
                "env_vars": {
                    "TPU_VISIBLE_CHIPS": "0",
                    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                    "VLLM_DEVICE": "tpu",
                    "PJRT_DEVICE": "TPU",
                },
                "pip": ["vllm-tpu", "transformers>=4.30.0", "numpy>=1.21.0"],
            }
        ).remote(
            model_dir=model_path,
            max_num_batched_tokens=config["max_num_batched_tokens"],
            max_num_seqs=config["max_num_seqs"],
        )

        # Warmup
        print("Warming up...")
        ray.get(actor.generate.remote(prompts[:5], max_tokens=512))

        # Benchmark
        print(f"Running inference on {num_problems} prompts...")
        result = ray.get(actor.generate.remote(prompts, max_tokens=512))

        print(f"\nResults:")
        print(f"  Total time: {result['elapsed']:.2f}s")
        print(f"  Throughput: {result['throughput']:.2f} prompts/sec")
        print(f"  Time per prompt: {result['elapsed']/result['num_prompts']*1000:.2f}ms")

        # Get TPU stats
        try:
            tpu_stats = ray.get(actor.get_tpu_stats.remote())
            if tpu_stats and isinstance(tpu_stats, list) and len(tpu_stats) > 0:
                stats = tpu_stats[0]
                if 'memory' in stats:
                    mem_gb = stats['memory'].get('bytes_in_use', 0) / 1e9
                    limit_gb = stats['memory'].get('bytes_limit', 1) / 1e9
                    util_pct = (mem_gb / limit_gb * 100) if limit_gb > 0 else 0
                    print(f"  HBM usage: {mem_gb:.2f} / {limit_gb:.2f} GiB ({util_pct:.1f}%)")
        except Exception as e:
            print(f"  Could not get TPU stats: {e}")

        results.append({
            'config': config['name'],
            'throughput': result['throughput'],
            'elapsed': result['elapsed'],
        })

        # Cleanup
        ray.kill(actor)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for r in results:
        print(f"{r['config']:30s}: {r['throughput']:6.2f} prompts/sec ({r['elapsed']:.2f}s total)")

    improvement = (results[-1]['throughput'] / results[0]['throughput'] - 1) * 100
    print(f"\nThroughput improvement: {improvement:.1f}%")

    ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_problems', type=int, default=50, help='Number of test prompts')
    args = parser.parse_args()

    test_batching(args.model_path, args.num_problems)
