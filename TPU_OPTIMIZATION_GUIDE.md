# TPU Utilization Optimization Guide

## Current Status
- **HBM Utilization**: 90.3% ✅ (Excellent)
- **Estimated Compute Utilization**: ~44% ⚠️ (Room for improvement)
- **Throughput**: ~11.7 problems/sec, ~85ms per problem

## Why Compute Utilization is Lower Than Memory

TPU compute (MXU units) can be underutilized even with high HBM usage because:
- **Small batch sizes** → Insufficient parallelism
- **Memory-bound operations** → Waiting on HBM bandwidth (KV cache reads during decode)
- **Model size** → 3B params may not saturate TPU v6e's 81 TFLOPS peak
- **Sequential decode** → Generate tokens one-by-one (autoregressive bottleneck)

## Optimization Strategies (Ranked by Impact)

### 1. ⭐ Tune vLLM Batching for Prefill/Decode Pipelining (IMPLEMENTED)
**Expected improvement**: 10-30% per-seed latency reduction

**Expert advice**: Match vLLM batch params to per-seed prompt count for better prefill/decode overlap

**Implementation** (es_fine-tuning_countdown_accl_vllm_tpu_jaxnnx.py:73-85):
```python
self._llm = LLM(
    model=model_dir,
    tensor_parallel_size=1,
    distributed_executor_backend="ray",
    worker_extension_cls="utils.nnx_es_worker.WorkerExtension",
    dtype=dtype,
    enable_prefix_caching=False,
    enforce_eager=False,
    max_num_seqs=200,                   # Match per-seed batch (200 prompts)
    max_num_batched_tokens=204800,      # 200 prompts × 1024 max_tokens
)
```

**Why this helps**:
- ✅ vLLM pipelines prefill vs decode internally when batch size matches KV budget
- ✅ Keeps both TPU cores busy during generation
- ✅ Reduces per-seed wall time through better scheduling

**Previous attempt (arbitrary large buffers) FAILED**:
- Setting max_num_seqs=64 and max_num_batched_tokens=8192 made things SLOWER (17s → 28s)
- Problem: Params didn't match actual workload, added overhead
- Solution: Size params to actual per-seed batch (200 prompts)

### 2. ❌ Multiple vLLM Engines on One Chip (NOT POSSIBLE)
**Why we can't do this**:
- ❌ Two processes cannot share one v6e-1 chip (JAX/TPU device exclusivity)
- ❌ Two vLLM engines inside one process is not a supported configuration
- ❌ Memory would be tight even if it worked (~27 GiB for 2 engines, only 31.25 GiB available)

**Expert clarification**: "Bottom line: two processes cannot share one v6e-1 chip, and two vLLM engines inside one process isn't a supported configuration either."

### 3. ⚠️ Use Multiple TPU Chips (Not Available on Single-Chip TPU)
**Expected improvement**: Near-linear scaling (2x with 2 chips, 4x with 4 chips)

**Limitation**: This machine has only 1 TPU chip (TPU v6e-1)

**If you had multiple chips**, you could:
1. Use tensor parallelism to spread model across chips (higher memory bandwidth)
2. Use multiple engines (one per chip) for parallel seed evaluation (rolling window)

**Current bottleneck**: Sequential seed evaluation on single chip

### 4. ⚡ Increase max_tokens (Medium Impact)
**Expected improvement**: 20-50% throughput

**Current**: `max_tokens=1024`

**Optimized**: `max_tokens=2048` (or higher if task allows)

```python
actor.generate.remote(prompts, max_tokens=2048, ...)
```

**Why it helps**:
- More work per forward pass
- Better amortization of prefill cost
- Longer sequences = more matmuls

### 5. 📊 Use Larger Model (Medium Impact)
**Expected improvement**: 30-80% compute utilization

**Current**: Qwen2.5-3B-Instruct

**Optimized**: Qwen2.5-7B-Instruct or Qwen2.5-14B-Instruct

```bash
python es_fine-tuning_countdown_accl_vllm_tpu_jaxnnx.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  ...
```

**Why it helps**:
- Larger matmuls → Better MXU saturation
- 7B params better matches TPU v6e capacity
- More compute per token

**Trade-off**: 2-3x slower per-problem latency (but better overall throughput with batching)

### 6. 🔧 Disable Prefix Caching (Low Impact, for testing only)
**Expected improvement**: Minor, mainly for measurement

```python
self._llm = LLM(
    model=model_dir,
    enable_prefix_caching=False,  # Already disabled
    ...
)
```

Already disabled in your code. Good!

### 7. 🎯 Enable KV Cache Chunked Prefill (Low-Medium Impact)
**Expected improvement**: 10-30% for long sequences

```python
self._llm = LLM(
    model=model_dir,
    enable_chunked_prefill=True,
    max_num_batched_tokens=8192,
    ...
)
```

**Why it helps**:
- Processes prefill in chunks → Better batching with mixed-length sequences
- Reduces bubble time in pipeline

## Quick Test: Batching Impact

Run the included test script to measure batching improvement:

```bash
# First, ensure you have a base model checkpoint
python test_batching_optimization.py \
  --model_path /path/to/model/checkpoint \
  --num_problems 50
```

Expected output:
```
Default (small batch):     10-15 prompts/sec
Medium batch:              25-40 prompts/sec  (2-3x improvement)
Large batch:               40-80 prompts/sec  (3-5x improvement)
```

## Recommended Configuration for Your Use Case

For ES fine-tuning with 200 problems × 30 seeds:

```python
# VllmTpuActor.__init__:
self._llm = LLM(
    model=model_dir,
    tensor_parallel_size=4,              # Use all 4 chips
    distributed_executor_backend="ray",
    worker_extension_cls="utils.nnx_es_worker.WorkerExtension",
    dtype="bfloat16",
    enable_prefix_caching=False,
    enforce_eager=False,
    # Optimize for throughput:
    max_num_batched_tokens=8192,        # Process more tokens in parallel
    max_num_seqs=64,                     # Allow more concurrent sequences
    gpu_memory_utilization=0.95,        # Use available HBM
)

# And in launch:
export TPU_VISIBLE_CHIPS=0,1,2,3
python es_fine-tuning_countdown_accl_vllm_tpu_jaxnnx.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \  # Larger model
  --num_engines 1 \
  --tpu_chips "0,1,2,3" \  # All chips
  --num_iterations 10 \
  --population_size 30 \
  --sigma 0.001 \
  --alpha 0.0005
```

**Expected throughput**: 40-80 problems/sec (3-7x improvement)

**Expected HBM usage**: 85-95% (with 7B model + TP=4)

**Expected compute utilization**: 65-85% (much better!)

## Monitoring Improvements

After optimization, check:

```python
# Throughput (problems/sec):
throughput = num_problems / elapsed_time

# Estimated compute utilization:
# For 7B model: ~12 GFLOPS/token (inference)
# TPU v6e: 81 TFLOPS peak per chip × 4 chips = 324 TFLOPS total
tokens_per_sec = throughput * avg_tokens_per_problem
effective_tflops = tokens_per_sec * 12 / 1000  # Convert GFLOPS → TFLOPS
compute_util = effective_tflops / 324 * 100  # Percentage
```

## Trade-offs to Consider

| Optimization | Throughput Gain | Latency Impact | Complexity |
|--------------|----------------|----------------|------------|
| Increase batch size | ⭐⭐⭐⭐⭐ | Minimal | Low |
| Use 4 chips (TP=4) | ⭐⭐⭐⭐⭐ | Minor (+5-10%) | Medium |
| Increase max_tokens | ⭐⭐⭐ | Higher (proportional) | Low |
| Larger model (7B) | ⭐⭐⭐⭐ | Higher (2-3x) | Low |
| Chunked prefill | ⭐⭐ | Minimal | Low |

## Next Steps

1. **Test batching** with current 3B model (quick win, no code changes needed)
2. **Add TP=4** to use all chips (medium effort, high reward)
3. **Switch to 7B model** if quality allows (easy, high impact)
4. **Re-run efficiency benchmark** and compare results

---

*Last updated: 2025-10-27*
