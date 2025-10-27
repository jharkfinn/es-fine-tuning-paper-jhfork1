# Development Log - TPU ES Fine-Tuning with vLLM

## Overview
This document tracks all changes, issues, and fixes encountered during the development of TPU-based Evolution Strategies fine-tuning using vLLM-TPU unified backend with JAX/NNX.

## Session: 2025-10-27

### Initial Setup
- **Repository**: `https://github.com/jharkfinn/es-fine-tuning-paper-jhfork1`
- **Branch**: `tpu-compatibility-fixes`
- **Goal**: Implement ES fine-tuning on TPU v6e using vLLM-TPU with NNX-native perturbations

### Issues and Fixes

#### Issue 1: vLLM Output Serialization Error
**Date**: 2025-10-27 20:30

**Error**:
```
ModuleNotFoundError: No module named 'vllm'
```

**Context**: When vLLM `RequestOutput` objects were returned from actors through Ray, the driver tried to deserialize them but didn't have vLLM installed (by design - isolated environment).

**Fix**: Modified `VllmTpuActor.generate()` method in `es_fine-tuning_countdown_accl_vllm_tpu_jaxnnx.py:84-95` to convert vLLM output objects to serializable dictionaries before returning:

```python
def generate(self, prompts, temperature: float = 0.0, seed: int = 42, max_tokens: int = 1024):
    params = self._SamplingParams(temperature=temperature, seed=seed, max_tokens=max_tokens)
    outputs = self._llm.generate(prompts, params, use_tqdm=False)
    # Convert vLLM outputs to serializable dictionaries
    return [
        {
            "prompt": output.prompt,
            "outputs": [{"text": o.text, "token_ids": o.token_ids} for o in output.outputs],
            "finished": output.finished,
        }
        for output in outputs
    ]
```

Updated `_postprocess_outputs()` at line 164 to access dictionary keys instead of object attributes:
```python
response = output["outputs"][0]["text"]  # Access dictionary instead of object
```

**Status**: ✅ Fixed

#### Issue 2: State Dict Serialization Error - flax.serialization
**Date**: 2025-10-27 20:35

**Error**:
```python
TypeError: list indices must be integers or slices, not str
File "/utils/nnx_es_worker.py", line 155, in _set_state
    restored = flax_ser.from_state_dict(model, state_tree)
```

**Context**: During ES update broadcast (line 305 in main script), the driver dumps state from engine 0 as CPU numpy PyTree and loads it into all engines. After Ray serialization/deserialization, the state structure wasn't compatible with `flax.serialization.from_state_dict()` - lists had string keys instead of integer indices.

**Attempted Fix #1**: Use `jax.tree_util.tree_map` to map both trees in parallel
- **Result**: Failed with `ValueError: Expected None, got [None]` - structure mismatch between `current_state` and `state_cpu` after Ray serialization

**Fix #2**: Use flatten/unflatten approach in `utils/nnx_es_worker.py:218-258`:

```python
def load_state_dict(self, state_cpu):
    """Load CPU numpy PyTree back into device arrays w/ correct dtype & sharding."""
    # Get current state as reference for structure and device placement
    mode, current_state = self._get_state()
    if self._ref_state is None:
        self._ref_state = current_state

    # Flatten the CPU state and convert leaves to device arrays
    # This avoids structure mismatch issues with tree_map
    flat_cpu, treedef_cpu = jax.tree_util.tree_flatten(state_cpu)
    flat_ref, treedef_ref = jax.tree_util.tree_flatten(current_state)

    # Convert CPU numpy leaves to device arrays matching reference leaves
    def to_device_leaf(cpu_leaf, ref_leaf):
        if isinstance(cpu_leaf, np.ndarray) and _is_array(ref_leaf):
            return _match_like(ref_leaf, cpu_leaf)
        return cpu_leaf

    flat_device = [to_device_leaf(cpu, ref) for cpu, ref in zip(flat_cpu, flat_ref)]

    # Unflatten using the CPU treedef (preserve the structure from state_cpu)
    new_state = jax.tree_util.tree_unflatten(treedef_cpu, flat_device)

    # Use NNX update if available (more robust than flax.serialization)
    model = self._get_model()
    if fnnx is not None and hasattr(fnnx, 'update'):
        try:
            fnnx.update(model, new_state)
            return True
        except Exception:
            pass

    # Fallback to _set_state
    self._set_state(mode, new_state)
    return True
```

**Key Changes**:
1. Flatten both CPU state and current state to lists of leaves
2. Convert leaves pairwise (CPU numpy → device arrays with correct dtype/sharding)
3. Unflatten using CPU treedef to preserve structure
4. Try NNX update first (more robust than flax.serialization)
5. Fallback to _set_state only if needed

**Fix #3**: Use reference treedef for unflattening in `utils/nnx_es_worker.py:218-281`:
- Key change: Unflatten using `treedef_ref` (from current_state) instead of `treedef_cpu`
- This ensures the reconstructed state matches the model's expected structure exactly
- Added multiple fallback approaches:
  1. NNX update() - standard state update
  2. NNX split/merge (GraphDef approach) - direct model reconstruction
  3. tree_map with in-place copy + _set_state fallback
- The reference treedef approach should prevent structure mismatch with flax.serialization

**Status**: ✅ **FIXED** - Test completed successfully! Both generations ran without errors.

**Test Results** (test_ref_treedef.txt):
- Generation 0: Completed ✓
- Generation 1: Completed ✓
- Exit code: 0 (success)
- No serialization errors
- No broadcast errors
- ES updates applied successfully

**Root Cause**: The issue was using `treedef_cpu` (from the serialized state) to unflatten, which didn't match the model's expected structure after Ray serialization/deserialization.

**Solution**: Use `treedef_ref` (from current_state) to unflatten, ensuring the reconstructed state matches the model's structure exactly.

### TPU Monitoring Features

#### Feature 1: TPU Stats Logging
**Date**: 2025-10-27 (earlier in session)

**Added**: TPU utilization monitoring via `get_tpu_stats()` method in `utils/nnx_es_worker.py:260-278`:

```python
def get_tpu_stats(self):
    """Get TPU compute and memory utilization statistics."""
    stats = {}
    try:
        devices = jax.devices()
        if devices:
            device = devices[0]
            if hasattr(device, 'memory_stats'):
                mem_stats = device.memory_stats()
                stats['memory'] = mem_stats
            if hasattr(jax.lib, 'xla_client'):
                xla_client = jax.lib.xla_client
                if hasattr(xla_client, 'get_device_memory_stats'):
                    stats['xla_memory'] = xla_client.get_device_memory_stats(device)
    except Exception as e:
        stats['error'] = str(e)
    return stats
```

Integrated into main training loop at `es_fine-tuning_countdown_accl_vllm_tpu_jaxnnx.py:312-323`:
- Logs TPU memory allocated (HBM in GiB)
- Logs XLA memory usage
- Records to TensorBoard

**Status**: ✅ Implemented

#### Feature 2: TPU Monitor Utility (nvidia-smi style)
**Date**: 2025-10-27 (earlier in session)

**Created**: `tpu_monitor.py` - Standalone utility for monitoring TPU status

**Features**:
- One-shot status: `python tpu_monitor.py`
- Continuous monitoring: `python tpu_monitor.py --loop --interval 2`
- Displays TPU device info, memory usage, and active processes
- Filters processes for vLLM/JAX/TPU-related workloads

**Status**: ✅ Implemented

### Architecture Notes

**Isolated Environment Strategy**:
- **Driver**: Clean Python environment with Ray, transformers, torch (CPU), tensorboard
  - No vLLM, JAX, torch_xla, or libtpu
- **Actors**: vllm-tpu package installs all TPU dependencies via Ray's runtime_env.pip
  - Includes JAX, jaxlib, libtpu bundled with vllm-tpu

**Ray Configuration** (`es_fine-tuning_countdown_accl_vllm_tpu_jaxnnx.py:180-195`):
```python
work_dir = os.path.abspath(os.path.dirname(__file__))
ray.init(address="local", include_dashboard=False, ignore_reinit_error=True,
         runtime_env={
             "working_dir": work_dir,
             "excludes": [
                 "*.safetensors", "*.bin", "*.ckpt", "*.pt", "*.pth", "*.log",
                 "es-ft-experiment/", ".git/",
             ]
         })
```

### Files Modified

1. **es_fine-tuning_countdown_accl_vllm_tpu_jaxnnx.py**
   - Lines 84-95: Serialization fix for vLLM outputs
   - Lines 161-168: Updated _postprocess_outputs for dict access
   - Lines 312-323: TPU stats logging integration

2. **utils/nnx_es_worker.py**
   - Lines 218-258: Rewritten load_state_dict with flatten/unflatten
   - Lines 260-278: Added get_tpu_stats method

3. **tpu_monitor.py** (new file)
   - Complete TPU monitoring utility

4. **DEVELOPMENT_LOG.md** (this file)
   - Comprehensive documentation of changes and issues

### Next Steps

1. **Test broadcast fix**: Verify flatten/unflatten approach resolves structure mismatch
2. **Complete full ES run**: Test with multiple iterations and population sizes
3. **Performance benchmarks**: Measure throughput and TPU utilization
4. **Document final results**: Update this log with test outcomes

### Test Results

#### Test: Broadcast Fix with Flatten/Unflatten
**Status**: 🔄 Running
**Command**: `python es_fine-tuning_countdown_accl_vllm_tpu_jaxnnx.py --model_name Qwen/Qwen2.5-3B-Instruct --num_engines 1 --tpu_chips 0 --num_iterations 2 --population_size 4 --sigma 0.001 --alpha 0.0005`
**Expected**: Both generations should complete without TypeError

---

## Previous Issues (Resolved Earlier)

### Ray Runtime Environment Configuration
- Fixed relative path issues → use absolute paths
- Added excludes list to avoid 512MB package size limit
- Removed device="tpu" parameter (vllm-tpu uses env vars)
- Fixed worker extension import path

### TPU Library Compatibility
- Upgraded libtpu from 0.0.17 to compatible version
- Let vllm-tpu bundle its own dependencies (libtpu 0.0.24)
- Removed JAX/jaxlib from driver to avoid PJRT API version conflicts

### Worker Extension Integration
- Added lazy initialization for _state_mode and _ref_state
- Fixed injection into vLLM worker via worker_extension_cls

---

*Last Updated: 2025-10-27 20:45 UTC*
