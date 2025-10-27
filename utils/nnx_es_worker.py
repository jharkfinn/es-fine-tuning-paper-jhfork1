"""
NNX-native ES worker extension for vLLM-TPU unified backend.

This module is loaded INSIDE vLLM's driver worker process. All parameter
access and mutations happen where the JAX/Flax-NNX model actually lives.

Implements:
  - perturb_self_weights(seed, sigma_or_scale, negate=False)
  - restore_self_weights(seed, sigma)
  - dump_state_dict() -> PyTree (CPU numpy leaves)
  - load_state_dict(state_cpu)  (rebuilds device arrays w/ correct dtype & sharding)
  - save_self_weights_to_disk(path_prefix)  -> writes <path_prefix>.npz + .treedef

Determinism:
  * For each leaf, noise is drawn from a PRNGKey derived from (global seed, leaf-path hash).
    This is independent of traversal order, host scheduling, etc.
  * Floating leaves only (bf16/fp32) are perturbed; non-floating leaves are passed through.
  * Restore is just perturb(..., negate=True) with the same seed.

Caveats:
  * Subtracting bf16 noise may not be bit-exact (±1 ULP). Use dump/load for exact restore.
"""

import hashlib
import io
import os
from typing import Any, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jrand
except Exception as e:  # pragma: no cover
    raise RuntimeError("JAX is required in the vLLM-TPU worker environment") from e

# Prefer Flax serialization APIs (work across Linen & many NNX builds)
from flax import serialization as flax_ser

# Optional NNX helpers (used if available for state writeback)
try:
    from flax import nnx as fnnx  # modern path
except Exception:
    try:
        from flax.experimental import nnx as fnnx  # older experimental path
    except Exception:
        fnnx = None


# --------------------- small helpers ---------------------

def _is_array(x: Any) -> bool:
    return isinstance(x, (jnp.ndarray, np.ndarray)) or hasattr(x, "dtype") and hasattr(x, "shape")


def _is_float_array(x: Any) -> bool:
    if not _is_array(x):
        return False
    try:
        return jnp.issubdtype(x.dtype, jnp.inexact)
    except Exception:
        return False


def _stable_hash32(s: str) -> int:
    # 32-bit integer from sha256 of the path string
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)


def _tree_map_with_paths(tree, fn, path=""):
    """Apply fn(path, leaf) to array leaves; recurse containers preserving structure.

    We handle dict / list / tuple (including Flax FrozenDict serialized as dict).
    """
    # dict-like
    if isinstance(tree, dict):
        # sort keys for stability
        return {k: _tree_map_with_paths(tree[k], fn, f"{path}/{k}") for k in sorted(tree.keys())}
    # list
    if isinstance(tree, list):
        return [_tree_map_with_paths(v, fn, f"{path}[{i}]") for i, v in enumerate(tree)]
    # tuple
    if isinstance(tree, tuple):
        return tuple(_tree_map_with_paths(v, fn, f"{path}({i})") for i, v in enumerate(tree))
    # leaf
    return fn(path or "/", tree)


def _to_cpu_numpy(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x) if _is_array(x) else x, tree)


def _match_like(ref, src_np):
    """Return src as a JAX array with ref's dtype and sharding (when possible)."""
    if not _is_array(ref) or not isinstance(src_np, np.ndarray):
        return src_np
    arr = jnp.asarray(src_np, dtype=ref.dtype)
    # Try to preserve sharding of the reference leaf (for jax>=0.4 Arrays)
    try:
        sharding = getattr(ref, "sharding", None)
        if sharding is not None:
            arr = jax.device_put(arr, sharding)
            return arr
    except Exception:
        pass
    # Fallback: place on the device of 'ref' (single-device case)
    try:
        dev = getattr(ref, "device", None)
        if callable(dev):
            dev = ref.device()
        if dev is not None:
            arr = jax.device_put(arr, dev)
    except Exception:
        pass
    return arr


# --------------------- core class ---------------------

class WorkerExtension:
    """Loaded by vLLM worker. vLLM sets self.model_runner before use."""
    def __init__(self):
        self._state_mode = None  # "flax_ser" or "nnx_state"
        self._ref_state = None   # reference state tree for dtype/sharding on load

    # --------- state i/o ---------
    def _get_model(self):
        return self.model_runner.model  # vLLM worker API

    def _get_state(self):
        """Return (mode, state_tree) where state_tree has JAX arrays."""
        model = self._get_model()
        # Preferred: flax.serialization
        try:
            state = flax_ser.to_state_dict(model)
            self._state_mode = "flax_ser"
            return self._state_mode, state
        except Exception:
            pass
        # Fallback: NNX state
        if fnnx is not None:
            try:
                state = fnnx.state(model)
                self._state_mode = "nnx_state"
                return self._state_mode, state
            except Exception:
                pass
        raise RuntimeError("Unable to extract model state (flax.serialization and flax.nnx.state both unavailable)")

    def _set_state(self, mode: str, state_tree):
        model = self._get_model()
        if mode == "flax_ser":
            restored = flax_ser.from_state_dict(model, state_tree)
            if restored is not model:
                self.model_runner.model = restored
            return True
        elif mode == "nnx_state":
            if fnnx is None:
                raise RuntimeError("flax.nnx not available to set state")
            if hasattr(fnnx, "update"):
                fnnx.update(model, state_tree)
                return True
            if hasattr(fnnx, "merge"):
                new_model = fnnx.merge(state_tree)
                if new_model is not None:
                    self.model_runner.model = new_model
                    return True
            raise RuntimeError("Could not update NNX model state; need fnnx.update or fnnx.merge")
        else:
            raise ValueError(f"Unknown state mode {mode!r}")

    # --------- ES ops ---------
    def perturb_self_weights(self, seed: int, sigma_or_scale: float, negate: bool = False):
        # Lazy initialization (since __init__ may not be called when injected)
        if not hasattr(self, '_state_mode'):
            self._state_mode = None
        if not hasattr(self, '_ref_state'):
            self._ref_state = None

        mode, state = self._get_state()
        if self._ref_state is None:
            self._ref_state = state  # cache first seen state for dtype/sharding refs

        key0 = jrand.PRNGKey(int(seed))
        sign = -1.0 if negate else 1.0
        scale = float(sigma_or_scale)

        def add_noise(path, leaf):
            if not _is_float_array(leaf):
                return leaf
            # Derive per-leaf key from (seed, path-hash)
            k = jrand.fold_in(key0, _stable_hash32(path))
            noise = jrand.normal(k, shape=leaf.shape, dtype=leaf.dtype)
            # Try to preserve sharding like the leaf
            try:
                sharding = getattr(leaf, "sharding", None)
                if sharding is not None:
                    noise = jax.device_put(noise, sharding)
            except Exception:
                pass
            return leaf + sign * scale * noise

        new_state = _tree_map_with_paths(state, add_noise)
        self._set_state(mode, new_state)
        return True

    def restore_self_weights(self, seed: int, sigma: float):
        return self.perturb_self_weights(seed=int(seed), sigma_or_scale=float(sigma), negate=True)

    # --------- checkpoint / transfer ---------
    def dump_state_dict(self):
        """Return CPU numpy PyTree representing parameters (safe for Ray transport)."""
        _, state = self._get_state()
        return _to_cpu_numpy(state)

    def load_state_dict(self, state_cpu):
        """Load CPU numpy PyTree back into device arrays w/ correct dtype & sharding."""
        # Lazy initialization (since __init__ may not be called when injected)
        if not hasattr(self, '_state_mode'):
            self._state_mode = None
        if not hasattr(self, '_ref_state'):
            self._ref_state = None

        # Get current state as reference for structure and device placement
        mode, current_state = self._get_state()
        if self._ref_state is None:
            self._ref_state = current_state

        # Flatten both states and convert leaves pairwise
        flat_cpu, _ = jax.tree_util.tree_flatten(state_cpu)
        flat_ref, treedef_ref = jax.tree_util.tree_flatten(current_state)

        # Convert CPU numpy leaves to device arrays matching reference leaves
        def to_device_leaf(cpu_leaf, ref_leaf):
            if isinstance(cpu_leaf, np.ndarray) and _is_array(ref_leaf):
                return _match_like(ref_leaf, cpu_leaf)
            return cpu_leaf

        flat_device = [to_device_leaf(cpu, ref) for cpu, ref in zip(flat_cpu, flat_ref)]

        # Unflatten using the REFERENCE treedef (preserve current model structure)
        new_state = jax.tree_util.tree_unflatten(treedef_ref, flat_device)

        # Direct parameter update via NNX
        model = self._get_model()
        if fnnx is not None:
            # Try NNX update first
            if hasattr(fnnx, 'update'):
                try:
                    fnnx.update(model, new_state)
                    return True
                except Exception as e:
                    pass

            # Try direct GraphDef update for NNX models
            try:
                # Get the model's graphdef and directly update state
                graphdef, _ = fnnx.split(model)
                updated_model = fnnx.merge(graphdef, new_state)
                if updated_model is not None:
                    self.model_runner.model = updated_model
                    return True
            except Exception as e:
                pass

        # Last resort: direct in-place mutation using tree_map
        # This works by traversing both trees and copying values
        def copy_leaf(src, dst):
            if _is_array(src) and _is_array(dst):
                # Copy data from src to dst (in-place if possible)
                dst_updated = jnp.array(src, dtype=dst.dtype)
                return dst_updated
            return src

        jax.tree_util.tree_map(copy_leaf, new_state, current_state)

        # Force model state update
        self._set_state(mode, new_state)
        return True

    def save_self_weights_to_disk(self, path_prefix: str):
        """Save to <path_prefix>.npz and <path_prefix>.treedef (portable)."""
        # Ensure parent directory exists
        parent_dir = os.path.dirname(path_prefix)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        state_cpu = self.dump_state_dict()
        leaves, treedef = jax.tree_util.tree_flatten(state_cpu)
        np.savez_compressed(path_prefix + ".npz", *leaves)
        with open(path_prefix + ".treedef", "wb") as f:
            f.write(treedef.to_pickle())  # type: ignore[attr-defined]
        return True

    def get_tpu_stats(self):
        """Get TPU compute and memory utilization statistics."""
        stats = {}
        try:
            # Get memory stats from JAX devices
            devices = jax.devices()
            if devices:
                device = devices[0]
                if hasattr(device, 'memory_stats'):
                    mem_stats = device.memory_stats()
                    stats['memory'] = mem_stats
                # Try to get device utilization info
                if hasattr(jax.lib, 'xla_client'):
                    xla_client = jax.lib.xla_client
                    if hasattr(xla_client, 'get_device_memory_stats'):
                        stats['xla_memory'] = xla_client.get_device_memory_stats(device)
        except Exception as e:
            stats['error'] = str(e)
        return stats
