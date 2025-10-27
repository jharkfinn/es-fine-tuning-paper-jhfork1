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

        # Ensure ref tree present for dtype/sharding
        if self._ref_state is None:
            _, ref = self._get_state()
            self._ref_state = ref

        def to_like(path, src_np, ref_leaf):
            if isinstance(src_np, np.ndarray):
                return _match_like(ref_leaf, src_np)
            return src_np

        # zip-with structure: map both trees together
        def map2(ref_sub, src_sub, path=""):
            if isinstance(ref_sub, dict) and isinstance(src_sub, dict):
                return {k: map2(ref_sub[k], src_sub[k], f"{path}/{k}") for k in sorted(src_sub.keys())}
            if isinstance(ref_sub, list) and isinstance(src_sub, list):
                return [map2(r, s, f"{path}[{i}]") for i, (r, s) in enumerate(zip(ref_sub, src_sub))]
            if isinstance(ref_sub, tuple) and isinstance(src_sub, tuple):
                return tuple(map2(r, s, f"{path}({i})") for i, (r, s) in enumerate(zip(ref_sub, src_sub)))
            return to_like(path or "/", src_sub, ref_sub)

        mode = self._state_mode or self._get_state()[0]
        new_state = map2(self._ref_state, state_cpu)
        self._set_state(mode, new_state)
        return True

    def save_self_weights_to_disk(self, path_prefix: str):
        """Save to <path_prefix>.npz and <path_prefix>.treedef (portable)."""
        state_cpu = self.dump_state_dict()
        leaves, treedef = jax.tree_util.tree_flatten(state_cpu)
        np.savez_compressed(path_prefix + ".npz", *leaves)
        with open(path_prefix + ".treedef", "wb") as f:
            f.write(treedef.to_pickle())  # type: ignore[attr-defined]
        return True
