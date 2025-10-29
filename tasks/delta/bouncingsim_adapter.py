"""
Placeholder for the BouncingSim adapter.

This file intentionally contains only a stub so that future work can plug
in the appropriate dataset loader and reward shaping logic without touching
the rest of the ES stack.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BouncingSimAdapter:
    """Stub adapter – implement when BouncingSim datasets are ready."""

    def __post_init__(self) -> None:  # pragma: no cover - placeholder
        raise NotImplementedError(
            "BouncingSimAdapter is not implemented yet. "
            "Add dataset loading and reward computation here."
        )


__all__ = ["BouncingSimAdapter"]

