"""
DELTA task adapters.

Each adapter handles dataset loading, prompt formatting, and reward
computation for a specific DELTA benchmark family.
"""

from .manufactoria_adapter import ManufactoriaAdapter, ScoreMode

__all__ = [
    "ManufactoriaAdapter",
    "ScoreMode",
]

