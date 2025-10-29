from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - informative error path
    raise ImportError(
        "The `datasets` package is required to load DELTA datasets. "
        "Install it with `pip install datasets`."
    ) from exc

from .common import (
    DeltaExample,
    ManufactoriaGrader,
    extract_manufactoria_program,
)

logger = logging.getLogger(__name__)


class ScoreMode(str, Enum):
    PASS_RATE = "pass_rate"
    FULL_PASS = "full_pass"


def _summarize_results(case_results: Sequence[Dict[str, Any]], limit: int = 5):
    if not case_results:
        return []
    return list(case_results[:limit])


@dataclass
class ManufactoriaAdapter:
    """Adapter that exposes Manufactoria datasets to the ES trainer."""

    dataset_repo: str
    split: str = "train"
    score_mode: ScoreMode = ScoreMode.PASS_RATE
    max_cases: int = 40
    max_runtime_seconds: float = 5.0
    load_dataset_kwargs: Optional[Dict[str, Any]] = None
    custom_repo_path: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.score_mode, str):
            self.score_mode = ScoreMode(self.score_mode)
        self._dataset = None
        self._examples: Optional[List[DeltaExample]] = None
        self.grader = ManufactoriaGrader(
            max_cases=self.max_cases,
            max_runtime_seconds=self.max_runtime_seconds,
            custom_repo_path=self._resolve_custom_repo_path(),
        )

    def _resolve_custom_repo_path(self) -> Optional[Path]:
        if self.custom_repo_path is None:
            return None
        return (
            self.custom_repo_path
            if isinstance(self.custom_repo_path, Path)
            else Path(self.custom_repo_path)
        )

    @property
    def dataset(self):
        if self._dataset is None:
            kwargs = dict(self.load_dataset_kwargs or {})
            split = kwargs.pop("split", self.split)
            logger.info(
                "Loading Manufactoria dataset %s split=%s with kwargs=%s",
                self.dataset_repo,
                split,
                kwargs,
            )
            self._dataset = load_dataset(self.dataset_repo, split=split, **kwargs)
        return self._dataset

    def _convert_row(self, row: Dict[str, Any]) -> DeltaExample:
        messages = row.get("messages") or []
        ground_truth = row.get("ground_truth") or []
        metadata = {
            key: row[key]
            for key in row.keys()
            if key not in {"messages", "ground_truth"}
        }
        return DeltaExample(
            id=metadata.pop("id", None),
            messages=list(messages),
            ground_truth=list(ground_truth),
            metadata=metadata,
        )

    def load_examples(self, limit: Optional[int] = None) -> List[DeltaExample]:
        if self._examples is None:
            built = [self._convert_row(row) for row in self.dataset]
            self._examples = built
        if limit is not None:
            return self._examples[:limit]
        return list(self._examples)

    def iter_messages(self, examples: Iterable[DeltaExample]):
        for example in examples:
            yield example.messages

    def format_messages_as_prompt(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        assistant_prefix: str = "",
        separator: str = "\n\n",
    ) -> str:
        """Render chat-style messages into a single string prompt."""
        rendered: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            rendered.append(f"[{role.upper()}]\n{content.strip()}")
        if assistant_prefix:
            rendered.append(assistant_prefix)
        return separator.join(rendered)

    def score_response(
        self, response: str, example: DeltaExample
    ) -> Dict[str, Any]:
        program = extract_manufactoria_program(response or "")
        grade = self.grader.grade_program(program or "", example.ground_truth)

        if "error" in grade:
            reward_value = 0.0
        elif self.score_mode == ScoreMode.PASS_RATE:
            reward_value = float(grade.get("pass_rate", 0.0))
        else:
            reward_value = 1.0 if grade.get("all_passed") else 0.0

        reward_info = {
            "score_mode": self.score_mode.value,
            "pass_rate": grade.get("pass_rate", 0.0),
            "all_passed": grade.get("all_passed", False),
            "timed_out": grade.get("timed_out", False),
            "evaluated_cases": grade.get("evaluated_cases", 0),
            "total_cases": grade.get("total_cases", 0),
            "program_found": bool(program),
            "error": grade.get("error"),
            "message": grade.get("message"),
            "case_results_preview": _summarize_results(
                grade.get("case_results", [])
            ),
        }

        return {"reward": reward_value, "reward_info": reward_info}

    def set_score_mode(self, score_mode: ScoreMode | str) -> None:
        self.score_mode = (
            score_mode if isinstance(score_mode, ScoreMode) else ScoreMode(score_mode)
        )


__all__ = ["ManufactoriaAdapter", "ScoreMode"]

