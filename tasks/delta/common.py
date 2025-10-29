from __future__ import annotations

import importlib
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

CODE_BLOCK_PATTERN = re.compile(
    r"```(?:manufactoria)?\s*(?P<code>[\s\S]*?)```", re.IGNORECASE
)
REGEX_META_CHARS = frozenset(".+*?|()[]{}^$\\")


@dataclass
class DeltaExample:
    """In-memory representation of a DELTA dataset row."""

    id: Optional[str]
    messages: List[Dict[str, Any]]
    ground_truth: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _maybe_add_repo_to_sys_path(repo_path: Path) -> None:
    if not repo_path.exists():
        return

    repo_str = str(repo_path.resolve())
    if repo_str not in sys.path:
        sys.path.append(repo_str)
        logger.debug("Added %s to sys.path for manufactoria imports.", repo_str)


def import_manufactoria_parser(custom_repo_path: Optional[Path] = None):
    """Import the manufactoria verifier module, adding repo path if needed."""
    module_name = "manufactoria.verifier.manufactoria_parser"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pass

    candidate_paths: List[Path] = []

    # Respect explicit overrides first.
    if custom_repo_path:
        candidate_paths.append(Path(custom_repo_path))

    env_override = os.getenv("MANUFACTORIA_REPO_PATH")
    if env_override:
        candidate_paths.append(Path(env_override))

    # Default heuristic: sibling checkout of rl-grok-recipe.
    repo_root = Path(__file__).resolve().parents[2]
    candidate_paths.append(repo_root.parent / "rl-grok-recipe")

    for path in candidate_paths:
        _maybe_add_repo_to_sys_path(path)
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            logger.debug("Failed to import %s after adding %s", module_name, path)
            continue

    raise ModuleNotFoundError(
        f"Could not import {module_name}. "
        "Set MANUFACTORIA_REPO_PATH to the rl-grok-recipe checkout."
    )


def extract_manufactoria_program(text: str) -> Optional[str]:
    """Extract the first manufactoria fenced code block from the model output."""
    if not text:
        return None
    match = CODE_BLOCK_PATTERN.search(text)
    if not match:
        return None
    program = match.group("code").strip()
    return program if program else None


def _looks_like_regex(pattern: str) -> bool:
    return any(ch in REGEX_META_CHARS for ch in pattern)


def _normalize_test_case(raw: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "input": str(raw.get("input", "")),
        "expected_output": str(raw.get("expected_output", "")),
        "expected_accepted": bool(raw.get("expected_accepted", True)),
        "check_output": bool(raw.get("check_output", False)),
        "description": str(raw.get("description", "")),
    }


class ManufactoriaGrader:
    """Grades Manufactoria solutions using the verifier from rl-grok-recipe."""

    def __init__(
        self,
        *,
        max_cases: int = 40,
        max_runtime_seconds: float = 5.0,
        custom_repo_path: Optional[Path] = None,
    ) -> None:
        self.max_cases = max_cases
        self.max_runtime_seconds = max_runtime_seconds
        parser_module = import_manufactoria_parser(custom_repo_path)
        self._create_robot_factory = parser_module.create_robot_factory

    def grade_program(
        self, program: str, raw_test_cases: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not program:
            return {
                "error": "missing_program",
                "message": "No Manufactoria code block found.",
                "pass_rate": 0.0,
                "all_passed": False,
                "case_results": [],
                "evaluated_cases": 0,
                "timed_out": False,
            }

        normalized_cases = [
            _normalize_test_case(tc) for tc in raw_test_cases[: self.max_cases]
        ]
        if not normalized_cases:
            return {
                "pass_rate": 1.0,
                "all_passed": True,
                "case_results": [],
                "evaluated_cases": 0,
                "timed_out": False,
            }

        try:
            factory = self._create_robot_factory(program)
        except Exception as exc:  # noqa: BLE001 - propagate parser feedback
            return {
                "error": "parse_error",
                "message": str(exc),
                "pass_rate": 0.0,
                "all_passed": False,
                "case_results": [],
                "evaluated_cases": 0,
                "timed_out": False,
            }

        start_time = time.perf_counter()
        case_results: List[Dict[str, Any]] = []
        passed = 0
        timed_out = False

        for test_case in normalized_cases:
            elapsed = time.perf_counter() - start_time
            if self.max_runtime_seconds and elapsed > self.max_runtime_seconds:
                timed_out = True
                break

            try:
                execution = factory.process_robot(test_case["input"])
            except Exception as exc:  # noqa: BLE001 - capture runtime issues
                case_results.append(
                    {
                        "input": test_case["input"],
                        "passed": False,
                        "error": str(exc),
                    }
                )
                continue

            if test_case["check_output"]:
                expected_output = test_case["expected_output"]
                actual_output = execution.final_tape
                output_matches = False
                if expected_output:
                    if _looks_like_regex(expected_output):
                        try:
                            output_matches = bool(
                                re.fullmatch(expected_output, actual_output)
                            )
                        except re.error:
                            output_matches = actual_output == expected_output
                    else:
                        output_matches = actual_output == expected_output
                else:
                    output_matches = actual_output == expected_output
                passed_case = (
                    output_matches and execution.finished
                ) == test_case["expected_accepted"]
            else:
                passed_case = execution.finished == test_case["expected_accepted"]

            case_results.append(
                {
                    "input": test_case["input"],
                    "expected_output": test_case["expected_output"],
                    "actual_output": execution.final_tape,
                    "expected_accepted": test_case["expected_accepted"],
                    "actual_accepted": execution.finished,
                    "check_output": test_case["check_output"],
                    "passed": bool(passed_case),
                    "path": getattr(execution, "path", []),
                    "rejection_reason": getattr(execution, "rejection_reason", None),
                    "description": test_case["description"],
                }
            )

            if passed_case:
                passed += 1

        evaluated_cases = len(case_results)
        pass_rate = passed / evaluated_cases if evaluated_cases else 0.0
        all_passed = evaluated_cases > 0 and passed == evaluated_cases and not timed_out

        return {
            "pass_rate": pass_rate,
            "all_passed": all_passed,
            "case_results": case_results,
            "evaluated_cases": evaluated_cases,
            "total_cases": len(normalized_cases),
            "timed_out": timed_out,
        }


__all__ = [
    "DeltaExample",
    "ManufactoriaGrader",
    "extract_manufactoria_program",
    "import_manufactoria_parser",
]

