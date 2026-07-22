from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class ExtractionMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    exact_match: bool
    valid_json: bool
    schema_valid: bool
    per_field: dict[str, dict[str, float | int | bool]]

    def to_dict(self) -> dict:
        return asdict(self)


def score_extraction(output: str, expected: Mapping[str, object]) -> ExtractionMetrics:
    """Score every JSON field/value atom against labeled expected output."""

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        expected_atoms = _atoms(expected)
        return ExtractionMetrics(
            true_positives=0,
            false_positives=0,
            false_negatives=len(expected_atoms),
            precision=0.0,
            recall=0.0,
            f1=0.0,
            exact_match=False,
            valid_json=False,
            schema_valid=False,
            per_field={},
        )

    expected_atoms = _atoms(expected)
    actual_atoms = _atoms(parsed)
    true_positives = len(expected_atoms & actual_atoms)
    false_positives = len(actual_atoms - expected_atoms)
    false_negatives = len(expected_atoms - actual_atoms)
    precision, recall, f1 = _prf(true_positives, false_positives, false_negatives)
    fields = sorted(set(expected) | set(parsed))
    return ExtractionMetrics(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
        exact_match=_canonical(parsed) == _canonical(dict(expected)),
        valid_json=True,
        schema_valid=_schema_valid(parsed, expected),
        per_field={
            field: _field_metrics(
                _atoms({field: parsed.get(field)}),
                _atoms({field: expected.get(field)}),
            )
            for field in fields
        },
    )


def aggregate_extraction(metrics: list[ExtractionMetrics]) -> dict[str, float | int] | None:
    if not metrics:
        return None
    true_positives = sum(item.true_positives for item in metrics)
    false_positives = sum(item.false_positives for item in metrics)
    false_negatives = sum(item.false_negatives for item in metrics)
    precision, recall, f1 = _prf(true_positives, false_positives, false_negatives)
    count = len(metrics)
    return {
        "examples": count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_rate": sum(item.exact_match for item in metrics) / count,
        "valid_json_rate": sum(item.valid_json for item in metrics) / count,
        "schema_valid_rate": sum(item.schema_valid for item in metrics) / count,
    }


def _atoms(value: Mapping[str, object]) -> set[str]:
    atoms: set[str] = set()
    for field, item in value.items():
        _flatten(atoms, field, item)
    return atoms


def _flatten(atoms: set[str], path: str, value: object) -> None:
    if isinstance(value, dict):
        if not value:
            atoms.add(f"{path}={{}}")
        for key, item in value.items():
            _flatten(atoms, f"{path}.{key}", item)
        return
    if isinstance(value, list):
        if not value:
            atoms.add(f"{path}=[]")
        for item in value:
            _flatten(atoms, path, item)
        return
    atoms.add(f"{path}={json.dumps(value, ensure_ascii=False, sort_keys=True)}")


def _field_metrics(actual: set[str], expected: set[str]) -> dict[str, float | int | bool]:
    true_positives = len(actual & expected)
    false_positives = len(actual - expected)
    false_negatives = len(expected - actual)
    precision, recall, f1 = _prf(true_positives, false_positives, false_negatives)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": actual == expected,
    }


def _prf(true_positives: int, false_positives: int, false_negatives: int) -> tuple[float, float, float]:
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _schema_valid(actual: Mapping[str, object], expected: Mapping[str, object]) -> bool:
    if set(actual) != set(expected):
        return False
    return all(_compatible_type(actual[key], expected[key]) for key in expected)


def _compatible_type(actual: object, expected: object) -> bool:
    if actual is None:
        return expected is None
    if isinstance(expected, bool):
        return isinstance(actual, bool)
    if isinstance(expected, (int, float)):
        return isinstance(actual, (int, float)) and not isinstance(actual, bool)
    if isinstance(expected, str):
        return isinstance(actual, str)
    if isinstance(expected, list):
        return isinstance(actual, list)
    if isinstance(expected, dict):
        return isinstance(actual, dict)
    return type(actual) is type(expected)
