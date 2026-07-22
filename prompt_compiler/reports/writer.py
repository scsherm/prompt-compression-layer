from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from prompt_compiler.hashing import stable_json

if TYPE_CHECKING:
    from prompt_compiler.optimize.feedback_optimizer import FeedbackOptimizationResult
    from prompt_compiler.optimize.legacy_optimizer import CompressionRunResult

    RunResult: TypeAlias = FeedbackOptimizationResult | CompressionRunResult


def write_run_artifacts(output_dir: Path, result: "RunResult") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "best_prompt.txt").write_text(result.best_prompt_template, encoding="utf-8")
    (output_dir / "best_prompt_template.json").write_text(
        stable_json({"best_prompt_template": result.best_prompt_template}),
        encoding="utf-8",
    )
    (output_dir / "compression_report.json").write_text(
        stable_json(result.evaluation_report.to_dict()),
        encoding="utf-8",
    )
    _write_candidate_prompt_audit(output_dir / "candidate_prompts.jsonl", result)
    _write_frontier_csv(output_dir / "pareto_frontier.csv", result)
    _write_frontier_csv(output_dir / "dev_frontier.csv", result)
    _write_failures(output_dir / "failures.json", result)
    _write_jsonl(output_dir / "reference_dataset.jsonl", [item.to_dict() for item in result.reference_dataset])
    _write_jsonl(output_dir / "candidate_reports.jsonl", [report.to_dict() for report in result.all_reports])
    _write_jsonl(output_dir / "holdout_reports.jsonl", [report.to_dict() for report in result.holdout_reports])
    _write_jsonl(
        output_dir / "candidate_outputs.jsonl",
        [record.__dict__ for report in result.all_reports for record in report.output_records],
    )


def _write_frontier_csv(path: Path, result: "RunResult") -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "instruction_tokens",
                "token_reduction",
                "normalized_token_reduction",
                "avg_semantic_drift",
                "avg_normalized_semantic_drift",
                "objective_score",
                "format_failure_rate",
                "task_failure_rate",
                "extraction_precision",
                "extraction_recall",
                "extraction_f1",
                "extraction_f1_delta",
            ],
        )
        writer.writeheader()
        for report in result.pareto_frontier:
            writer.writerow(
                {
                    "candidate_id": report.candidate_id,
                    "instruction_tokens": report.instruction_tokens,
                    "token_reduction": report.token_reduction,
                    "normalized_token_reduction": report.normalized_token_reduction,
                    "avg_semantic_drift": report.avg_semantic_drift,
                    "avg_normalized_semantic_drift": report.avg_normalized_semantic_drift,
                    "objective_score": report.objective_score,
                    "format_failure_rate": report.format_failure_rate,
                    "task_failure_rate": report.task_failure_rate,
                    "extraction_precision": (
                        report.candidate_extraction.get("precision")
                        if report.candidate_extraction
                        else None
                    ),
                    "extraction_recall": (
                        report.candidate_extraction.get("recall")
                        if report.candidate_extraction
                        else None
                    ),
                    "extraction_f1": (
                        report.candidate_extraction.get("f1")
                        if report.candidate_extraction
                        else None
                    ),
                    "extraction_f1_delta": report.extraction_f1_delta,
                }
            )


def _write_failures(path: Path, result: "RunResult") -> None:
    failures = [failure.__dict__ for report in result.all_reports for failure in report.examples_failed]
    path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_candidate_prompt_audit(path: Path, result: "RunResult") -> None:
    seen: set[str] = set()
    rows: list[dict] = []
    for report in result.all_reports:
        if report.candidate_id in seen:
            continue
        seen.add(report.candidate_id)
        rows.append(
            {
                "candidate_id": report.candidate_id,
                "instruction_tokens": report.instruction_tokens,
                "token_reduction": report.token_reduction,
                "operator_summary": report.operator_summary,
                "prompt_template": report.prompt_template,
            }
        )
    _write_jsonl(path, rows)
