from __future__ import annotations

from prompt_compiler.eval.evaluator import CandidateReport


def compute_pareto_frontier(reports: list[CandidateReport]) -> list[CandidateReport]:
    frontier: list[CandidateReport] = []
    for report in reports:
        if any(_dominates(other, report) for other in reports if other is not report):
            continue
        frontier.append(report)
    return sorted(
        frontier,
        key=lambda item: (
            item.format_failure_rate,
            item.task_failure_rate,
            item.objective_score,
            -item.token_reduction,
        ),
    )


def _dominates(a: CandidateReport, b: CandidateReport) -> bool:
    better_or_equal = (
        a.token_reduction >= b.token_reduction
        and a.avg_semantic_drift <= b.avg_semantic_drift
    )
    strictly_better = (
        a.token_reduction > b.token_reduction
        or a.avg_semantic_drift < b.avg_semantic_drift
    )
    return better_or_equal and strictly_better
