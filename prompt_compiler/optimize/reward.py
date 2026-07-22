from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import fmean, stdev
from typing import Any


@dataclass(frozen=True)
class RewardEstimate:
    """Mean losses and uncertainty observed for one candidate.

    Uncertainty is the standard error of the mean. It is ``None`` until at
    least two observations exist, because a single rollout cannot estimate
    its own noise.
    """

    candidate_id: str
    observations: int
    mean_behavior_loss: float
    behavior_uncertainty: float | None
    mean_objective: float | None
    objective_uncertainty: float | None


@dataclass(frozen=True)
class BaselineNoise:
    """Natural output distance observed when the original prompt is repeated."""

    observations: int
    mean_distance: float
    standard_deviation: float
    standard_error: float | None


def aggregate_observations(reports: Iterable[Any]) -> dict[str, RewardEstimate]:
    """Aggregate repeated CandidateReport-like objects by candidate id.

    Objects may expose attributes or mapping keys. Behavior loss is read from
    ``behavior_loss``, ``avg_normalized_semantic_drift``, or
    ``avg_semantic_drift`` (in that order). Objective is optional and is read
    from ``objective``, ``objective_score``, or ``loss``.
    """

    grouped: dict[str, list[Any]] = {}
    for report in reports:
        candidate_id = str(_field(report, "candidate_id", "id"))
        grouped.setdefault(candidate_id, []).append(report)

    estimates: dict[str, RewardEstimate] = {}
    for candidate_id, candidate_reports in grouped.items():
        behavior_losses = [
            float(
                _field(
                    report,
                    "behavior_loss",
                    "avg_normalized_semantic_drift",
                    "avg_semantic_drift",
                )
            )
            for report in candidate_reports
        ]
        objectives = [
            float(value)
            for report in candidate_reports
            if (value := _field(report, "objective", "objective_score", "loss", default=None)) is not None
        ]
        behavior_mean, behavior_uncertainty = _mean_and_uncertainty(behavior_losses)
        objective_mean, objective_uncertainty = _mean_and_uncertainty(objectives) if objectives else (None, None)
        estimates[candidate_id] = RewardEstimate(
            candidate_id=candidate_id,
            observations=len(candidate_reports),
            mean_behavior_loss=behavior_mean,
            behavior_uncertainty=behavior_uncertainty,
            mean_objective=objective_mean,
            objective_uncertainty=objective_uncertainty,
        )
    return estimates


def select_for_additional_rollouts(
    estimates: Iterable[RewardEstimate],
    *,
    metric: str = "objective",
    close_within: float = 0.0,
    baseline_noise: float = 0.0,
    confidence_multiplier: float = 1.96,
    limit: int | None = None,
) -> list[str]:
    """Return candidates whose loss is close to, or may overlap, the best.

    Lower values are better. ``baseline_noise`` is the per-rollout standard
    deviation estimated from repeated original-prompt completions. It prevents
    one apparently strong completion from being treated as noise-free.
    """

    candidates = list(estimates)
    if len(candidates) < 2:
        return []
    if metric not in {"objective", "behavior"}:
        raise ValueError("metric must be 'objective' or 'behavior'")

    scored = [(_score(estimate, metric), estimate) for estimate in candidates]
    scored = [(score, estimate) for score, estimate in scored if score is not None]
    if not scored:
        return []

    scored.sort(key=lambda item: (item[0], item[1].candidate_id))
    if len(scored) < 2:
        return []

    best_score, best = scored[0]
    best_radius = confidence_multiplier * _effective_uncertainty(best, metric, baseline_noise)
    best_upper_bound = best_score + best_radius

    promoted: list[tuple[float, float, str]] = []
    for score, estimate in scored:
        radius = confidence_multiplier * _effective_uncertainty(estimate, metric, baseline_noise)
        gap = score - best_score
        close = gap <= close_within and estimate.observations < 2
        plausibly_best = score - radius <= best_upper_bound + close_within
        uncertain = radius > 0.0 and plausibly_best
        if close or uncertain:
            promoted.append((max(0.0, gap - radius), score, estimate.candidate_id))

    promoted.sort()
    candidate_ids = [candidate_id for _, _, candidate_id in promoted]
    return candidate_ids[:limit] if limit is not None else candidate_ids


def estimate_baseline_noise(distances: Iterable[float]) -> BaselineNoise:
    """Summarize distances among repeated completions of the original prompt."""

    values = [float(distance) for distance in distances]
    if not values:
        raise ValueError("at least one baseline distance is required")
    mean_distance, standard_error = _mean_and_uncertainty(values)
    return BaselineNoise(
        observations=len(values),
        mean_distance=mean_distance,
        standard_deviation=stdev(values) if len(values) > 1 else 0.0,
        standard_error=standard_error,
    )


def deployment_utility(
    *,
    tokens_saved: float,
    expected_reuse_volume: float,
    behavior_loss: float,
    behavior_penalty: float,
) -> float:
    """Value recurring token savings against the cost of behavioral drift."""

    return (tokens_saved * expected_reuse_volume) - (behavior_loss * behavior_penalty)


def _mean_and_uncertainty(values: list[float]) -> tuple[float, float | None]:
    average = fmean(values)
    if len(values) < 2:
        return average, None
    return average, stdev(values) / math.sqrt(len(values))


def _score(estimate: RewardEstimate, metric: str) -> float | None:
    return estimate.mean_objective if metric == "objective" else estimate.mean_behavior_loss


def _effective_uncertainty(estimate: RewardEstimate, metric: str, baseline_noise: float) -> float:
    observed = estimate.objective_uncertainty if metric == "objective" else estimate.behavior_uncertainty
    baseline_standard_error = baseline_noise / math.sqrt(max(estimate.observations, 1))
    return max(observed or 0.0, baseline_standard_error)


def _field(value: Any, *names: str, default: Any = ...) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    if default is not ...:
        return default
    joined = ", ".join(names)
    raise AttributeError(f"expected one of these fields: {joined}")
