from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from difflib import unified_diff
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence

from prompt_compiler.candidates.prompt_candidate import PromptCandidate
from prompt_compiler.data.dataset_builder import (
    InputExample,
    ReferenceExample,
    build_reference_dataset,
    normalize_inputs,
)
from prompt_compiler.data.splits import split_dataset
from prompt_compiler.eval.contract_checks import OutputContract
from prompt_compiler.eval.embedding_distance import DriftScorer
from prompt_compiler.eval.evaluator import CandidateOutputRecord, CandidateReport, EvaluationWeights, Evaluator
from prompt_compiler.eval.extraction_metrics import score_extraction
from prompt_compiler.models.base import GenerateParams, ModelClient
from prompt_compiler.observability import RunLogger
from prompt_compiler.operators.full_prompt_proposer import (
    CandidateFeedback,
    ExampleResidual,
    PromptProposal,
    PromptProposer,
    ProposalContext,
    RoundSummary,
)
from prompt_compiler.optimize.reward import (
    BaselineNoise,
    RewardEstimate,
    aggregate_observations,
    deployment_utility,
    estimate_baseline_noise,
    select_for_additional_rollouts,
)
from prompt_compiler.optimize.search_state import ExampleFeedback, SearchArchive, Trial
from prompt_compiler.prompt.chunk import PLACEHOLDER_RE
from prompt_compiler.prompt.template import PromptTemplate
from prompt_compiler.reports.writer import write_run_artifacts
from prompt_compiler.tokenizer import ApproxTokenizer, Tokenizer


@dataclass(frozen=True)
class FeedbackEvaluationReport:
    original_instruction_tokens: int
    best_instruction_tokens: int
    token_reduction: float
    semantic_drift_normalization: float
    dev_semantic_drift: float
    dev_normalized_semantic_drift: float
    dev_behavior_loss: float
    dev_format_failure_rate: float
    dev_task_failure_rate: float
    dev_reference_extraction: dict | None
    dev_candidate_extraction: dict | None
    dev_extraction_f1_delta: float | None
    holdout_semantic_drift: float | None
    holdout_normalized_semantic_drift: float | None
    holdout_behavior_loss: float | None
    holdout_format_failure_rate: float | None
    holdout_task_failure_rate: float | None
    holdout_reference_extraction: dict | None
    holdout_candidate_extraction: dict | None
    holdout_extraction_f1_delta: float | None
    train_size: int
    dev_size: int
    holdout_size: int
    rounds_completed: int
    candidates_evaluated: int
    unique_candidates_evaluated: int
    best_candidate_id: str
    convergence_reason: str
    baseline_noise: BaselineNoise | None
    frontier_history: tuple[tuple[int, float], ...]
    selection_behavior_penalty: float
    usage_summary: dict[str, int]
    proposer_usage_summary: dict[str, int]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["validation_semantic_drift"] = self.dev_semantic_drift
        data["validation_normalized_semantic_drift"] = self.dev_normalized_semantic_drift
        data["validation_behavior_loss"] = self.dev_behavior_loss
        data["format_failure_rate"] = self.dev_format_failure_rate
        data["task_failure_rate"] = self.dev_task_failure_rate
        return data


@dataclass(frozen=True)
class FeedbackOptimizationResult:
    best_prompt_template: str
    pareto_frontier: list[CandidateReport]
    dev_frontier: list[CandidateReport]
    holdout_reports: list[CandidateReport]
    evaluation_report: FeedbackEvaluationReport
    reference_dataset: list[ReferenceExample]
    all_reports: list[CandidateReport]
    search_archive: SearchArchive


def optimize_prompt(
    *,
    target_model: ModelClient,
    prompt_proposer: PromptProposer,
    original_prompt: PromptTemplate,
    inputs: Iterable[InputExample | Mapping[str, object] | str],
    output_dir: Path,
    rounds: int = 8,
    batch_size: int = 8,
    convergence_patience: int = 3,
    min_frontier_improvement: float = 1e-4,
    parent_limit: int | None = None,
    recent_limit: int | None = None,
    worst_example_limit: int | None = None,
    repeat_top_k: int | None = None,
    max_candidate_rollouts: int = 2,
    repeat_close_within: float = 0.02,
    baseline_repeats: int = 2,
    feedback_enabled: bool = True,
    selection_behavior_penalty: float = 1.0,
    tokenizer: Tokenizer | None = None,
    output_contract: OutputContract | None = None,
    evaluation_weights: EvaluationWeights | None = None,
    drift_scorer: DriftScorer | None = None,
    params: GenerateParams | None = None,
    max_concurrency: int = 1,
    log_to_stderr: bool = True,
    live_log_path: Path | None = None,
) -> FeedbackOptimizationResult:
    """Optimize a reusable instruction prompt through measured full-prompt feedback.

    The original prompt supplies behavioral references but is never a search
    candidate. Each round evaluates complete compressed prompts on the same
    examples, then conditions the next proposal batch on the observed frontier
    and the most informative completion residuals.
    """

    tokenizer = tokenizer or ApproxTokenizer()
    params = params or GenerateParams()
    weights = evaluation_weights or EvaluationWeights()
    output_contract = output_contract or OutputContract()
    if rounds < 1:
        raise ValueError("rounds must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_candidate_rollouts < 1:
        raise ValueError("max_candidate_rollouts must be at least 1")
    if selection_behavior_penalty < 0:
        raise ValueError("selection_behavior_penalty must be non-negative")

    normalized_inputs = normalize_inputs(inputs)
    if not normalized_inputs:
        raise ValueError("at least one input example is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir / "run_events.jsonl", echo=log_to_stderr, mirror_path=live_log_path)
    evaluator = Evaluator(
        tokenizer=tokenizer,
        output_contract=output_contract,
        weights=weights,
        drift_scorer=drift_scorer,
    )
    original_instruction_tokens = tokenizer.count(original_prompt.instruction_text())
    if original_instruction_tokens == 0:
        raise ValueError("the original prompt has no instruction tokens to reduce")

    logger.event(
        "run_start",
        optimizer="feedback_full_prompt",
        model=target_model.config(),
        input_count=len(normalized_inputs),
        rounds=rounds,
        batch_size=batch_size,
        convergence_patience=convergence_patience,
        min_frontier_improvement=min_frontier_improvement,
        max_candidate_rollouts=max_candidate_rollouts,
        feedback_enabled=feedback_enabled,
        selection_behavior_penalty=selection_behavior_penalty,
    )
    logger.event("reference_build_start", example_count=len(normalized_inputs))
    references = build_reference_dataset(
        model=target_model,
        prompt=original_prompt,
        inputs=normalized_inputs,
        tokenizer=tokenizer,
        params=params,
    )
    logger.event("reference_build_done", reference_count=len(references))
    split = split_dataset(references)
    search_set = split.train
    dev_set = split.validation or split.train
    holdout_set = split.test
    logger.event(
        "dataset_split",
        train_size=len(search_set),
        dev_size=len(dev_set),
        holdout_size=len(holdout_set),
    )

    baseline_noise, baseline_usage = _measure_baseline_noise(
        target_model=target_model,
        references=search_set,
        evaluator=evaluator,
        params=params,
        repeats=max(0, baseline_repeats),
    )
    logger.event(
        "baseline_noise_measured",
        observations=baseline_noise.observations if baseline_noise else 0,
        mean_distance=baseline_noise.mean_distance if baseline_noise else None,
        standard_deviation=baseline_noise.standard_deviation if baseline_noise else None,
    )

    archive = SearchArchive(
        original_prompt=original_prompt.text,
        original_instruction_tokens=original_instruction_tokens,
    )
    candidates: dict[str, PromptCandidate] = {}
    reports_by_candidate: dict[str, list[CandidateReport]] = {}
    round_history: list[RoundSummary] = []
    all_reports: list[CandidateReport] = []
    convergence_reason = f"round budget reached ({rounds})"

    for round_index in range(rounds):
        context = (
            _proposal_context(
                archive=archive,
                reports_by_candidate=reports_by_candidate,
                original_prompt=original_prompt.text,
                target_model_name=target_model.name,
                target_tokenizer_name=tokenizer.name,
                round_index=round_index,
                round_history=round_history,
                parent_limit=parent_limit,
                recent_limit=recent_limit,
                worst_example_limit=worst_example_limit,
                baseline_noise=baseline_noise,
            )
            if feedback_enabled
            else ProposalContext(
                original_prompt=original_prompt.text,
                target_model_name=target_model.name,
                target_tokenizer_name=tokenizer.name,
                round_index=round_index,
            )
        )
        logger.event(
            "proposal_round_start",
            round=round_index,
            batch_size=batch_size,
            frontier_count=len(archive.pareto_frontier()),
            trial_count=archive.trial_count,
        )
        proposals = prompt_proposer.propose(context, batch_size=batch_size)
        population = _accept_proposals(
            proposals=proposals,
            archive=archive,
            original_prompt=original_prompt,
            original_instruction_tokens=original_instruction_tokens,
            tokenizer=tokenizer,
            round_index=round_index,
        )
        logger.event(
            "proposal_batch_generated",
            round=round_index,
            candidate_count=len(population),
            rejected_or_duplicate_count=len(proposals) - len(population),
        )
        if not population:
            convergence_reason = "the proposal policy produced no new valid compressed prompts"
            logger.event("search_converged", round=round_index, reason=convergence_reason)
            break

        candidates.update({candidate.id: candidate for candidate in population})
        round_reports = _evaluate_population(
            population=population,
            evaluator=evaluator,
            model=target_model,
            references=search_set,
            original_instruction_tokens=original_instruction_tokens,
            params=params,
            max_concurrency=max_concurrency,
            logger=logger,
            stage="search",
            round_index=round_index,
        )
        all_reports.extend(round_reports)
        for report in round_reports:
            reports_by_candidate.setdefault(report.candidate_id, []).append(report)

        for rollout_index in range(1, max_candidate_rollouts):
            repeat_ids = _select_repeats(
                candidate_ids=[candidate.id for candidate in population],
                reports_by_candidate=reports_by_candidate,
                repeat_top_k=repeat_top_k,
                repeat_close_within=repeat_close_within,
                baseline_noise=baseline_noise,
            )
            repeated = [candidates[candidate_id] for candidate_id in repeat_ids if candidate_id in candidates]
            if not repeated:
                break
            logger.event(
                "adaptive_rollout_start",
                round=round_index,
                rollout=rollout_index + 1,
                candidate_ids=repeat_ids,
            )
            repeat_reports = _evaluate_population(
                population=repeated,
                evaluator=evaluator,
                model=target_model,
                references=search_set,
                original_instruction_tokens=original_instruction_tokens,
                params=params,
                max_concurrency=max_concurrency,
                logger=logger,
                stage="repeat",
                round_index=round_index,
            )
            all_reports.extend(repeat_reports)
            for report in repeat_reports:
                reports_by_candidate.setdefault(report.candidate_id, []).append(report)

        before_hypervolume = archive.hypervolume()
        for candidate in population:
            candidate_reports = reports_by_candidate[candidate.id]
            proposal = next(item for item in proposals if item.id == candidate.id)
            archive.record(
                round_index=round_index,
                prompt=candidate.prompt_template,
                instruction_tokens=candidate_reports[0].instruction_tokens,
                behavior_loss=_aggregate_behavior_loss(candidate_reports, baseline_noise),
                examples=_aggregate_example_feedback(candidate_reports, baseline_noise),
                parent_ids=proposal.based_on_candidate_ids,
            )

        after_hypervolume = archive.hypervolume()
        round_trials = archive.trials_for_round(round_index)
        summary = RoundSummary(
            round_index=round_index,
            candidates_evaluated=len(round_trials),
            best_token_savings=max((item.token_reduction for item in round_trials), default=0),
            best_behavior_loss=min((item.behavior_loss for item in round_trials), default=1.0),
            frontier_improvement=after_hypervolume - before_hypervolume,
        )
        round_history.append(summary)
        logger.event(
            "search_round_complete",
            round=round_index,
            frontier_count=len(archive.pareto_frontier()),
            best_token_reduction=summary.best_token_savings,
            best_behavior_loss=round(summary.best_behavior_loss, 6),
            frontier_improvement=round(summary.frontier_improvement or 0.0, 6),
        )
        if archive.converged(
            patience=convergence_patience,
            min_improvement=min_frontier_improvement,
        ):
            convergence_reason = (
                f"no material frontier improvement for {convergence_patience} rounds"
            )
            logger.event("search_converged", round=round_index, reason=convergence_reason)
            break

    if not archive.trials:
        raise ValueError("optimization produced no evaluated compressed prompts")

    search_frontier = archive.pareto_frontier()
    finalist_candidates = [candidates[trial.id] for trial in search_frontier]
    logger.event("dev_start", candidate_count=len(finalist_candidates), example_count=len(dev_set))
    dev_reports = _evaluate_population(
        population=finalist_candidates,
        evaluator=evaluator,
        model=target_model,
        references=dev_set,
        original_instruction_tokens=original_instruction_tokens,
        params=params,
        max_concurrency=max_concurrency,
        logger=logger,
        stage="dev",
        round_index=None,
    )
    all_reports.extend(dev_reports)
    dev_frontier = _pareto_reports(dev_reports, baseline_noise)
    best = _choose_best(
        dev_frontier or dev_reports,
        baseline_noise,
        behavior_penalty=selection_behavior_penalty,
    )

    holdout_candidates = [candidates[report.candidate_id] for report in dev_frontier or [best]]
    holdout_reports: list[CandidateReport] = []
    if holdout_set:
        logger.event(
            "holdout_start",
            candidate_count=len(holdout_candidates),
            example_count=len(holdout_set),
        )
        holdout_reports = _evaluate_population(
            population=holdout_candidates,
            evaluator=evaluator,
            model=target_model,
            references=holdout_set,
            original_instruction_tokens=original_instruction_tokens,
            params=params,
            max_concurrency=max_concurrency,
            logger=logger,
            stage="holdout",
            round_index=None,
        )
        all_reports.extend(holdout_reports)

    best_holdout = next(
        (report for report in holdout_reports if report.candidate_id == best.candidate_id),
        None,
    )
    proposer_usage = dict(getattr(prompt_proposer, "usage_summary", {}))
    usage_summary = _total_usage(references, all_reports, baseline_usage, proposer_usage)
    evaluation_report = FeedbackEvaluationReport(
        original_instruction_tokens=original_instruction_tokens,
        best_instruction_tokens=best.instruction_tokens,
        token_reduction=best.token_reduction,
        semantic_drift_normalization=weights.semantic_drift_normalization,
        dev_semantic_drift=best.avg_semantic_drift,
        dev_normalized_semantic_drift=best.avg_normalized_semantic_drift,
        dev_behavior_loss=_behavior_loss(best, baseline_noise),
        dev_format_failure_rate=best.format_failure_rate,
        dev_task_failure_rate=best.task_failure_rate,
        dev_reference_extraction=best.reference_extraction,
        dev_candidate_extraction=best.candidate_extraction,
        dev_extraction_f1_delta=best.extraction_f1_delta,
        holdout_semantic_drift=best_holdout.avg_semantic_drift if best_holdout else None,
        holdout_normalized_semantic_drift=(
            best_holdout.avg_normalized_semantic_drift if best_holdout else None
        ),
        holdout_behavior_loss=_behavior_loss(best_holdout, baseline_noise) if best_holdout else None,
        holdout_format_failure_rate=best_holdout.format_failure_rate if best_holdout else None,
        holdout_task_failure_rate=best_holdout.task_failure_rate if best_holdout else None,
        holdout_reference_extraction=(best_holdout.reference_extraction if best_holdout else None),
        holdout_candidate_extraction=(best_holdout.candidate_extraction if best_holdout else None),
        holdout_extraction_f1_delta=(best_holdout.extraction_f1_delta if best_holdout else None),
        train_size=len(search_set),
        dev_size=len(dev_set),
        holdout_size=len(holdout_set),
        rounds_completed=len(archive.rounds),
        candidates_evaluated=len(all_reports),
        unique_candidates_evaluated=archive.trial_count,
        best_candidate_id=best.candidate_id,
        convergence_reason=convergence_reason,
        baseline_noise=baseline_noise,
        frontier_history=archive.hypervolume_history(),
        selection_behavior_penalty=selection_behavior_penalty,
        usage_summary=usage_summary,
        proposer_usage_summary=proposer_usage,
    )
    result = FeedbackOptimizationResult(
        best_prompt_template=best.prompt_template,
        pareto_frontier=dev_frontier,
        dev_frontier=dev_frontier,
        holdout_reports=holdout_reports,
        evaluation_report=evaluation_report,
        reference_dataset=references,
        all_reports=all_reports,
        search_archive=archive,
    )
    write_run_artifacts(output_dir, result)
    (output_dir / "search_archive.json").write_text(
        json.dumps(archive.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.event(
        "run_done",
        optimizer="feedback_full_prompt",
        best_candidate_id=best.candidate_id,
        token_reduction=round(best.token_reduction, 6),
        dev_behavior_loss=round(evaluation_report.dev_behavior_loss, 6),
        rounds_completed=evaluation_report.rounds_completed,
        convergence_reason=convergence_reason,
        usage_summary=usage_summary,
    )
    return result


def _accept_proposals(
    *,
    proposals: Sequence[PromptProposal],
    archive: SearchArchive,
    original_prompt: PromptTemplate,
    original_instruction_tokens: int,
    tokenizer: Tokenizer,
    round_index: int,
) -> list[PromptCandidate]:
    original_placeholders = _placeholder_sequence(original_prompt.text)
    accepted: list[PromptCandidate] = []
    seen: set[str] = set()
    for proposal in proposals:
        prompt = proposal.prompt_template.strip()
        if not prompt or prompt in seen or archive.contains_prompt(prompt):
            continue
        seen.add(prompt)
        if _placeholder_sequence(prompt) != original_placeholders:
            continue
        instruction_tokens = tokenizer.count(PromptTemplate(prompt).instruction_text())
        if instruction_tokens >= original_instruction_tokens:
            continue
        accepted.append(
            PromptCandidate(
                prompt_template=prompt,
                round_index=round_index,
                rationale=proposal.rationale,
                parent_ids=proposal.based_on_candidate_ids,
            )
        )
    return accepted


def _proposal_context(
    *,
    archive: SearchArchive,
    reports_by_candidate: Mapping[str, list[CandidateReport]],
    original_prompt: str,
    target_model_name: str,
    target_tokenizer_name: str,
    round_index: int,
    round_history: Sequence[RoundSummary],
    parent_limit: int | None,
    recent_limit: int | None,
    worst_example_limit: int | None,
    baseline_noise: BaselineNoise | None,
) -> ProposalContext:
    if not archive.trials:
        return ProposalContext(
            original_prompt=original_prompt,
            target_model_name=target_model_name,
            target_tokenizer_name=target_tokenizer_name,
            round_index=round_index,
            round_history=tuple(round_history),
        )

    parents = (
        archive.select_frontier_parents(parent_limit)
        if parent_limit is not None
        else archive.pareto_frontier()
    )
    failures = sorted(
        archive.trials,
        key=lambda item: (-item.behavior_loss, -item.token_reduction, item.id),
    )
    parent_ids = {trial.id for trial in parents}
    failures = [trial for trial in failures if trial.id not in parent_ids]
    if recent_limit is not None:
        failures = failures[:recent_limit]
    return ProposalContext(
        original_prompt=original_prompt,
        target_model_name=target_model_name,
        target_tokenizer_name=target_tokenizer_name,
        round_index=round_index,
        frontier=tuple(
            _candidate_feedback(
                trial,
                original_prompt,
                reports_by_candidate.get(trial.id, []),
                worst_example_limit,
                baseline_noise,
            )
            for trial in parents
        ),
        informative_failures=tuple(
            _candidate_feedback(
                trial,
                original_prompt,
                reports_by_candidate.get(trial.id, []),
                worst_example_limit,
                baseline_noise,
            )
            for trial in failures
        ),
        round_history=tuple(round_history),
    )


def _candidate_feedback(
    trial: Trial,
    original_prompt: str,
    reports: Sequence[CandidateReport],
    worst_example_limit: int | None,
    baseline_noise: BaselineNoise | None,
) -> CandidateFeedback:
    estimates = aggregate_observations(
        {
            "candidate_id": report.candidate_id,
            "behavior_loss": _behavior_loss(report, baseline_noise),
        }
        for report in reports
    )
    estimate = estimates.get(trial.id)
    metrics: dict[str, float] = {}
    if reports:
        metrics = {
            "token_reduction_fraction": fmean(report.token_reduction for report in reports),
            "normalized_semantic_drift": fmean(
                report.avg_normalized_semantic_drift for report in reports
            ),
            "format_failure_rate": fmean(report.format_failure_rate for report in reports),
            "task_failure_rate": fmean(report.task_failure_rate for report in reports),
            "observations": float(len(reports)),
        }
        if reports[0].candidate_extraction is not None:
            candidate_extraction = reports[0].candidate_extraction
            reference_extraction = reports[0].reference_extraction or {}
            metrics.update(
                {
                    "extraction_precision": float(candidate_extraction["precision"]),
                    "extraction_recall": float(candidate_extraction["recall"]),
                    "extraction_f1": float(candidate_extraction["f1"]),
                    "reference_extraction_f1": float(reference_extraction.get("f1", 0.0)),
                    "extraction_f1_delta": float(reports[0].extraction_f1_delta or 0.0),
                }
            )
        if estimate and estimate.behavior_uncertainty is not None:
            metrics["behavior_standard_error"] = estimate.behavior_uncertainty

    worst = sorted(trial.examples, key=lambda item: (-item.behavior_loss, item.input_id))
    if worst_example_limit is not None:
        worst = worst[:worst_example_limit]
    return CandidateFeedback(
        candidate_id=trial.id,
        prompt_template=trial.prompt,
        instruction_tokens=trial.instruction_tokens,
        token_savings=trial.token_reduction,
        behavior_loss=trial.behavior_loss,
        diff_from_original=_prompt_diff(original_prompt, trial.prompt),
        metrics=metrics,
        worst_residuals=tuple(
            ExampleResidual(
                example_id=example.input_id,
                residual=example.behavior_loss,
                reference_completion=example.reference_output,
                candidate_completion=example.candidate_output,
                observation=example.reason,
            )
            for example in worst
        ),
    )


def _select_repeats(
    *,
    candidate_ids: Sequence[str],
    reports_by_candidate: Mapping[str, list[CandidateReport]],
    repeat_top_k: int | None,
    repeat_close_within: float,
    baseline_noise: BaselineNoise | None,
) -> list[str]:
    if (repeat_top_k is not None and repeat_top_k <= 0) or len(candidate_ids) < 2:
        return []
    estimates = aggregate_observations(
        {
            "candidate_id": report.candidate_id,
            "behavior_loss": _behavior_loss(report, baseline_noise),
        }
        for candidate_id in candidate_ids
        for report in reports_by_candidate.get(candidate_id, ())
    )
    frontier_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if candidate_id in estimates
        and not any(
            _estimate_dominates(
                other_id,
                candidate_id,
                estimates,
                reports_by_candidate,
            )
            for other_id in candidate_ids
            if other_id != candidate_id and other_id in estimates
        )
    ]
    return select_for_additional_rollouts(
        (estimates[candidate_id] for candidate_id in frontier_ids),
        metric="behavior",
        close_within=max(0.0, repeat_close_within),
        baseline_noise=baseline_noise.standard_deviation if baseline_noise else 0.0,
        limit=repeat_top_k,
    )


def _estimate_dominates(
    a_id: str,
    b_id: str,
    estimates: Mapping[str, RewardEstimate],
    reports_by_candidate: Mapping[str, list[CandidateReport]],
) -> bool:
    a_reduction = reports_by_candidate[a_id][0].token_reduction
    b_reduction = reports_by_candidate[b_id][0].token_reduction
    a_loss = estimates[a_id].mean_behavior_loss
    b_loss = estimates[b_id].mean_behavior_loss
    return (
        a_reduction >= b_reduction
        and a_loss <= b_loss
        and (a_reduction > b_reduction or a_loss < b_loss)
    )


def _aggregate_behavior_loss(
    reports: Sequence[CandidateReport],
    baseline_noise: BaselineNoise | None,
) -> float:
    return fmean(_behavior_loss(report, baseline_noise) for report in reports)


def _behavior_loss(report: CandidateReport, baseline_noise: BaselineNoise | None) -> float:
    if report.task_regression_loss is not None and not report.output_records:
        return report.task_regression_loss
    natural_distance = baseline_noise.mean_distance if baseline_noise else 0.0
    if not report.output_records:
        residual_distance = max(0.0, report.avg_normalized_semantic_drift - natural_distance)
        return residual_distance + report.format_failure_rate + report.task_failure_rate
    failure_types = _failure_types_by_input(report)
    return fmean(
        _example_behavior_loss(record, failure_types.get(record.input_id, set()), natural_distance)
        for record in report.output_records
    )


def _aggregate_example_feedback(
    reports: Sequence[CandidateReport],
    baseline_noise: BaselineNoise | None,
) -> tuple[ExampleFeedback, ...]:
    observations: dict[str, list[tuple[float, CandidateOutputRecord]]] = {}
    reasons: dict[str, set[str]] = {}
    natural_distance = baseline_noise.mean_distance if baseline_noise else 0.0
    for report in reports:
        failure_types = _failure_types_by_input(report)
        for record in report.output_records:
            loss = _example_behavior_loss(
                record,
                failure_types.get(record.input_id, set()),
                natural_distance,
            )
            observations.setdefault(record.input_id, []).append((loss, record))
            if record.candidate_extraction is not None:
                candidate_metric = record.candidate_extraction
                reference_metric = record.reference_extraction or {}
                reasons.setdefault(record.input_id, set()).add(
                    "ground-truth extraction: "
                    f"precision={float(candidate_metric['precision']):.4f}, "
                    f"recall={float(candidate_metric['recall']):.4f}, "
                    f"f1={float(candidate_metric['f1']):.4f}, "
                    f"reference_f1={float(reference_metric.get('f1', 0.0)):.4f}"
                )
        for failure in report.examples_failed:
            reasons.setdefault(failure.input_id, set()).add(f"{failure.failure_type}: {failure.reason}")

    examples: list[ExampleFeedback] = []
    for input_id, input_observations in observations.items():
        _, representative = max(input_observations, key=lambda item: item[0])
        examples.append(
            ExampleFeedback(
                input_id=input_id,
                behavior_loss=fmean(loss for loss, _ in input_observations),
                reason="; ".join(sorted(reasons.get(input_id, set()))),
                reference_output=representative.reference_output,
                candidate_output=representative.candidate_output,
            )
        )
    return tuple(sorted(examples, key=lambda item: item.input_id))


def _failure_types_by_input(report: CandidateReport) -> dict[str, set[str]]:
    failures: dict[str, set[str]] = {}
    for failure in report.examples_failed:
        failures.setdefault(failure.input_id, set()).add(failure.failure_type)
    return failures


def _example_behavior_loss(
    record: CandidateOutputRecord,
    failure_types: set[str],
    natural_distance: float,
) -> float:
    if record.task_regression_loss is not None:
        return record.task_regression_loss
    return (
        max(0.0, record.normalized_semantic_drift - natural_distance)
        + ("format_failure" in failure_types)
        + ("task_failure" in failure_types)
    )


def _measure_baseline_noise(
    *,
    target_model: ModelClient,
    references: Sequence[ReferenceExample],
    evaluator: Evaluator,
    params: GenerateParams,
    repeats: int,
) -> tuple[BaselineNoise | None, dict[str, int]]:
    rollout_distances: list[float] = []
    usage: dict[str, int] = {}
    for _ in range(repeats):
        example_distances: list[float] = []
        for reference in references:
            response = target_model.generate(reference.rendered_prompt, params)
            expected = (reference.metadata or {}).get("expected")
            if isinstance(expected, dict):
                reference_metric = score_extraction(reference.reference_output, expected)
                repeated_metric = score_extraction(response.text, expected)
                example_distances.append(max(reference_metric.f1 - repeated_metric.f1, 0.0))
            else:
                comparison = evaluator.compare_outputs(
                    response.text,
                    reference.reference_output,
                    reference.id,
                )
                example_distances.append(comparison.normalized_semantic_drift)
            _add_usage(usage, response.usage)
        if example_distances:
            rollout_distances.append(fmean(example_distances))
    return (estimate_baseline_noise(rollout_distances) if rollout_distances else None), usage


def _pareto_reports(
    reports: list[CandidateReport],
    baseline_noise: BaselineNoise | None,
) -> list[CandidateReport]:
    frontier = [
        report
        for report in reports
        if not any(
            _report_dominates(other, report, baseline_noise)
            for other in reports
            if other is not report
        )
    ]
    return sorted(
        frontier,
        key=lambda item: (_behavior_loss(item, baseline_noise), -item.token_reduction, item.candidate_id),
    )


def _report_dominates(
    a: CandidateReport,
    b: CandidateReport,
    baseline_noise: BaselineNoise | None,
) -> bool:
    a_loss = _behavior_loss(a, baseline_noise)
    b_loss = _behavior_loss(b, baseline_noise)
    return (
        a.token_reduction >= b.token_reduction
        and a_loss <= b_loss
        and (a.token_reduction > b.token_reduction or a_loss < b_loss)
    )


def _choose_best(
    reports: Sequence[CandidateReport],
    baseline_noise: BaselineNoise | None,
    *,
    behavior_penalty: float,
) -> CandidateReport:
    if not reports:
        raise ValueError("no candidate reports were produced")
    return max(
        reports,
        key=lambda report: (
            deployment_utility(
                tokens_saved=report.normalized_token_reduction,
                expected_reuse_volume=1.0,
                behavior_loss=_behavior_loss(report, baseline_noise),
                behavior_penalty=behavior_penalty,
            ),
            -_behavior_loss(report, baseline_noise),
            report.token_reduction,
        ),
    )


def _evaluate_population(
    *,
    population: Sequence[PromptCandidate],
    evaluator: Evaluator,
    model: ModelClient,
    references: list[ReferenceExample],
    original_instruction_tokens: int,
    params: GenerateParams,
    max_concurrency: int,
    logger: RunLogger,
    stage: str,
    round_index: int | None,
) -> list[CandidateReport]:
    if max_concurrency <= 1 or len(population) <= 1:
        return [
            _evaluate_one(
                candidate=candidate,
                evaluator=evaluator,
                model=model,
                references=references,
                original_instruction_tokens=original_instruction_tokens,
                params=params,
                logger=logger,
                stage=stage,
                round_index=round_index,
            )
            for candidate in population
        ]

    reports: dict[str, CandidateReport] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_concurrency)) as executor:
        futures = {
            executor.submit(
                _evaluate_one,
                candidate=candidate,
                evaluator=evaluator,
                model=model,
                references=references,
                original_instruction_tokens=original_instruction_tokens,
                params=params,
                logger=logger,
                stage=stage,
                round_index=round_index,
            ): candidate.id
            for candidate in population
        }
        for future in as_completed(futures):
            report = future.result()
            reports[report.candidate_id] = report
    return [reports[candidate.id] for candidate in population]


def _evaluate_one(
    *,
    candidate: PromptCandidate,
    evaluator: Evaluator,
    model: ModelClient,
    references: list[ReferenceExample],
    original_instruction_tokens: int,
    params: GenerateParams,
    logger: RunLogger,
    stage: str,
    round_index: int | None,
) -> CandidateReport:
    started = time.perf_counter()
    report = evaluator.evaluate_candidate(
        candidate=candidate,
        model=model,
        references=references,
        original_instruction_tokens=original_instruction_tokens,
        params=params,
    )
    logger.event(
        "candidate_evaluated",
        stage=stage,
        round=round_index,
        candidate_id=report.candidate_id,
        instruction_tokens=report.instruction_tokens,
        token_reduction=round(report.token_reduction, 6),
        avg_semantic_drift=round(report.avg_semantic_drift, 6),
        avg_normalized_semantic_drift=round(report.avg_normalized_semantic_drift, 6),
        behavior_loss=round(_behavior_loss(report, None), 6),
        objective_score=round(report.objective_score, 6),
        format_failure_rate=round(report.format_failure_rate, 6),
        task_failure_rate=round(report.task_failure_rate, 6),
        usage_summary=report.usage_summary,
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )
    return report


def _prompt_diff(original: str, candidate: str, limit: int = 3000) -> str:
    diff = "\n".join(
        unified_diff(
            original.splitlines(),
            candidate.splitlines(),
            fromfile="original",
            tofile="candidate",
            lineterm="",
        )
    )
    if len(diff) <= limit:
        return diff
    return diff[: limit - 1].rstrip() + "…"


def _placeholder_sequence(prompt: str) -> tuple[str, ...]:
    return tuple(match.group(1) for match in PLACEHOLDER_RE.finditer(prompt))


def _total_usage(
    references: Sequence[ReferenceExample],
    reports: Sequence[CandidateReport],
    baseline_usage: Mapping[str, int],
    proposer_usage: Mapping[str, int],
) -> dict[str, int]:
    total = dict(baseline_usage)
    for reference in references:
        _add_usage(total, reference.usage)
    for report in reports:
        _add_usage(total, report.usage_summary)
    for key, value in proposer_usage.items():
        if isinstance(value, int):
            total[f"proposer_{key}"] = value
    return total


def _add_usage(total: dict[str, int], usage: Mapping[str, int] | None) -> None:
    if not usage:
        return
    for key, value in usage.items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value
