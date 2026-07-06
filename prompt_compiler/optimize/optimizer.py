from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from prompt_compiler.candidates.candidate import Candidate
from prompt_compiler.candidates.generation import describe_chunkings, next_generation_from_frontier, seed_population
from prompt_compiler.data.dataset_builder import InputExample, ReferenceExample, build_reference_dataset, normalize_inputs
from prompt_compiler.data.splits import split_dataset
from prompt_compiler.eval.contract_checks import OutputContract
from prompt_compiler.eval.embedding_distance import DriftScorer
from prompt_compiler.eval.evaluator import CandidateReport, Evaluator
from prompt_compiler.eval.pareto import compute_pareto_frontier
from prompt_compiler.models.base import GenerateParams, ModelClient
from prompt_compiler.observability import RunLogger
from prompt_compiler.operators.proposer import RewriteProposer
from prompt_compiler.optimize.credit_assignment import OperatorDiagnostic, summarize_operator_diagnostics
from prompt_compiler.optimize.curriculum import curriculum_subset
from prompt_compiler.prompt.template import PromptTemplate
from prompt_compiler.reports.writer import write_run_artifacts
from prompt_compiler.tokenizer import ApproxTokenizer, Tokenizer


@dataclass(frozen=True)
class EvaluationReport:
    original_instruction_tokens: int
    best_instruction_tokens: int
    token_reduction: float
    dev_semantic_drift: float
    dev_format_failure_rate: float
    dev_task_failure_rate: float
    holdout_semantic_drift: float | None
    holdout_format_failure_rate: float | None
    holdout_task_failure_rate: float | None
    train_size: int
    dev_size: int
    holdout_size: int
    epochs_completed: int
    candidates_evaluated: int
    best_candidate_id: str
    usage_summary: dict[str, int]
    estimated_cost_usd: float | None
    diagnostics: list[OperatorDiagnostic]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["diagnostics"] = [asdict(item) for item in self.diagnostics]
        data["validation_semantic_drift"] = self.dev_semantic_drift
        data["format_failure_rate"] = self.dev_format_failure_rate
        data["task_failure_rate"] = self.dev_task_failure_rate
        return data


@dataclass(frozen=True)
class CompressionRunResult:
    best_prompt_template: str
    pareto_frontier: list[CandidateReport]
    dev_frontier: list[CandidateReport]
    holdout_reports: list[CandidateReport]
    evaluation_report: EvaluationReport
    reference_dataset: list[ReferenceExample]
    all_reports: list[CandidateReport]


def optimize_prompt(
    *,
    target_model: ModelClient,
    original_prompt: PromptTemplate,
    inputs: Iterable[InputExample | Mapping[str, object] | str],
    output_dir: Path,
    epochs: int = 3,
    population_size: int = 32,
    tokenizer: Tokenizer | None = None,
    output_contract: OutputContract | None = None,
    drift_scorer: DriftScorer | None = None,
    params: GenerateParams | None = None,
    rewrite_proposer: RewriteProposer | None = None,
    max_concurrency: int = 1,
    log_to_stderr: bool = True,
    live_log_path: Path | None = None,
    chunker_names: tuple[str, ...] | None = None,
) -> CompressionRunResult:
    tokenizer = tokenizer or ApproxTokenizer()
    params = params or GenerateParams()
    output_contract = output_contract or OutputContract()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir / "run_events.jsonl", echo=log_to_stderr, mirror_path=live_log_path)
    max_concurrency = max(1, max_concurrency)
    normalized_inputs = normalize_inputs(inputs)

    logger.event(
        "run_start",
        model=target_model.config(),
        input_count=len(normalized_inputs),
        epochs=epochs,
        population_size=population_size,
        max_concurrency=max_concurrency,
        generation_params=params,
        chunkers=chunker_names,
    )

    logger.event("reference_build_start", example_count=len(normalized_inputs))
    references = build_reference_dataset(
        model=target_model,
        prompt=original_prompt,
        inputs=normalized_inputs,
        tokenizer=tokenizer,
        params=params,
    )
    reference_usage = _reference_usage(references)
    logger.event(
        "reference_build_done",
        reference_count=len(references),
        usage_summary=reference_usage,
        estimated_cost_usd=_estimate_cost_usd(target_model.name, reference_usage),
    )
    split = split_dataset(references)
    dev_set = split.validation or split.train
    holdout_set = split.test
    logger.event(
        "dataset_split",
        train_size=len(split.train),
        dev_size=len(dev_set),
        holdout_size=len(holdout_set),
    )
    original_instruction_tokens = tokenizer.count(original_prompt.instruction_text())
    evaluator = Evaluator(tokenizer=tokenizer, output_contract=output_contract, drift_scorer=drift_scorer)
    chunking_plan = describe_chunkings(original_prompt.text, tokenizer=tokenizer, chunker_names=chunker_names)
    logger.event(
        "chunking_plan",
        active_chunkers=list(chunking_plan),
        chunk_counts={name: plan["chunk_count"] for name, plan in chunking_plan.items()},
        chunks=chunking_plan,
    )
    logger.event(
        "population_seed_start",
        population_size=population_size,
        proposer="llm" if rewrite_proposer else "rule",
        original_instruction_tokens=original_instruction_tokens,
    )
    population = seed_population(
        original_prompt.text,
        tokenizer=tokenizer,
        population_size=population_size,
        proposer=rewrite_proposer,
        chunker_names=chunker_names,
    )
    candidate_index: dict[str, Candidate] = {candidate.id: candidate for candidate in population}
    logger.event(
        "population_seeded",
        candidate_count=len(population),
        original_instruction_tokens=original_instruction_tokens,
        summary=_population_summary(population),
    )

    epoch_reports: list[CandidateReport] = []
    all_evaluated_reports: list[CandidateReport] = []
    for epoch in range(max(epochs, 1)):
        subset = curriculum_subset(split.train, epoch)
        logger.event(
            "epoch_start",
            epoch=epoch,
            candidate_count=len(population),
            example_count=len(subset),
        )
        epoch_reports = _evaluate_population(
            population=population,
            evaluator=evaluator,
            model=target_model,
            references=subset,
            original_instruction_tokens=original_instruction_tokens,
            params=params,
            max_concurrency=max_concurrency,
            logger=logger,
            stage="train",
            epoch=epoch,
        )
        all_evaluated_reports.extend(epoch_reports)
        frontier = compute_pareto_frontier(epoch_reports)
        if epoch < max(epochs, 1) - 1:
            population = next_generation_from_frontier(
                population=population,
                frontier_ids={report.candidate_id for report in frontier},
                tokenizer=tokenizer,
                population_size=population_size,
                proposer=rewrite_proposer,
                chunker_names=chunker_names,
            )
            candidate_index.update({candidate.id: candidate for candidate in population})
            logger.event(
                "mutation_done",
                epoch=epoch,
                frontier_count=len(frontier),
                next_population_count=len(population),
                summary=_population_summary(population),
            )
        logger.event(
            "epoch_frontier",
            epoch=epoch,
            frontier_count=len(frontier),
            next_population_count=len(population),
        )

    finalists = compute_pareto_frontier(epoch_reports)
    dev_candidates = [_candidate_by_id(candidate_index, report.candidate_id) for report in finalists if report.candidate_id in candidate_index]
    logger.event("dev_start", candidate_count=len(dev_candidates), example_count=len(dev_set))
    dev_reports = _evaluate_population(
        population=dev_candidates,
        evaluator=evaluator,
        model=target_model,
        references=dev_set,
        original_instruction_tokens=original_instruction_tokens,
        params=params,
        max_concurrency=max_concurrency,
        logger=logger,
        stage="dev",
        epoch=None,
    )
    final_reports = dev_reports or finalists
    dev_frontier = compute_pareto_frontier(final_reports)
    best = _choose_best(dev_frontier or final_reports)

    holdout_candidates = [
        _candidate_by_id(candidate_index, report.candidate_id)
        for report in (dev_frontier or [best])
        if report.candidate_id in candidate_index
    ]
    holdout_reports: list[CandidateReport] = []
    if holdout_set and holdout_candidates:
        logger.event("holdout_start", candidate_count=len(holdout_candidates), example_count=len(holdout_set))
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
            epoch=None,
        )

    best_holdout = _report_by_id(holdout_reports, best.candidate_id)
    diagnostics = summarize_operator_diagnostics(final_reports)
    usage_summary = _total_usage(references, all_evaluated_reports + final_reports + holdout_reports)

    evaluation_report = EvaluationReport(
        original_instruction_tokens=original_instruction_tokens,
        best_instruction_tokens=best.instruction_tokens,
        token_reduction=best.token_reduction,
        dev_semantic_drift=best.avg_semantic_drift,
        dev_format_failure_rate=best.format_failure_rate,
        dev_task_failure_rate=best.task_failure_rate,
        holdout_semantic_drift=best_holdout.avg_semantic_drift if best_holdout else None,
        holdout_format_failure_rate=best_holdout.format_failure_rate if best_holdout else None,
        holdout_task_failure_rate=best_holdout.task_failure_rate if best_holdout else None,
        train_size=len(split.train),
        dev_size=len(dev_set),
        holdout_size=len(holdout_set),
        epochs_completed=max(epochs, 1),
        candidates_evaluated=len(all_evaluated_reports) + len(final_reports) + len(holdout_reports),
        best_candidate_id=best.candidate_id,
        usage_summary=usage_summary,
        estimated_cost_usd=_estimate_cost_usd(target_model.name, usage_summary),
        diagnostics=diagnostics,
    )
    result = CompressionRunResult(
        best_prompt_template=best.prompt_template,
        pareto_frontier=dev_frontier,
        dev_frontier=dev_frontier,
        holdout_reports=holdout_reports,
        evaluation_report=evaluation_report,
        reference_dataset=references,
        all_reports=all_evaluated_reports + final_reports + holdout_reports,
    )
    write_run_artifacts(output_dir, result)
    logger.event(
        "run_done",
        best_candidate_id=best.candidate_id,
        token_reduction=round(best.token_reduction, 6),
        dev_semantic_drift=round(best.avg_semantic_drift, 6),
        holdout_semantic_drift=round(best_holdout.avg_semantic_drift, 6) if best_holdout else None,
        usage_summary=evaluation_report.usage_summary,
        estimated_cost_usd=evaluation_report.estimated_cost_usd,
    )
    return result


def _choose_best(reports: list[CandidateReport]) -> CandidateReport:
    if not reports:
        raise ValueError("No candidate reports were produced")
    compressed_reports = [report for report in reports if report.token_reduction > 0]
    reports = compressed_reports or reports
    return sorted(
        reports,
        key=lambda item: (
            item.format_failure_rate,
            item.task_failure_rate,
            item.objective_score,
            -item.token_reduction,
        ),
    )[0]


def _evaluate_population(
    *,
    population: list[Candidate],
    evaluator: Evaluator,
    model: ModelClient,
    references: list[ReferenceExample],
    original_instruction_tokens: int,
    params: GenerateParams,
    max_concurrency: int,
    logger: RunLogger,
    stage: str,
    epoch: int | None,
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
                epoch=epoch,
            )
            for candidate in population
        ]

    reports_by_id: dict[str, CandidateReport] = {}
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
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
                epoch=epoch,
            ): candidate.id
            for candidate in population
        }
        for future in as_completed(futures):
            report = future.result()
            reports_by_id[report.candidate_id] = report
    return [reports_by_id[candidate.id] for candidate in population if candidate.id in reports_by_id]


def _evaluate_one(
    *,
    candidate: Candidate,
    evaluator: Evaluator,
    model: ModelClient,
    references: list[ReferenceExample],
    original_instruction_tokens: int,
    params: GenerateParams,
    logger: RunLogger,
    stage: str,
    epoch: int | None,
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
        epoch=epoch,
        candidate_id=report.candidate_id,
        instruction_tokens=report.instruction_tokens,
        token_reduction=round(report.token_reduction, 6),
        avg_semantic_drift=round(report.avg_semantic_drift, 6),
        objective_score=round(report.objective_score, 6),
        format_failure_rate=round(report.format_failure_rate, 6),
        task_failure_rate=round(report.task_failure_rate, 6),
        usage_summary=report.usage_summary,
        estimated_cost_usd=_estimate_cost_usd(model.name, report.usage_summary),
        elapsed_seconds=round(time.perf_counter() - started, 3),
    )
    return report


def _candidate_by_id(candidates: Mapping[str, Candidate], candidate_id: str) -> Candidate:
    if candidate_id in candidates:
        return candidates[candidate_id]
    raise KeyError(candidate_id)


def _report_by_id(reports: list[CandidateReport], candidate_id: str) -> CandidateReport | None:
    for report in reports:
        if report.candidate_id == candidate_id:
            return report
    return None


def _population_summary(population: list[Candidate]) -> dict:
    chunkers: dict[str, int] = {}
    operators: dict[str, int] = {}
    for candidate in population:
        chunkers[candidate.genome.chunker_name] = chunkers.get(candidate.genome.chunker_name, 0) + 1
        for chunk in candidate.chunks:
            key = f"{chunk.chunk_type.value}:{chunk.operator.value}"
            operators[key] = operators.get(key, 0) + 1
    return {
        "chunkers": chunkers,
        "top_operators": dict(sorted(operators.items(), key=lambda item: item[1], reverse=True)[:12]),
    }


def _reference_usage(references: list[ReferenceExample]) -> dict[str, int]:
    total: dict[str, int] = {}
    for reference in references:
        _add_usage(total, reference.usage)
    return total


def _total_usage(references: list[ReferenceExample], reports: list[CandidateReport]) -> dict[str, int]:
    total = _reference_usage(references)
    for report in reports:
        _add_usage(total, report.usage_summary)
    return total


def _add_usage(total: dict[str, int], usage: dict[str, int] | None) -> None:
    if not usage:
        return
    for key, value in usage.items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def _estimate_cost_usd(model_name: str, usage: dict[str, int]) -> float | None:
    if "gpt-5-nano" not in model_name:
        return None
    input_cost = usage.get("input_tokens", 0) * 0.05 / 1_000_000
    output_cost = usage.get("output_tokens", 0) * 0.40 / 1_000_000
    return round(input_cost + output_cost, 6)
