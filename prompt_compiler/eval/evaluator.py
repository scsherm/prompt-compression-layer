from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Iterable

from prompt_compiler.candidates.candidate import Candidate
from prompt_compiler.eval.contract_checks import OutputContract
from prompt_compiler.eval.embedding_distance import DriftScorer, LexicalDriftScorer, normalize_distance
from prompt_compiler.models.base import GenerateParams, ModelClient
from prompt_compiler.prompt.template import PromptTemplate
from prompt_compiler.tokenizer import Tokenizer


@dataclass(frozen=True)
class EvaluationWeights:
    token: float = 0.50
    embedding: float = 0.50
    semantic_drift_normalization: float = 1.0


@dataclass(frozen=True)
class FailureCase:
    input_id: str
    failure_type: str
    reason: str
    reference_output: str
    candidate_output: str


@dataclass(frozen=True)
class OutputComparison:
    input_id: str
    semantic_drift: float
    normalized_semantic_drift: float
    equivalence_distance: float
    contract_ok: bool
    task_ok: bool
    failures: tuple[FailureCase, ...] = ()


@dataclass(frozen=True)
class CandidateOutputRecord:
    candidate_id: str
    input_id: str
    rendered_prompt: str
    reference_output: str
    candidate_output: str
    equivalence_distance: float
    semantic_drift: float
    normalized_semantic_drift: float
    model: str
    usage: dict[str, int] | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class CandidateReport:
    candidate_id: str
    prompt_template: str
    instruction_tokens: int
    token_reduction: float
    avg_semantic_drift: float
    objective_score: float
    format_failure_rate: float
    task_failure_rate: float
    output_variance: float
    examples_failed: list[FailureCase]
    operator_summary: dict[str, int]
    usage_summary: dict[str, int] = field(default_factory=dict)
    output_records: list[CandidateOutputRecord] = field(default_factory=list)
    avg_normalized_semantic_drift: float = 0.0
    normalized_token_reduction: float = 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["examples_failed"] = [asdict(item) for item in self.examples_failed]
        data["output_records"] = [asdict(item) for item in self.output_records]
        return data


class Evaluator:
    def __init__(
        self,
        tokenizer: Tokenizer,
        output_contract: OutputContract | None = None,
        weights: EvaluationWeights | None = None,
        drift_scorer: DriftScorer | None = None,
    ):
        self.tokenizer = tokenizer
        self.output_contract = output_contract or OutputContract()
        self.weights = weights or EvaluationWeights()
        self.drift_scorer = drift_scorer or LexicalDriftScorer()

    def compare_outputs(self, candidate_output: str, reference_output: str, input_id: str) -> OutputComparison:
        contract_result = self.output_contract.validate(candidate_output)
        semantic_drift = self.drift_scorer.distance(candidate_output, reference_output)
        normalized_semantic_drift = _normalize_drift(semantic_drift, self.drift_scorer, self.weights)
        task_ok = _task_equivalent(candidate_output, reference_output)

        failures: list[FailureCase] = []
        if not contract_result.ok:
            failures.append(
                FailureCase(
                    input_id=input_id,
                    failure_type="format_failure",
                    reason=",".join(contract_result.failures),
                    reference_output=reference_output,
                    candidate_output=candidate_output,
                )
            )
        if not task_ok:
            failures.append(
                FailureCase(
                    input_id=input_id,
                    failure_type="task_failure",
                    reason="parsed task fields differ",
                    reference_output=reference_output,
                    candidate_output=candidate_output,
                )
            )

        equivalence_distance = normalized_semantic_drift

        return OutputComparison(
            input_id=input_id,
            semantic_drift=semantic_drift,
            normalized_semantic_drift=normalized_semantic_drift,
            equivalence_distance=equivalence_distance,
            contract_ok=contract_result.ok,
            task_ok=task_ok,
            failures=tuple(failures),
        )

    def evaluate_candidate(
        self,
        *,
        candidate: Candidate,
        model: ModelClient,
        references: Iterable,
        original_instruction_tokens: int,
        params: GenerateParams,
    ) -> CandidateReport:
        prompt = PromptTemplate(candidate.prompt_template)
        comparisons: list[OutputComparison] = []
        output_records: list[CandidateOutputRecord] = []
        usage_summary: dict[str, int] = {}
        for reference in references:
            rendered = prompt.render({"input": reference.input_text})
            response = model.generate(rendered, params)
            _add_usage(usage_summary, response.usage)
            comparison = self.compare_outputs(response.text, reference.reference_output, reference.id)
            comparisons.append(comparison)
            output_records.append(
                CandidateOutputRecord(
                    candidate_id=candidate.id,
                    input_id=reference.id,
                    rendered_prompt=rendered,
                    reference_output=reference.reference_output,
                    candidate_output=response.text,
                    equivalence_distance=comparison.equivalence_distance,
                    semantic_drift=comparison.semantic_drift,
                    normalized_semantic_drift=comparison.normalized_semantic_drift,
                    model=response.model,
                    usage=response.usage,
                    metadata=response.metadata,
                )
            )

        instruction_tokens = self.tokenizer.count(prompt.instruction_text())
        token_reduction = 1.0 - (instruction_tokens / max(original_instruction_tokens, 1))
        normalized_token_reduction = _clamp(token_reduction)
        failures = [failure for comparison in comparisons for failure in comparison.failures]
        count = max(len(comparisons), 1)
        operator_summary: dict[str, int] = {}
        for chunk in candidate.chunks:
            key = f"{chunk.chunk_type.value}:{chunk.operator.value}"
            operator_summary[key] = operator_summary.get(key, 0) + 1

        avg_drift = mean([comparison.semantic_drift for comparison in comparisons]) if comparisons else 1.0
        avg_normalized_drift = (
            mean([comparison.normalized_semantic_drift for comparison in comparisons]) if comparisons else 1.0
        )
        objective_score = _weighted_loss(
            token_loss=1.0 - normalized_token_reduction,
            normalized_semantic_drift=avg_normalized_drift,
            weights=self.weights,
        )

        return CandidateReport(
            candidate_id=candidate.id,
            prompt_template=candidate.prompt_template,
            instruction_tokens=instruction_tokens,
            token_reduction=token_reduction,
            normalized_token_reduction=normalized_token_reduction,
            avg_semantic_drift=avg_drift,
            objective_score=objective_score,
            format_failure_rate=sum(not item.contract_ok for item in comparisons) / count,
            task_failure_rate=sum(not item.task_ok for item in comparisons) / count,
            output_variance=0.0,
            examples_failed=failures,
            operator_summary=operator_summary,
            usage_summary=usage_summary,
            output_records=output_records,
            avg_normalized_semantic_drift=avg_normalized_drift,
        )


def _task_equivalent(candidate_output: str, reference_output: str) -> bool:
    candidate_json = _json_dict(candidate_output)
    reference_json = _json_dict(reference_output)
    if candidate_json is not None and reference_json is not None:
        comparable_fields = set(candidate_json) & set(reference_json) & {"status", "label", "answer"}
        if comparable_fields:
            return all(candidate_json[field] == reference_json[field] for field in comparable_fields)
    return True


def _json_dict(text: str) -> dict | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _add_usage(total: dict[str, int], usage: dict[str, int] | None) -> None:
    if not usage:
        return
    for key, value in usage.items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def _normalize_drift(semantic_drift: float, scorer: DriftScorer, weights: EvaluationWeights) -> float:
    if getattr(scorer, "name", "") == "embedding_euclidean":
        return normalize_distance(semantic_drift, weights.semantic_drift_normalization)
    return _clamp(semantic_drift)


def _weighted_loss(
    *,
    token_loss: float,
    normalized_semantic_drift: float,
    weights: EvaluationWeights,
) -> float:
    weight_sum = weights.token + weights.embedding
    if weight_sum <= 0.0:
        return 1.0
    return ((weights.token * token_loss) + (weights.embedding * normalized_semantic_drift)) / weight_sum


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)
