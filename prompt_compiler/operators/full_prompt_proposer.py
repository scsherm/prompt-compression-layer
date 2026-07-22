from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Protocol

from prompt_compiler.hashing import stable_hash
from prompt_compiler.models.base import GenerateParams, ModelClient
from prompt_compiler.prompt.chunk import PLACEHOLDER_RE
from prompt_compiler.tokenizer import Tokenizer


@dataclass(frozen=True)
class ExampleResidual:
    """One completion pair that is informative for the next proposal round."""

    example_id: str
    residual: float
    reference_completion: str
    candidate_completion: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    observation: str = ""


@dataclass(frozen=True)
class CandidateFeedback:
    """Measured outcome for a complete prompt candidate."""

    candidate_id: str
    prompt_template: str
    instruction_tokens: int
    token_savings: int
    behavior_loss: float
    diff_from_original: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)
    worst_residuals: tuple[ExampleResidual, ...] = ()
    observation: str = ""


@dataclass(frozen=True)
class RoundSummary:
    """Small learning-curve record; detailed trials live in candidate feedback."""

    round_index: int
    candidates_evaluated: int
    best_token_savings: int
    best_behavior_loss: float
    frontier_improvement: float | None = None
    observation: str = ""


@dataclass(frozen=True)
class ProposalContext:
    """Empirical context used to update the prompt proposal distribution."""

    original_prompt: str
    target_model_name: str
    target_tokenizer_name: str
    round_index: int = 0
    frontier: tuple[CandidateFeedback, ...] = ()
    informative_failures: tuple[CandidateFeedback, ...] = ()
    round_history: tuple[RoundSummary, ...] = ()


@dataclass(frozen=True)
class PromptProposal:
    """A complete, deployable prompt template measured by the target tokenizer."""

    prompt_template: str
    instruction_tokens: int
    token_savings: int
    rationale: str
    based_on_candidate_ids: tuple[str, ...] = ()
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            object.__setattr__(self, "id", stable_hash(self.prompt_template)[:12])


class PromptProposer(Protocol):
    def propose(
        self,
        context: ProposalContext,
        *,
        batch_size: int,
    ) -> tuple[PromptProposal, ...]:
        ...


PROPOSAL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "prompt_template": {"type": "string"},
                    "rationale": {"type": "string"},
                    "based_on_candidate_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "prompt_template",
                    "rationale",
                    "based_on_candidate_ids",
                ],
            },
        }
    },
    "required": ["proposals"],
}


@dataclass
class LLMFullPromptProposer:
    """Propose full prompts conditioned on measured outcomes from earlier rounds.

    The model provides the proposal policy. Token counting and template integrity
    are local observations, so model estimates never determine candidate cost.
    """

    model: ModelClient
    tokenizer: Tokenizer
    params: GenerateParams = field(
        default_factory=lambda: GenerateParams(reasoning_effort="minimal")
    )
    trace_path: Path | None = None
    usage_summary: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.params.response_json_schema:
            return
        self.params = replace(
            self.params,
            response_json_schema=PROPOSAL_JSON_SCHEMA,
            response_json_schema_name="full_prompt_proposals",
        )

    def propose(
        self,
        context: ProposalContext,
        *,
        batch_size: int,
    ) -> tuple[PromptProposal, ...]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not context.original_prompt.strip():
            raise ValueError("original_prompt must not be empty")

        proposer_prompt = self.render_prompt(context, batch_size=batch_size)
        response = self.model.generate(proposer_prompt, self.params)
        _add_usage(self.usage_summary, response.usage)
        parsed = _extract_json_object(response.text)
        proposals, rejected = self._parse_proposals(parsed, context)
        proposals = proposals[:batch_size]

        self._write_trace(
            context=context,
            requested=batch_size,
            proposer_prompt=proposer_prompt,
            response_text=response.text,
            parsed=parsed,
            accepted=proposals,
            rejected=rejected,
            usage=response.usage,
            response_metadata=response.metadata,
        )
        return tuple(proposals)

    def render_prompt(self, context: ProposalContext, *, batch_size: int) -> str:
        original_tokens = _instruction_token_count(context.original_prompt, self.tokenizer)
        feedback = {
            "frontier": [_candidate_feedback_dict(item) for item in context.frontier],
            "informative_failures": [
                _candidate_feedback_dict(item) for item in context.informative_failures
            ],
            "round_history": [asdict(item) for item in context.round_history],
        }
        return f"""You are the proposal policy in a black-box prompt optimization loop.

Objective: reduce the recurring instruction-token cost of ORIGINAL_PROMPT while preserving its observed completion behavior on the target model. Previous candidate results are empirical learning signals. Use them to make this batch better than prior batches: retain changes associated with strong outcomes, repair changes associated with residuals, and explore promising compression opportunities not yet tested.

Return exactly {batch_size} distinct proposals as structured JSON. Each proposal must be a COMPLETE deployable prompt template, not a rewritten fragment, patch, commentary, or set of editing instructions.

Proposal requirements:
- Every proposal must use fewer instruction tokens than ORIGINAL_PROMPT. The counter will measure this; do not return the original prompt.
- Preserve every {{{{placeholder}}}} in the same order. Do not add, remove, rename, or reorder placeholders.
- Preserve the intended task and completion behavior. Candidate prompts may use any internal wording or notation that the target model understands.
- Search at multiple scales. You may reorganize the whole prompt, merge distant redundancy, replace a section, or make surgical edits. Choose scope from the evidence rather than a predefined rewrite category.
- Generate meaningfully different hypotheses. Do not create superficial wording variants of one idea.
- `based_on_candidate_ids` identifies prior candidates whose measured evidence informed a proposal; it may be empty in round 0.
- `rationale` briefly states the observed evidence or compression hypothesis behind the proposal. Do not put rationale inside `prompt_template`.

TARGET
model: {context.target_model_name}
tokenizer: {context.target_tokenizer_name}
active counter: {self.tokenizer.name}
round: {context.round_index}
original instruction tokens: {original_tokens}

MEASURED SEARCH FEEDBACK
{json.dumps(feedback, ensure_ascii=False, sort_keys=True)}

ORIGINAL_PROMPT
<original_prompt>
{context.original_prompt}
</original_prompt>
"""

    def _parse_proposals(
        self,
        parsed: dict[str, Any],
        context: ProposalContext,
    ) -> tuple[list[PromptProposal], list[dict[str, str]]]:
        original_tokens = _instruction_token_count(context.original_prompt, self.tokenizer)
        original_placeholders = _placeholder_sequence(context.original_prompt)
        raw_proposals = parsed.get("proposals", [])
        if not isinstance(raw_proposals, list):
            raw_proposals = []

        accepted: list[PromptProposal] = []
        rejected: list[dict[str, str]] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_proposals):
            if not isinstance(raw, dict):
                rejected.append({"index": str(index), "reason": "proposal is not an object"})
                continue
            prompt = str(raw.get("prompt_template", "")).strip()
            if not prompt:
                rejected.append({"index": str(index), "reason": "empty prompt_template"})
                continue
            if prompt in seen:
                rejected.append({"index": str(index), "reason": "duplicate prompt_template"})
                continue
            seen.add(prompt)

            placeholders = _placeholder_sequence(prompt)
            if placeholders != original_placeholders:
                rejected.append(
                    {
                        "index": str(index),
                        "reason": (
                            "placeholder sequence changed: "
                            f"expected={original_placeholders}, actual={placeholders}"
                        ),
                    }
                )
                continue

            instruction_tokens = _instruction_token_count(prompt, self.tokenizer)
            if instruction_tokens >= original_tokens:
                rejected.append(
                    {
                        "index": str(index),
                        "reason": f"not shorter: {instruction_tokens}>={original_tokens}",
                    }
                )
                continue

            parent_ids = raw.get("based_on_candidate_ids", [])
            if not isinstance(parent_ids, list):
                parent_ids = []
            accepted.append(
                PromptProposal(
                    prompt_template=prompt,
                    instruction_tokens=instruction_tokens,
                    token_savings=original_tokens - instruction_tokens,
                    rationale=str(raw.get("rationale", "")).strip(),
                    based_on_candidate_ids=tuple(str(item) for item in parent_ids),
                )
            )
        return accepted, rejected

    def _write_trace(
        self,
        *,
        context: ProposalContext,
        requested: int,
        proposer_prompt: str,
        response_text: str,
        parsed: dict[str, Any],
        accepted: list[PromptProposal],
        rejected: list[dict[str, str]],
        usage: dict[str, int] | None,
        response_metadata: dict[str, Any] | None,
    ) -> None:
        if not self.trace_path:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "event": "full_prompt_proposals",
            "round_index": context.round_index,
            "requested": requested,
            "proposer_model": self.model.name,
            "target_model": context.target_model_name,
            "target_tokenizer": context.target_tokenizer_name,
            "proposer_prompt": proposer_prompt,
            "proposer_response": response_text,
            "parsed_response": parsed,
            "accepted": [asdict(item) for item in accepted],
            "rejected": rejected,
            "usage": usage,
            "response_metadata": response_metadata,
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _candidate_feedback_dict(feedback: CandidateFeedback) -> dict[str, Any]:
    return {
        "candidate_id": feedback.candidate_id,
        "prompt_template": feedback.prompt_template,
        "instruction_tokens": feedback.instruction_tokens,
        "token_savings": feedback.token_savings,
        "behavior_loss": feedback.behavior_loss,
        "diff_from_original": feedback.diff_from_original,
        "metrics": dict(feedback.metrics),
        "worst_residuals": [
            {
                "example_id": item.example_id,
                "residual": item.residual,
                "reference_completion": item.reference_completion,
                "candidate_completion": item.candidate_completion,
                "metrics": dict(item.metrics),
                "observation": item.observation,
            }
            for item in feedback.worst_residuals
        ],
        "observation": feedback.observation,
    }


def _instruction_token_count(prompt_template: str, tokenizer: Tokenizer) -> int:
    return tokenizer.count(PLACEHOLDER_RE.sub("", prompt_template))


def _placeholder_sequence(prompt_template: str) -> tuple[str, ...]:
    return tuple(match.group(1) for match in PLACEHOLDER_RE.finditer(prompt_template))


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            parsed = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _add_usage(total: dict[str, int], usage: Mapping[str, int] | None) -> None:
    if not usage:
        return
    for key, value in usage.items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value
