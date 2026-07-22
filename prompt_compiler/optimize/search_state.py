from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class ExampleFeedback:
    """Observed behavior for one evaluation example."""

    input_id: str
    behavior_loss: float
    reason: str = ""
    reference_output: str = ""
    candidate_output: str = ""

    def __post_init__(self) -> None:
        _validate_loss(self.behavior_loss, "example behavior_loss")

    def to_dict(self) -> dict[str, object]:
        return {
            "input_id": self.input_id,
            "behavior_loss": self.behavior_loss,
            "reason": self.reason,
            "reference_output": self.reference_output,
            "candidate_output": self.candidate_output,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ExampleFeedback:
        return cls(
            input_id=str(value["input_id"]),
            behavior_loss=float(value["behavior_loss"]),
            reason=str(value.get("reason", "")),
            reference_output=str(value.get("reference_output", "")),
            candidate_output=str(value.get("candidate_output", "")),
        )


@dataclass(frozen=True, slots=True)
class Trial:
    """One evaluated, deployable prompt produced by the search policy."""

    id: str
    round_index: int
    prompt: str
    instruction_tokens: int
    token_reduction: int
    behavior_loss: float
    examples: tuple[ExampleFeedback, ...] = ()
    parent_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative")
        if self.instruction_tokens < 0:
            raise ValueError("instruction_tokens must be non-negative")
        _validate_loss(self.behavior_loss, "behavior_loss")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "round_index": self.round_index,
            "prompt": self.prompt,
            "instruction_tokens": self.instruction_tokens,
            "token_reduction": self.token_reduction,
            "behavior_loss": self.behavior_loss,
            "examples": [example.to_dict() for example in self.examples],
            "parent_ids": list(self.parent_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Trial:
        examples = value.get("examples", ())
        parent_ids = value.get("parent_ids", ())
        if not isinstance(examples, Iterable) or isinstance(examples, (str, bytes)):
            raise TypeError("examples must be an iterable")
        if not isinstance(parent_ids, Iterable) or isinstance(parent_ids, (str, bytes)):
            raise TypeError("parent_ids must be an iterable")
        return cls(
            id=str(value["id"]),
            round_index=int(value["round_index"]),
            prompt=str(value["prompt"]),
            instruction_tokens=int(value["instruction_tokens"]),
            token_reduction=int(value["token_reduction"]),
            behavior_loss=float(value["behavior_loss"]),
            examples=tuple(ExampleFeedback.from_dict(item) for item in examples if isinstance(item, Mapping)),
            parent_ids=tuple(str(item) for item in parent_ids),
        )


class SearchArchive:
    """In-memory history and learning state for full-prompt optimization."""

    def __init__(self, *, original_prompt: str, original_instruction_tokens: int) -> None:
        if original_instruction_tokens < 0:
            raise ValueError("original_instruction_tokens must be non-negative")
        self.original_prompt = original_prompt
        self.original_instruction_tokens = original_instruction_tokens
        self._trials_by_id: dict[str, Trial] = {}
        self._trial_id_by_prompt: dict[str, str] = {}

    @property
    def trials(self) -> tuple[Trial, ...]:
        return tuple(sorted(self._trials_by_id.values(), key=lambda item: (item.round_index, item.id)))

    @property
    def trial_count(self) -> int:
        return len(self._trials_by_id)

    @property
    def rounds(self) -> tuple[int, ...]:
        return tuple(sorted({trial.round_index for trial in self._trials_by_id.values()}))

    def contains_prompt(self, prompt: str) -> bool:
        return prompt == self.original_prompt or prompt in self._trial_id_by_prompt

    def get(self, trial_id: str) -> Trial | None:
        return self._trials_by_id.get(trial_id)

    def record(
        self,
        *,
        round_index: int,
        prompt: str,
        instruction_tokens: int,
        behavior_loss: float,
        examples: Iterable[ExampleFeedback] = (),
        parent_ids: Iterable[str] = (),
    ) -> Trial:
        """Record a unique prompt, returning the existing trial on duplicates."""

        if prompt == self.original_prompt:
            raise ValueError("the original prompt is reference metadata, not a trial")
        existing_id = self._trial_id_by_prompt.get(prompt)
        if existing_id is not None:
            return self._trials_by_id[existing_id]

        trial = Trial(
            id=_prompt_id(prompt),
            round_index=round_index,
            prompt=prompt,
            instruction_tokens=instruction_tokens,
            token_reduction=self.original_instruction_tokens - instruction_tokens,
            behavior_loss=behavior_loss,
            examples=tuple(examples),
            parent_ids=tuple(dict.fromkeys(parent_ids)),
        )
        self._insert(trial)
        return trial

    def trials_for_round(self, round_index: int) -> tuple[Trial, ...]:
        return tuple(trial for trial in self.trials if trial.round_index == round_index)

    def pareto_frontier(self, *, through_round: int | None = None) -> tuple[Trial, ...]:
        trials = self._through_round(through_round)
        frontier = [
            trial
            for trial in trials
            if not any(_dominates(other, trial) for other in trials if other.id != trial.id)
        ]
        return tuple(sorted(frontier, key=lambda item: (item.behavior_loss, -item.token_reduction, item.id)))

    def select_frontier_parents(self, limit: int = 4) -> tuple[Trial, ...]:
        """Select objective-space-diverse points from the current frontier."""

        frontier = self.pareto_frontier()
        limit = max(0, limit)
        if len(frontier) <= limit:
            return frontier
        if limit == 0:
            return ()

        selected: list[Trial] = [min(frontier, key=lambda item: (item.behavior_loss, -item.token_reduction, item.id))]
        if limit > 1:
            most_compressed = max(frontier, key=lambda item: (item.token_reduction, -item.behavior_loss, item.id))
            if most_compressed.id != selected[0].id:
                selected.append(most_compressed)

        coordinates = _normalized_coordinates(frontier)
        while len(selected) < limit:
            selected_ids = {trial.id for trial in selected}
            remaining = [trial for trial in frontier if trial.id not in selected_ids]
            if not remaining:
                break
            next_trial = max(
                remaining,
                key=lambda trial: (
                    min(_distance(coordinates[trial.id], coordinates[item.id]) for item in selected),
                    trial.token_reduction,
                    -trial.behavior_loss,
                    trial.id,
                ),
            )
            selected.append(next_trial)

        return tuple(sorted(selected, key=lambda item: (item.behavior_loss, -item.token_reduction, item.id)))

    def hypervolume(self, *, through_round: int | None = None) -> float:
        """Return normalized dominated area in reduction/behavior-quality space."""

        frontier = self.pareto_frontier(through_round=through_round)
        if not frontier or self.original_instruction_tokens == 0:
            return 0.0

        points = sorted(
            (
                max(0.0, trial.token_reduction / self.original_instruction_tokens),
                1.0 / (1.0 + trial.behavior_loss),
            )
            for trial in frontier
            if trial.token_reduction > 0
        )
        area = 0.0
        previous_reduction = 0.0
        for reduction, quality in points:
            if reduction > previous_reduction:
                area += (reduction - previous_reduction) * quality
                previous_reduction = reduction
        return area

    def hypervolume_history(self) -> tuple[tuple[int, float], ...]:
        return tuple((round_index, self.hypervolume(through_round=round_index)) for round_index in self.rounds)

    def converged(self, *, patience: int = 3, min_improvement: float = 1e-4) -> bool:
        """Return true after ``patience`` rounds without material frontier gain."""

        if patience < 1:
            raise ValueError("patience must be at least 1")
        if min_improvement < 0:
            raise ValueError("min_improvement must be non-negative")
        history = self.hypervolume_history()
        if len(history) <= patience:
            return False

        last_material_value = history[0][1]
        last_improvement_index = 0
        for index, (_, value) in enumerate(history[1:], start=1):
            if value > last_material_value + min_improvement:
                last_improvement_index = index
                last_material_value = value
        return len(history) - 1 - last_improvement_index >= patience

    def to_dict(self) -> dict[str, object]:
        return {
            "original_prompt": self.original_prompt,
            "original_instruction_tokens": self.original_instruction_tokens,
            "trials": [trial.to_dict() for trial in self.trials],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SearchArchive:
        archive = cls(
            original_prompt=str(value["original_prompt"]),
            original_instruction_tokens=int(value["original_instruction_tokens"]),
        )
        trials = value.get("trials", ())
        if not isinstance(trials, Iterable) or isinstance(trials, (str, bytes)):
            raise TypeError("trials must be an iterable")
        for item in trials:
            if isinstance(item, Mapping):
                archive._insert(Trial.from_dict(item))
        return archive

    def _through_round(self, round_index: int | None) -> tuple[Trial, ...]:
        if round_index is None:
            return self.trials
        return tuple(trial for trial in self.trials if trial.round_index <= round_index)

    def _insert(self, trial: Trial) -> None:
        if trial.prompt == self.original_prompt:
            raise ValueError("the original prompt is reference metadata, not a trial")
        expected_reduction = self.original_instruction_tokens - trial.instruction_tokens
        if trial.token_reduction != expected_reduction:
            raise ValueError("trial token_reduction does not match the archive reference")
        existing_prompt = self._trial_id_by_prompt.get(trial.prompt)
        if existing_prompt is not None:
            return
        existing_trial = self._trials_by_id.get(trial.id)
        if existing_trial is not None and existing_trial.prompt != trial.prompt:
            raise ValueError(f"trial id collision: {trial.id}")
        self._trials_by_id[trial.id] = trial
        self._trial_id_by_prompt[trial.prompt] = trial.id


def _dominates(a: Trial, b: Trial) -> bool:
    no_worse = a.token_reduction >= b.token_reduction and a.behavior_loss <= b.behavior_loss
    strictly_better = a.token_reduction > b.token_reduction or a.behavior_loss < b.behavior_loss
    return no_worse and strictly_better


def _normalized_coordinates(trials: tuple[Trial, ...]) -> dict[str, tuple[float, float]]:
    reductions = [trial.token_reduction for trial in trials]
    losses = [trial.behavior_loss for trial in trials]
    reduction_span = max(reductions) - min(reductions)
    loss_span = max(losses) - min(losses)
    return {
        trial.id: (
            (trial.token_reduction - min(reductions)) / reduction_span if reduction_span else 0.0,
            (trial.behavior_loss - min(losses)) / loss_span if loss_span else 0.0,
        )
        for trial in trials
    }


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def _validate_loss(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
