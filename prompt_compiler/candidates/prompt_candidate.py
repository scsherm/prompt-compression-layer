from __future__ import annotations

from dataclasses import dataclass, field

from prompt_compiler.hashing import stable_hash


@dataclass(frozen=True)
class PromptCandidate:
    """A complete compressed prompt proposed during feedback search."""

    prompt_template: str
    round_index: int
    rationale: str = ""
    parent_ids: tuple[str, ...] = ()
    id: str = field(default="")

    def __post_init__(self) -> None:
        if self.id:
            return
        object.__setattr__(self, "id", stable_hash(self.prompt_template)[:12])
