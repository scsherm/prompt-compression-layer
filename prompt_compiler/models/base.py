from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerateParams:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int | None = None
    system_prompt: str = ""
    reasoning_effort: str | None = None
    response_json_schema: dict[str, Any] | None = None
    response_json_schema_name: str | None = None
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelResponse:
    text: str
    model: str
    params: GenerateParams
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None


class ModelClient(Protocol):
    name: str

    def config(self) -> dict[str, Any]:
        ...

    def generate(self, prompt: str, params: GenerateParams) -> ModelResponse:
        ...
