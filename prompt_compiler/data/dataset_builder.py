from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

from prompt_compiler.cache import ModelCallCache, ModelCallKey
from prompt_compiler.hashing import stable_hash
from prompt_compiler.models.base import GenerateParams, ModelClient
from prompt_compiler.prompt.template import PromptTemplate
from prompt_compiler.tokenizer import Tokenizer


@dataclass(frozen=True)
class InputExample:
    id: str
    input_text: str
    metadata: dict | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], index: int) -> "InputExample":
        return cls(
            id=str(value.get("id", f"ex_{index:04d}")),
            input_text=str(value.get("input", value.get("input_text", ""))),
            metadata={key: item for key, item in value.items() if key not in {"id", "input", "input_text"}},
        )


@dataclass(frozen=True)
class ReferenceExample:
    id: str
    input_text: str
    reference_output: str
    model: str
    params: dict
    rendered_prompt: str
    prompt_hash: str
    prompt_tokens: int
    output_tokens: int
    usage: dict[str, int] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_inputs(inputs: Iterable[InputExample | Mapping[str, object] | str]) -> list[InputExample]:
    normalized: list[InputExample] = []
    for index, item in enumerate(inputs, start=1):
        if isinstance(item, InputExample):
            normalized.append(item)
        elif isinstance(item, str):
            normalized.append(InputExample(id=f"ex_{index:04d}", input_text=item))
        else:
            normalized.append(InputExample.from_mapping(item, index))
    return normalized


def build_reference_dataset(
    *,
    model: ModelClient,
    prompt: PromptTemplate,
    inputs: Iterable[InputExample | Mapping[str, object] | str],
    tokenizer: Tokenizer,
    params: GenerateParams,
    cache: ModelCallCache | None = None,
) -> list[ReferenceExample]:
    cache = cache or ModelCallCache()
    references: list[ReferenceExample] = []
    for example in normalize_inputs(inputs):
        rendered = prompt.render({"input": example.input_text})
        key = ModelCallKey.from_call(
            model_config=model.config(),
            prompt=rendered,
            input_id=example.id,
            params=params,
        )
        response = cache.get_or_generate(key, lambda rendered=rendered: model.generate(rendered, params))
        references.append(
            ReferenceExample(
                id=example.id,
                input_text=example.input_text,
                reference_output=response.text,
                model=response.model,
                params=response.params.__dict__,
                rendered_prompt=rendered,
                prompt_hash=stable_hash(rendered),
                prompt_tokens=tokenizer.count(rendered),
                output_tokens=tokenizer.count(response.text),
                usage=response.usage,
            )
        )
    return references
