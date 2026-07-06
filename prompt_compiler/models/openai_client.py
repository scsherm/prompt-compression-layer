from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from prompt_compiler.models.base import GenerateParams, ModelResponse


@dataclass
class OpenAIResponsesModel:
    name: str
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    timeout: float = 120.0
    max_retries: int = 2
    send_sampling_params: bool = False
    _client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install the OpenAI adapter with `pip install openai`.") from exc

        kwargs: dict[str, Any] = {
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.organization:
            kwargs["organization"] = self.organization
        if self.project:
            kwargs["project"] = self.project
        self._client = OpenAI(**kwargs)

    def config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": "openai",
            "endpoint": "responses",
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "send_sampling_params": self.send_sampling_params,
        }

    def generate(self, prompt: str, params: GenerateParams) -> ModelResponse:
        if params.tools:
            raise NotImplementedError("Tool calls are not wired into the OpenAI adapter yet.")

        request: dict[str, Any] = {
            "model": self.name,
            "input": prompt,
        }
        if params.system_prompt:
            request["instructions"] = params.system_prompt
        if params.max_tokens is not None:
            request["max_output_tokens"] = params.max_tokens
        if params.reasoning_effort:
            request["reasoning"] = {"effort": params.reasoning_effort}
        if params.response_json_schema:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": params.response_json_schema_name or "structured_response",
                    "schema": params.response_json_schema,
                    "strict": True,
                }
            }
        if self.send_sampling_params and params.temperature is not None:
            request["temperature"] = params.temperature
        if self.send_sampling_params and params.top_p is not None:
            request["top_p"] = params.top_p

        metadata: dict[str, Any] = {"provider": "openai", "endpoint": "responses"}
        if not self.send_sampling_params:
            metadata["sampling_params_omitted"] = True
        try:
            response = self._client.responses.create(**request)
        except Exception as exc:
            if not _sampling_retryable(exc, request):
                raise
            retry_request = {key: value for key, value in request.items() if key not in {"temperature", "top_p"}}
            response = self._client.responses.create(**retry_request)
            metadata["sampling_params_omitted_after_retry"] = True

        return ModelResponse(
            text=_response_text(response),
            model=getattr(response, "model", self.name) or self.name,
            params=params,
            usage=_usage_summary(response),
            metadata={**metadata, **_response_metadata(response)},
        )


def _sampling_retryable(exc: Exception, request: dict[str, Any]) -> bool:
    message = str(exc).lower()
    return (
        ("temperature" in request or "top_p" in request)
        and any(term in message for term in ("temperature", "top_p"))
        and any(term in message for term in ("unsupported", "not supported", "invalid"))
    )


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str):
        return text

    data = _dump(response)
    output = data.get("output", [])
    pieces: list[str] = []
    for item in output if isinstance(output, list) else []:
        for content in item.get("content", []) if isinstance(item, dict) else []:
            if isinstance(content, dict):
                value = content.get("text") or content.get("output_text")
                if isinstance(value, str):
                    pieces.append(value)
    return "".join(pieces)


def _usage_summary(response: Any) -> dict[str, int] | None:
    usage = _dump(response).get("usage")
    if not isinstance(usage, dict):
        return None
    return {key: value for key, value in usage.items() if isinstance(value, int)}


def _response_metadata(response: Any) -> dict[str, Any]:
    data = _dump(response)
    metadata = {}
    for key in ("id", "status", "incomplete_details"):
        if data.get(key) is not None:
            metadata[key] = data[key]
    return metadata


def _dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}
