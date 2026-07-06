from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol


DEFAULT_EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"


class DriftScorer(Protocol):
    name: str

    def distance(self, candidate: str, reference: str) -> float:
        ...


class EmbeddingClient(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass
class LexicalDriftScorer:
    """Deterministic offline fallback.

    Real embedding clients can replace this layer. The fallback uses Jaccard
    distance over normalized lexical tokens so local runs remain dependency-free.
    """

    name: str = "lexical"

    def distance(self, candidate: str, reference: str) -> float:
        candidate_tokens = set(_tokens(candidate))
        reference_tokens = set(_tokens(reference))
        if not candidate_tokens and not reference_tokens:
            return 0.0
        union = candidate_tokens | reference_tokens
        intersection = candidate_tokens & reference_tokens
        return 1.0 - (len(intersection) / len(union))


@dataclass
class EmbeddingDriftScorer:
    client: EmbeddingClient
    name: str = "embedding_euclidean"

    def distance(self, candidate: str, reference: str) -> float:
        candidate_vec, reference_vec = self.client.embed([candidate, reference])
        return euclidean_distance(candidate_vec, reference_vec)


class SentenceTransformersEmbeddingClient:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        try:
            sentence_transformers = import_module("sentence_transformers")
        except ModuleNotFoundError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        self.model_name = model_name
        self.name = f"sentence-transformers:{model_name}"
        self._model = sentence_transformers.SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        encoded = self._model.encode(texts)
        return _to_vector_list(encoded)


class HuggingFaceInferenceEmbeddingClient:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        api_key: str | None = None,
        provider: str | None = None,
        normalize: bool = False,
    ):
        try:
            huggingface_hub = import_module("huggingface_hub")
        except ModuleNotFoundError as exc:
            raise RuntimeError("huggingface_hub is not installed") from exc
        self.model_name = model_name
        self.name = f"hf-inference:{model_name}"
        self.normalize = normalize
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if provider:
            kwargs["provider"] = provider
        self._client = huggingface_hub.InferenceClient(**kwargs)

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.feature_extraction(
            texts,
            model=self.model_name,
            normalize=self.normalize,
        )
        vectors = _to_vector_list(result)
        return [_pool_vector(vector) for vector in vectors]


def make_drift_scorer(
    provider: str = "lexical",
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
    hf_provider: str | None = None,
) -> DriftScorer:
    if provider == "lexical":
        return LexicalDriftScorer()
    if provider == "sentence-transformers":
        return EmbeddingDriftScorer(SentenceTransformersEmbeddingClient(model_name))
    if provider == "hf-inference":
        return EmbeddingDriftScorer(
            HuggingFaceInferenceEmbeddingClient(
                model_name=model_name,
                api_key=api_key,
                provider=hf_provider,
            )
        )
    raise ValueError(f"Unknown drift scorer provider: {provider}")


def euclidean_distance(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 1.0
    length = min(len(a), len(b))
    a = a[:length]
    b = b[:length]
    return sum((left - right) ** 2 for left, right in zip(a, b)) ** 0.5


def normalize_distance(distance: float, scale: float) -> float:
    if scale <= 0.0:
        return 1.0
    return _clamp(distance / scale)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9_]+", text.lower())


def _clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _to_vector_list(value) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TypeError(f"Expected embedding list, got {type(value)!r}")
    return value


def _pool_vector(value) -> list[float]:
    if not value:
        return []
    if isinstance(value[0], (int, float)):
        return [float(item) for item in value]
    rows = [_pool_vector(row) for row in value]
    rows = [row for row in rows if row]
    if not rows:
        return []
    width = min(len(row) for row in rows)
    return [sum(row[index] for row in rows) / len(rows) for index in range(width)]
