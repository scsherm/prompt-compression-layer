from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_-]*)\s*\}\}")
INPUT_LABELS = frozenset({"input:", "user request:", "alert:", "document:", "query:"})


class ChunkType(Enum):
    ROLE = "role"
    TASK = "task"
    CONSTRAINT = "constraint"
    NEGATIVE_CONSTRAINT = "negative_constraint"
    OUTPUT_SCHEMA = "output_schema"
    EXAMPLE = "example"
    STYLE = "style"
    INPUT_SLOT = "input_slot"
    SAFETY = "safety"
    TOOL_INSTRUCTION = "tool_instruction"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PromptChunk:
    id: str
    text: str
    chunk_type: ChunkType
    start_char: int
    end_char: int
    protected: bool = False


def detect_chunk_type(text: str) -> ChunkType:
    lowered = text.lower()
    stripped = lowered.strip()
    if PLACEHOLDER_RE.search(text) or stripped in INPUT_LABELS:
        return ChunkType.INPUT_SLOT
    if any(term in lowered for term in ("json", "schema", "field", "return", "status", "response:")):
        return ChunkType.OUTPUT_SCHEMA
    if any(term in lowered for term in ("do not", "don't", "never", "no markdown", "without")):
        return ChunkType.NEGATIVE_CONSTRAINT
    if any(term in lowered for term in ("you are", "role:", "system:")):
        return ChunkType.ROLE
    if any(term in lowered for term in ("must", "should", "only", "constraint", "rule")):
        return ChunkType.CONSTRAINT
    if any(term in lowered for term in ("example", "few-shot")):
        return ChunkType.EXAMPLE
    if any(term in lowered for term in ("style", "tone", "voice")):
        return ChunkType.STYLE
    if any(term in lowered for term in ("tool", "function call")):
        return ChunkType.TOOL_INSTRUCTION
    if any(term in lowered for term in ("safe", "policy", "harm")):
        return ChunkType.SAFETY
    if any(term in lowered for term in ("task", "triage", "classify", "summarize")):
        return ChunkType.TASK
    return ChunkType.UNKNOWN


def is_protected(text: str) -> bool:
    stripped = text.strip().lower()
    return bool(PLACEHOLDER_RE.search(text)) or stripped in INPUT_LABELS or stripped == "response:"


def placeholder_names(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(1) for match in PLACEHOLDER_RE.finditer(text)))
