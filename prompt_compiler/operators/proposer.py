from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from string import Template
from typing import Protocol

from prompt_compiler.models.base import GenerateParams, ModelClient
from prompt_compiler.operators.rewrite_ops import RewriteOperator
from prompt_compiler.prompt.chunk import ChunkType, PromptChunk
from prompt_compiler.tokenizer import Tokenizer


@dataclass(frozen=True)
class RewriteVariant:
    operator: RewriteOperator
    text: str
    token_count: int
    gloss: str
    attempt_id: str = "default"
    jitter: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RewriteAttempt:
    id: str
    target_multiplier: float = 1.0
    aggression: str = "medium"
    style_hint: str = "Use the operator's default compression style."

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_REWRITE_ATTEMPT = RewriteAttempt(id="default")

JITTER_ATTEMPT_POOL = (
    RewriteAttempt(
        id="compact_symbols",
        target_multiplier=0.85,
        aggression="medium",
        style_hint="Prefer compact operators like =>, ;, /, +, no_*, preserve, unless.",
    ),
    RewriteAttempt(
        id="aggressive_short",
        target_multiplier=0.70,
        aggression="aggressive",
        style_hint="Prefer the shortest clear wording; allow terse grammar and stable abbreviations.",
    ),
    RewriteAttempt(
        id="conservative_exact",
        target_multiplier=1.10,
        aggression="conservative",
        style_hint="Compress less aggressively; preserve explicit wording when ambiguity risk is high.",
    ),
    RewriteAttempt(
        id="abbrev_heavy",
        target_multiplier=0.80,
        aggression="medium",
        style_hint="Favor stable abbreviations such as req, fmt, len, EN, out, info.",
    ),
    RewriteAttempt(
        id="dsl_dense",
        target_multiplier=0.65,
        aggression="aggressive",
        style_hint="Favor compact DSL-like notation with clear keys and constraints.",
    ),
)


def make_rewrite_attempts(count: int, seed: int = 0) -> tuple[RewriteAttempt, ...]:
    if count <= 1:
        return (DEFAULT_REWRITE_ATTEMPT,)
    pool = list(JITTER_ATTEMPT_POOL)
    random.Random(seed).shuffle(pool)
    return (DEFAULT_REWRITE_ATTEMPT, *pool[: max(0, count - 1)])


class RewriteProposer(Protocol):
    def rewrite(self, chunk: PromptChunk, operator: RewriteOperator) -> tuple[str, str]:
        ...


REWRITE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rewritten_chunk": {"type": "string"},
        "rationale": {"type": "string"},
        "risk_notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["rewritten_chunk", "rationale", "risk_notes"],
}


class RuleRewriteProposer:
    """Deterministic baseline proposer.

    Real deployments can replace this with an LLM proposer. Keeping this local
    proposer makes the compiler runnable and gives the optimizer broad initial
    search coverage.
    """

    def rewrite(self, chunk: PromptChunk, operator: RewriteOperator) -> tuple[str, str]:
        if chunk.protected:
            return chunk.text, "protected input slot preserved verbatim"
        if operator == RewriteOperator.KEEP:
            return chunk.text, "original chunk"
        if operator == RewriteOperator.DELETE:
            return "", "deleted chunk"
        if operator == RewriteOperator.SHORT_ENGLISH:
            return self._short_english(chunk), "compact English rewrite"
        if operator == RewriteOperator.TELEGRAPH_ENGLISH:
            return self._telegraph(chunk.text), "telegraphic English rewrite"
        if operator == RewriteOperator.SYMBOLIC_DSL:
            return self._symbolic(chunk), "symbolic DSL rewrite"
        if operator == RewriteOperator.SCHEMA_ABBREVIATION:
            return self._schema_abbrev(chunk), "schema abbreviation"
        if operator == RewriteOperator.HYBRID_SYMBOLIC_ENGLISH:
            return self._hybrid(chunk), "hybrid symbolic English"
        if operator == RewriteOperator.SHORT_MANDARIN:
            return self._short_mandarin(chunk), "short Mandarin instruction"
        if operator == RewriteOperator.FORMAL_CHINESE:
            return self._formal_chinese(chunk), "formal Chinese instruction"
        if operator == RewriteOperator.CLASSICAL_CHINESE_LIKE:
            return self._classical_like(chunk), "classical-Chinese-like compression"
        if operator == RewriteOperator.MANDARIN_SYMBOLIC:
            return self._mandarin_symbolic(chunk), "Mandarin-symbolic compression"
        if operator == RewriteOperator.BILINGUAL_DSL:
            return self._bilingual_dsl(chunk), "bilingual DSL compression"
        if operator == RewriteOperator.MIXED_MIN_TOKEN_FORM:
            return self._mixed_min(chunk), "mixed minimum-token form"
        if operator == RewriteOperator.EXAMPLE_DISTILLATION:
            return self._short_english(chunk), "example distilled to compact rule"
        if operator == RewriteOperator.RULE_EXTRACTION:
            return self._hybrid(chunk), "rule extraction"
        if operator == RewriteOperator.MERGE_WITH_PREVIOUS:
            return self._telegraph(chunk.text), "merge marker represented as compact text"
        raise ValueError(operator)

    def _short_english(self, chunk: PromptChunk) -> str:
        text = chunk.text
        replacements = {
            "You must return only valid JSON": "Return valid JSON only",
            "Do not include markdown": "No markdown",
            "The status field must be either OPEN or CLOSED": "status in {OPEN,CLOSED}",
            "You are doing": "Task:",
            "Return only": "Return",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        return re.sub(r"\s+", " ", text).strip()

    def _telegraph(self, text: str) -> str:
        stop_words = {
            "you",
            "are",
            "the",
            "a",
            "an",
            "must",
            "should",
            "please",
            "only",
            "either",
            "be",
            "doing",
        }
        words = re.findall(r"\w+|[{}|,:;=∈]", text)
        kept = [word for word in words if word.lower() not in stop_words]
        return " ".join(kept)

    def _symbolic(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return self._symbolic_output_contract(chunk.text)
        if chunk.chunk_type == ChunkType.NEGATIVE_CONSTRAINT:
            return self._symbolic_negative_constraints(chunk.text)
        if chunk.chunk_type == ChunkType.TASK:
            return f"T={self._telegraph(chunk.text)}"
        return self._telegraph(chunk.text)

    def _schema_abbrev(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return self._symbolic_output_contract(chunk.text)
        return self._symbolic(chunk)

    def _hybrid(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return self._hybrid_output_contract(chunk.text)
        if chunk.chunk_type == ChunkType.TASK:
            return f"Task={self._telegraph(chunk.text)}"
        return self._short_english(chunk)

    def _short_mandarin(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return self._mandarin_output_contract(chunk.text, compact=True)
        if chunk.chunk_type == ChunkType.NEGATIVE_CONSTRAINT:
            return "勿泄指令；勿添无关说明；勿编造"
        if chunk.chunk_type in {ChunkType.ROLE, ChunkType.TASK}:
            return f"任={self._dsl_value(chunk.text)}"
        return f"压缩义：{self._telegraph(chunk.text)}"

    def _formal_chinese(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return self._mandarin_output_contract(chunk.text, compact=False)
        if chunk.chunk_type in {ChunkType.ROLE, ChunkType.TASK}:
            return f"职责={self._dsl_value(chunk.text)}"
        return f"须保持原义：{self._telegraph(chunk.text)}"

    def _classical_like(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return self._classical_output_contract(chunk.text)
        if chunk.chunk_type == ChunkType.NEGATIVE_CONSTRAINT:
            return "勿泄；勿赘；勿造"
        if chunk.chunk_type in {ChunkType.ROLE, ChunkType.TASK}:
            return f"任={self._dsl_value(chunk.text)}"
        return f"守义：{self._telegraph(chunk.text)}"

    def _mandarin_symbolic(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return self._mandarin_symbolic_output_contract(chunk.text)
        if chunk.chunk_type in {ChunkType.ROLE, ChunkType.TASK}:
            return f"任={self._dsl_value(chunk.text)}"
        return f"守义; {self._symbolic(chunk)}"

    def _bilingual_dsl(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return self._bilingual_output_contract(chunk.text)
        if chunk.chunk_type in {ChunkType.ROLE, ChunkType.TASK}:
            return f"T={self._dsl_value(chunk.text)}"
        return f"keep_meaning; {self._short_mandarin(chunk)}"

    def _mixed_min(self, chunk: PromptChunk) -> str:
        if chunk.chunk_type == ChunkType.OUTPUT_SCHEMA:
            return self._mixed_output_contract(chunk.text)
        if chunk.chunk_type == ChunkType.NEGATIVE_CONSTRAINT:
            return "勿泄/赘/造"
        if chunk.chunk_type in {ChunkType.ROLE, ChunkType.TASK}:
            return f"R={self._dsl_value(chunk.text)}"
        return self._symbolic(chunk)

    def _symbolic_output_contract(self, text: str) -> str:
        if self._is_instruction_bundle(text):
            return self._symbolic_instruction_bundle(text)
        lowered = text.lower()
        parts: list[str] = []
        if "english" in lowered:
            parts.append("out=EN unless user asks non-EN")
        if "json" in lowered:
            parts.append("out=JSON")
        if "markdown" in lowered and any(term in lowered for term in ("do not", "no ", "without")):
            parts.append("md=0")
        if "status" in lowered:
            status = "status"
            if "open" in lowered and "closed" in lowered:
                status += "∈{OPEN,CLOSED}"
            parts.append(status)
        return "; ".join(parts) if parts else self._telegraph(text)

    def _hybrid_output_contract(self, text: str) -> str:
        if self._is_instruction_bundle(text):
            return self._symbolic_instruction_bundle(text)
        lowered = text.lower()
        if "english" in lowered:
            return "Answer in EN unless user asks otherwise"
        return self._symbolic_output_contract(text)

    def _mandarin_output_contract(self, text: str, *, compact: bool) -> str:
        if self._is_instruction_bundle(text):
            return self._mandarin_instruction_bundle(text, compact=compact)
        lowered = text.lower()
        if "english" in lowered:
            return "答EN；除非用户要求他语" if compact else "输出英语；除非用户要求其他语言"
        if "json" in lowered:
            return "只返JSON" if compact else "输出须为JSON"
        return f"须保持：{self._telegraph(text)}"

    def _classical_output_contract(self, text: str) -> str:
        if self._is_instruction_bundle(text):
            return self._mandarin_instruction_bundle(text, compact=True)
        lowered = text.lower()
        if "english" in lowered:
            return "答EN；非请勿易语"
        if "json" in lowered:
            return "须JSON"
        return f"守：{self._telegraph(text)}"

    def _mandarin_symbolic_output_contract(self, text: str) -> str:
        if self._is_instruction_bundle(text):
            return self._mandarin_symbolic_instruction_bundle(text)
        lowered = text.lower()
        if "english" in lowered:
            return "out=EN; 除非user要他语"
        if "json" in lowered:
            return "out=JSON"
        return self._symbolic_output_contract(text)

    def _bilingual_output_contract(self, text: str) -> str:
        if self._is_instruction_bundle(text):
            return self._symbolic_instruction_bundle(text)
        lowered = text.lower()
        if "english" in lowered:
            return "out=EN; unless user asks other_lang"
        if "json" in lowered:
            return "out=JSON"
        return self._symbolic_output_contract(text)

    def _mixed_output_contract(self, text: str) -> str:
        if self._is_instruction_bundle(text):
            return self._symbolic_instruction_bundle(text)
        lowered = text.lower()
        if "english" in lowered:
            return "out=EN unless asked otherwise"
        if "json" in lowered:
            parts = ["out=JSON"]
            if "markdown" in lowered and any(term in lowered for term in ("do not", "no ", "without")):
                parts.append("md=0")
            return ";".join(parts)
        return self._symbolic_output_contract(text)

    def _symbolic_negative_constraints(self, text: str) -> str:
        lowered = text.lower()
        parts: list[str] = []
        if "mention these instructions" in lowered:
            parts.append("no_policy_echo")
        if any(term in lowered for term in ("caveats", "disclaimers", "process commentary")):
            parts.append("no_extra_meta")
        if "do not refuse" in lowered or "don't refuse" in lowered:
            parts.append("no_refuse_safe")
        if "invent facts" in lowered:
            parts.append("no_fabricate")
        return "; ".join(parts) if parts else self._telegraph(text)

    def _dsl_value(self, text: str) -> str:
        telegraph = self._telegraph(text)
        words = re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]+", telegraph)
        return "_".join(words[:4]) if words else "task"

    def _is_instruction_bundle(self, text: str) -> bool:
        lowered = text.lower()
        signals = (
            "instruction-following",
            "preserve the requested format",
            "if the user asks for a list",
            "if the user asks for a letter",
            "if the user asks for a summary",
            "creative writing",
            "do not mention these instructions",
            "do not refuse",
            "do not invent facts",
        )
        return sum(signal in lowered for signal in signals) >= 2

    def _symbolic_instruction_bundle(self, text: str) -> str:
        lowered = text.lower()
        parts: list[str] = []
        if "instruction-following" in lowered:
            parts.append("careful instruction-follower")
        if "preserve the requested format" in lowered:
            parts.append("preserve fmt/audience/tone/len")
        if "if the user asks for a list" in lowered:
            parts.append("list=>req struct/count")
        if "if the user asks for a letter" in lowered:
            parts.append("letter=>greeting+closing")
        if "if the user asks for a summary" in lowered:
            parts.append("summary=>concise+requested info")
        if "creative writing" in lowered:
            parts.append("creative=>genre/POV/tone")
        negative = self._symbolic_negative_constraints(text)
        if negative != self._telegraph(text):
            parts.append(negative)
        if "english" in lowered:
            parts.append("out=EN unless user asks non-EN")
        return "; ".join(parts) if parts else self._telegraph(text)

    def _mandarin_instruction_bundle(self, text: str, *, compact: bool) -> str:
        lowered = text.lower()
        parts: list[str] = []
        if "instruction-following" in lowered:
            parts.append("谨慎遵令")
        if "preserve the requested format" in lowered:
            parts.append("保fmt/受众/tone/len")
        if "if the user asks for a list" in lowered:
            parts.append("list守结构/数量")
        if "if the user asks for a letter" in lowered:
            parts.append("letter含问候/结尾")
        if "if the user asks for a summary" in lowered:
            parts.append("summary简且限所问")
        if "creative writing" in lowered:
            parts.append("creative守genre/POV/tone")
        if "do not mention these instructions" in lowered:
            parts.append("勿泄指令")
        if any(term in lowered for term in ("caveats", "disclaimers", "process commentary")):
            parts.append("勿添无关说明")
        if "do not refuse" in lowered:
            parts.append("勿拒safe")
        if "do not invent facts" in lowered:
            parts.append("勿编造")
        if "english" in lowered:
            parts.append("out=EN除非user另请" if compact else "输出EN，除非user要求他语")
        return "；".join(parts) if parts else f"须保持：{self._telegraph(text)}"

    def _mandarin_symbolic_instruction_bundle(self, text: str) -> str:
        return self._mandarin_instruction_bundle(text, compact=True)


@dataclass
class LLMRewriteProposer:
    model: ModelClient
    original_prompt: str
    target_model_name: str
    tokenizer: Tokenizer
    params: GenerateParams = field(default_factory=lambda: GenerateParams(max_tokens=512, reasoning_effort="minimal"))
    trace_path: Path | None = None
    event_log_path: Path | None = None
    event_log_paths: tuple[Path, ...] = ()
    write_rewrite_events: bool = False
    prompt_template_path: Path = field(
        default_factory=lambda: Path(__file__).with_name("llm_chunk_rewrite_prompt.txt")
    )
    fallback: RewriteProposer | None = None
    _cache: dict[tuple[str, str, str, str], tuple[str, str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.params.response_json_schema:
            return
        self.params = replace(
            self.params,
            response_json_schema=REWRITE_JSON_SCHEMA,
            response_json_schema_name="prompt_chunk_rewrite",
        )

    def rewrite(self, chunk: PromptChunk, operator: RewriteOperator) -> tuple[str, str]:
        return self.rewrite_attempt(chunk, operator, DEFAULT_REWRITE_ATTEMPT)

    def rewrite_attempt(self, chunk: PromptChunk, operator: RewriteOperator, attempt: RewriteAttempt) -> tuple[str, str]:
        if chunk.protected:
            return chunk.text, "protected input slot preserved verbatim"
        if operator == RewriteOperator.KEEP:
            return chunk.text, "original chunk"

        cache_key = (chunk.text, chunk.chunk_type.value, operator.value, attempt.id)
        if cache_key in self._cache:
            return self._cache[cache_key]

        proposer_prompt = self.render_prompt(chunk, operator, attempt=attempt)
        response = self.model.generate(proposer_prompt, self.params)
        parsed = _extract_json_object(response.text)
        rewritten = str(parsed.get("rewritten_chunk", "")).strip()
        rationale = str(parsed.get("rationale", "")).strip() or "LLM rewrite"

        validation_error = self._validate_rewrite(chunk, rewritten)
        if validation_error:
            self._write_trace(
                chunk=chunk,
                operator=operator,
                proposer_prompt=proposer_prompt,
                response_text=response.text,
                parsed=parsed,
                result=(rewritten, rationale),
                usage=response.usage,
                response_metadata=response.metadata,
                validation_error=validation_error,
                attempt=attempt,
            )
            raise ValueError(f"Invalid LLM rewrite for {chunk.id}/{operator.value}: {validation_error}")

        result = (rewritten, rationale)
        self._cache[cache_key] = result
        self._write_trace(
            chunk=chunk,
            operator=operator,
            proposer_prompt=proposer_prompt,
            response_text=response.text,
            parsed=parsed,
            result=result,
            usage=response.usage,
            response_metadata=response.metadata,
            validation_error=validation_error,
            attempt=attempt,
        )
        if self.write_rewrite_events:
            self._write_event(chunk=chunk, operator=operator, result=result, usage=response.usage, attempt=attempt)
        return result

    def render_prompt(
        self,
        chunk: PromptChunk,
        operator: RewriteOperator,
        *,
        attempt: RewriteAttempt = DEFAULT_REWRITE_ATTEMPT,
    ) -> str:
        template = Template(self.prompt_template_path.read_text(encoding="utf-8"))
        original_tokens = self.tokenizer.count(chunk.text)
        base_target_tokens = _target_rewrite_tokens(operator, original_tokens)
        target_tokens = max(1, int(base_target_tokens * attempt.target_multiplier))
        return template.safe_substitute(
            operator=operator.value,
            operator_instruction=_operator_instruction(operator),
            operator_hard_requirements=_operator_hard_requirements(operator),
            proposal_variation=_proposal_variation_instruction(attempt),
            original_prompt=self.original_prompt,
            chunk_type=chunk.chunk_type.value,
            original_chunk_tokens=original_tokens,
            target_rewrite_tokens=target_tokens,
            chunk_text=chunk.text,
        )

    def _validate_rewrite(self, chunk: PromptChunk, rewritten: str) -> str | None:
        if not rewritten.strip():
            return "empty"
        forbidden = (
            "original_chunk_tokens",
            "hard constraints",
            "original full prompt template",
            "chunk to rewrite",
            "return json only",
            "rewritten_chunk",
        )
        lowered = rewritten.lower()
        for marker in forbidden:
            if marker in lowered:
                return f"copied_scaffold:{marker}"
        original_tokens = max(self.tokenizer.count(chunk.text), 1)
        rewritten_tokens = self.tokenizer.count(rewritten)
        if rewritten_tokens > max(original_tokens * 2, original_tokens + 24):
            return f"too_long:{rewritten_tokens}>{original_tokens}"
        return None

    def _write_trace(
        self,
        *,
        chunk: PromptChunk,
        operator: RewriteOperator,
        proposer_prompt: str,
        response_text: str,
        parsed: dict,
        result: tuple[str, str],
        usage: dict[str, int] | None,
        response_metadata: dict | None,
        validation_error: str | None,
        attempt: RewriteAttempt,
    ) -> None:
        if not self.trace_path:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "chunk_id": chunk.id,
            "chunk_type": chunk.chunk_type.value,
            "operator": operator.value,
            "attempt": attempt.to_dict(),
            "proposer_model": self.model.name,
            "target_model_name": self.target_model_name,
            "proposer_prompt": proposer_prompt,
            "proposer_response": response_text,
            "parsed_response": parsed,
            "rewritten_chunk": result[0],
            "rationale": result[1],
            "usage": usage,
            "response_metadata": response_metadata,
            "validation_error": validation_error,
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _write_event(
        self,
        *,
        chunk: PromptChunk,
        operator: RewriteOperator,
        result: tuple[str, str],
        usage: dict[str, int] | None,
        attempt: RewriteAttempt,
    ) -> None:
        paths = self._event_paths()
        if not paths:
            return
        row = {
            "event": "proposer_rewrite",
            "chunk_id": chunk.id,
            "chunk_type": chunk.chunk_type.value,
            "operator": operator.value,
            "attempt": attempt.to_dict(),
            "rewritten_chunk": result[0],
            "rationale": result[1],
            "usage": usage,
        }
        line = json.dumps(row, ensure_ascii=False, sort_keys=True)
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _event_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        if self.event_log_path:
            paths.append(self.event_log_path)
        paths.extend(self.event_log_paths)
        unique: list[Path] = []
        for path in paths:
            if path not in unique:
                unique.append(path)
        return tuple(unique)


class TokenizerAwareRewritePlanner:
    def __init__(self, tokenizer: Tokenizer, proposer: RewriteProposer | None = None):
        self.tokenizer = tokenizer
        self.proposer = proposer or RuleRewriteProposer()

    def plan(
        self,
        chunk: PromptChunk,
        operators: tuple[RewriteOperator, ...] | None = None,
        attempts: tuple[RewriteAttempt, ...] | None = None,
    ) -> list[RewriteVariant]:
        operators = operators or tuple(RewriteOperator)
        attempts = attempts or (DEFAULT_REWRITE_ATTEMPT,)
        variants: list[RewriteVariant] = []
        seen: set[str] = set()
        for operator in operators:
            operator_attempts = (DEFAULT_REWRITE_ATTEMPT,) if operator == RewriteOperator.KEEP or chunk.protected else attempts
            for attempt in operator_attempts:
                try:
                    text, gloss = _rewrite_with_attempt(self.proposer, chunk, operator, attempt)
                except ValueError:
                    continue
                if _non_keep_rewrite_is_not_shorter(chunk, operator, text, self.tokenizer):
                    continue
                normalized = _normalize_rewrite_text(text)
                if normalized in seen:
                    continue
                seen.add(normalized)
                variants.append(
                    RewriteVariant(
                        operator=operator,
                        text=text,
                        token_count=self.tokenizer.count(text),
                        gloss=gloss,
                        attempt_id=attempt.id,
                        jitter=attempt.to_dict(),
                    )
                )
        return sorted(variants, key=lambda item: (item.token_count, item.operator.value, item.attempt_id))


def _non_keep_rewrite_is_not_shorter(
    chunk: PromptChunk,
    operator: RewriteOperator,
    text: str,
    tokenizer: Tokenizer,
) -> bool:
    if chunk.protected or operator == RewriteOperator.KEEP:
        return False
    return tokenizer.count(text) >= tokenizer.count(chunk.text)


def _rewrite_with_attempt(
    proposer: RewriteProposer,
    chunk: PromptChunk,
    operator: RewriteOperator,
    attempt: RewriteAttempt,
) -> tuple[str, str]:
    rewrite_attempt = getattr(proposer, "rewrite_attempt", None)
    if callable(rewrite_attempt):
        return rewrite_attempt(chunk, operator, attempt)
    return proposer.rewrite(chunk, operator)


def _normalize_rewrite_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _proposal_variation_instruction(attempt: RewriteAttempt) -> str:
    return (
        f"attempt_id={attempt.id}; aggression={attempt.aggression}; "
        f"target_multiplier={attempt.target_multiplier:.2f}; {attempt.style_hint}"
    )


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _operator_instruction(operator: RewriteOperator) -> str:
    instructions = {
        RewriteOperator.KEEP: "Return CHUNK_TEXT unchanged.",
        RewriteOperator.DELETE: "Return a minimal marker only if deletion would preserve behavior; otherwise return CHUNK_TEXT unchanged.",
        RewriteOperator.SHORT_ENGLISH: "Use short, plain English. Preserve every specific rule, condition, source limit, negation, and output constraint.",
        RewriteOperator.TELEGRAPH_ENGLISH: "Use telegraphic English: omit filler words, keep exact triggers/actions, source limits, and negations.",
        RewriteOperator.SYMBOLIC_DSL: "Use compact symbolic DSL where clear: key=value, arrows, sets, semicolons. Preserve literals/placeholders.",
        RewriteOperator.SCHEMA_ABBREVIATION: "Compress schemas/output constraints with field abbreviations and symbolic value sets. Preserve required fields and literals.",
        RewriteOperator.HYBRID_SYMBOLIC_ENGLISH: "Use compact English plus symbols. Preserve specific conditions/actions, negations, and output language.",
        RewriteOperator.SHORT_MANDARIN: "Use compact Mandarin shorthand for the instruction body, not English. Preserve literal field names, enum values, placeholders, and required English output behavior.",
        RewriteOperator.FORMAL_CHINESE: "Use concise formal Mandarin for the instruction body, not English. Preserve literal field names, enum values, placeholders, and required English output behavior.",
        RewriteOperator.CLASSICAL_CHINESE_LIKE: "Use ultra-compact literary/classical-style Chinese for the instruction body where understandable. Preserve literals/placeholders/output language.",
        RewriteOperator.MANDARIN_SYMBOLIC: "Use Mandarin plus symbolic DSL for the instruction body, not all-English. Preserve literals, placeholders, enum values, and required English output behavior.",
        RewriteOperator.BILINGUAL_DSL: "Use a compact bilingual DSL. Mix Mandarin/English only where it saves tokens and stays clear.",
        RewriteOperator.MIXED_MIN_TOKEN_FORM: "Use the shortest clear mixture of symbols, English, and Mandarin. Never drop triggers, actions, or negations.",
        RewriteOperator.EXAMPLE_DISTILLATION: "Distill examples into the shortest rule that preserves the example's behavioral lesson.",
        RewriteOperator.RULE_EXTRACTION: "Extract the governing rule in compact form. Preserve all specific conditions/actions from CHUNK_TEXT.",
        RewriteOperator.MERGE_WITH_PREVIOUS: "Rewrite as a compact fragment that can merge with the previous chunk without losing meaning.",
    }
    return instructions[operator]


def _operator_hard_requirements(operator: RewriteOperator) -> str:
    requirements = {
        RewriteOperator.KEEP: "rewritten_chunk must equal CHUNK_TEXT.",
        RewriteOperator.DELETE: "Do not return empty text; if deletion is unsafe, return CHUNK_TEXT unchanged.",
        RewriteOperator.SHORT_ENGLISH: "Use English only; no Chinese characters. Preserve all semantic constraints.",
        RewriteOperator.TELEGRAPH_ENGLISH: "Use English only; no Chinese characters. Telegraphic grammar is allowed, but constraints must remain explicit.",
        RewriteOperator.SYMBOLIC_DSL: "Use English keys and symbols only; no Chinese characters. Preserve all semantic constraints.",
        RewriteOperator.SCHEMA_ABBREVIATION: "Use English keys/symbols only; no Chinese characters. Preserve every required field, literal value, enum, and output-format constraint.",
        RewriteOperator.HYBRID_SYMBOLIC_ENGLISH: "Use English plus symbols only; no Chinese characters. Preserve all semantic constraints.",
        RewriteOperator.SHORT_MANDARIN: "If CHUNK_TEXT is not a protected literal, rewritten_chunk must contain Mandarin Chinese characters.",
        RewriteOperator.FORMAL_CHINESE: "If CHUNK_TEXT is not a protected literal, rewritten_chunk must contain Mandarin Chinese characters.",
        RewriteOperator.CLASSICAL_CHINESE_LIKE: "If CHUNK_TEXT is not a protected literal, rewritten_chunk must contain Mandarin Chinese characters.",
        RewriteOperator.MANDARIN_SYMBOLIC: "If CHUNK_TEXT is not a protected literal, rewritten_chunk must contain Mandarin Chinese characters and may use symbols like =>, /, ;.",
        RewriteOperator.BILINGUAL_DSL: "Use compact bilingual or DSL form. Preserve all semantic constraints.",
        RewriteOperator.MIXED_MIN_TOKEN_FORM: "Use the shortest clear form. Preserve all semantic constraints even if that costs tokens.",
        RewriteOperator.EXAMPLE_DISTILLATION: "Preserve the example's behavioral lesson. Do not introduce a new task.",
        RewriteOperator.RULE_EXTRACTION: "Preserve every specific condition/action/scope qualifier.",
        RewriteOperator.MERGE_WITH_PREVIOUS: "Return a non-empty mergeable fragment preserving this chunk's meaning.",
    }
    return requirements[operator]


def _target_rewrite_tokens(operator: RewriteOperator, original_tokens: int) -> int:
    if original_tokens <= 8:
        return original_tokens + 2
    ratios = {
        RewriteOperator.SHORT_ENGLISH: 0.70,
        RewriteOperator.TELEGRAPH_ENGLISH: 0.50,
        RewriteOperator.SYMBOLIC_DSL: 0.45,
        RewriteOperator.SCHEMA_ABBREVIATION: 0.45,
        RewriteOperator.HYBRID_SYMBOLIC_ENGLISH: 0.55,
        RewriteOperator.SHORT_MANDARIN: 0.45,
        RewriteOperator.FORMAL_CHINESE: 0.55,
        RewriteOperator.CLASSICAL_CHINESE_LIKE: 0.35,
        RewriteOperator.MANDARIN_SYMBOLIC: 0.40,
        RewriteOperator.BILINGUAL_DSL: 0.45,
        RewriteOperator.MIXED_MIN_TOKEN_FORM: 0.35,
        RewriteOperator.EXAMPLE_DISTILLATION: 0.50,
        RewriteOperator.RULE_EXTRACTION: 0.55,
        RewriteOperator.MERGE_WITH_PREVIOUS: 0.50,
        RewriteOperator.DELETE: 0.10,
        RewriteOperator.KEEP: 1.00,
    }
    return max(6, int(original_tokens * ratios[operator]))
