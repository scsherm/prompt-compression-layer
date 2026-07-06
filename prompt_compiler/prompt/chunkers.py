from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from prompt_compiler.prompt.chunk import PLACEHOLDER_RE, ChunkType, PromptChunk, detect_chunk_type, is_protected
from prompt_compiler.tokenizer import ApproxTokenizer, Tokenizer


class Chunker(Protocol):
    name: str

    def chunk(self, prompt: str) -> list[PromptChunk]:
        ...


def _make_chunk(prefix: str, index: int, text: str, start: int, end: int) -> PromptChunk:
    chunk_type = detect_chunk_type(text)
    return PromptChunk(
        id=f"{prefix}_{index}",
        text=text,
        chunk_type=chunk_type,
        start_char=start,
        end_char=end,
        protected=is_protected(text),
    )


INPUT_LABEL_RE = re.compile(r"(?i)(?:^|\n)\s*(input|user request|alert|document|query)\s*:\s*$")


def _append_text_chunk(
    chunks: list[PromptChunk],
    *,
    chunk_id: str,
    text: str,
    start: int,
) -> None:
    text = text.strip()
    if not text:
        return
    match = INPUT_LABEL_RE.search(text)
    if match and match.start() > 0:
        prefix = text[: match.start()].strip()
        label = text[match.start() :].strip()
        if prefix:
            chunks.append(
                PromptChunk(
                    id=f"{chunk_id}_body",
                    text=prefix,
                    chunk_type=detect_chunk_type(prefix),
                    start_char=start,
                    end_char=start + len(prefix),
                    protected=is_protected(prefix),
                )
            )
        label_start = start + text.rfind(label)
        chunks.append(
            PromptChunk(
                id=f"{chunk_id}_label",
                text=label,
                chunk_type=ChunkType.INPUT_SLOT,
                start_char=label_start,
                end_char=label_start + len(label),
                protected=True,
            )
        )
        return
    chunks.append(
        PromptChunk(
            id=chunk_id,
            text=text,
            chunk_type=detect_chunk_type(text),
            start_char=start,
            end_char=start + len(text),
            protected=is_protected(text),
        )
    )


def _split_placeholder_chunks(chunks: list[PromptChunk]) -> list[PromptChunk]:
    split: list[PromptChunk] = []
    for chunk in chunks:
        matches = list(PLACEHOLDER_RE.finditer(chunk.text))
        if not matches:
            split.append(chunk)
            continue

        cursor = 0
        part_index = 0
        for match in matches:
            before = chunk.text[cursor : match.start()].strip()
            if before:
                start = chunk.start_char + cursor + chunk.text[cursor : match.start()].find(before)
                _append_text_chunk(
                    split,
                    chunk_id=f"{chunk.id}_{part_index}",
                    text=before,
                    start=start,
                )
                part_index += 1

            placeholder = match.group(0)
            split.append(
                PromptChunk(
                    id=f"{chunk.id}_{part_index}",
                    text=placeholder,
                    chunk_type=ChunkType.INPUT_SLOT,
                    start_char=chunk.start_char + match.start(),
                    end_char=chunk.start_char + match.end(),
                    protected=True,
                )
            )
            part_index += 1
            cursor = match.end()

        after = chunk.text[cursor:].strip()
        if after:
            start = chunk.start_char + cursor + chunk.text[cursor:].find(after)
            _append_text_chunk(
                split,
                chunk_id=f"{chunk.id}_{part_index}",
                text=after,
                start=start,
            )
    return split


@dataclass
class ParagraphChunker:
    name: str = "paragraph"

    def chunk(self, prompt: str) -> list[PromptChunk]:
        chunks: list[PromptChunk] = []
        start = 0
        index = 0
        for match in re.finditer(r"\n\s*\n", prompt):
            text = prompt[start : match.start()].strip()
            if text:
                chunks.append(_make_chunk(self.name, index, text, start, match.start()))
                index += 1
            start = match.end()
        text = prompt[start:].strip()
        if text:
            chunks.append(_make_chunk(self.name, index, text, start, len(prompt)))
        return chunks


@dataclass
class SentenceChunker:
    name: str = "sentence"

    def chunk(self, prompt: str) -> list[PromptChunk]:
        spans = [match for match in re.finditer(r"[^.!?\n]+[.!?]?|[^\S\n]*\n+[^\S\n]*", prompt)]
        chunks: list[PromptChunk] = []
        for match in spans:
            text = match.group(0).strip()
            if not text:
                continue
            chunks.append(_make_chunk(self.name, len(chunks), text, match.start(), match.end()))
        return chunks


@dataclass
class MarkdownHeadingChunker:
    name: str = "markdown"

    def chunk(self, prompt: str) -> list[PromptChunk]:
        heading_matches = list(re.finditer(r"(?m)^#{1,6}\s+.*$", prompt))
        if not heading_matches:
            return ParagraphChunker(name=self.name).chunk(prompt)
        chunks: list[PromptChunk] = []
        if heading_matches[0].start() > 0:
            text = prompt[: heading_matches[0].start()].strip()
            if text:
                chunks.append(_make_chunk(self.name, len(chunks), text, 0, heading_matches[0].start()))
        for i, match in enumerate(heading_matches):
            end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(prompt)
            text = prompt[match.start() : end].strip()
            chunks.append(_make_chunk(self.name, len(chunks), text, match.start(), end))
        return chunks


@dataclass
class SchemaAwareChunker:
    name: str = "schema_aware"

    def chunk(self, prompt: str) -> list[PromptChunk]:
        chunks: list[PromptChunk] = []
        last = 0
        schema_re = r"```.*?```|(?<!\{)\{(?!\{)[\s\S]*?(?<!\})\}(?!\})|\[[\s\S]*?\]"
        for match in re.finditer(schema_re, prompt, re.DOTALL):
            before = prompt[last : match.start()].strip()
            if before:
                chunks.extend(ParagraphChunker(name=self.name).chunk(before))
            schema = match.group(0).strip()
            chunks.append(_make_chunk(self.name, len(chunks), schema, match.start(), match.end()))
            last = match.end()
        tail = prompt[last:].strip()
        if tail:
            base = len(chunks)
            for chunk in ParagraphChunker(name=self.name).chunk(tail):
                chunks.append(
                    PromptChunk(
                        id=f"{self.name}_{base}",
                        text=chunk.text,
                        chunk_type=chunk.chunk_type,
                        start_char=last + chunk.start_char,
                        end_char=last + chunk.end_char,
                        protected=chunk.protected,
                    )
                )
                base += 1
        return chunks or ParagraphChunker(name=self.name).chunk(prompt)


@dataclass
class InstructionRoleChunker:
    name: str = "instruction_role"

    def chunk(self, prompt: str) -> list[PromptChunk]:
        chunks: list[PromptChunk] = []
        start = 0
        current: list[str] = []
        current_start = 0
        role_line = re.compile(r"(?i)^\s*(role|task|rules?|constraints?|input|user request|output|return|schema)\s*:")
        for line in prompt.splitlines(keepends=True):
            if role_line.search(line) and current:
                text = "".join(current).strip()
                chunks.append(_make_chunk(self.name, len(chunks), text, current_start, start))
                current = []
                current_start = start
            if not current:
                current_start = start
            current.append(line)
            start += len(line)
        if current:
            chunks.append(_make_chunk(self.name, len(chunks), "".join(current).strip(), current_start, len(prompt)))
        return chunks


@dataclass
class TokenWindowChunker:
    max_tokens: int = 80
    tokenizer: Tokenizer | None = None
    name: str = "token_window"

    def chunk(self, prompt: str) -> list[PromptChunk]:
        tokenizer = self.tokenizer or ApproxTokenizer()
        words = re.finditer(r"\S+\s*", prompt)
        chunks: list[PromptChunk] = []
        current: list[str] = []
        current_start: int | None = None
        current_end = 0
        for match in words:
            token = match.group(0)
            next_text = "".join(current) + token
            if current and tokenizer.count(next_text) > self.max_tokens:
                text = "".join(current).strip()
                chunks.append(_make_chunk(self.name, len(chunks), text, current_start or 0, current_end))
                current = []
                current_start = match.start()
            if current_start is None:
                current_start = match.start()
            current.append(token)
            current_end = match.end()
        if current:
            chunks.append(_make_chunk(self.name, len(chunks), "".join(current).strip(), current_start or 0, current_end))
        return chunks


def generate_chunkings(prompt: str, tokenizer: Tokenizer | None = None) -> dict[str, list[PromptChunk]]:
    chunkers: list[Chunker] = [
        ParagraphChunker(),
        SentenceChunker(),
        MarkdownHeadingChunker(),
        SchemaAwareChunker(),
        InstructionRoleChunker(),
        TokenWindowChunker(tokenizer=tokenizer),
    ]
    return {chunker.name: _split_placeholder_chunks(chunker.chunk(prompt)) for chunker in chunkers}
