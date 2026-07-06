from __future__ import annotations

import json
from pathlib import Path

from prompt_compiler.candidates.candidate import Candidate
from prompt_compiler.prompt.chunk import PLACEHOLDER_RE
from prompt_compiler.prompt.template import PromptTemplate
from prompt_compiler.tokenizer import Tokenizer


def write_candidate_template_preview(
    *,
    output_dir: Path,
    original_prompt: PromptTemplate,
    candidates: list[Candidate],
    tokenizer: Tokenizer,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = candidate_template_preview_rows(
        original_prompt=original_prompt,
        candidates=candidates,
        tokenizer=tokenizer,
    )
    _write_jsonl(output_dir / "candidate_templates.jsonl", rows)
    (output_dir / "candidate_templates.md").write_text(_preview_markdown(rows), encoding="utf-8")
    return rows


def candidate_template_preview_rows(
    *,
    original_prompt: PromptTemplate,
    candidates: list[Candidate],
    tokenizer: Tokenizer,
) -> list[dict]:
    original_placeholders = _placeholder_sequence(original_prompt.text)
    original_instruction_tokens = tokenizer.count(original_prompt.instruction_text())
    rows: list[dict] = []
    for candidate in candidates:
        prompt = PromptTemplate(candidate.prompt_template)
        candidate_placeholders = _placeholder_sequence(candidate.prompt_template)
        instruction_tokens = tokenizer.count(prompt.instruction_text())
        rows.append(
            {
                "candidate_id": candidate.id,
                "chunker": candidate.genome.chunker_name,
                "assembly_strategy": candidate.genome.assembly_strategy,
                "original_instruction_tokens": original_instruction_tokens,
                "instruction_tokens": instruction_tokens,
                "token_reduction_estimate": 1.0 - (instruction_tokens / max(original_instruction_tokens, 1)),
                "original_placeholder_sequence": list(original_placeholders),
                "placeholder_sequence": list(candidate_placeholders),
                "placeholder_sequence_ok": candidate_placeholders == original_placeholders,
                "operator_summary": _operator_summary(candidate),
                "chunks": [
                    {
                        "id": chunk.id,
                        "type": chunk.chunk_type.value,
                        "operator": chunk.operator.value,
                        "protected": chunk.protected,
                        "original_tokens": chunk.original_tokens,
                        "compressed_tokens": chunk.compressed_tokens,
                        "text": chunk.text,
                    }
                    for chunk in candidate.chunks
                ],
                "prompt_template": candidate.prompt_template,
            }
        )
    return rows


def _placeholder_sequence(text: str) -> tuple[str, ...]:
    return tuple(match.group(1) for match in PLACEHOLDER_RE.finditer(text))


def _operator_summary(candidate: Candidate) -> dict[str, int]:
    summary: dict[str, int] = {}
    for chunk in candidate.chunks:
        key = f"{chunk.chunk_type.value}:{chunk.operator.value}"
        summary[key] = summary.get(key, 0) + 1
    return summary


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _preview_markdown(rows: list[dict]) -> str:
    lines = ["# Candidate Prompt Templates", ""]
    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {index}. {row['candidate_id']}",
                "",
                f"- chunker: `{row['chunker']}`",
                f"- assembly: `{row['assembly_strategy']}`",
                f"- instruction tokens: `{row['instruction_tokens']}`",
                f"- token reduction estimate: `{row['token_reduction_estimate']:.6f}`",
                f"- placeholders: `{row['placeholder_sequence']}`",
                f"- placeholders ok: `{row['placeholder_sequence_ok']}`",
                "",
                "```text",
                row["prompt_template"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)
