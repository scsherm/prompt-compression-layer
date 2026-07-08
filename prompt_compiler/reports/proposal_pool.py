from __future__ import annotations

import json
from pathlib import Path

from prompt_compiler.candidates.generation import ChunkProposalPool
from prompt_compiler.tokenizer import Tokenizer


def write_proposal_pool(output_dir: Path, pools: tuple[ChunkProposalPool, ...], tokenizer: Tokenizer) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "proposal_pool.jsonl"
    markdown_path = output_dir / "proposal_pool.md"
    rows = _proposal_rows(pools, tokenizer)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    markdown_path.write_text(_proposal_markdown(pools, tokenizer), encoding="utf-8")
    return {
        "proposal_pool_jsonl": str(jsonl_path),
        "proposal_pool_markdown": str(markdown_path),
    }


def _proposal_rows(pools: tuple[ChunkProposalPool, ...], tokenizer: Tokenizer) -> list[dict]:
    rows: list[dict] = []
    for pool in pools:
        for chunk in pool.chunks:
            original_tokens = tokenizer.count(chunk.text)
            variants = pool.variants_by_chunk.get(chunk.id, [])
            for variant in variants:
                rows.append(
                    {
                        "chunker": pool.chunker_name,
                        "chunk_id": chunk.id,
                        "chunk_type": chunk.chunk_type.value,
                        "protected": chunk.protected,
                        "original_tokens": original_tokens,
                        "original_text": chunk.text,
                        "operator": variant.operator.value,
                        "attempt_id": variant.attempt_id,
                        "jitter": variant.jitter,
                        "token_count": variant.token_count,
                        "token_delta": original_tokens - variant.token_count,
                        "rewritten_text": variant.text,
                        "gloss": variant.gloss,
                    }
                )
    return rows


def _proposal_markdown(pools: tuple[ChunkProposalPool, ...], tokenizer: Tokenizer) -> str:
    lines = ["# Proposal Pool", ""]
    for pool in pools:
        lines.extend([f"## Chunker: `{pool.chunker_name}`", ""])
        for chunk in pool.chunks:
            original_tokens = tokenizer.count(chunk.text)
            lines.extend(
                [
                    f"### `{chunk.id}` `{chunk.chunk_type.value}` protected={str(chunk.protected).lower()} tokens={original_tokens}",
                    "",
                    "Original:",
                    "",
                    "```text",
                    chunk.text,
                    "```",
                    "",
                ]
            )
            variants = pool.variants_by_chunk.get(chunk.id, [])
            if not variants:
                lines.extend(["No variants generated for this chunk.", ""])
                continue
            for index, variant in enumerate(variants, start=1):
                lines.extend(
                    [
                        (
                            f"{index}. operator=`{variant.operator.value}` "
                            f"attempt=`{variant.attempt_id}` tokens={variant.token_count} "
                            f"delta={original_tokens - variant.token_count}"
                        ),
                        "",
                        "```text",
                        variant.text,
                        "```",
                        "",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"
