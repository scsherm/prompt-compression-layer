from __future__ import annotations

from prompt_compiler.candidates.candidate import Candidate, CompressedChunk
from prompt_compiler.candidates.genome import CandidateGenome
from prompt_compiler.operators.proposer import RewriteProposer, TokenizerAwareRewritePlanner
from prompt_compiler.operators.rewrite_ops import RewriteOperator
from prompt_compiler.prompt.assembly import assemble_candidate
from prompt_compiler.prompt.chunk import PLACEHOLDER_RE, PromptChunk
from prompt_compiler.prompt.chunkers import generate_chunkings
from prompt_compiler.tokenizer import Tokenizer


class CandidateBuildError(RuntimeError):
    pass


BROAD_OPERATORS = (
    RewriteOperator.KEEP,
    RewriteOperator.SHORT_ENGLISH,
    RewriteOperator.TELEGRAPH_ENGLISH,
    RewriteOperator.SYMBOLIC_DSL,
    RewriteOperator.SCHEMA_ABBREVIATION,
    RewriteOperator.HYBRID_SYMBOLIC_ENGLISH,
    RewriteOperator.SHORT_MANDARIN,
    RewriteOperator.FORMAL_CHINESE,
    RewriteOperator.CLASSICAL_CHINESE_LIKE,
    RewriteOperator.MANDARIN_SYMBOLIC,
    RewriteOperator.BILINGUAL_DSL,
    RewriteOperator.MIXED_MIN_TOKEN_FORM,
)

MULTILINGUAL_OPERATORS = {
    RewriteOperator.SHORT_MANDARIN,
    RewriteOperator.FORMAL_CHINESE,
    RewriteOperator.CLASSICAL_CHINESE_LIKE,
    RewriteOperator.MANDARIN_SYMBOLIC,
    RewriteOperator.BILINGUAL_DSL,
    RewriteOperator.MIXED_MIN_TOKEN_FORM,
}


def seed_population(
    prompt_template: str,
    tokenizer: Tokenizer,
    population_size: int,
    proposer: RewriteProposer | None = None,
    chunker_names: tuple[str, ...] | None = None,
) -> list[Candidate]:
    planner = TokenizerAwareRewritePlanner(tokenizer, proposer=proposer)
    population: list[Candidate] = []
    seen_prompts: set[str] = set()
    chunkings = _selected_chunkings(prompt_template, tokenizer, chunker_names)

    for chunker_name, chunks in chunkings:
        _append_unique(
            population,
            seen_prompts,
            _candidate_from_uniform_operator(chunker_name, chunks, RewriteOperator.KEEP, tokenizer, planner),
        )
        if len(population) >= population_size:
            return population

    for operator in BROAD_OPERATORS[1:]:
        added_for_operator = False
        for chunker_name, chunks in chunkings:
            added_for_operator = _append_unique(
                population,
                seen_prompts,
                _candidate_from_uniform_operator(chunker_name, chunks, operator, tokenizer, planner),
            ) or added_for_operator
            if added_for_operator:
                break
        if len(population) >= population_size:
            return population

    for operator in BROAD_OPERATORS[1:]:
        for chunker_name, chunks in chunkings:
            _append_unique(
                population,
                seen_prompts,
                _candidate_from_uniform_operator(chunker_name, chunks, operator, tokenizer, planner),
            )
            if len(population) >= population_size:
                return population

    for chunker_name, chunks in chunkings:
        _append_unique(population, seen_prompts, _candidate_from_min_tokens(chunker_name, chunks, tokenizer, planner))
        if len(population) >= population_size:
            return population
    return population[:population_size]


def next_generation_from_frontier(
    population: list[Candidate],
    frontier_ids: set[str],
    tokenizer: Tokenizer,
    population_size: int,
    proposer: RewriteProposer | None = None,
    chunker_names: tuple[str, ...] | None = None,
) -> list[Candidate]:
    planner = TokenizerAwareRewritePlanner(tokenizer, proposer=proposer)
    parents = [candidate for candidate in population if candidate.id in frontier_ids] or population[:1]
    children: list[Candidate] = []
    seen_prompts: set[str] = set()

    for parent in parents:
        _append_unique(children, seen_prompts, parent)
        if len(children) >= population_size:
            return children[:population_size]
        for chunk_index, chunk in enumerate(parent.chunks):
            if chunk.protected:
                continue
            prompt_chunk = PromptChunk(
                id=chunk.id,
                text=chunk.original_text,
                chunk_type=chunk.chunk_type,
                start_char=0,
                end_char=len(chunk.original_text),
                protected=chunk.protected,
            )
            for variant in planner.plan(prompt_chunk, BROAD_OPERATORS):
                if variant.operator == chunk.operator:
                    continue
                new_chunks = list(parent.chunks)
                new_chunks[chunk_index] = _compressed_chunk(
                    prompt_chunk,
                    variant.operator,
                    variant.text,
                    variant.token_count,
                    variant.gloss,
                    tokenizer,
                )
                operator_map = dict(parent.genome.chunk_operator_map)
                operator_map[chunk.id] = variant.operator
                child = _build_candidate(
                    parent.genome.chunker_name,
                    operator_map,
                    new_chunks,
                    assembly_strategy=parent.genome.assembly_strategy,
                )
                _append_unique(children, seen_prompts, child)
                if len(children) >= population_size:
                    return children[:population_size]

    return children[:population_size] or parents[:population_size]


def describe_chunkings(
    prompt_template: str,
    tokenizer: Tokenizer,
    chunker_names: tuple[str, ...] | None = None,
) -> dict[str, dict[str, object]]:
    return {
        name: {
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "id": chunk.id,
                    "type": chunk.chunk_type.value,
                    "protected": chunk.protected,
                    "tokens": tokenizer.count(chunk.text),
                    "preview": chunk.text[:160].replace("\n", "\\n"),
                }
                for chunk in chunks
            ],
        }
        for name, chunks in _selected_chunkings(prompt_template, tokenizer, chunker_names)
    }


def _selected_chunkings(
    prompt_template: str,
    tokenizer: Tokenizer,
    chunker_names: tuple[str, ...] | None,
) -> list[tuple[str, list[PromptChunk]]]:
    all_chunkings = generate_chunkings(prompt_template, tokenizer)
    if not chunker_names:
        return list(all_chunkings.items())
    missing = [name for name in chunker_names if name not in all_chunkings]
    if missing:
        available = ", ".join(all_chunkings)
        raise ValueError(f"Unknown chunker(s): {', '.join(missing)}. Available: {available}")
    return [(name, all_chunkings[name]) for name in chunker_names]


def _candidate_from_uniform_operator(
    chunker_name: str,
    chunks: list[PromptChunk],
    operator: RewriteOperator,
    tokenizer: Tokenizer,
    planner: TokenizerAwareRewritePlanner,
) -> Candidate:
    compressed: list[CompressedChunk] = []
    operator_map: dict[str, RewriteOperator] = {}
    for chunk in chunks:
        variants = [item for item in planner.plan(chunk, (operator,)) if item.operator == operator]
        if not variants:
            raise CandidateBuildError(f"no valid rewrite for {chunk.id}/{operator.value}")
        variant = variants[0]
        operator_map[chunk.id] = operator
        compressed.append(_compressed_chunk(chunk, variant.operator, variant.text, variant.token_count, variant.gloss, tokenizer))
    return _build_candidate(chunker_name, operator_map, compressed)


def _candidate_from_min_tokens(
    chunker_name: str,
    chunks: list[PromptChunk],
    tokenizer: Tokenizer,
    planner: TokenizerAwareRewritePlanner,
) -> Candidate:
    compressed: list[CompressedChunk] = []
    operator_map: dict[str, RewriteOperator] = {}
    for chunk in chunks:
        variants = [variant for variant in planner.plan(chunk, BROAD_OPERATORS) if variant.text.strip() or chunk.protected]
        if chunk.protected:
            variants = [variant for variant in variants if variant.text == chunk.text]
        if not variants:
            raise CandidateBuildError(f"no valid rewrite for {chunk.id}")
        variant = variants[0]
        operator_map[chunk.id] = variant.operator
        compressed.append(_compressed_chunk(chunk, variant.operator, variant.text, variant.token_count, variant.gloss, tokenizer))
    return _build_candidate(chunker_name, operator_map, compressed)


def _compressed_chunk(
    chunk: PromptChunk,
    operator: RewriteOperator,
    text: str,
    token_count: int,
    gloss: str,
    tokenizer: Tokenizer,
) -> CompressedChunk:
    return CompressedChunk(
        id=chunk.id,
        original_text=chunk.text,
        text=text,
        chunk_type=chunk.chunk_type,
        operator=operator,
        original_tokens=tokenizer.count(chunk.text),
        compressed_tokens=token_count,
        gloss=gloss,
        protected=chunk.protected,
    )


def _build_candidate(
    chunker_name: str,
    operator_map: dict[str, RewriteOperator],
    chunks: list[CompressedChunk],
    assembly_strategy: str = "newline",
) -> Candidate:
    original_placeholders = _placeholder_sequence(chunk.original_text for chunk in chunks)
    prompt = assemble_candidate(chunks, strategy=assembly_strategy)
    assembled_placeholders = _placeholder_sequence((prompt,))
    if assembled_placeholders != original_placeholders:
        raise CandidateBuildError(
            "assembled prompt changed placeholder boundary: "
            f"original={original_placeholders}, assembled={assembled_placeholders}"
        )
    if _needs_output_language_guard(operator_map) and "out_lang=EN" not in prompt:
        prompt = f"out_lang=EN\n{prompt}"
    genome = CandidateGenome(
        chunker_name=chunker_name,
        chunk_operator_map=operator_map,
        assembly_strategy=assembly_strategy,
    )
    return Candidate(genome=genome, chunks=tuple(chunks), prompt_template=prompt)


def _needs_output_language_guard(operator_map: dict[str, RewriteOperator]) -> bool:
    return any(operator in MULTILINGUAL_OPERATORS for operator in operator_map.values())


def _placeholder_sequence(texts) -> tuple[str, ...]:
    names: list[str] = []
    for text in texts:
        names.extend(match.group(1) for match in PLACEHOLDER_RE.finditer(text))
    return tuple(names)


def _append_unique(population: list[Candidate], seen_prompts: set[str], candidate: Candidate) -> bool:
    if candidate.prompt_template in seen_prompts:
        return False
    seen_prompts.add(candidate.prompt_template)
    population.append(candidate)
    return True
