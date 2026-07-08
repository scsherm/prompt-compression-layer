from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from prompt_compiler.candidates.candidate import Candidate, CompressedChunk
from prompt_compiler.candidates.genome import CandidateGenome
from prompt_compiler.operators.proposer import (
    RewriteAttempt,
    RewriteProposer,
    RewriteVariant,
    TokenizerAwareRewritePlanner,
    make_rewrite_attempts,
)
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


@dataclass
class ChunkProposalPool:
    chunker_name: str
    chunks: tuple[PromptChunk, ...]
    variants_by_chunk: dict[str, list[RewriteVariant]] = field(default_factory=dict)
    attempted: set[tuple[str, str]] = field(default_factory=set)


@dataclass(frozen=True)
class SeedPopulationResult:
    candidates: list[Candidate]
    proposal_pools: tuple[ChunkProposalPool, ...]


def seed_population(
    prompt_template: str,
    tokenizer: Tokenizer,
    population_size: int,
    proposer: RewriteProposer | None = None,
    chunker_names: tuple[str, ...] | None = None,
    token_window_size: int = 80,
    proposal_attempts: int = 1,
    proposal_jitter_seed: int = 0,
) -> list[Candidate]:
    return seed_population_with_proposals(
        prompt_template,
        tokenizer,
        population_size,
        proposer=proposer,
        chunker_names=chunker_names,
        token_window_size=token_window_size,
        proposal_attempts=proposal_attempts,
        proposal_jitter_seed=proposal_jitter_seed,
    ).candidates


def seed_population_with_proposals(
    prompt_template: str,
    tokenizer: Tokenizer,
    population_size: int,
    proposer: RewriteProposer | None = None,
    chunker_names: tuple[str, ...] | None = None,
    token_window_size: int = 80,
    proposal_attempts: int = 1,
    proposal_jitter_seed: int = 0,
) -> SeedPopulationResult:
    planner = TokenizerAwareRewritePlanner(tokenizer, proposer=proposer)
    population: list[Candidate] = []
    seen_prompts: set[str] = set()
    chunkings = _selected_chunkings(prompt_template, tokenizer, chunker_names, token_window_size=token_window_size)
    attempts = make_rewrite_attempts(proposal_attempts, seed=proposal_jitter_seed)
    pools = [ChunkProposalPool(chunker_name=name, chunks=tuple(chunks)) for name, chunks in chunkings]

    for pool in pools:
        _append_candidate_if_valid(
            population,
            seen_prompts,
            lambda pool=pool: _candidate_from_uniform_operator_pool(
                pool,
                RewriteOperator.KEEP,
                tokenizer,
                planner,
                attempts,
            ),
        )
        if len(population) >= population_size:
            return SeedPopulationResult(population, tuple(pools))

    for operator in BROAD_OPERATORS[1:]:
        for pool in pools:
            _append_candidate_if_valid(
                population,
                seen_prompts,
                lambda pool=pool, operator=operator: _candidate_from_uniform_operator_pool(
                    pool,
                    operator,
                    tokenizer,
                    planner,
                    attempts,
                ),
            )
            if len(population) >= population_size:
                return SeedPopulationResult(population, tuple(pools))
        if len(population) >= population_size:
            return SeedPopulationResult(population, tuple(pools))

    for operator in BROAD_OPERATORS[1:]:
        for pool in pools:
            _append_candidate_if_valid(
                population,
                seen_prompts,
                lambda pool=pool, operator=operator: _candidate_from_uniform_operator_pool(
                    pool,
                    operator,
                    tokenizer,
                    planner,
                    attempts,
                ),
            )
            if len(population) >= population_size:
                return SeedPopulationResult(population, tuple(pools))

    for variant_rank in range(1, max(1, proposal_attempts)):
        for operator in BROAD_OPERATORS[1:]:
            for pool in pools:
                _append_candidate_if_valid(
                    population,
                    seen_prompts,
                    lambda pool=pool, operator=operator, variant_rank=variant_rank: _candidate_from_uniform_operator_pool(
                        pool,
                        operator,
                        tokenizer,
                        planner,
                        attempts,
                        variant_rank=variant_rank,
                    ),
                )
                if len(population) >= population_size:
                    return SeedPopulationResult(population, tuple(pools))

    for pool in pools:
        _append_candidate_if_valid(
            population,
            seen_prompts,
            lambda pool=pool: _candidate_from_min_tokens_pool(
                pool,
                tokenizer,
                planner,
                attempts,
            ),
        )
        if len(population) >= population_size:
            return SeedPopulationResult(population, tuple(pools))
    return SeedPopulationResult(population[:population_size], tuple(pools))


def next_generation_from_frontier(
    population: list[Candidate],
    frontier_ids: set[str],
    tokenizer: Tokenizer,
    population_size: int,
    proposer: RewriteProposer | None = None,
    chunker_names: tuple[str, ...] | None = None,
    frontier_order: tuple[str, ...] | None = None,
    elite_ids: tuple[str, ...] = (),
    proposal_attempts: int = 1,
    proposal_jitter_seed: int = 0,
) -> list[Candidate]:
    planner = TokenizerAwareRewritePlanner(tokenizer, proposer=proposer)
    attempts = make_rewrite_attempts(proposal_attempts, seed=proposal_jitter_seed)
    parents = _ordered_frontier_parents(population, frontier_ids, frontier_order) or population[:1]
    children: list[Candidate] = []
    seen_prompts: set[str] = set()

    candidates_by_id = {candidate.id: candidate for candidate in population}
    for elite_id in elite_ids:
        elite = candidates_by_id.get(elite_id)
        if elite:
            _append_unique(children, seen_prompts, elite)
        if len(children) >= population_size:
            return children[:population_size]

    if not elite_ids:
        for parent in parents:
            _append_unique(children, seen_prompts, parent)
            if len(children) >= population_size:
                return children[:population_size]

    mutation_streams = [_candidate_mutations(parent, tokenizer, planner, attempts) for parent in parents]
    while mutation_streams and len(children) < population_size:
        active_streams = []
        for stream in mutation_streams:
            try:
                child = next(stream)
            except StopIteration:
                continue
            _append_unique(children, seen_prompts, child)
            if len(children) >= population_size:
                return children[:population_size]
            active_streams.append(stream)
        mutation_streams = active_streams

    return children[:population_size] or parents[:population_size]


def _ordered_frontier_parents(
    population: list[Candidate],
    frontier_ids: set[str],
    frontier_order: tuple[str, ...] | None,
) -> list[Candidate]:
    candidates_by_id = {candidate.id: candidate for candidate in population}
    ordered_ids: list[str] = []
    if frontier_order:
        ordered_ids.extend(candidate_id for candidate_id in frontier_order if candidate_id in candidates_by_id)
    ordered_ids.extend(candidate.id for candidate in population if candidate.id in frontier_ids and candidate.id not in ordered_ids)
    return [candidates_by_id[candidate_id] for candidate_id in ordered_ids]


def _candidate_mutations(
    parent: Candidate,
    tokenizer: Tokenizer,
    planner: TokenizerAwareRewritePlanner,
    attempts: tuple[RewriteAttempt, ...],
) -> Iterator[Candidate]:
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
        for variant in planner.plan(prompt_chunk, BROAD_OPERATORS, attempts=attempts):
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
            yield child


def describe_chunkings(
    prompt_template: str,
    tokenizer: Tokenizer,
    chunker_names: tuple[str, ...] | None = None,
    token_window_size: int = 80,
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
        for name, chunks in _selected_chunkings(
            prompt_template,
            tokenizer,
            chunker_names,
            token_window_size=token_window_size,
        )
    }


def _selected_chunkings(
    prompt_template: str,
    tokenizer: Tokenizer,
    chunker_names: tuple[str, ...] | None,
    *,
    token_window_size: int = 80,
) -> list[tuple[str, list[PromptChunk]]]:
    all_chunkings = generate_chunkings(prompt_template, tokenizer, token_window_size=token_window_size)
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


def _candidate_from_uniform_operator_pool(
    pool: ChunkProposalPool,
    operator: RewriteOperator,
    tokenizer: Tokenizer,
    planner: TokenizerAwareRewritePlanner,
    attempts: tuple[RewriteAttempt, ...],
    *,
    variant_rank: int = 0,
) -> Candidate:
    compressed: list[CompressedChunk] = []
    operator_map: dict[str, RewriteOperator] = {}
    for chunk in pool.chunks:
        effective_operator = RewriteOperator.KEEP if chunk.protected else operator
        variants = _variants_for(pool, chunk, effective_operator, planner, attempts)
        if not variants:
            variants = _variants_for(pool, chunk, RewriteOperator.KEEP, planner, attempts)
        if not variants:
            raise CandidateBuildError(f"no valid rewrite for {chunk.id}/{effective_operator.value}")
        variant = variants[min(variant_rank, len(variants) - 1)]
        operator_map[chunk.id] = variant.operator
        compressed.append(_compressed_chunk(chunk, variant.operator, variant.text, variant.token_count, variant.gloss, tokenizer))
    return _build_candidate(pool.chunker_name, operator_map, compressed)


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


def _candidate_from_min_tokens_pool(
    pool: ChunkProposalPool,
    tokenizer: Tokenizer,
    planner: TokenizerAwareRewritePlanner,
    attempts: tuple[RewriteAttempt, ...],
) -> Candidate:
    compressed: list[CompressedChunk] = []
    operator_map: dict[str, RewriteOperator] = {}
    for chunk in pool.chunks:
        variants: list[RewriteVariant] = []
        for operator in BROAD_OPERATORS:
            variants.extend(_variants_for(pool, chunk, RewriteOperator.KEEP if chunk.protected else operator, planner, attempts))
        variants = [variant for variant in variants if variant.text.strip() or chunk.protected]
        if chunk.protected:
            variants = [variant for variant in variants if variant.text == chunk.text]
        if not variants:
            raise CandidateBuildError(f"no valid rewrite for {chunk.id}")
        variant = sorted(variants, key=lambda item: (item.token_count, item.operator.value, item.attempt_id))[0]
        operator_map[chunk.id] = variant.operator
        compressed.append(_compressed_chunk(chunk, variant.operator, variant.text, variant.token_count, variant.gloss, tokenizer))
    return _build_candidate(pool.chunker_name, operator_map, compressed)


def _variants_for(
    pool: ChunkProposalPool,
    chunk: PromptChunk,
    operator: RewriteOperator,
    planner: TokenizerAwareRewritePlanner,
    attempts: tuple[RewriteAttempt, ...],
) -> list[RewriteVariant]:
    attempted_key = (chunk.id, operator.value)
    if attempted_key not in pool.attempted:
        existing = {
            _variant_dedupe_key(variant)
            for variant in pool.variants_by_chunk.get(chunk.id, [])
        }
        new_variants = planner.plan(chunk, (operator,), attempts=attempts)
        for variant in new_variants:
            key = _variant_dedupe_key(variant)
            if key in existing:
                continue
            existing.add(key)
            pool.variants_by_chunk.setdefault(chunk.id, []).append(variant)
        pool.attempted.add(attempted_key)
    return [
        variant
        for variant in pool.variants_by_chunk.get(chunk.id, [])
        if variant.operator == operator
    ]


def _variant_dedupe_key(variant: RewriteVariant) -> str:
    return " ".join(variant.text.split())


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


def _append_candidate_if_valid(population: list[Candidate], seen_prompts: set[str], build_candidate) -> bool:
    try:
        candidate = build_candidate()
    except CandidateBuildError:
        return False
    return _append_unique(population, seen_prompts, candidate)
