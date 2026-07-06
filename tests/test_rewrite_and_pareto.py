import unittest

from prompt_compiler.eval.pareto import compute_pareto_frontier
from prompt_compiler.eval.evaluator import CandidateReport
from prompt_compiler.candidates.generation import seed_population
from prompt_compiler.operators.proposer import TokenizerAwareRewritePlanner
from prompt_compiler.operators.rewrite_ops import RewriteOperator
from prompt_compiler.prompt.chunk import ChunkType, PromptChunk
from prompt_compiler.tokenizer import ApproxTokenizer


class CompactTestProposer:
    def rewrite(self, chunk: PromptChunk, operator: RewriteOperator) -> tuple[str, str]:
        if chunk.protected or operator == RewriteOperator.KEEP:
            return chunk.text, "keep"
        rewrites = {
            RewriteOperator.SHORT_ENGLISH: "Return JSON only; no markdown.",
            RewriteOperator.TELEGRAPH_ENGLISH: "return JSON no markdown",
            RewriteOperator.SYMBOLIC_DSL: "out=JSON; md=0",
            RewriteOperator.SCHEMA_ABBREVIATION: "JSON; md=0",
            RewriteOperator.HYBRID_SYMBOLIC_ENGLISH: "JSON only; md=0",
            RewriteOperator.SHORT_MANDARIN: "只返JSON；禁md",
            RewriteOperator.FORMAL_CHINESE: "输出JSON；禁markdown",
            RewriteOperator.CLASSICAL_CHINESE_LIKE: "须JSON；禁md",
            RewriteOperator.MANDARIN_SYMBOLIC: "out=JSON；禁md",
            RewriteOperator.BILINGUAL_DSL: "out=JSON; 禁md",
            RewriteOperator.MIXED_MIN_TOKEN_FORM: "JSON;禁md",
        }
        return rewrites.get(operator, "JSON"), "test rewrite"


class RewriteAndParetoTests(unittest.TestCase):
    def test_tokenizer_aware_planner_includes_mandarin_and_mixed_min_token_forms(self):
        chunk = PromptChunk(
            id="c1",
            text="You must return only valid JSON. Do not include markdown.",
            chunk_type=ChunkType.OUTPUT_SCHEMA,
            start_char=0,
            end_char=58,
        )

        variants = TokenizerAwareRewritePlanner(ApproxTokenizer(), proposer=CompactTestProposer()).plan(chunk)
        operators = {variant.operator for variant in variants}

        self.assertIn(RewriteOperator.SHORT_MANDARIN, operators)
        self.assertIn(RewriteOperator.CLASSICAL_CHINESE_LIKE, operators)
        self.assertIn(RewriteOperator.MANDARIN_SYMBOLIC, operators)
        self.assertIn(RewriteOperator.MIXED_MIN_TOKEN_FORM, operators)
        self.assertEqual(variants, sorted(variants, key=lambda item: item.token_count))

    def test_pareto_frontier_removes_dominated_candidates(self):
        reports = [
            CandidateReport(candidate_id="slow_good", prompt_template="a", instruction_tokens=80, token_reduction=0.20, avg_semantic_drift=0.05, objective_score=0.20, format_failure_rate=0.0, task_failure_rate=0.0, output_variance=0.0, examples_failed=[], operator_summary={}),
            CandidateReport(candidate_id="fast_good", prompt_template="b", instruction_tokens=40, token_reduction=0.60, avg_semantic_drift=0.05, objective_score=0.10, format_failure_rate=0.0, task_failure_rate=0.0, output_variance=0.0, examples_failed=[], operator_summary={}),
            CandidateReport(candidate_id="fast_risky", prompt_template="c", instruction_tokens=20, token_reduction=0.80, avg_semantic_drift=0.30, objective_score=0.50, format_failure_rate=0.1, task_failure_rate=0.2, output_variance=0.0, examples_failed=[], operator_summary={}),
        ]

        frontier = compute_pareto_frontier(reports)
        frontier_ids = {report.candidate_id for report in frontier}

        self.assertNotIn("slow_good", frontier_ids)
        self.assertIn("fast_good", frontier_ids)
        self.assertIn("fast_risky", frontier_ids)

    def test_seed_population_reaches_multilingual_candidates_when_first_chunking_duplicates(self):
        prompt = (
            "You are doing alert triage.\n"
            "Return only valid JSON. Do not include markdown.\n"
            "The status field must be either OPEN or CLOSED.\n"
            "Input:\n{{input}}"
        )

        population = seed_population(prompt, ApproxTokenizer(), population_size=16, proposer=CompactTestProposer())
        operators = {
            chunk.operator
            for candidate in population
            for chunk in candidate.chunks
        }

        self.assertIn(RewriteOperator.SHORT_MANDARIN, operators)
        self.assertIn(RewriteOperator.MANDARIN_SYMBOLIC, operators)


if __name__ == "__main__":
    unittest.main()
