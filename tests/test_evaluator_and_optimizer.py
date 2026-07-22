import json
import tempfile
import unittest
from pathlib import Path

from prompt_compiler.eval.contract_checks import OutputContract
from prompt_compiler.eval.embedding_distance import EmbeddingDriftScorer, euclidean_distance
from prompt_compiler.eval.evaluator import Evaluator
from prompt_compiler.models.mock import RuleBasedMockModel
from prompt_compiler.optimize.optimizer import optimize_prompt
from prompt_compiler.operators.full_prompt_proposer import PromptProposal
from prompt_compiler.prompt.template import PromptTemplate
from prompt_compiler.tokenizer import ApproxTokenizer


class FeedbackTestProposer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.contexts = []

    def propose(self, context, *, batch_size):
        self.contexts.append(context)
        prompts = (
            ("Triage alert. Return JSON status OPEN or CLOSED.\nInput: {{input}}",),
            ("JSON status OPEN|CLOSED for: {{input}}",),
        )[min(context.round_index, 1)]
        parent_ids = tuple(item.candidate_id for item in context.frontier)
        return tuple(
            PromptProposal(
                prompt_template=prompt,
                instruction_tokens=self.tokenizer.count(PromptTemplate(prompt).instruction_text()),
                token_savings=0,
                rationale="compress while preserving measured mock behavior",
                based_on_candidate_ids=parent_ids,
            )
            for prompt in prompts[:batch_size]
        )


class EvaluatorAndOptimizerTests(unittest.TestCase):
    def test_embedding_drift_scorer_uses_euclidean_distance(self):
        class FakeEmbeddingClient:
            name = "fake"

            def embed(self, texts):
                return [[2.0, 0.0], [0.0, 1.0]]

        scorer = EmbeddingDriftScorer(FakeEmbeddingClient())

        self.assertEqual(euclidean_distance([1.0, 0.0], [1.0, 0.0]), 0.0)
        self.assertAlmostEqual(scorer.distance("a", "b"), 5**0.5)

    def test_invalid_json_is_reported_separately_from_equivalence_distance(self):
        evaluator = Evaluator(tokenizer=ApproxTokenizer(), output_contract=OutputContract(require_json=True))

        result = evaluator.compare_outputs(
            candidate_output="status: OPEN",
            reference_output=json.dumps({"status": "OPEN"}),
            input_id="ex1",
        )

        self.assertFalse(result.contract_ok)
        self.assertTrue(result.failures)
        self.assertLess(result.equivalence_distance, 10.0)

    def test_optimizer_writes_core_artifacts_and_keeps_input_slot(self):
        prompt = PromptTemplate(
            "You are doing alert triage.\n"
            "Return only valid JSON. Do not include markdown.\n"
            "The status field must be either OPEN or CLOSED.\n"
            "Input:\n{{input}}"
        )
        inputs = [
            {"id": "ex1", "input": "benign login from known device"},
            {"id": "ex2", "input": "malware beacon detected"},
            {"id": "ex3", "input": "routine patch completed"},
            {"id": "ex4", "input": "credential theft alert"},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tokenizer = ApproxTokenizer()
            proposer = FeedbackTestProposer(tokenizer)
            result = optimize_prompt(
                target_model=RuleBasedMockModel(),
                prompt_proposer=proposer,
                original_prompt=prompt,
                inputs=inputs,
                output_dir=Path(tmp),
                rounds=2,
                batch_size=1,
                tokenizer=tokenizer,
                output_contract=OutputContract(require_json=True, required_fields=("status",)),
                baseline_repeats=0,
                max_candidate_rollouts=1,
                log_to_stderr=False,
            )

            self.assertIn("{{input}}", result.best_prompt_template)
            self.assertTrue(result.pareto_frontier)
            self.assertEqual(len(proposer.contexts), 2)
            self.assertTrue(proposer.contexts[1].frontier)
            feedback = proposer.contexts[1].frontier[0]
            self.assertEqual(
                feedback.instruction_tokens,
                tokenizer.count(PromptTemplate(feedback.prompt_template).instruction_text()),
            )
            self.assertEqual(feedback.behavior_loss, 0.0)
            self.assertTrue(feedback.diff_from_original)
            self.assertTrue(feedback.worst_residuals)
            self.assertTrue(feedback.worst_residuals[0].reference_completion)
            self.assertTrue(feedback.worst_residuals[0].candidate_completion)
            self.assertNotIn(prompt.text, [trial.prompt for trial in result.search_archive.trials])
            self.assertTrue(
                all(
                    trial.instruction_tokens < result.evaluation_report.original_instruction_tokens
                    for trial in result.search_archive.trials
                )
            )
            self.assertTrue((Path(tmp) / "best_prompt.txt").exists())
            self.assertTrue((Path(tmp) / "compression_report.json").exists())
            self.assertTrue((Path(tmp) / "reference_dataset.jsonl").exists())
            self.assertTrue((Path(tmp) / "candidate_reports.jsonl").exists())
            self.assertTrue((Path(tmp) / "candidate_outputs.jsonl").exists())
            self.assertTrue((Path(tmp) / "search_archive.json").exists())


if __name__ == "__main__":
    unittest.main()
