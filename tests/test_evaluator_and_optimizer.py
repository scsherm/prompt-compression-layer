import json
import tempfile
import unittest
from pathlib import Path

from prompt_compiler.eval.contract_checks import OutputContract
from prompt_compiler.eval.embedding_distance import EmbeddingDriftScorer, euclidean_distance
from prompt_compiler.eval.evaluator import Evaluator
from prompt_compiler.models.mock import RuleBasedMockModel
from prompt_compiler.optimize.optimizer import optimize_prompt
from prompt_compiler.prompt.template import PromptTemplate
from prompt_compiler.tokenizer import ApproxTokenizer


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
            result = optimize_prompt(
                target_model=RuleBasedMockModel(),
                original_prompt=prompt,
                inputs=inputs,
                output_dir=Path(tmp),
                epochs=2,
                population_size=10,
                tokenizer=ApproxTokenizer(),
                output_contract=OutputContract(require_json=True, required_fields=("status",)),
            )

            self.assertIn("{{input}}", result.best_prompt_template)
            self.assertTrue(result.pareto_frontier)
            self.assertTrue((Path(tmp) / "best_prompt.txt").exists())
            self.assertTrue((Path(tmp) / "compression_report.json").exists())
            self.assertTrue((Path(tmp) / "reference_dataset.jsonl").exists())
            self.assertTrue((Path(tmp) / "candidate_reports.jsonl").exists())
            self.assertTrue((Path(tmp) / "candidate_outputs.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
