import unittest
import json

from prompt_compiler.eval.extraction_metrics import aggregate_extraction, score_extraction
from prompt_compiler.optimize.reward import (
    RewardEstimate,
    estimate_baseline_noise,
    select_for_additional_rollouts,
)
from prompt_compiler.optimize.search_state import ExampleFeedback, SearchArchive


class FeedbackSearchTests(unittest.TestCase):
    def test_ground_truth_extraction_reports_precision_and_recall(self):
        expected = {"malicious_ips": ["10.0.0.1", "10.0.0.2"]}
        scored = score_extraction(
            json.dumps({"malicious_ips": ["10.0.0.1", "192.168.1.10"]}),
            expected,
        )

        self.assertEqual(scored.true_positives, 1)
        self.assertEqual(scored.false_positives, 1)
        self.assertEqual(scored.false_negatives, 1)
        self.assertEqual(scored.precision, 0.5)
        self.assertEqual(scored.recall, 0.5)
        self.assertEqual(aggregate_extraction([scored])["f1"], 0.5)

    def test_archive_keeps_cross_round_pareto_state_and_rejects_original(self):
        archive = SearchArchive(original_prompt="Original {{input}}", original_instruction_tokens=10)
        first = archive.record(
            round_index=0,
            prompt="Short A {{input}}",
            instruction_tokens=7,
            behavior_loss=0.1,
            examples=(ExampleFeedback("x", 0.1, candidate_output="A"),),
        )
        second = archive.record(
            round_index=1,
            prompt="Short B {{input}}",
            instruction_tokens=5,
            behavior_loss=0.2,
            parent_ids=(first.id,),
        )

        self.assertEqual({item.id for item in archive.pareto_frontier()}, {first.id, second.id})
        self.assertEqual(second.parent_ids, (first.id,))
        self.assertTrue(archive.select_frontier_parents())
        with self.assertRaises(ValueError):
            archive.record(
                round_index=2,
                prompt="Original {{input}}",
                instruction_tokens=10,
                behavior_loss=0.0,
            )

    def test_rollout_selection_uses_noise_for_close_candidates(self):
        noise = estimate_baseline_noise([0.00, 0.04, 0.02])
        estimates = [
            RewardEstimate("a", 1, 0.10, None, None, None),
            RewardEstimate("b", 1, 0.11, None, None, None),
            RewardEstimate("c", 1, 0.80, None, None, None),
        ]

        selected = select_for_additional_rollouts(
            estimates,
            metric="behavior",
            baseline_noise=noise.standard_deviation,
            close_within=0.0,
        )

        self.assertIn("a", selected)
        self.assertIn("b", selected)
        self.assertNotIn("c", selected)

    def test_single_candidate_is_not_repeated_without_a_close_rival(self):
        selected = select_for_additional_rollouts(
            [RewardEstimate("only", 1, 0.1, None, None, None)],
            metric="behavior",
            baseline_noise=0.0,
        )

        self.assertEqual(selected, [])

    def test_deterministic_close_candidates_stop_after_second_observation(self):
        selected = select_for_additional_rollouts(
            [
                RewardEstimate("a", 2, 0.10, 0.0, None, None),
                RewardEstimate("b", 2, 0.11, 0.0, None, None),
            ],
            metric="behavior",
            close_within=0.02,
            baseline_noise=0.0,
        )

        self.assertEqual(selected, [])


if __name__ == "__main__":
    unittest.main()
