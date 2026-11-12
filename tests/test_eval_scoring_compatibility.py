"""Real scorer provenance across the explicit legacy-to-tagged evaluation change."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from agmem.evaluate_checkpoints import _run_suite, validate_evaluation_scoring
from agmem.pipeline import make_reward_scorer
from agmem.prompts import task_input_contract


SNAPSHOT = Path(__file__).resolve().parents[1] / "configs/protocols/format05_training_judge.json"


class EvaluationScoringCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.http = self.enterContext(patch("httpx.Client.request", side_effect=AssertionError("HTTP forbidden")))
        self.enterContext(patch("agmem.policy.model_fingerprint", return_value={"fingerprint": "fixture-qwen-revision"}))
        self.legacy = {
            "models": {"task_path": "/fixture/no-model-is-loaded"},
            "service": {"model": "agmem-27b", "model_revision": "fixture-qwen-revision",
                        "base_url": "http://127.0.0.1:50543/v1", "retries": 0},
            "retrieval": {"max_ids_per_call": 1},
            "task_agent": {"max_output_tokens": 512, "max_context_tokens": 65536},
            "reward": {"formula": "judge_cost_v2", "read_call_cost": .01,
                       "format_penalty": .5, "format_penalty_mode": "per_id",
                       "judge": {"template_file": str(SNAPSHOT), "max_output_tokens": 128, "seed": 42},
                       "likelihood": {"mode": "off", "coefficient": .1, "tau": 1.0,
                                      "reference_aggregation": "mean"}},
        }
        self.config = deepcopy(self.legacy)
        self.config["task_agent"].update(prompt_version="history_answer_v1", max_output_tokens=2048)
        self.config["service"]["request_base_url"] = "http://127.0.0.1:59589/v1"
        self.expected = self.provenance(self.legacy)
        self.actual = self.provenance(self.config)

    def provenance(self, config):
        scorer = make_reward_scorer(config, None)
        try:
            return scorer.provenance()
        finally:
            scorer.close()

    def test_real_off_scorer_reproduces_original_mismatch_and_accepts_explicit_task_change(self):
        self.assertNotEqual(self.actual, self.expected)
        self.assertEqual(self.actual["judge"], self.expected["judge"])
        self.assertEqual(self.expected["likelihood"]["task_input_contract"], task_input_contract("legacy_v1"))
        self.assertEqual(self.actual["likelihood"]["task_input_contract"], task_input_contract("history_answer_v1"))
        before = deepcopy((self.config, self.actual, self.expected))
        validate_evaluation_scoring(self.config, self.actual, self.expected)
        self.assertEqual((self.config, self.actual, self.expected), before)
        self.http.assert_not_called()

    def test_unchanged_legacy_and_tagged_provenance_remain_accepted(self):
        validate_evaluation_scoring(self.legacy, self.expected, self.expected)
        validate_evaluation_scoring(self.config, self.actual, self.actual)

    def test_task_change_without_explicit_supported_override_is_rejected(self):
        for version in ("legacy_v1", "history_answer_query_v1", "history_answer_evidence_v2"):
            with self.subTest(version=version):
                config = deepcopy(self.config)
                config["task_agent"]["prompt_version"] = version
                with self.assertRaisesRegex(ValueError, "Judge/reward provenance"):
                    validate_evaluation_scoring(config, self.actual, self.expected)

    def test_judge_changes_are_still_rejected(self):
        for field, value in (("max_output_tokens", 256), ("seed", 3), ("parser", "lenient"),
                             ("temperature", .1), ("model", {"model_revision": "other-model"}),
                             ("template_sha256", {"longmemeval_official": "changed"})):
            with self.subTest(field=field):
                actual = deepcopy(self.actual)
                actual["judge"][field] = value
                with self.assertRaisesRegex(ValueError, "Judge/reward provenance"):
                    validate_evaluation_scoring(self.config, actual, self.expected)

    def test_disabled_likelihood_nonprompt_settings_are_still_strict(self):
        for field, value in (("tau", 2.0), ("model", {"model_revision": "other-model"}),
                             ("protocol", "changed"), ("max_context_tokens", 32768),
                             ("reference_aggregation", "max"), ("unexpected", "new")):
            with self.subTest(field=field):
                actual = deepcopy(self.actual)
                actual["likelihood"][field] = value
                with self.assertRaisesRegex(ValueError, "Judge/reward provenance"):
                    validate_evaluation_scoring(self.config, actual, self.expected)

    def test_prompt_derived_fields_must_match_the_exact_new_contract(self):
        for field, value in (("assistant_prefix", "<answer>"), ("target_encoding", "changed"),
                             ("task_input_contract", {"task_prompt_version": "history_answer_v1"})):
            with self.subTest(field=field):
                actual = deepcopy(self.actual)
                actual["likelihood"][field] = value
                with self.assertRaisesRegex(ValueError, "Judge/reward provenance"):
                    validate_evaluation_scoring(self.config, actual, self.expected)

    def test_reward_and_format_protocol_changes_are_rejected(self):
        for field, value in (("formula", "f1_cost_v1"), ("format_penalty_mode", "per_decision"),
                             ("format_counting_version", "changed"), ("task_output_policy", "changed")):
            with self.subTest(field=field):
                actual = deepcopy(self.actual)
                actual[field] = value
                with self.assertRaisesRegex(ValueError, "Judge/reward provenance"):
                    validate_evaluation_scoring(self.config, actual, self.expected)

    def test_active_likelihood_cannot_use_prompt_compatibility(self):
        for position in ("config", "actual", "expected"):
            for mode in ("diagnostic", "reward"):
                with self.subTest(position=position, mode=mode):
                    config, actual, expected = deepcopy((self.config, self.actual, self.expected))
                    target = config["reward"]["likelihood"] if position == "config" else {
                        "actual": actual, "expected": expected}[position]["likelihood"]
                    target["mode"] = mode
                    with self.assertRaisesRegex(ValueError, "Judge/reward provenance"):
                        validate_evaluation_scoring(config, actual, expected)

    def test_unrecognized_original_prompt_cannot_be_relabelled(self):
        expected = deepcopy(self.expected)
        expected["likelihood"]["task_input_contract"]["task_prompt_sha256"] = "different-training-prompt"
        with self.assertRaisesRegex(ValueError, "Judge/reward provenance"):
            validate_evaluation_scoring(self.config, self.actual, expected)

    def test_cpu_prepare_rejects_mismatch_before_reading_directories_and_closes_scorer(self):
        source = "split_longmemeval_s"
        suite = {"datasets": [source], "checkpoints": {"best": "fixture-checkpoint"},
                 "task_agents": ["local"], "output_root": "/fixture/not-created",
                 "required_prepared_sources": {source: "/fixture/not-opened"}}
        args = Namespace(local_base_url=None, remote_model=None, datasets=None,
                         checkpoints=None, task_agents=None, limit=None, stage="prepare")
        actual = deepcopy(self.actual)
        actual["judge"]["seed"] += 1
        scorer = Mock()
        scorer.provenance.return_value = actual
        with patch("agmem.evaluate_checkpoints.make_reward_scorer", return_value=scorer), \
             patch("agmem.evaluation_data.load_existing_source") as directories, \
             patch("agmem.evaluate_checkpoints.announce"):
            with self.assertRaisesRegex(ValueError, "Judge/reward provenance"):
                _run_suite(args, suite, self.config, {"scoring_protocol": self.expected})
        directories.assert_not_called()
        scorer.close.assert_called_once_with()
        self.http.assert_not_called()


if __name__ == "__main__":
    unittest.main()
