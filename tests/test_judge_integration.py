"""Judge reward integration without model weights, CUDA or a running server."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq

from agmem.checkpoints import best_checkpoint, validation_scores
from agmem.core import Completion, Decision, Example, Trajectory, atomic_json, file_digest
from agmem.data import InvalidSample, adapt_row, task_examples
from agmem.judges import prompt_for
from agmem.pipeline import make_reward_scorer, metrics
from agmem.prompts import reader_messages, task_messages
from agmem.rewards import score_trajectory


class JudgeIntegrationTests(unittest.TestCase):
    reward = {"formula": "judge_gated_v1", "read_call_cost": 0.01, "format_penalty": 0.1,
              "likelihood": {"mode": "off", "coefficient": 0.1}}

    def scored(self, *, prediction="azure", gold="blue", judge_score=1, calls=3,
               failure=False, likelihood=None, source="locomo", index=0, reward=None):
        example = Example(f"{source}/train/{index}", source, "train", "memory", "What color?",
                          gold_answers=[gold])
        row = Trajectory(example.sample_id, 42, example.memory_id, prediction=prediction,
                         n_read_calls=calls, termination_reason="protocol_failure" if failure else "stop")
        row.decisions = [Decision([], Completion('{"stop":true,"ids":[]}'))]
        judge = SimpleNamespace(score=Mock(return_value=judge_score))
        return score_trajectory(example, row, reward or self.reward, judge=judge, likelihood=likelihood)

    def test_semantic_outcome_is_separate_from_lexical_f1(self):
        semantically_correct = self.scored(prediction="azure", gold="blue", judge_score=1)
        self.assertEqual(semantically_correct.answer_f1, 0)
        self.assertEqual(semantically_correct.outcome, 1)
        self.assertEqual(semantically_correct.judge_score, 1)
        self.assertAlmostEqual(semantically_correct.reward, 0.97)
        judged_wrong = self.scored(prediction="blue", gold="blue", judge_score=0, failure=True)
        self.assertEqual(judged_wrong.answer_f1, 1)
        self.assertEqual(judged_wrong.outcome, 0)
        self.assertEqual(judged_wrong.read_cost, 0)
        self.assertAlmostEqual(judged_wrong.reward, -1.1)

    def test_metrics_preserve_judge_accuracy_f1_and_source_macro(self):
        rows = [self.scored(source="locomo", index=i) for i in range(3)]
        rows.append(self.scored(source="split_longmemeval_s", prediction="blue", judge_score=0))
        report = metrics(rows)
        self.assertEqual(report["pooled"]["judge_accuracy"], 0.75)
        self.assertEqual(report["pooled"]["answer_f1"], 0.25)
        self.assertEqual(report["pooled"]["outcome_f1"], 0.25)
        self.assertEqual(report["source_macro"]["judge_accuracy"], 0.5)
        self.assertEqual(report["source_macro"]["answer_f1"], 0.5)
        self.assertEqual(report["by_source"]["locomo"]["judge_accuracy"], 1)
        self.assertEqual(report["by_source"]["locomo"]["answer_f1"], 0)

    def test_checkpoint_selection_uses_judge_not_higher_f1(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step, judge_score, f1 in [(1, 0.9, 0.1), (2, 0.1, 0.9)]:
                path = root / f"checkpoint-{step:08d}"
                atomic_json(path / "model/config.json", {})
                (path / "trainer.pt").write_bytes(b"fixture")
                report = {"source_macro": {"reward": float(step), "judge_accuracy": judge_score,
                                            "answer_f1": f1, "outcome_f1": f1}}
                atomic_json(path / "validation_metrics.json", report)
                self.assertEqual(validation_scores(report)["best_outcome"], judge_score)
            self.assertEqual(best_checkpoint(root, "best_outcome").name, "checkpoint-00000001")
            self.assertEqual(best_checkpoint(root, "best_reward").name, "checkpoint-00000002")
        self.assertEqual(validation_scores({"source_macro": {"reward": 0.4, "outcome_f1": 0.6}})
                         ["best_outcome"], 0.6)

    def test_likelihood_off_does_not_load_tokenizer_or_construct_http_client(self):
        config = {"reward": deepcopy(self.reward), "service": {"model": "agmem-27b", "base_url": "http://127.0.0.1/v1"},
                  "models": {"task_path": "fixture-27b"}, "task_agent": {"max_context_tokens": 65536},
                  "retrieval": {"max_ids_per_call": 4}}
        judge = SimpleNamespace(score=Mock(return_value=0.0), provenance=lambda: {"fixture": True}, close=Mock())
        example = Example("locomo/train/0", "locomo", "train", "m", "What color?", gold_answers=["blue"])
        row = Trajectory(example.sample_id, 1, "m", prediction="red", evidence=[{"id": "m1", "raw_text": "blue"}])
        with patch("agmem.judges.JudgeScorer", return_value=judge), \
             patch("agmem.policy.model_fingerprint", return_value={"fingerprint": "frozen27b"}), \
             patch("agmem.policy.load_tokenizer", side_effect=AssertionError("D is disabled")) as tokenizer, \
             patch("httpx.Client", side_effect=AssertionError("D must not create a client")) as client:
            scorer = make_reward_scorer(config, SimpleNamespace())
            try:
                scored = scorer.score(example, row)
                self.assertEqual(scored.reward, -1)
                self.assertEqual(scored.likelihood["mode"], "off")
                self.assertIsNone(scored.likelihood["d"])
                self.assertIsNone(scored.likelihood["logprob"])
                self.assertEqual(scored.auxiliary_reward, 0)
                self.assertEqual(scorer.provenance()["likelihood"]["mode"], "off")
            finally:
                scorer.close()
            tokenizer.assert_not_called()
            client.assert_not_called()
        judge.score.assert_called_once_with(example, "red")

    def test_diagnostic_likelihood_is_recorded_but_never_added_to_reward(self):
        likelihood = SimpleNamespace(score=Mock(return_value={"mode": "diagnostic", "d": 0.75,
                                                              "logprob": -1.0, "baseline_logprob": -2.0}))
        row = self.scored(judge_score=0, likelihood=likelihood, failure=True)
        self.assertEqual(row.likelihood["d"], 0.75)
        self.assertEqual(row.auxiliary_reward, 0)
        self.assertAlmostEqual(row.reward, -1.1)
        likelihood.score.assert_called_once()

    def test_auxiliary_reward_is_signed_and_gated_by_judge_correctness(self):
        for judge_score in (0, 1):
            for d in (-0.8, 0.8):
                with self.subTest(judge_score=judge_score, d=d):
                    likelihood = SimpleNamespace(score=Mock(return_value={"mode": "reward", "d": d}))
                    row = self.scored(judge_score=judge_score, calls=4, failure=True, likelihood=likelihood)
                    expected_aux = 0.1 * (1 - judge_score) * d
                    self.assertAlmostEqual(row.auxiliary_reward, expected_aux)
                    self.assertAlmostEqual(row.reward, 2 * judge_score - 1 - 0.04 * judge_score - 0.1 + expected_aux)

    def test_judge_cost_diagnostic_likelihood_does_not_change_reward_or_trigger_auxiliary_reward(self):
        reward = {**self.reward, "formula": "judge_cost_v2", "likelihood": {"mode": "diagnostic"}}
        for judge in (0, 1):
            likelihood = SimpleNamespace(score=Mock(return_value={"mode": "diagnostic", "d": 0.75,
                                                                  "logprob": -1.0, "baseline_logprob": -2.0}))
            row = self.scored(judge_score=judge, calls=3, failure=True, likelihood=likelihood, reward=reward)
            self.assertEqual(row.likelihood["d"], 0.75)
            self.assertEqual(row.auxiliary_reward, 0)
            self.assertAlmostEqual(row.reward, 2 * judge - 1 - 0.03 - 0.1)
        invalid_reward = {**reward, "likelihood": {"mode": "reward", "coefficient": 0.1}}
        with self.assertRaisesRegex(ValueError, "off/diagnostic"):
            self.scored(reward=invalid_reward)
        # A misconfigured scorer cannot sneak a reward contribution into a
        # diagnostic experiment even when its config initially looked valid.
        invalid_scorer = SimpleNamespace(score=Mock(return_value={"mode": "reward", "d": 1.0}))
        with self.assertRaisesRegex(ValueError, "off/diagnostic"):
            self.scored(reward=reward, likelihood=invalid_scorer)

    def test_judge_cost_off_is_constructed_without_loading_likelihood_dependencies(self):
        config = {"reward": {**self.reward, "formula": "judge_cost_v2"},
                  "service": {"model": "agmem-27b", "base_url": "http://127.0.0.1/v1"},
                  "models": {"task_path": "fixture-27b"}, "task_agent": {"max_context_tokens": 65536},
                  "retrieval": {"max_ids_per_call": 4}}
        judge = SimpleNamespace(score=Mock(return_value=0.0), provenance=lambda: {"fixture": True}, close=Mock())
        example = Example("locomo/train/0", "locomo", "train", "m", "?", gold_answers=["blue"])
        row = Trajectory(example.sample_id, 1, "m", prediction="blue", n_read_calls=4)
        with patch("agmem.judges.JudgeScorer", return_value=judge), \
             patch("agmem.policy.model_fingerprint", return_value={"fingerprint": "frozen27b"}), \
             patch("agmem.policy.load_tokenizer", side_effect=AssertionError("D is disabled")), \
             patch("httpx.Client", side_effect=AssertionError("D must not create a client")):
            scorer = make_reward_scorer(config, SimpleNamespace())
            try:
                scored = scorer.score(example, row)
                self.assertAlmostEqual(scored.reward, -1.04)
                self.assertEqual(scored.answer_f1, 1)
                self.assertEqual(scored.judge_score, 0)
                self.assertEqual(scored.likelihood["mode"], "off")
                self.assertEqual(scorer.provenance()["formula"], "judge_cost_v2")
            finally:
                scorer.close()

    def test_legacy_f1_reward_does_not_construct_judge_or_likelihood(self):
        config = {"reward": {"formula": "outcome_gated_v2", "read_call_cost": 0.01, "format_penalty": 0.1},
                  "retrieval": {"max_ids_per_call": 4}}
        with patch("agmem.judges.JudgeScorer", side_effect=AssertionError("legacy F1")), \
             patch("agmem.likelihood.ReferenceLikelihoodScorer", side_effect=AssertionError("legacy F1")):
            scorer = make_reward_scorer(config, SimpleNamespace())
            row = scorer.score(Example("locomo/train/0", "locomo", "train", "m", "?", gold_answers=["blue"]),
                               Trajectory("locomo/train/0", 0, "m", prediction="blue", n_read_calls=2))
        self.assertAlmostEqual(row.reward, 0.98)
        self.assertIsNone(row.judge_score)
        self.assertNotIn("judge_accuracy", metrics([row])["source_macro"])


class JudgeMetadataIntegrationTests(unittest.TestCase):
    def fixture(self, root, *, include_judge_metadata=True):
        source = "split_longmemeval_s"
        directory = root / source
        directory.mkdir()
        prepared, files = {}, {}
        for split in ("train", "validation", "test"):
            row = {"question": "What changed?", "answer": "SECRET_GOLD", "question_date": "2023/04/20"}
            if include_judge_metadata:
                row.update(question_type="knowledge-update", question_id=f"SECRET_{split}_abs")
            path = directory / f"{split}.parquet"
            pq.write_table(pa.Table.from_pylist([{**row, "question": ""}, row]), path)
            files[split] = {"sha256": file_digest(path)}
            prepared[split] = [Example(f"{source}/{split}/000001", source, split, "unchanged-memory",
                                       row["question"], gold_answers=[row["answer"]])]
        atomic_json(root / "manifest.json", {"sources": [{"source": source, "output_files": files}]})
        identity = {"input_manifest_sha256": file_digest(root / "manifest.json")}
        store = SimpleNamespace(examples=lambda split: deepcopy(prepared[split]),
                                get_metadata=lambda key: {"layout": "by_source", "datasets": {source: identity}})
        config = {"data": {"input_root": str(root)}, "reward": {"formula": "judge_gated_v1"}}
        return config, store, prepared

    def test_legacy_metadata_restored_for_all_splits_without_changing_public_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, store, prepared = self.fixture(Path(temporary))
            for split in ("train", "validation", "test"):
                original = deepcopy(prepared[split][0])
                restored = task_examples(config, store, split)[0]
                self.assertEqual(restored.judge_metadata,
                                 {"question_type": "knowledge-update", "question_id": f"SECRET_{split}_abs"})
                self.assertEqual(prepared[split][0], original)
                self.assertEqual(restored.memory_id, original.memory_id)
                prompt, _ = prompt_for(restored, restored.gold_answers[0], "unknown", "longmemeval_official")
                self.assertIn("unanswerable question", prompt)
                self.assertNotIn("previous information along with an updated answer", prompt)
                for messages in (reader_messages(restored, [], []), task_messages(restored, [])):
                    self.assertNotIn("SECRET", json.dumps(messages))
                    self.assertNotIn("knowledge-update", json.dumps(messages))
                self.assertNotIn("judge_metadata", restored.public_task())
                with patch("pyarrow.parquet.ParquetFile", side_effect=AssertionError("cache should be reused")):
                    self.assertEqual(task_examples(config, store, split), [restored])

    def test_f1_cache_cannot_hide_missing_or_unrestored_judge_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, store, prepared = self.fixture(Path(temporary))
            legacy_config = {**config, "reward": {"formula": "outcome_gated_v2"}}
            self.assertEqual(task_examples(legacy_config, store, "train")[0].judge_metadata, {})
            restored = task_examples(config, store, "train")[0]
            self.assertTrue(restored.judge_metadata["question_id"].endswith("_abs"))
            prepared["train"][0].judge_metadata = {"question_type": "multi-session", "question_id": "wrong"}
            with self.assertRaisesRegex(ValueError, "Conflicting prepared Judge"):
                task_examples(config, store, "train")

    def test_old_f1_data_without_types_remains_usable_but_judge_fails_early(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, store, _ = self.fixture(Path(temporary), include_judge_metadata=False)
            legacy_config = {**config, "reward": {"formula": "outcome_gated_v2"}}
            self.assertTrue(task_examples(legacy_config, store, "train")[0].observation)
            with self.assertRaisesRegex(InvalidSample, "Judge requires question_type"):
                task_examples(config, store, "train")

    def test_new_adapter_preserves_private_judge_metadata_without_label_leakage(self):
        row = {"question": "When?", "answer": "SECRET_GOLD", "question_date": "2023/04/20",
               "question_type": "temporal-reasoning", "question_id": "SECRET_abs"}
        example = adapt_row(row, "split_longmemeval_s", "train", 0)
        self.assertEqual(example.judge_metadata, {"question_type": "temporal-reasoning", "question_id": "SECRET_abs"})
        self.assertNotIn("SECRET", json.dumps(example.public_task()))


if __name__ == "__main__":
    unittest.main()
