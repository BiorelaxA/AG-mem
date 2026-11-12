"""Round-trip exact local Task inputs through portable answers and the fixed Judge."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agmem.core import Example, Trajectory, atomic_json, digest
from agmem.evaluate_checkpoints import EpisodeJournal, _run_suite, score_evidence
from agmem.evaluation_offline import export_task_requests, judge_remote_answers
from agmem.remote_task_runner import load_bundle, response_manifest
from agmem.rewards import score_trajectory
from agmem.services import TaskAgent


class OfflineEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / "best" / "locomo"
        self.examples = [Example(str(i), "locomo", "test", "history", f"question {i}",
                                 gold_answers=["GOLD_MUST_STAY_LOCAL"],
                                 judge_metadata={"secret": "SCORING_ONLY"}) for i in range(2)]
        self.config = {
            "reader": {"prompt_version": "evidence_loop_v3"},
            "retrieval": {"max_ids_per_call": 1, "max_read_calls": 8, "id_protocol": "strict_v2"},
            "task_agent": {"max_output_tokens": 512},
            "reward": {"formula": "judge_cost_v2", "read_call_cost": 0.01,
                       "format_penalty": 0.5, "format_penalty_mode": "per_id",
                       "text_outcome": "qa_f1_v1", "choice_outcome": "label_f1"},
        }
        self.remote = {"model": "gpt-5-mini", "model_revision": "gpt-5-mini",
                       "request_profile": "openai_chat", "send_seed": False,
                       "temperature": None, "max_tokens_parameter": "max_completion_tokens",
                       "api_key_env": "OPENAI_API_KEY", "base_url_env": "OPENAI_BASE_URL"}
        self.retrieval = EpisodeJournal(self.target / "retrieval", {
            "protocol": "fixture", "sample_ids": [e.sample_id for e in self.examples]})
        for e in self.examples:
            self.retrieval.write(Trajectory(
                e.sample_id, 42, e.memory_id, n_read_calls=1, termination_reason="stop",
                evidence=[{"id": "session1", "raw_text": f"blue unicode 中文 {e.sample_id}"}]))
        self.retrieval.finish()
        local_service = SimpleNamespace(
            identity={"model": "local-qwen", "base_url": "http://local/v1", "model_revision": "fixture"},
            config={}, client=SimpleNamespace(close=lambda: None), complete=self.complete_local)
        self.judge_calls = []
        self.scorer = SimpleNamespace(provenance=lambda: {"judge": "frozen-training"},
                                      score=self.score, close=lambda: None)
        with patch("agmem.evaluate_checkpoints.make_task_agent", return_value=TaskAgent(local_service)), \
             patch("agmem.evaluate_checkpoints.make_reward_scorer", return_value=self.scorer), \
             patch("agmem.evaluate_checkpoints.announce"):
            score_evidence(self.config, None, self.examples, self.retrieval,
                           self.target / "answers" / "local", workers=2)
        self.judge_calls.clear()

    def complete_local(self, messages, max_tokens, metadata):
        metadata.update(protocol="score_nonempty_truncated_v1", finish_reason="stop",
                        truncated=False, usage=None, max_tokens=max_tokens)
        return "QWEN_ANSWER_MUST_STAY_LOCAL"

    def score(self, example, row):
        self.judge_calls.append((example.sample_id, row.prediction))
        judge = SimpleNamespace(score=lambda e, text: float(text == "blue"))
        score_trajectory(example, row, self.config["reward"], judge=judge, max_ids=1)

    def export(self, target=None):
        return export_task_requests(self.config, self.examples, self.retrieval,
                                    target or self.target, self.remote, "best", "locomo")

    def responses(self, bundle_path):
        bundle = load_bundle(bundle_path)
        root = self.root / "returned"
        manifest = response_manifest(bundle, "https://provider.example/v1")
        atomic_json(root / "manifest.json", manifest)
        for index, request in enumerate(bundle["requests"]):
            atomic_json(root / "episodes" / f"{index:08d}.json", {
                "manifest_sha256": digest(manifest), "sample_id": request["sample_id"],
                "task_input_hash": request["task_input_hash"],
                "completion": {"protocol": "score_nonempty_truncated_v1", "text": "blue",
                               "finish_reason": "stop", "truncated": False,
                               "usage": {"completion_tokens": 1}, "max_tokens": 512},
            })
        atomic_json(root / "complete.json", {"manifest_sha256": digest(manifest), "n": 2})
        return root

    def judge(self, bundle, responses):
        with patch.dict("os.environ", {}, clear=True), \
             patch("agmem.evaluate_checkpoints.make_task_agent", side_effect=AssertionError("No remote call")), \
             patch("agmem.evaluate_checkpoints.make_reward_scorer", return_value=self.scorer), \
             patch("agmem.evaluate_checkpoints.announce"):
            return judge_remote_answers(self.config, None, self.examples, self.retrieval,
                                        self.target, bundle, responses, {"judge": "frozen-training"}, workers=2)

    def test_round_trip_exact_messages_no_gold_and_resumable_local_judging(self):
        with patch.dict("os.environ", {}, clear=True):
            path = self.export()
        before = path.read_bytes()
        self.assertEqual(self.export(), path)
        self.assertEqual(path.read_bytes(), before)
        bundle = load_bundle(path)
        text = json.dumps(bundle)
        for private in ("GOLD_MUST_STAY_LOCAL", "SCORING_ONLY", "QWEN_ANSWER_MUST_STAY_LOCAL", "api_key"):
            self.assertNotIn(private, text)
        local_manifest = json.loads((self.target / "answers/local/manifest.json").read_text())
        local = EpisodeJournal(self.target / "answers/local", local_manifest)
        for request in bundle["requests"]:
            self.assertEqual(request["task_input_hash"], local.read(request["sample_id"])["task_input_hash"])
        responses = self.responses(path)
        report = self.judge(path, responses)
        self.assertEqual(report["pooled"]["judge_accuracy"], 1)
        self.assertAlmostEqual(report["pooled"]["reward"], .99)
        self.assertEqual(report["task"]["identity"]["model"], "gpt-5-mini")
        self.assertEqual(len(self.judge_calls), 2)
        self.judge_calls.clear()
        self.assertEqual(self.judge(path, responses), report)
        self.assertEqual(self.judge_calls, [])
        manifest = json.loads((self.target / "answers/remote/manifest.json").read_text())
        remote = EpisodeJournal(self.target / "answers/remote", manifest)
        for e in self.examples:
            self.assertEqual(remote.read(e.sample_id)["evidence"], self.retrieval.read(e.sample_id)["evidence"])

    def test_export_rejects_partial_local_evaluation(self):
        (self.target / "answers/local/complete.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.export()
        self.assertFalse((self.target / "offline/requests.json.gz").exists())

    def test_export_rejects_changed_saved_input(self):
        path = self.target / "answers/local/episodes/00000001.json.gz"
        with gzip.open(path, "rt") as stream:
            record = json.load(stream)
        record["trajectory"]["task_input_hash"] = "invalid"
        with gzip.open(path, "wt") as stream:
            json.dump(record, stream)
        with self.assertRaisesRegex(ValueError, "input hash"):
            self.export()

    def test_export_rejects_changed_model_settings(self):
        path = self.export()
        before = path.read_bytes()
        self.remote["model"] = "different-model"
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            self.export()
        self.assertEqual(path.read_bytes(), before)

    def test_incomplete_or_foreign_returned_answers_fail_before_judge(self):
        path = self.export()
        responses = self.responses(path)
        last = responses / "episodes/00000001.json"
        good = json.loads(last.read_text())
        last.unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.judge(path, responses)
        bad = deepcopy(good)
        bad["task_input_hash"] = "foreign"
        atomic_json(last, bad)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.judge(path, responses)
        self.assertEqual(self.judge_calls, [])

    def test_changed_returned_answer_cannot_reuse_existing_judge_result(self):
        path = self.export()
        responses = self.responses(path)
        self.judge(path, responses)
        self.judge_calls.clear()
        record_path = responses / "episodes/00000001.json"
        record = json.loads(record_path.read_text())
        record["completion"]["text"] = "red"
        atomic_json(record_path, record)
        with self.assertRaisesRegex(ValueError, "differs from imported"):
            self.judge(path, responses)
        self.assertEqual(self.judge_calls, [])

    def test_judge_provenance_remains_frozen(self):
        path = self.export()
        responses = self.responses(path)
        self.scorer.provenance = lambda: {"judge": "changed"}
        with self.assertRaisesRegex(ValueError, "Judge/reward provenance"):
            self.judge(path, responses)
        self.assertEqual(self.judge_calls, [])

    def test_cached_retrieval_changes_fail_before_any_judge_request(self):
        path = self.export()
        responses = self.responses(path)
        episode = self.retrieval.path(self.examples[-1].sample_id)
        with gzip.open(episode, "rt") as stream:
            record = json.load(stream)
        record["trajectory"]["evidence"][0]["raw_text"] = "new evidence"
        with gzip.open(episode, "wt") as stream:
            json.dump(record, stream)
        with self.assertRaisesRegex(ValueError, "evidence mismatch"):
            self.judge(path, responses)
        self.assertEqual(self.judge_calls, [])

    def test_adaptive_budget_answers_import_with_explicit_policy_and_legacy_rows(self):
        path = self.export()
        responses = self.responses(path)
        bundle = load_bundle(path)
        legacy = json.loads((responses / "manifest.json").read_text())
        local = json.loads((self.target / "answers/local/manifest.json").read_text())
        local["task"] = deepcopy(legacy["task"])
        EpisodeJournal(self.target / "answers/remote", local)  # Previously failed online run.
        updated = response_manifest(bundle, "https://provider.example/v1", empty_answer_max_tokens=16384)
        atomic_json(responses / "manifest.json", updated)
        second = responses / "episodes/00000001.json"
        record = json.loads(second.read_text())
        record["manifest_sha256"] = digest(updated)
        record["completion"]["max_tokens"] = 2048
        atomic_json(second, record)
        atomic_json(responses / "complete.json", {"manifest_sha256": digest(updated), "n": 2})
        report = self.judge(path, responses)
        self.assertEqual(report["pooled"]["judge_accuracy"], 1)
        self.assertEqual(report["task"]["request_options"]["empty_answer_retry"]["max_completion_tokens"], 16384)
        output = self.target / "answers/remote"
        journal = EpisodeJournal(output, json.loads((output / "manifest.json").read_text()))
        self.assertEqual(journal.read(self.examples[0].sample_id)["task_completion"]["max_tokens"], 512)
        self.assertEqual(journal.read(self.examples[1].sample_id)["task_completion"]["max_tokens"], 2048)
        self.assertTrue((self.target / "offline/legacy_remote_answer_manifest.json").exists())
        self.judge_calls.clear()
        self.assertEqual(self.judge(path, responses), report)
        self.assertEqual(self.judge_calls, [])

    def test_local_only_suite_reaches_both_checkpoints_and_exports_without_remote_credentials(self):
        manifest = json.loads((self.retrieval.root / "manifest.json").read_text())
        final_target = self.root / "final/locomo"
        final_retrieval = EpisodeJournal(final_target / "retrieval", manifest)
        for example in self.examples:
            from agmem.core import trajectory_from_dict
            row = trajectory_from_dict(self.retrieval.read(example.sample_id))
            row.policy_version = 32
            row.evidence[0]["raw_text"] = "distinct checkpoint 32 evidence"
            final_retrieval.write(row)
        final_retrieval.finish()
        suite = {"output_root": str(self.root), "prepared_root": str(self.root / "prepared"),
                 "datasets": ["locomo"], "checkpoints": {"best": "20", "final": "32"},
                 "task_agents": ["local"], "export_remote_inputs": True, "remote_task": self.remote}
        args = SimpleNamespace(stage="all", local_base_url=None, remote_model=None,
                               datasets=None, checkpoints=None, task_agents=None, limit=None)
        store = SimpleNamespace(close=lambda: None)
        service = SimpleNamespace(identity={"model": "local-qwen", "base_url": "http://local/v1",
                                            "model_revision": "fixture"}, config={},
                                  client=SimpleNamespace(close=lambda: None), complete=Mock(side_effect=self.complete_local))
        prepared = {"config": self.config, "status": "ready"}
        with patch.dict("os.environ", {}, clear=True), \
             patch("agmem.evaluation_data.prepare_source", return_value=prepared), \
             patch("agmem.evaluation_data.build_source_directories", side_effect=lambda value: value), \
             patch("agmem.evaluate_checkpoints.checkpoint_model", side_effect=lambda s, label: Path(label)), \
             patch("agmem.evaluate_checkpoints.open_prepared", return_value=store), \
             patch("agmem.evaluate_checkpoints.validate_directories"), \
             patch("agmem.evaluate_checkpoints.select_examples", return_value=self.examples), \
             patch("agmem.evaluate_checkpoints.retrieval_manifest", return_value=manifest), \
             patch("agmem.evaluate_checkpoints.make_task_agent", return_value=TaskAgent(service)), \
             patch("agmem.evaluate_checkpoints.make_reward_scorer", return_value=self.scorer), \
             patch("agmem.services.resolve_task_service", side_effect=AssertionError("Remote credentials requested")), \
             patch("agmem.policy.load_model", side_effect=AssertionError("Reader rerun")), \
             patch("agmem.evaluate_checkpoints.announce"):
            _run_suite(args, suite, self.config, {"scoring_protocol": {"judge": "frozen-training"}})
        self.assertEqual(service.complete.call_count, 2)  # Only final needed Task answers.
        best = load_bundle(self.target / "offline/requests.json.gz")
        final = load_bundle(final_target / "offline/requests.json.gz")
        self.assertNotEqual(best["requests"][0]["task_input_hash"], final["requests"][0]["task_input_hash"])
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual({row["checkpoint"] for row in summary["results"]}, {"best", "final"})
        self.assertEqual({row["task_agent"] for row in summary["results"]}, {"local"})


if __name__ == "__main__":
    unittest.main()
