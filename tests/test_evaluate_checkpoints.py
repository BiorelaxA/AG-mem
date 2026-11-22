"""CPU-only tests of paired evidence, atomic recovery and evaluation identities."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from threading import Event, Lock
from types import SimpleNamespace
import csv
import gzip
import io
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from agmem.core import Completion, Decision, Example, InfrastructureError, Trajectory
from agmem.evaluate_checkpoints import (EpisodeJournal, compact_for_metrics,
                                        run_suite, score_evidence, validate_selections, write_summary)
from agmem.pipeline import metrics
from agmem.rewards import score_trajectory


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.examples = [Example(f"locomo/test/{i}", "locomo", "test", "memory", f"q{i}",
                                 gold_answers=["blue"]) for i in range(2)]
        self.config = {
            "reader": {"prompt_version": "evidence_loop_v3"},
            "retrieval": {"max_ids_per_call": 1, "max_read_calls": 8, "id_protocol": "strict_v2"},
            "task_agent": {"max_output_tokens": 512},
            "reward": {"formula": "judge_cost_v2", "read_call_cost": 0.01,
                       "format_penalty": 0.5, "format_penalty_mode": "per_id",
                       "text_outcome": "qa_f1_v1", "choice_outcome": "label_f1"},
        }
        self.manifest = {"sample_ids": [e.sample_id for e in self.examples], "protocol": "fixture"}
        self.retrieval = EpisodeJournal(self.root / "retrieval", self.manifest)
        for example in self.examples:
            row = self.row(example.sample_id)
            self.retrieval.write(row)
        self.retrieval.finish()

    def row(self, sample_id):
        return Trajectory(sample_id, 42, "memory", policy_version=20, n_read_calls=1,
                          termination_reason="stop", evidence=[{"id": "r1", "raw_text": "The color is blue."}],
                          decisions=[Decision([{"role": "user", "content": "context"}],
                                              Completion('{"ids":["r1"],"stop":false}', [1, 2], [3, 4]),
                                              action={"ids": ["r1"], "stop": False},
                                              tool_result={"records": [{"id": "r1", "raw_text": "blue"}],
                                                           "new_ids": ["r1"], "read_call_consumed": True})])

    def agent(self, name="local", fail_at=None):
        calls = []
        def answer(example, evidence, metadata):
            calls.append((example.sample_id, deepcopy(evidence)))
            if example.sample_id == fail_at:
                raise InfrastructureError("Provider temporarily unavailable")
            metadata.update(truncated=False)
            return "blue", "taskhash"
        service = SimpleNamespace(identity={"model": name, "base_url": "http://fixture/v1",
                                            "model_revision": name},
                                  config={}, client=SimpleNamespace(close=Mock()))
        return SimpleNamespace(service=service, answer=Mock(side_effect=answer)), calls

    def scorer(self):
        judge = SimpleNamespace(score=lambda example, prediction: float(prediction == "blue"))
        return SimpleNamespace(
            score=lambda e, row: score_trajectory(e, row, self.config["reward"], judge=judge, max_ids=1),
            provenance=lambda: {"judge": "fixed-training-judge"}, close=Mock())

    def evaluate(self, output, agent, expected=None):
        with patch("agmem.evaluate_checkpoints.make_task_agent", return_value=agent), \
             patch("agmem.evaluate_checkpoints.make_reward_scorer", return_value=self.scorer()):
            return score_evidence(self.config, None, self.examples, self.retrieval, output, expected)

    def test_resume_does_not_duplicate_or_reanswer_completed_examples(self):
        output = self.root / "answers"
        failing, calls = self.agent(fail_at=self.examples[1].sample_id)
        with self.assertRaises(InfrastructureError):
            self.evaluate(output, failing)
        self.assertFalse((output / "metrics.json").exists())
        self.assertFalse((output / "complete.json").exists())
        self.assertEqual(len(list((output / "episodes").glob("*.json.gz"))), 1)
        succeeding, calls = self.agent()
        report = self.evaluate(output, succeeding)
        self.assertEqual([call[0] for call in calls], [self.examples[1].sample_id])
        self.assertEqual(report["pooled"]["n"], 2)
        self.assertEqual(report["pooled"]["judge_accuracy"], 1)
        self.assertTrue((output / "complete.json").exists())
        again, calls = self.agent()
        self.evaluate(output, again)
        self.assertEqual(calls, [])
        self.assertEqual(len(list((output / "episodes").glob("*.json.gz"))), 2)

    def test_two_answering_models_receive_identical_immutable_evidence(self):
        local, a = self.agent("local")
        remote, b = self.agent("remote")
        first = self.evaluate(self.root / "local", local)
        second = self.evaluate(self.root / "remote", remote)
        self.assertEqual(a, b)
        for name in ["read_calls", "format_reward", "format_failure_rate", "reader_generated_tokens"]:
            self.assertEqual(first["pooled"][name], second["pooled"][name])
        self.assertEqual(first["retrieval_manifest_sha256"], second["retrieval_manifest_sha256"])
        raw = self.retrieval.read(self.examples[0].sample_id)
        self.assertEqual(raw["prediction"], "")
        self.assertIn("raw_text", raw["evidence"][0])

    def test_task_identity_change_rejects_previous_output(self):
        output = self.root / "answers"
        self.evaluate(output, self.agent("local")[0])
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            self.evaluate(output, self.agent("remote")[0])

    def test_judge_must_match_saved_training_protocol(self):
        with self.assertRaisesRegex(ValueError, "differs from the saved training run"):
            self.evaluate(self.root / "answers", self.agent()[0], {"judge": "different"})
        self.assertFalse((self.root / "answers/metrics.json").exists())

    def test_manifest_change_and_unknown_episode_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            EpisodeJournal(self.root / "retrieval", {**self.manifest, "protocol": "new"})
        (self.root / "retrieval/episodes/99999999.json.gz").write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "Unknown episode"):
            EpisodeJournal(self.root / "retrieval", self.manifest)

    def test_partial_write_is_not_counted_and_tampered_sample_is_rejected(self):
        journal = EpisodeJournal(self.root / "partial", self.manifest)
        (journal.root / "episodes/00000000.json.gz.tmp").write_bytes(b"partial gzip")
        self.assertIsNone(journal.read(self.examples[0].sample_id))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            journal.finish()
        journal.write(self.row(self.examples[0].sample_id))
        path = journal.path(self.examples[0].sample_id)
        with gzip.open(path, "rt") as stream:
            value = json.load(stream)
        value["trajectory"]["sample_id"] = "another sample"
        with gzip.open(path, "wt") as stream:
            json.dump(value, stream)
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            journal.read(self.examples[0].sample_id)

    def test_metric_compaction_preserves_full_metrics(self):
        row = self.row(self.examples[0].sample_id)
        row.prediction = "blue"
        self.scorer().score(self.examples[0], row)
        before = metrics([row])
        after = metrics([compact_for_metrics(row)])
        self.assertEqual(before, after)

    def test_summary_includes_only_complete_cells(self):
        root = self.root / "results"
        self.evaluate(root / "best/locomo/answers/local", self.agent()[0])
        self.evaluate(root / "final/locomo/answers/local", self.agent()[0])
        (root / "final/locomo/answers/local/complete.json").unlink()
        rows = write_summary(root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["checkpoint"], "best")
        self.assertTrue((root / "summary.csv").exists())

    def test_invalid_task_names_cannot_silently_run_the_local_model(self):
        suite = {"datasets": ["locomo"], "checkpoints": {"best": "checkpoint-20"}}
        for tasks in (["local", "remtoe"], [], ["remote", "remote"]):
            with self.subTest(tasks=tasks), self.assertRaisesRegex(ValueError, "task_agents"):
                validate_selections(suite, ["locomo"], ["best"], tasks)
        with self.assertRaisesRegex(ValueError, "datasets"):
            validate_selections(suite, [], ["best"], ["local"])

    def test_empty_summary_replaces_old_csv(self):
        root = self.root / "results"
        output = root / "best/locomo/answers/local"
        self.evaluate(output, self.agent()[0])
        self.assertEqual(len(write_summary(root)), 1)
        (output / "complete.json").unlink()
        self.assertEqual(write_summary(root), [])
        self.assertEqual(len((root / "summary.csv").read_text().splitlines()), 1)

    def test_summary_rejects_missing_manifest_or_false_completion_count(self):
        root = self.root / "results"
        output = root / "best/locomo/answers/local"
        self.evaluate(output, self.agent()[0])
        done = json.loads((output / "complete.json").read_text())
        done["n"] = 999
        (output / "complete.json").write_text(json.dumps(done))
        with self.assertRaisesRegex(ValueError, "stale metrics"):
            write_summary(root)
        (output / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "no manifest"):
            write_summary(root)

    def suite(self):
        return {"prepared_root": str(self.root / "prepared"),
                "output_root": str(self.root / "results"),
                "datasets": ["locomo", "split_longmemeval_s"],
                "checkpoints": {"best": "checkpoint-20"}, "task_agents": ["local"]}

    def suite_args(self, **changes):
        values = {"config": "fixture", "stage": "all", "datasets": ["locomo"],
                  "checkpoints": None, "task_agents": None, "local_base_url": None,
                  "remote_model": None, "limit": None}
        values.update(changes)
        return SimpleNamespace(**values)

    def test_legacy_global_lock_prevents_new_worker_from_modifying_stores(self):
        from agmem.jobs import file_lock
        suite = self.suite()
        with file_lock(self.root / "prepared/.evaluation.lock"), \
             patch("agmem.evaluate_checkpoints.load_suite", return_value=(suite, {}, {})), \
             patch("agmem.evaluate_checkpoints._run_suite") as execute:
            with self.assertRaisesRegex(RuntimeError, "Another job holds"):
                run_suite(self.suite_args())
            execute.assert_not_called()

    def test_different_dataset_workers_can_run_while_both_stores_stay_locked(self):
        from agmem.jobs import file_lock
        suite = self.suite()
        global_path = self.root / "prepared/.evaluation.lock"
        local_path = self.root / "prepared/.evaluation_locks/locomo.lock"
        long_path = self.root / "prepared/.evaluation_locks/split_longmemeval_s.lock"

        def execute(*args):
            # The foreground worker owns its source lock; all new workers also
            # prevent an older all-dataset worker from acquiring its global lock.
            for path in (global_path, local_path, long_path):
                with self.assertRaisesRegex(RuntimeError, "Another job holds"):
                    with file_lock(path):
                        self.fail("An active worker's lock was not held")
            return "locomo finished"

        with file_lock(global_path, shared=True), file_lock(long_path), \
             patch("agmem.evaluate_checkpoints.load_suite", return_value=(suite, {}, {})), \
             patch("agmem.evaluate_checkpoints._run_suite", side_effect=execute):
            self.assertEqual(run_suite(self.suite_args()), "locomo finished")

    def test_every_store_stage_excludes_another_worker_on_the_same_dataset(self):
        from agmem.jobs import file_lock
        suite = self.suite()
        with file_lock(self.root / "prepared/.evaluation.lock", shared=True), \
             file_lock(self.root / "prepared/.evaluation_locks/locomo.lock"), \
             patch("agmem.evaluate_checkpoints.load_suite", return_value=(suite, {}, {})), \
             patch("agmem.evaluate_checkpoints._run_suite") as execute:
            for stage in ("all", "build", "prepare", "score", "retrieve"):
                with self.subTest(stage=stage), self.assertRaisesRegex(RuntimeError, "Another job holds"):
                    run_suite(self.suite_args(stage=stage))
            execute.assert_not_called()

    def test_source_locks_are_acquired_in_a_stable_order(self):
        from agmem.jobs import file_lock
        suite, acquired = self.suite(), []

        def tracked(path, **kwargs):
            acquired.append((str(path), kwargs))
            return file_lock(path, **kwargs)

        with patch("agmem.evaluate_checkpoints.load_suite", return_value=(suite, {}, {})), \
             patch("agmem.evaluate_checkpoints.file_lock", side_effect=tracked), \
             patch("agmem.evaluate_checkpoints._run_suite"):
            run_suite(self.suite_args(datasets=["split_longmemeval_s", "locomo"]))
        self.assertEqual([Path(path).name for path, _ in acquired],
                         [".evaluation.lock", "locomo.lock", "split_longmemeval_s.lock"])
        self.assertEqual(acquired[0][1], {"shared": True})

    def test_source_selection_is_validated_before_lock_paths_are_created(self):
        suite = self.suite()
        for source in ("", "../escape", "/tmp/escape", "..", "_runtime"):
            suite["datasets"] = [source]
            with self.subTest(source=source), \
                 patch("agmem.evaluate_checkpoints.load_suite", return_value=(suite, {}, {})), \
                 patch("agmem.evaluate_checkpoints._run_suite") as execute:
                with self.assertRaisesRegex(ValueError, "Invalid evaluation source"):
                    run_suite(self.suite_args(datasets=[source]))
                execute.assert_not_called()
            self.assertFalse((self.root / "prepared").exists())

    def test_summarize_does_not_acquire_store_locks(self):
        from agmem.jobs import file_lock
        suite = self.suite()
        with file_lock(self.root / "prepared/.evaluation.lock"), \
             file_lock(self.root / "prepared/.evaluation_locks/locomo.lock"), \
             patch("agmem.evaluate_checkpoints.load_suite", return_value=(suite, {}, {})), \
             patch("agmem.evaluation_data.prepare_source") as prepare, \
             redirect_stdout(io.StringIO()):
            run_suite(self.suite_args(stage="summarize"))
        prepare.assert_not_called()
        self.assertEqual(json.loads((self.root / "results/summary.json").read_text()), {"results": []})

    def test_concurrent_summaries_serialize_publication_and_keep_json_csv_consistent(self):
        from agmem.core import atomic_json
        root = self.root / "results"
        self.evaluate(root / "best/locomo/answers/local", self.agent()[0])
        self.evaluate(root / "best/split_longmemeval_s/answers/local", self.agent()[0])
        first_writing, release_first = Event(), Event()
        second_attempted, second_writing = Event(), Event()
        guard, writes = Lock(), 0

        def delayed_json(path, value):
            nonlocal writes
            if Path(path).name == "summary.json":
                with guard:
                    writes += 1
                    number = writes
                if number == 1:
                    first_writing.set()
                    self.assertTrue(release_first.wait(timeout=3))
                else:
                    second_writing.set()
            return atomic_json(path, value)

        def second_summary():
            second_attempted.set()
            return write_summary(root)

        with patch("agmem.evaluate_checkpoints.atomic_json", side_effect=delayed_json), \
             ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(write_summary, root)
            try:
                self.assertTrue(first_writing.wait(timeout=3))
                second = pool.submit(second_summary)
                self.assertTrue(second_attempted.wait(timeout=3))
                self.assertFalse(second_writing.wait(timeout=0.1))
            finally:
                release_first.set()
            first_rows, second_rows = first.result(timeout=3), second.result(timeout=3)
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(len(second_rows), 2)
        self.assertEqual(json.loads((root / "summary.json").read_text())["results"], second_rows)
        with (root / "summary.csv").open() as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual([(r["checkpoint"], r["dataset"], r["task_agent"]) for r in csv_rows],
                         [(r["checkpoint"], r["dataset"], r["task_agent"]) for r in second_rows])


if __name__ == "__main__":
    unittest.main()
