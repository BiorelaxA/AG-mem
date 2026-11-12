"""Reader, Task and Judge integration with real journals and CPU-only fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from threading import Event, get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agmem.core import CapacityError
from agmem.evaluate_checkpoints import (EpisodeJournal, evaluation_settings,
                                        iter_retrieved, score_evidence)
from agmem.rewards import score_trajectory
from tests.test_evaluation_retrieval import Environment, Reader, example


class EvaluationPipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.examples = [example(i) for i in range(3)]
        for item in self.examples:
            item.gold_answers = [item.sample_id]
        self.examples[0].judge_metadata["question_type"] = "short"
        self.examples[1].judge_metadata["question_type"] = "long"
        self.examples[2].judge_metadata["question_type"] = "short"
        self.config = {
            "seed": 17, "reader": {"prompt_version": "evidence_loop_v3"},
            "retrieval": {"max_ids_per_call": 1, "max_read_calls": 8, "id_protocol": "strict_v2"},
            "task_agent": {"max_output_tokens": 512},
            "reward": {"formula": "judge_cost_v2", "read_call_cost": 0.01,
                       "format_penalty": 0.5, "format_penalty_mode": "per_id",
                       "text_outcome": "qa_f1_v1", "choice_outcome": "label_f1"},
        }
        self.journal = EpisodeJournal(self.root / "retrieval", {
            "sample_ids": [e.sample_id for e in self.examples], "protocol": "fixture"})
        self.calls = []
        service = SimpleNamespace(identity={"model": "fixture"}, config={},
                                  client=SimpleNamespace(close=Mock()))
        self.agent = SimpleNamespace(service=service, answer=self.answer)
        self.scorer = SimpleNamespace(score=self.score, provenance=lambda: {"fixture": True}, close=Mock())
        self.owner = get_ident()

    def answer(self, item, evidence, metadata):
        self.assertNotEqual(get_ident(), self.owner)
        self.calls.append(item.sample_id)
        self.assertTrue(all(row["id"].startswith(item.memory_id + "-") for row in evidence))
        return item.sample_id, "hash"

    def score(self, item, row):
        judge = SimpleNamespace(score=lambda e, prediction: float(e == self.examples[0] or e == self.examples[2]))
        return score_trajectory(item, row, self.config["reward"], judge=judge, max_ids=1)

    def reader(self):
        reader = Reader({e.query: count for e, count in zip(self.examples, (0, 2, 1))})
        reader.temperature = 0.7
        return reader

    def run_pipeline(self, reader, env, settings=None):
        with patch("agmem.evaluate_checkpoints.validate_directories"), \
             patch("agmem.evaluate_checkpoints.make_environment", return_value=env), \
             patch("agmem.evaluate_checkpoints.make_task_agent", return_value=self.agent), \
             patch("agmem.evaluate_checkpoints.make_reward_scorer", return_value=self.scorer), \
             patch("agmem.evaluate_checkpoints.announce"):
            stream = iter_retrieved(self.config, None, reader, self.examples, self.journal, settings)
            return score_evidence(self.config, None, self.examples, self.journal,
                                  self.root / "answers", workers=2, retrieval_rows=stream)

    def test_reader_overlaps_judge_and_saves_paired_evidence_and_ordered_categories(self):
        reader, env = self.reader(), Environment()
        judge_started, reader_advanced = Event(), Event()

        def before_batch():
            self.assertEqual(reader.temperature, 0.0)
            if len(reader.batches) == 2:
                self.assertTrue(judge_started.wait(timeout=3))
                reader_advanced.set()

        def score(item, row):
            if item == self.examples[0]:
                # Reader must save evidence before handing a query to this worker.
                self.assertTrue(self.journal.path(item.sample_id).is_file())
                judge_started.set()
                self.assertTrue(reader_advanced.wait(timeout=3))
            return self.score(item, row)

        reader.before_batch = before_batch
        self.scorer.score = score
        report = self.run_pipeline(reader, env)
        self.assertTrue(reader_advanced.is_set())
        self.assertEqual(reader.temperature, 0.7)
        self.assertEqual(env.active, set())
        self.assertEqual(report["pooled"]["n"], 3)
        self.assertEqual(report["by_question_type"]["short"]["judge_accuracy"], 1)
        self.assertEqual(report["by_question_type"]["long"]["judge_accuracy"], 0)
        self.assertTrue((self.journal.root / "complete.json").is_file())
        self.assertTrue((self.root / "answers/complete.json").is_file())
        manifest = json.loads((self.root / "answers/manifest.json").read_text())
        answers = EpisodeJournal(self.root / "answers", manifest)
        for item, calls in zip(self.examples, (0, 2, 1)):
            raw, scored = self.journal.read(item.sample_id), answers.read(item.sample_id)
            self.assertEqual(raw["prediction"], "")
            self.assertEqual(raw["n_read_calls"], calls)
            self.assertEqual(raw["evidence"], scored["evidence"])
            self.assertEqual(scored["prediction"], item.sample_id)

    def test_reader_failure_saves_inflight_answer_and_resume_skips_completed_query(self):
        reader, env = self.reader(), Environment()

        def fail():
            if len(reader.batches) == 2:
                raise CapacityError("fixture context failure")

        reader.before_batch = fail
        with self.assertRaisesRegex(CapacityError, "fixture context failure"):
            self.run_pipeline(reader, env)
        self.assertEqual(reader.temperature, 0.7)
        self.assertEqual(env.active, set())
        self.assertEqual(self.calls, [self.examples[0].sample_id])
        self.assertTrue((self.root / "answers/episodes/00000000.json.gz").is_file())
        for root in (self.journal.root, self.root / "answers"):
            self.assertFalse((root / "complete.json").exists())
        self.assertFalse((self.root / "answers/metrics.json").exists())
        first_bytes = self.journal.path(self.examples[0].sample_id).read_bytes()

        self.calls.clear()
        fresh = self.reader()
        report = self.run_pipeline(fresh, Environment())
        self.assertCountEqual(self.calls, [e.sample_id for e in self.examples[1:]])
        self.assertNotIn(self.examples[0].query,
                         [state["query"] for batch in fresh.batches for state in batch["states"]])
        self.assertEqual(self.journal.path(self.examples[0].sample_id).read_bytes(), first_bytes)
        self.assertEqual(report["pooled"]["n"], 3)

    def test_resuming_with_different_concurrency_keeps_saved_results_and_needs_no_reader(self):
        self.run_pipeline(self.reader(), Environment())
        before = deepcopy(self.calls)
        settings = evaluation_settings({"evaluation_execution": {
            "reader_query_concurrency": 1, "reader_batch_size": 1}})
        self.run_pipeline(None, None, settings)
        self.assertEqual(self.calls, before)

    def test_evaluation_settings_are_independent_and_validate_typos_and_values(self):
        settings = evaluation_settings({"execution": {"mini_batch_groups": 16}})
        self.assertEqual(settings["reader_batch_size"], 4)
        self.assertEqual(settings["local_scoring_workers"], 2)
        self.assertEqual(settings["remote_scoring_workers"], 4)
        for value in ({"local_scoring_workers": 0}, {"reader_batch_size": True},
                      {"reader_batch_size": 1.5}, {"unknown_workers": 4}, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluation_settings({"evaluation_execution": value})


if __name__ == "__main__":
    unittest.main()
