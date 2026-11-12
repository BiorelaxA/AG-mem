"""Task/Judge concurrency and recovery without HTTP, model weights or CUDA."""

from copy import deepcopy
from concurrent.futures import wait
from threading import Barrier, Event, Lock, get_ident
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agmem.core import Example, InfrastructureError, Trajectory
from agmem.evaluation_scoring import score_pending


class Journal:
    def __init__(self, rows=None, written=None):
        self.owner = get_ident()
        self.rows = {row.sample_id: row.to_dict() for row in rows or []}
        self.reads = []
        self.writes = []
        self.written = written

    def read(self, sample_id):
        if get_ident() != self.owner:
            raise AssertionError("Journal reads must stay on the coordinator")
        self.reads.append(sample_id)
        return deepcopy(self.rows.get(sample_id))

    def write(self, row):
        if get_ident() != self.owner:
            raise AssertionError("Journal writes must stay on the coordinator")
        self.writes.append(row.sample_id)
        self.rows[row.sample_id] = row.to_dict()
        if self.written:
            self.written(row)


class ScoringConcurrencyTests(unittest.TestCase):
    def examples(self, count):
        return [Example(f"locomo/test/{i}", "locomo", "test", f"memory-{i}", f"q{i}")
                for i in range(count)]

    def row(self, example):
        return Trajectory(example.sample_id, 42, example.memory_id,
                          evidence=[{"id": "r1", "raw_text": "The evidence."}],
                          n_read_calls=1, termination_reason="stop")

    def scorer(self, calls=None):
        def score(example, row):
            if calls is not None:
                calls.append(("judge", example.sample_id, get_ident()))
            row.judge_score = 1.0
            row.reward = 0.99
            return row
        return SimpleNamespace(score=score)

    def test_queries_overlap_but_journals_remain_on_coordinator(self):
        examples = self.examples(3)
        retrieval = Journal([self.row(example) for example in examples])
        release_first = Event()
        journal = Journal(written=lambda row: release_first.set()
                          if row.sample_id == examples[1].sample_id else None)
        barrier, lock = Barrier(3), Lock()
        active = peak = 0
        calls = []

        def answer(example, evidence, metadata):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(active, peak)
                calls.append(("task", example.sample_id, get_ident()))
            barrier.wait(timeout=3)
            if example.sample_id == examples[0].sample_id:
                self.assertTrue(release_first.wait(timeout=3))
            metadata["truncated"] = False
            with lock:
                active -= 1
            return "answer", "task-hash"

        outputs = list(score_pending(examples, retrieval, journal,
                                     SimpleNamespace(answer=answer), self.scorer(calls), workers=3))
        self.assertEqual(peak, 3)
        self.assertNotEqual(outputs[0][0], 0)
        self.assertEqual(sorted(result[0] for result in outputs), [0, 1, 2])
        ordered = [None] * len(examples)
        for index, example, row, is_new in outputs:
            self.assertTrue(is_new)
            ordered[index] = row.sample_id
        self.assertEqual(ordered, [example.sample_id for example in examples])
        self.assertTrue(all(thread_id != get_ident() for _, _, thread_id in calls))
        for example in examples:
            phases = [phase for phase, sample_id, _ in calls if sample_id == example.sample_id]
            self.assertEqual(phases, ["task", "judge"])
        self.assertEqual(len(journal.writes), 3)

    def test_worker_bound_limits_full_trajectory_prefetch(self):
        examples = self.examples(8)
        base = Journal([self.row(example) for example in examples])
        outstanding = peak = 0

        def read(sample_id):
            nonlocal outstanding, peak
            outstanding += 1
            peak = max(peak, outstanding)
            return base.read(sample_id)

        def written(row):
            nonlocal outstanding
            outstanding -= 1

        journal = Journal(written=written)
        outputs = list(score_pending(examples, SimpleNamespace(read=read), journal,
                                     SimpleNamespace(answer=lambda *args, **kwargs: ("answer", "hash")),
                                     self.scorer(), workers=2))
        self.assertEqual(len(outputs), 8)
        self.assertLessEqual(peak, 2)
        self.assertEqual(outstanding, 0)

    def test_resume_skips_cached_task_and_judge_and_preserves_input_index(self):
        examples = self.examples(3)
        retrieval = Journal([self.row(example) for example in examples])
        cached = self.row(examples[1])
        cached.prediction, cached.judge_score = "cached answer", 1.0
        journal = Journal([cached])
        calls = []

        def answer(example, evidence, metadata):
            calls.append(example.sample_id)
            return "answer", "hash"

        outputs = list(score_pending(examples, retrieval, journal, SimpleNamespace(answer=answer),
                                     self.scorer(), workers=2))
        self.assertEqual(outputs[0][0], 1)
        self.assertFalse(outputs[0][3])
        self.assertEqual(outputs[0][2].prediction, "cached answer")
        self.assertCountEqual(calls, [examples[0].sample_id, examples[2].sample_id])
        self.assertNotIn(examples[1].sample_id, retrieval.reads)
        self.assertNotIn(examples[1].sample_id, journal.writes)
        calls.clear()
        outputs = list(score_pending(examples, retrieval, journal, SimpleNamespace(answer=answer),
                                     self.scorer(), workers=2))
        self.assertEqual(calls, [])
        self.assertTrue(all(not result[3] for result in outputs))

    def test_failure_drains_successes_without_zero_score_or_more_submissions(self):
        examples = self.examples(5)
        retrieval = Journal([self.row(example) for example in examples])
        journal = Journal()
        barrier = Barrier(3)
        release_successes = Event()
        calls = []

        def answer(example, evidence, metadata):
            calls.append(example.sample_id)
            barrier.wait(timeout=3)
            if example.sample_id == examples[0].sample_id:
                raise InfrastructureError("provider unavailable")
            self.assertTrue(release_successes.wait(timeout=3))
            return "answer", "hash"

        def observed_failure(*args, **kwargs):
            completed = wait(*args, **kwargs)
            # The first completed future must be the failing query. Release
            # the successful workers only after the coordinator observes it.
            release_successes.set()
            return completed

        outputs = []
        with patch("agmem.execution.wait", side_effect=observed_failure), \
                self.assertRaisesRegex(InfrastructureError, "provider unavailable"):
            for output in score_pending(examples, retrieval, journal, SimpleNamespace(answer=answer),
                                        self.scorer(), workers=3):
                outputs.append(output)
        self.assertCountEqual(calls, [example.sample_id for example in examples[:3]])
        self.assertCountEqual(journal.writes, [examples[1].sample_id, examples[2].sample_id])
        self.assertNotIn(examples[0].sample_id, journal.rows)
        self.assertTrue(all(row.judge_score == 1 for _, _, row, _ in outputs))

    def test_missing_retrieval_drains_already_started_success(self):
        examples = self.examples(3)
        retrieval = Journal([self.row(examples[0])])
        journal = Journal()
        with self.assertRaisesRegex(ValueError, "Missing retrieval"):
            list(score_pending(examples, retrieval, journal,
                               SimpleNamespace(answer=lambda *args, **kwargs: ("answer", "hash")),
                               self.scorer(), workers=2))
        self.assertEqual(journal.writes, [examples[0].sample_id])

    def test_stream_advances_reader_on_coordinator_while_task_is_running(self):
        examples = self.examples(3)
        journal = Journal()
        task_started, stream_advanced = Event(), Event()
        coordinator = get_ident()
        consumed = []

        def stream():
            for index in (2, 0, 1):
                self.assertEqual(get_ident(), coordinator)
                if index == 0:
                    self.assertTrue(task_started.wait(timeout=3))
                    stream_advanced.set()
                consumed.append(index)
                yield examples[index], self.row(examples[index])

        def answer(example, evidence, metadata):
            if example.sample_id == examples[2].sample_id:
                task_started.set()
                self.assertTrue(stream_advanced.wait(timeout=3))
            return "answer", "hash"

        outputs = list(score_pending(examples, None, journal, SimpleNamespace(answer=answer),
                                     self.scorer(), workers=2, retrieval_rows=stream()))
        self.assertEqual(consumed, [2, 0, 1])
        self.assertEqual(sorted(result[0] for result in outputs), [0, 1, 2])
        self.assertCountEqual(journal.writes, [example.sample_id for example in examples])

    def test_stream_is_fully_consumed_even_when_all_answers_are_cached(self):
        examples = self.examples(3)
        journal = Journal([self.row(example) for example in examples])
        consumed = []

        def stream():
            for example in examples:
                consumed.append(example.sample_id)
                yield example, self.row(example)

        outputs = list(score_pending(examples, None, journal, None, None,
                                     workers=2, retrieval_rows=stream()))
        self.assertEqual(consumed, [example.sample_id for example in examples])
        self.assertEqual(journal.writes, [])
        self.assertTrue(all(not result[3] for result in outputs))

    def test_stream_failure_preserves_successful_inflight_queries(self):
        examples = self.examples(3)
        for failure in ("short", "duplicate", "unknown", "provider"):
            with self.subTest(failure=failure):
                journal = Journal()

                def stream():
                    yield examples[0], self.row(examples[0])
                    if failure == "duplicate":
                        yield examples[0], self.row(examples[0])
                    elif failure == "unknown":
                        unknown = Example("unknown", "locomo", "test", "memory", "q")
                        yield unknown, self.row(unknown)
                    elif failure == "provider":
                        raise InfrastructureError("reader unavailable")

                expected_error = InfrastructureError if failure == "provider" else ValueError
                with self.assertRaises(expected_error):
                    list(score_pending(examples, None, journal,
                                       SimpleNamespace(answer=lambda *args, **kwargs: ("answer", "hash")),
                                       self.scorer(), workers=2, retrieval_rows=stream()))
                self.assertEqual(journal.writes, [examples[0].sample_id])

    def test_judge_failure_never_persists_incomplete_answer_as_zero_score(self):
        examples = self.examples(1)
        retrieval = Journal([self.row(examples[0])])
        journal = Journal()

        def judge(example, row):
            raise InfrastructureError("judge unavailable")

        with self.assertRaisesRegex(InfrastructureError, "judge unavailable"):
            list(score_pending(examples, retrieval, journal,
                               SimpleNamespace(answer=lambda *args, **kwargs: ("answer", "hash")),
                               SimpleNamespace(score=judge), workers=2))
        self.assertEqual(journal.writes, [])

    def test_retrieval_and_cached_history_must_match_example(self):
        examples = self.examples(1)
        row = self.row(examples[0])
        row.memory_id = "wrong-memory"
        for cached in (False, True):
            with self.subTest(cached=cached):
                retrieval = Journal([row])
                journal = Journal([row] if cached else [])
                with self.assertRaisesRegex(ValueError, "history does not match"):
                    list(score_pending(examples, retrieval, journal,
                                       SimpleNamespace(answer=lambda *args, **kwargs: self.fail("unexpected request")),
                                       self.scorer(), workers=2))

    def test_invalid_workers_and_duplicate_examples_fail_before_io(self):
        examples = self.examples(1)
        no_io = SimpleNamespace(read=lambda *args: self.fail("unexpected read"))
        for workers in (0, -1, 1.5, True, None):
            with self.subTest(workers=workers), self.assertRaisesRegex(ValueError, "positive integer"):
                list(score_pending(examples, no_io, no_io, None, None, workers=workers))
        with self.assertRaisesRegex(ValueError, "unique sample IDs"):
            list(score_pending(examples * 2, no_io, no_io, None, None, workers=1))

    def test_single_worker_keeps_same_question_answer_before_judge(self):
        examples = self.examples(2)
        retrieval = Journal([self.row(example) for example in examples])
        journal = Journal()
        calls = []

        def answer(example, evidence, metadata):
            calls.append(("task", example.sample_id, get_ident()))
            return "answer", "hash"

        outputs = list(score_pending(examples, retrieval, journal, SimpleNamespace(answer=answer),
                                     self.scorer(calls), workers=1))
        self.assertEqual([result[0] for result in outputs], [0, 1])
        self.assertEqual([(phase, sample_id) for phase, sample_id, _ in calls],
                         [(phase, example.sample_id) for example in examples for phase in ("task", "judge")])


if __name__ == "__main__":
    unittest.main()
