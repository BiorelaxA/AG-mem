import json
from threading import Barrier, Event, Lock, Thread, enumerate as threads, get_ident
from types import SimpleNamespace
import unittest

from agmem.core import Completion, Example, InfrastructureError, seeded_int
from agmem.env import MemoryEnvironment
from agmem.rollout_pipeline import collect_candidate_groups


def configuration(**execution):
    return {"seed": 17, "training": {"group_size": 8, "advantage_epsilon": 1e-6,
                                    "group_infrastructure_retries": 0},
            "execution": {"sampling_group_concurrency": 4, "reader_batch_size": 4,
                          "reader_batch_max_tokens": 16384, "task_agent_workers": 2,
                          **execution}}


def example(index):
    return Example(f"fixture/train/{index}", "fixture", "train", f"memory-{index}",
                   f"query-{index}", gold_answers=["yes"])


class MemoryStore:
    """Each query has an intentionally disjoint directory and source record."""
    def __init__(self, owner):
        self.owner = owner

    def _record(self, memory_id):
        assert get_ident() == self.owner, "Environment/cache access left its owner thread"
        return {"id": f"{memory_id}-id", "source_session_id": memory_id,
                "time": "2026-01-01", "order": 0, "raw_text": f"fact for {memory_id}"}

    def memory(self, memory_id):
        return [self._record(memory_id)]

    def directory(self, memory_id, version):
        return [{"id": self._record(memory_id)["id"], "content": f"summary for {memory_id}"}]

    def record(self, reference):
        return self._record(reference["source_session_id"])


class Reader:
    def __init__(self, owner, before_batch=None):
        self.owner, self.before_batch = owner, before_batch
        self.policy_version = 7
        self.batches = []

    def complete_batch(self, messages, seeds, *, max_batch_tokens):
        assert get_ident() == self.owner, "Reader must have exactly one owner thread"
        assert max_batch_tokens == 16384
        states = [json.loads(item[1]["content"]) for item in messages]
        self.batches.append([(state["query"], seed, bool(state["retrieved_evidence"]))
                             for state, seed in zip(states, seeds)])
        if self.before_batch:
            self.before_batch(len(self.batches))
        completions = []
        for state in states:
            evidence = state["retrieved_evidence"]
            if evidence:
                assert evidence[0]["id"] == state["directory"][0]["id"], "Cross-query evidence leak"
            action = {"ids": [] if evidence else [state["directory"][0]["id"]],
                      "stop": bool(evidence)}
            completions.append(Completion(json.dumps(action), token_ids=[1], old_logprobs=[-0.3]))
        return completions

    def complete(self, messages, *, seed):
        return self.complete_batch([messages], [seed], max_batch_tokens=16384)[0]


def score(example, trajectory):
    trajectory.judge_score = float(trajectory.seed % 2)
    trajectory.reward = 2 * trajectory.judge_score - 1
    return trajectory


class RolloutPipelineTests(unittest.TestCase):
    def components(self, answer=None, before_batch=None):
        owner = get_ident()
        env = MemoryEnvironment(MemoryStore(owner), "directory-v1", id_protocol="strict_v2")
        reader = Reader(owner, before_batch)
        def default_answer(item, evidence, metadata):
            self.assertNotEqual(get_ident(), owner)
            self.assertEqual(evidence[0]["id"], f"{item.memory_id}-id")
            return "yes", item.sample_id
        return env, reader, SimpleNamespace(answer=answer or default_answer), SimpleNamespace(score=score)

    def test_sampler_target_drains_real_window_and_leaves_next_queries_unconsumed(self):
        from agmem.dapo_sampling import DynamicGroupSampler
        from tests.test_dapo_sampling import judge_cost_settings

        config = judge_cost_settings(group_size=8, candidate_groups=32, target_groups=2)
        config["execution"] = configuration()["execution"]
        sampler = DynamicGroupSampler(config, [example(i) for i in range(12)])
        env, reader, agent, _ = self.components()
        counts, lock = {}, Lock()

        def mixed_score(item, trajectory):
            with lock:
                count = counts.get(item.sample_id, 0)
                counts[item.sample_id] = count + 1
            trajectory.judge_score = float(count % 2)
            trajectory.reward = 2 * trajectory.judge_score - 1
            return trajectory

        progress, records = [], []
        scorer = SimpleNamespace(score=mixed_score)

        def stream(candidates):
            return collect_candidate_groups(config, env, reader, agent, scorer, candidates,
                                            on_progress=progress.append)

        groups, report = sampler.collect(None, 7, collect_candidates=stream,
            record_candidate=lambda group, result: records.append(result))
        self.assertEqual(len(groups), 2)
        self.assertEqual(report["candidate_groups"], 4)
        self.assertEqual([row["selected"] for row in records], [True, True, False, False])
        self.assertEqual(sorted(counts.values()), [8] * 4)
        self.assertEqual(progress[-1]["groups_completed"], 4)
        self.assertEqual(progress[-1]["windows_completed"], 1)
        self.assertEqual(sampler.next_index, 4)
        self.assertFalse(any(t.name.startswith("agmem-scoring") and t.is_alive() for t in threads()))

        # A resumed sampler must start with the next unsubmitted query; no
        # background worker may survive into the new policy version.
        restored = DynamicGroupSampler(config, sampler.examples, **sampler.state_dict())
        reader.policy_version = 8
        restored.collect(None, 8, collect_candidates=stream,
            record_candidate=lambda group, result: records.append(result))
        self.assertEqual([row["candidate_id"] for row in records], list(range(8)))
        self.assertEqual(len({row["sample_id"] for row in records}), 8)
        self.assertEqual(restored.next_index, 8)
        self.assertEqual(sorted(counts.values()), [8] * 8)
        self.assertFalse(any(t.name.startswith("agmem-scoring") and t.is_alive() for t in threads()))

    def test_cross_query_batching_http_overlap_order_and_bounded_admission(self):
        both_scoring, reader_advanced = Barrier(3), Event()
        lock = Lock()
        active = peak = calls = 0
        def answer(item, evidence, metadata):
            nonlocal active, peak, calls
            with lock:
                active += 1
                peak = max(peak, active)
                calls += 1
                current = calls
            try:
                self.assertEqual(evidence[0]["id"], f"{item.memory_id}-id")
                if current <= 2:
                    both_scoring.wait(timeout=5)
                    self.assertTrue(reader_advanced.wait(timeout=5))
                return "yes", item.sample_id
            finally:
                with lock:
                    active -= 1
        def before_batch(number):
            # First four trajectories finish in batch 9. Scoring must already
            # run while the next Reader batch is still being generated.
            if number == 10:
                both_scoring.wait(timeout=5)
                reader_advanced.set()
        env, reader, agent, scorer = self.components(answer, before_batch)
        delivered, admitted, progress = [], [], []
        def candidates():
            for index in range(6):
                if index == 4:
                    self.assertEqual(delivered, [0, 1, 2, 3])
                admitted.append(index)
                yield example(index), index
        for item, candidate_id, rows in collect_candidate_groups(
                configuration(), env, reader, agent, scorer, candidates(), on_progress=progress.append):
            self.assertEqual(candidate_id, len(delivered))
            self.assertLessEqual(len(admitted) - len(delivered), 4)
            self.assertEqual(len(rows), 8)
            self.assertEqual([row.seed for row in rows], [
                seeded_int(17, item.sample_id, 7, candidate_id, i) % (2 ** 31) for i in range(8)])
            self.assertTrue(all(row.policy_version == 7 and len(row.decisions) == 2 for row in rows))
            self.assertTrue(all(row.evidence[0]["id"] == f"{item.memory_id}-id" for row in rows))
            delivered.append(candidate_id)
        self.assertEqual(peak, 2)
        self.assertEqual(active, 0)
        self.assertEqual(delivered, list(range(6)))
        self.assertEqual({entry[0] for entry in reader.batches[0]}, {f"query-{i}" for i in range(4)})
        self.assertTrue(all(len(batch) <= 4 for batch in reader.batches))
        self.assertEqual(progress[-1]["peak_groups"], 4)
        self.assertLessEqual(progress[-1]["peak_scoring_inflight"], 32)
        self.assertEqual(progress[-1]["reader_decisions"], 6 * 8 * 2)
        self.assertEqual(progress[-1]["groups_completed"], 6)
        self.assertEqual(progress[-1]["windows_completed"], 2)
        self.assertGreater(progress[-1]["scoring_seconds_sum"], 0)

    def test_http_completion_order_does_not_change_reader_batch_seeds(self):
        observed = []
        for delayed_query in ("query-0", "query-1"):
            early, release = Event(), Event()
            seen = set()
            lock = Lock()
            def answer(item, evidence, metadata):
                with lock:
                    first = item.query not in seen
                    seen.add(item.query)
                if first and item.query == delayed_query:
                    self.assertTrue(early.wait(timeout=5))
                    self.assertTrue(release.wait(timeout=5))
                else:
                    early.set()
                return "yes", item.sample_id
            def before_batch(number):
                if number == 10:
                    self.assertTrue(early.wait(timeout=5))
                    release.set()
            env, reader, agent, scorer = self.components(answer, before_batch)
            rows = list(collect_candidate_groups(configuration(), env, reader, agent, scorer,
                                                 ((example(i), i) for i in range(4))))
            self.assertEqual([item[1] for item in rows], [0, 1, 2, 3])
            observed.append(reader.batches)
        self.assertEqual(observed[0], observed[1])

    def test_infrastructure_retry_repeats_entire_group_with_original_seeds(self):
        failures, scored = [], []
        first = True
        lock = Lock()
        def retry_score(item, trajectory):
            nonlocal first
            with lock:
                scored.append(trajectory.seed)
                fail = first
                first = False
            if fail:
                raise InfrastructureError("temporary HTTP failure")
            return score(item, trajectory)
        env, reader, agent, _ = self.components()
        config = configuration(sampling_group_concurrency=1)
        config["training"]["group_infrastructure_retries"] = 1
        groups = list(collect_candidate_groups(config, env, reader, agent, SimpleNamespace(score=retry_score),
                      [(example(0), 33)], on_failure=lambda *args: failures.append(args)))
        self.assertEqual(len(groups[0][2]), 8)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0][1:3], (33, 0))
        expected = [seeded_int(17, example(0).sample_id, 7, 33, i) % (2 ** 31) for i in range(8)]
        self.assertEqual([row.seed for row in groups[0][2]], expected)
        # All eight trajectories, including any previously successful ones,
        # were freshly generated and scored on the replacement attempt.
        self.assertCountEqual(scored[-8:], expected)
        first_decision_seeds = [seed for batch in reader.batches for _, seed, has_evidence in batch if not has_evidence]
        self.assertEqual(first_decision_seeds, expected + expected)

    def test_reader_version_mutation_aborts_without_returning_any_group(self):
        env, reader, agent, scorer = self.components()
        reader.before_batch = lambda number: setattr(reader, "policy_version", 8)
        with self.assertRaisesRegex(ValueError, "unchanged policy version"):
            list(collect_candidate_groups(configuration(), env, reader, agent, scorer, [(example(0), 0)]))

    def test_exception_cancels_queued_scoring_and_drains_running_worker(self):
        started, fail_now, drained = Event(), Event(), Event()
        calls, lock = [], Lock()
        def answer(item, evidence, metadata):
            with lock:
                calls.append(item.query)
                index = len(calls)
            if index == 1:
                started.set()
                self.assertTrue(fail_now.wait(timeout=5))
                drained.set()
                return "yes", item.sample_id
            raise RuntimeError("CUDA out of memory")
        env, reader, agent, scorer = self.components(answer)
        def release():
            self.assertTrue(started.wait(timeout=5))
            fail_now.set()
        thread = Thread(target=release)
        thread.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                list(collect_candidate_groups(configuration(), env, reader, agent, scorer,
                                             ((example(i), i) for i in range(4))))
        finally:
            fail_now.set()
            thread.join(timeout=5)
        self.assertTrue(drained.is_set())
        self.assertFalse(thread.is_alive())
        # shutdown has completed before this assertion, so no scoring worker
        # can mutate application state after the optimizer would resume.
        self.assertFalse(any(t.name.startswith("agmem-scoring") and t.is_alive() for t in threads()))

    def test_generator_close_drains_worker_and_does_not_admit_next_window(self):
        blocked, release, drained, second_pass = Event(), Event(), Event(), Event()
        close_started, close_finished = Event(), Event()
        admitted = []
        lock = Lock()
        scored_first = 0
        def answer(item, evidence, metadata):
            nonlocal scored_first
            if item.query == "query-0":
                with lock:
                    scored_first += 1
                    last = scored_first == 8
                if last:
                    self.assertTrue(second_pass.wait(timeout=5))
            if item.query == "query-1":
                blocked.set()
                self.assertTrue(release.wait(timeout=5))
                drained.set()
            return "yes", item.sample_id
        def before_batch(number):
            if number == 9:
                second_pass.set()
        env, reader, agent, scorer = self.components(answer, before_batch)
        # Group 0 has only stops; its eight scores finish before group 1 starts
        # waiting, which permits an early ordered yield with HTTP still active.
        original = reader.complete_batch
        def generate(messages, seeds, **kwargs):
            values = original(messages, seeds, **kwargs)
            for index, message in enumerate(messages):
                if json.loads(message[1]["content"])["query"] == "query-0":
                    values[index] = Completion('{"ids":[],"stop":true}')
            return values
        reader.complete_batch = generate
        def candidates():
            for index in range(8):
                admitted.append(index)
                yield example(index), index
        iterator = collect_candidate_groups(configuration(), env, reader, agent, scorer, candidates())
        self.assertEqual(next(iterator)[1], 0)
        self.assertTrue(blocked.wait(timeout=5))
        close_returned_early = []
        def release_after_close():
            close_started.wait(timeout=5)
            close_returned_early.append(close_finished.wait(timeout=0.02))
            release.set()
        helper = Thread(target=release_after_close)
        helper.start()
        close_started.set()
        iterator.close()
        close_finished.set()
        helper.join(timeout=5)
        self.assertTrue(drained.is_set())
        self.assertEqual(close_returned_early, [False])
        self.assertEqual(admitted, [0, 1, 2, 3])
        self.assertFalse(any(t.name.startswith("agmem-scoring") and t.is_alive() for t in threads()))

    def test_exhausted_retries_report_both_attempts_and_leave_no_workers(self):
        env, reader, agent, _ = self.components()
        failures = []
        def unavailable(item, row):
            raise InfrastructureError("Judge unavailable")
        config = configuration(sampling_group_concurrency=1)
        config["training"]["group_infrastructure_retries"] = 1
        with self.assertRaisesRegex(InfrastructureError, "Judge unavailable"):
            list(collect_candidate_groups(config, env, reader, agent, SimpleNamespace(score=unavailable),
                 [(example(0), 0)], on_failure=lambda *args: failures.append(args)))
        self.assertEqual([failure[2] for failure in failures], [0, 1])
        self.assertFalse(any(t.name.startswith("agmem-scoring") and t.is_alive() for t in threads()))

    def test_infrastructure_retry_preserves_other_groups_and_order(self):
        env, reader, agent, _ = self.components()
        failed = False
        lock = Lock()
        def fail_once(item, row):
            nonlocal failed
            with lock:
                should_fail = item.query == "query-1" and not failed
                if should_fail:
                    failed = True
            if should_fail:
                raise InfrastructureError("temporary Judge failure")
            return score(item, row)
        config = configuration()
        config["training"]["group_infrastructure_retries"] = 1
        groups = list(collect_candidate_groups(config, env, reader, agent, SimpleNamespace(score=fail_once),
                                               ((example(i), i) for i in range(4))))
        self.assertEqual([group[1] for group in groups], [0, 1, 2, 3])
        for index in (0, 2, 3):
            generated = [seed for batch in reader.batches for query, seed, has_evidence in batch
                         if query == f"query-{index}" and not has_evidence]
            self.assertEqual(len(generated), 8)
        self.assertTrue(all(len(group[2]) == 8 for group in groups))

    def test_version_change_while_yielded_is_detected_on_resume(self):
        env, reader, agent, scorer = self.components()
        iterator = collect_candidate_groups(configuration(), env, reader, agent, scorer,
                                            ((example(i), i) for i in range(4)))
        self.assertEqual(next(iterator)[1], 0)
        reader.policy_version += 1
        with self.assertRaisesRegex(ValueError, "unchanged policy version"):
            next(iterator)


if __name__ == "__main__":
    unittest.main()
