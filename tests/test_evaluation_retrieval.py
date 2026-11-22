"""CPU-only checks of batched query state, early delivery and bounded recovery."""
import json
from threading import get_ident
import unittest

from agmem.core import CapacityError, Completion, Example
from agmem.env import MemoryEnvironment
from agmem.evaluation_retrieval import iter_retrievals


def example(index):
    return Example(f"fixture/test/{index}", "fixture", "test", f"memory-{index}", f"query-{index}")


class MemoryStore:
    def __init__(self):
        self.owner = get_ident()

    def memory(self, memory_id):
        assert get_ident() == self.owner, "Store access left the coordinating thread"
        return [{"id": f"{memory_id}-id-{index}", "source_session_id": memory_id,
                 "time": "2026-01-01", "order": index, "raw_text": f"fact for {memory_id}/{index}"}
                for index in range(3)]

    def directory(self, memory_id, version):
        return [{"id": row["id"], "content": f"summary for {row['id']}"} for row in self.memory(memory_id)]

    def record(self, reference):
        return self.memory(reference["source_session_id"])[reference["order"]]


class Environment(MemoryEnvironment):
    def __init__(self):
        super().__init__(MemoryStore(), "fixture", max_ids=1, max_calls=8,
                         id_protocol="strict_v2", prompt_version="evidence_loop_v3")
        self.opened, self.closed, self.active = [], [], set()
        self.peak = 0

    def start_episode(self, item, seed, policy_version=0):
        original = super().start_episode(item, seed, policy_version)

        def track():
            self.opened.append(item.sample_id)
            self.active.add(item.sample_id)
            self.peak = max(self.peak, len(self.active))
            try:
                return (yield from original)
            finally:
                self.active.remove(item.sample_id)
                self.closed.append(item.sample_id)

        return track()


class Reader:
    def __init__(self, targets):
        self.targets = targets
        self.owner = get_ident()
        self.policy_version = 20
        self.batches = []
        self.before_batch = None

    def complete_batch(self, messages, seeds, *, max_batch_tokens):
        assert get_ident() == self.owner, "Reader generation left the coordinating thread"
        states = [json.loads(row[1]["content"]) for row in messages]
        self.batches.append({"states": states, "seeds": seeds, "budget": max_batch_tokens})
        if self.before_batch:
            self.before_batch()
        completions = []
        for state in states:
            read = len(state["retrieved_evidence"])
            directory = state["directory"]
            assert [row["id"] for row in state["retrieved_evidence"]] == [row["id"] for row in directory[:read]]
            stop = read >= self.targets[state["query"]]
            action = {"ids": [] if stop else [directory[read]["id"]], "stop": stop}
            completions.append(Completion(json.dumps(action), sampling_batch_size=len(states)))
        return completions


class EvaluationRetrievalTests(unittest.TestCase):
    def components(self, counts=(0, 2, 1, 1, 0)):
        examples = [example(index) for index in range(len(counts))]
        return Environment(), Reader({item.query: count for item, count in zip(examples, counts)}), examples

    def test_cross_query_isolation_round_robin_and_different_stopping_times(self):
        env, policy, examples = self.components()
        rows = list(iter_retrievals(env, policy, examples, 17, batch_size=2,
                                    max_batch_tokens=1234, query_concurrency=3))
        self.assertEqual([item.sample_id for item, _ in rows],
                         [examples[index].sample_id for index in (0, 2, 1, 3, 4)])
        self.assertEqual(env.peak, 3)
        self.assertEqual(env.active, set())
        self.assertCountEqual(env.closed, [item.sample_id for item in examples])
        self.assertEqual([state["query"] for state in policy.batches[0]["states"]], ["query-0", "query-1"])
        self.assertEqual([state["query"] for state in policy.batches[1]["states"]], ["query-2", "query-1"])
        self.assertTrue(all(len(batch["states"]) <= 2 and batch["budget"] == 1234 for batch in policy.batches))
        self.assertEqual(len(policy.batches[-1]["states"]), 1)
        for item, row in rows:
            reads = policy.targets[item.query]
            self.assertEqual(row.sample_id, item.sample_id)
            self.assertEqual(row.seed, 17)
            self.assertEqual(row.policy_version, 20)
            self.assertEqual(row.n_read_calls, reads)
            self.assertEqual(row.termination_reason, "stop")
            self.assertEqual(len(row.decisions), reads + 1)
            self.assertEqual([record["id"] for record in row.evidence],
                             [f"{item.memory_id}-id-{index}" for index in range(reads)])
            decision_seeds = [seed for batch in policy.batches for state, seed in zip(batch["states"], batch["seeds"])
                              if state["query"] == item.query]
            self.assertEqual(decision_seeds, list(range(17, 18 + reads)))

    def test_first_completed_query_is_yielded_before_remaining_decisions_and_close_releases_all(self):
        env, policy, examples = self.components()
        stream = iter_retrievals(env, policy, examples, 17, batch_size=2, query_concurrency=3)
        item, row = next(stream)
        self.assertEqual(item.sample_id, examples[0].sample_id)
        self.assertEqual(row.n_read_calls, 0)
        self.assertEqual(len(policy.batches), 1)
        self.assertEqual(env.opened, [item.sample_id for item in examples[:3]])
        self.assertEqual(env.closed, [examples[0].sample_id])
        stream.close()
        self.assertEqual(env.active, set())
        self.assertCountEqual(env.closed, [item.sample_id for item in examples[:3]])
        self.assertEqual(len(env.opened), 3)

    def test_capacity_failure_has_query_context_and_closes_all_episodes(self):
        env, policy, examples = self.components()
        def fail():
            raise CapacityError("Reader requires more context; no truncation")
        policy.before_batch = fail
        with self.assertRaisesRegex(CapacityError, r"fixture/test/0.*fixture/test/1.*no truncation"):
            list(iter_retrievals(env, policy, examples, 17, batch_size=2, query_concurrency=3))
        self.assertEqual(env.active, set())
        self.assertCountEqual(env.closed, [item.sample_id for item in examples[:3]])

    def test_wrong_completion_count_has_query_context_and_closes_all_episodes(self):
        env, policy, examples = self.components()
        policy.complete_batch = lambda *args, **kwargs: []
        with self.assertRaisesRegex(ValueError, r"fixture/test/0.*expected 2, got 0"):
            list(iter_retrievals(env, policy, examples, 17, batch_size=2, query_concurrency=3))
        self.assertEqual(env.active, set())
        self.assertEqual(len(env.closed), 3)

    def test_unexpected_reader_error_preserves_type_and_adds_query_context(self):
        env, policy, examples = self.components()
        def fail():
            raise RuntimeError("GPU failure")
        policy.before_batch = fail
        with self.assertRaisesRegex(RuntimeError, "GPU failure") as caught:
            list(iter_retrievals(env, policy, examples, 17))
        self.assertIn("fixture/test/0", " ".join(caught.exception.__notes__))
        self.assertEqual(env.active, set())

    def test_input_failure_does_not_leave_active_episodes_open(self):
        env, policy, examples = self.components()
        def source():
            yield examples[0]
            raise ValueError("Broken query source")
        with self.assertRaisesRegex(ValueError, "Broken query source"):
            list(iter_retrievals(env, policy, source(), 17))
        self.assertEqual(env.closed, [examples[0].sample_id])
        self.assertEqual(policy.batches, [])

    def test_environment_initialization_failure_closes_previously_started_queries(self):
        env, policy, examples = self.components()
        directory = env.store.directory
        def broken_directory(memory_id, version):
            if memory_id == examples[1].memory_id:
                raise ValueError("Directory snapshot is missing")
            return directory(memory_id, version)
        env.store.directory = broken_directory
        with self.assertRaisesRegex(ValueError, "Directory snapshot is missing") as caught:
            list(iter_retrievals(env, policy, examples, 17))
        self.assertIn(examples[1].sample_id, " ".join(caught.exception.__notes__))
        self.assertEqual(env.active, set())
        self.assertCountEqual(env.closed, [item.sample_id for item in examples[:2]])

    def test_capacity_rejected_tool_result_remains_query_feedback(self):
        env, policy, examples = self.components((1, 0))
        def reject_evidence(*args):
            raise CapacityError("Task context would overflow")
        env.capacity = reject_evidence
        rows = list(iter_retrievals(env, policy, examples, 17, batch_size=2))
        self.assertEqual([item for item, _ in rows], examples[::-1])
        row = rows[1][1]
        self.assertEqual(row.n_read_calls, 8)
        self.assertEqual(row.termination_reason, "max_read_calls")
        self.assertEqual(row.evidence, [])
        self.assertEqual([decision.tool_result["error"]["code"] for decision in row.decisions],
                         ["context_capacity_exceeded"] * 8 + ["max_read_calls"])
        self.assertEqual(env.active, set())

    def test_policy_version_cannot_change_during_generation_or_between_yields(self):
        env, policy, examples = self.components()
        policy.before_batch = lambda: setattr(policy, "policy_version", 21)
        with self.assertRaisesRegex(ValueError, "unchanged policy version"):
            list(iter_retrievals(env, policy, examples, 17))
        self.assertEqual(env.active, set())
        env, policy, examples = self.components()
        stream = iter_retrievals(env, policy, examples, 17)
        next(stream)
        policy.policy_version = 21
        with self.assertRaisesRegex(ValueError, "unchanged policy version"):
            next(stream)
        self.assertEqual(env.active, set())

    def test_empty_input_and_single_query_concurrency(self):
        env, policy, examples = self.components()
        self.assertEqual(list(iter_retrievals(env, policy, [], 17)), [])
        rows = list(iter_retrievals(env, policy, examples[:2], 17, batch_size=4, query_concurrency=1))
        self.assertEqual([item for item, _ in rows], examples[:2])
        self.assertTrue(all(len(batch["states"]) == 1 for batch in policy.batches))
        self.assertEqual(env.peak, 1)

    def test_only_positive_integer_execution_settings_are_accepted(self):
        for name in ("batch_size", "max_batch_tokens", "query_concurrency"):
            for value in (0, -1, True, 1.5, "4"):
                with self.subTest(name=name, value=value):
                    env, policy, examples = self.components()
                    with self.assertRaisesRegex(ValueError, name):
                        list(iter_retrievals(env, policy, examples, 17, **{name: value}))
                    self.assertEqual(env.opened, [])
                    self.assertEqual(policy.batches, [])


if __name__ == "__main__":
    unittest.main()
