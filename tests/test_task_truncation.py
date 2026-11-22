"""Exercise truncated Task answers through the actual sampling/scoring pipeline."""

from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import Mock

import httpx

from agmem.core import (Completion, Decision, Example, InfrastructureError,
                        Trajectory, seeded_int, trajectory_from_dict)
from agmem.dapo import trajectory_truncated, valid_training_tokens
from agmem.judges import JudgeScorer, prompt_for
from agmem.pipeline import collect_group, metrics, scored_rollout
from agmem.rewards import RewardScorer
from agmem.services import ChatService, TaskAgent
from agmem.store import Store
from agmem.traces import compact_group
from agmem.training_dapo import run_training_loop


class TaskTruncationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "store")
        self.example = Example("locomo/train/0", "locomo", "train", "memory", "What is the answer?",
                               gold_answers=["yes"])

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def fixture(self, *, failure=None, workers=2):
        config = {
            "seed": 42, "reader": {"rollout_temperature": 1.0},
            "execution": {"reader_batch_size": 1, "task_agent_workers": workers},
            "training": {"group_size": 8, "advantage_epsilon": 1e-6, "epochs": 1,
                         "candidate_groups": 1, "target_groups": 1, "max_num_gen_batches": 1,
                         "group_infrastructure_retries": 0},
            "reward": {"formula": "judge_gated_v1", "read_call_cost": 0.01,
                       "format_penalty": 0.1, "likelihood": {"mode": "off"}},
        }
        service_config = {"model": "fixture-27b", "model_revision": str(failure or "healthy"),
                          "base_url": "http://fixture/v1", "retries": 0}
        policy = SimpleNamespace(policy_version=0)
        seeds = [seeded_int(42, self.example.sample_id, 0, 0, i) % (2 ** 31) for i in range(8)]
        truncated = {0, 3, 4, 7}
        answers = [f"item {i}: " + ("yes" if i % 2 == 0 else "no")
                   + (", and the" if i in truncated else ".") for i in range(8)]
        task_calls, judged, created = [], [], []
        lock, barrier = Lock(), Barrier(2)

        def run(example, current_policy, seed, version):
            self.assertEqual(current_policy.policy_version, 0)
            index = seeds.index(seed)
            row = Trajectory(example.sample_id, seed, example.memory_id, version,
                             decisions=[Decision([], Completion('{"ids":[],"stop":true}',
                                                                 token_ids=[10, 11], old_logprobs=[-0.2, -0.3]),
                                                 {"ids": [], "stop": True})],
                             evidence=[{"index": index}], n_read_calls=1, termination_reason="stop")
            created.append(row)
            return row

        def failure_response(role):
            if not failure or failure[0] != role:
                return None
            kind = failure[1]
            if kind in {"http", "oom"}:
                message = "CUDA out of memory" if kind == "oom" else "service unavailable"
                return httpx.Response(500, json={"error": {"message": message}})
            content = '{"label":"CORRECT"}' if kind == "length" else '{"label":"MAYBE"}'
            return httpx.Response(200, json={"choices": [{
                "finish_reason": "length" if kind == "length" else "stop",
                "message": {"content": content}}]})

        def task_response(request):
            payload = json.loads(request.content)
            index = json.loads(payload["messages"][1]["content"])["retrieved_evidence"][0]["index"]
            with lock:
                task_calls.append(index)
            # Force two different completion records to be in flight together.
            if workers > 1 and index in {0, 1}:
                barrier.wait(timeout=5)
            failed = failure_response("task_agent")
            if failed is not None:
                return failed
            return httpx.Response(200, json={"choices": [{
                "finish_reason": "length" if index in truncated else "stop",
                "message": {"content": answers[index]}}],
                "usage": {"prompt_tokens": 100 + index,
                          "completion_tokens": 512 if index in truncated else index + 2}})

        def judge_response(request):
            payload = json.loads(request.content)
            prompt = payload["messages"][0]["content"]
            matches = [i for i, answer in enumerate(answers)
                       if prompt == prompt_for(self.example, "yes", answer, "locomo_refined")[0]]
            self.assertEqual(len(matches), 1, "Judge must receive the exact generated text")
            index = matches[0]
            with lock:
                judged.append(index)
            failed = failure_response("judge")
            if failed is not None:
                return failed
            label = "CORRECT" if index % 2 == 0 else "WRONG"
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                "message": {"content": json.dumps({"label": label})}}]})

        service = ChatService(service_config, self.store, "task_agent")
        service.client.close()
        service.client = httpx.Client(transport=httpx.MockTransport(task_response))
        self.addCleanup(service.client.close)
        agent = TaskAgent(service, max_output_tokens=512)
        judge = JudgeScorer(service_config, self.store)
        judge.service.client.close()
        judge.service.client = httpx.Client(transport=httpx.MockTransport(judge_response))
        self.addCleanup(judge.close)
        scorer = RewardScorer(config["reward"], judge=judge)
        return SimpleNamespace(config=config, policy=policy, env=SimpleNamespace(run=run, max_ids=4),
                               agent=agent, scorer=scorer, seeds=seeds, answers=answers,
                               truncated=truncated, task_calls=task_calls, judged=judged, created=created)

    def collect(self, fixture):
        return collect_group(fixture.config, fixture.env, fixture.policy, fixture.agent,
                             self.example, 0, scorer=fixture.scorer)

    def test_eight_rollouts_keep_truncated_answers_metadata_and_advantages(self):
        fixture = self.fixture()
        group = self.collect(fixture)
        self.assertEqual(len(group), 8)
        self.assertEqual([row.seed for row in group], fixture.seeds)
        self.assertEqual([row.prediction for row in group], fixture.answers)
        self.assertCountEqual(fixture.task_calls, range(8))
        self.assertCountEqual(fixture.judged, range(8))
        self.assertEqual([row.judge_score for row in group], [1, 0] * 4)
        for index, row in enumerate(group):
            self.assertEqual(row.task_completion["truncated"], index in fixture.truncated)
            self.assertEqual(row.task_completion["finish_reason"],
                             "length" if index in fixture.truncated else "stop")
            self.assertEqual(row.task_completion["usage"]["prompt_tokens"], 100 + index)
            self.assertEqual(row.task_completion["max_tokens"], 512)
            self.assertAlmostEqual(row.reward, 0.99 if index % 2 == 0 else -1)
            self.assertEqual(row.advantage > 0, index % 2 == 0)
            self.assertFalse(trajectory_truncated(row))
            self.assertEqual(trajectory_from_dict(row.to_dict()).to_dict(), row.to_dict())
        self.assertEqual(valid_training_tokens(group, overlong_filter=True), 16)
        self.assertEqual(metrics(group)["pooled"]["task_truncation_rate"], 0.5)
        compact = compact_group({"group_id": "group0", "directory_version": "version0",
                                 "rollouts": [row.to_dict() for row in group]})
        self.assertEqual([row["task_completion"] for row in compact["rollouts"]],
                         [row.task_completion for row in group])
        self.assertFalse((self.store.root / "service_failures.jsonl").exists())

        # Reopen the disk cache and confirm both text and metadata survive.
        self.store.close()
        self.store = Store(self.store.root)
        fixture.agent.service.store = self.store
        fixture.scorer.judge.service.store = self.store
        group[0].task_completion["usage"]["prompt_tokens"] = -1
        cached = self.collect(fixture)
        self.assertEqual(cached[0].task_completion["usage"]["prompt_tokens"], 100)
        self.assertEqual([row.prediction for row in cached], fixture.answers)
        self.assertEqual([row.task_completion["truncated"] for row in cached],
                         [i in fixture.truncated for i in range(8)])
        self.assertEqual(len(fixture.task_calls), 8)
        self.assertEqual(len(fixture.judged), 8)

    def test_evaluation_rollout_scores_the_same_nonempty_truncated_answer(self):
        fixture = self.fixture(workers=1)
        row = scored_rollout(fixture.config, fixture.env, fixture.policy, fixture.agent,
                             self.example, fixture.seeds[0], scorer=fixture.scorer)
        self.assertEqual(row.prediction, fixture.answers[0])
        self.assertTrue(row.task_completion["truncated"])
        self.assertEqual(row.judge_score, 1)
        self.assertEqual(fixture.judged, [0])

    def test_bad_judge_or_http_oom_propagates_out_of_training_without_an_update(self):
        failures = [("judge", "length"), ("judge", "malformed"), ("judge", "http"),
                    ("judge", "oom"), ("task_agent", "http"), ("task_agent", "oom")]
        for failure in failures:
            with self.subTest(failure=failure):
                fixture = self.fixture(failure=failure, workers=1)
                updater = Mock()
                run_dir = Path(self.temp.name) / "-".join(failure)
                with self.assertRaisesRegex(InfrastructureError, f"{failure[0]} service failed"):
                    run_training_loop(fixture.config, self.store, fixture.policy, None, updater,
                                      fixture.env, fixture.agent, fixture.scorer, [self.example], run_dir)
                updater.begin_batch.assert_not_called()
                updater.update.assert_not_called()
                failures_log = [json.loads(line) for line in
                                (run_dir / "infrastructure_failures.jsonl").read_text().splitlines()]
                self.assertEqual(len(failures_log), 1)
                self.assertEqual(failures_log[0]["sample_id"], self.example.sample_id)
                self.assertEqual(failures_log[0]["step"], 0)
                self.assertFalse((run_dir / "candidate_metrics.jsonl").exists())
                self.assertIsNone(fixture.created[0].judge_score)


if __name__ == "__main__":
    unittest.main()
