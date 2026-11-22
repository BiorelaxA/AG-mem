"""PRM scores available answer-tag text without promoting transport failures."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

from agmem.core import Completion, Decision, Example, InfrastructureError, Trajectory, digest
from agmem.dapo import trajectory_truncated
from agmem.pipeline import metrics
from agmem.prompts import task_messages
from agmem.rewards import RewardScorer
from agmem.services import (ChatService, TASK_OUTPUT_POLICY, TASK_TAGGED_PARTIAL_OUTPUT_POLICY,
                            TaskAgent, extract_partial_tagged_answer)
from agmem.store import Store


class PartialTaskAnswerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name) / "store")
        self.addCleanup(lambda: self.store.close())
        self.service_number = 0
        self.example = Example("fixture/train/0", "fixture", "train", "memory", "Which city?",
                               observation="Current date: 2026-11-09", gold_answers=["Paris"])
        self.evidence = [{"id": "r_0", "time": "2026-11-08", "content": "Ada visited Paris.",
                          "raw_text": "Ada: I visited Paris."}]

    @staticmethod
    def response(content, finish_reason="stop"):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 64, "completion_tokens": 2048 if finish_reason == "length" else 8,
                      "prompt_tokens_details": {"cached_tokens": 0}}})

    def service(self, handler, *, retries=2):
        self.service_number += 1
        config = {"model": "fixture-qwen", "model_revision": f"fixture-v{self.service_number}",
                  "base_url": "http://fixture/v1", "retries": retries}
        service = ChatService(config, self.store, "task_agent")
        service.client.close()
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            return handler(request)
        service.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(service.client.close)
        return service, requests

    @staticmethod
    def agent(service):
        return TaskAgent(service, 2048, prompt_version="history_answer_evidence_v2")

    def test_parser_returns_only_answer_body_or_available_open_tag_suffix(self):
        cases = [
            ("analysis\n<answer> Paris </answer> trailing analysis", "Paris", "complete"),
            ("analysis\n<answer> Paris", "Paris", "open_tag_only"),
            ("analysis but no answer", "", "missing_open_tag"),
            ("Paris</answer>", "", "missing_open_tag"),
            ("<answ", "", "missing_open_tag"),
            ("<answer>", "", "open_tag_only"),
            ("<answer> </answer>", "", "complete"),
            ("", "", "missing_open_tag"),
            (" \n ", "", "missing_open_tag"),
            ("<answer>Paris</answer><answer>ignored</answer>", "Paris", "complete"),
        ]
        for raw, expected, status in cases:
            with self.subTest(raw=raw):
                metadata = {}
                self.assertEqual(extract_partial_tagged_answer(raw, metadata), expected)
                self.assertEqual(metadata, {"answer_status": status, "answer_tag_complete": status == "complete"})
        for non_text in (None, ["Paris"], {"content": "Paris"}, 123):
            with self.subTest(non_text=non_text), self.assertRaises(ValueError):
                extract_partial_tagged_answer(non_text)

    def test_stop_and_length_keep_raw_text_and_tag_completion_distinct(self):
        cases = [
            ("<answer>Paris</answer>", "Paris", True),
            ("<answer>Paris</answer>unfinished trailing analysis", "Paris", True),
            ("analysis before answer\n<answer>Par", "Par", False),
            ("only analysis", "", False),
            ("Paris</answer>", "", False),
            ("", "", False),
            (" \n ", "", False),
        ]
        for finish in ("stop", "length"):
            for raw, expected, complete in cases:
                with self.subTest(finish=finish, raw=raw):
                    service, requests = self.service(lambda _, raw=raw, finish=finish: self.response(raw, finish))
                    metadata = {}
                    with redirect_stdout(io.StringIO()):
                        answer, input_hash = self.agent(service).answer(self.example, self.evidence, metadata=metadata)
                    self.assertEqual(answer, expected)
                    self.assertEqual(input_hash, digest(task_messages(self.example, self.evidence, "history_answer_evidence_v2")))
                    self.assertEqual(metadata["protocol"], TASK_TAGGED_PARTIAL_OUTPUT_POLICY)
                    self.assertEqual(metadata["answer_parser"], "answer_tag_partial_v1")
                    self.assertEqual(metadata["raw_text"], raw)
                    self.assertEqual(metadata["finish_reason"], finish)
                    self.assertEqual(metadata["truncated"], finish == "length")
                    self.assertEqual(metadata["answer_tag_complete"], complete)
                    self.assertEqual(metadata["max_tokens"], 2048)
                    self.assertEqual([request["max_tokens"] for request in requests], [2048])
        self.assertFalse((self.store.root / "service_failures.jsonl").exists())

    def test_length_is_not_retried_or_expanded_and_survives_cache_reopen(self):
        raw = "some analysis\n<answer>Paris"
        service, requests = self.service(lambda _: self.response(raw, "length"), retries=2)
        agent = self.agent(service)
        metadata = {}
        with redirect_stdout(io.StringIO()), patch("agmem.services.time.sleep") as sleep:
            answer = agent.answer(self.example, self.evidence, metadata=metadata)
        sleep.assert_not_called()
        expected = deepcopy(metadata)
        metadata["usage"]["prompt_tokens_details"]["cached_tokens"] = -1
        self.store.close()
        self.store = Store(self.store.root)
        service.store = self.store
        cached = {}
        self.assertEqual(agent.answer(self.example, self.evidence, metadata=cached), answer)
        self.assertEqual(cached, expected)
        self.assertEqual([request["max_tokens"] for request in requests], [2048])
        self.assertFalse((self.store.root / "service_failures.jsonl").exists())

    def test_partial_policy_cache_is_separate_from_strict_and_legacy_policies(self):
        raw = "<answer>Paris</answer>"
        service, requests = self.service(lambda _: self.response(raw, "length"), retries=0)
        messages = task_messages(self.example, self.evidence, "history_answer_evidence_v2")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(service.complete(messages, 2048, answer_policy=TASK_TAGGED_PARTIAL_OUTPUT_POLICY), raw)
            with self.assertRaises(InfrastructureError):
                service.complete(messages, 2048, require_stop=True)
            legacy_metadata = {}
            self.assertEqual(service.complete(messages, 2048, metadata=legacy_metadata), raw)
        self.assertEqual(legacy_metadata["protocol"], TASK_OUTPUT_POLICY)
        self.assertEqual(len(requests), 3)
        self.assertEqual(service.complete(messages, 2048, answer_policy=TASK_TAGGED_PARTIAL_OUTPUT_POLICY), raw)
        self.assertEqual(len(requests), 3)
        self.assertEqual(len(self.store.completions.keys()), 2)

    def test_old_history_task_still_rejects_length_with_complete_tags(self):
        service, requests = self.service(lambda _: self.response("<answer>Paris</answer>", "length"))
        for version in ("history_answer_v1", "history_answer_query_v1"):
            with self.subTest(version=version), redirect_stdout(io.StringIO()), \
                    patch("agmem.services.time.sleep"), self.assertRaises(InfrastructureError):
                TaskAgent(service, 2048, prompt_version=version).answer(self.example, self.evidence)
        self.assertEqual(len(requests), 6)
        self.assertEqual(len(self.store.completions.keys()), 0)

    def test_nontext_http_and_other_finish_reasons_remain_infrastructure_failures(self):
        failures = [(None, "length"), (["Paris"], "stop"), (123, "stop"),
                    ("<answer>Paris</answer>", "content_filter"), ("answer", "tool_calls"), ("answer", None)]
        handlers = [lambda _, text=text, finish=finish: self.response(text, finish)
                    for text, finish in failures]
        handlers.extend([lambda _: httpx.Response(403), lambda _: httpx.Response(500)])
        for index, handler in enumerate(handlers):
            service, requests = self.service(handler)
            with self.subTest(index=index), redirect_stdout(io.StringIO()), \
                    patch("agmem.services.time.sleep"), self.assertRaises(InfrastructureError):
                self.agent(service).answer(self.example, self.evidence)
            self.assertEqual(len(requests), 3)
        self.assertEqual(len(self.store.completions.keys()), 0)

    def test_judge_receives_empty_partial_or_complete_answer_without_reader_format_penalty(self):
        cases = [("<answer>Paris</answer>unfinished tail", "Paris"),
                 ("analysis\n<answer>Par", "Par"), ("only analysis", ""), ("", ""), (" \n ", "")]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                service, _ = self.service(lambda _, raw=raw: self.response(raw, "length"))
                row = Trajectory(self.example.sample_id, 42, self.example.memory_id,
                    evidence=deepcopy(self.evidence), n_read_calls=1, termination_reason="stop", decisions=[
                        Decision([], Completion('{"ids":["r_0"],"stop":false}', token_ids=[10]),
                                 action={"ids": ["r_0"], "stop": False},
                                 tool_result={"records": deepcopy(self.evidence), "ok": True}),
                        Decision([], Completion('{"ids":[],"stop":true}', token_ids=[11]),
                                 action={"ids": [], "stop": True})])
                judge = SimpleNamespace(score=Mock(return_value=float(expected == "Paris")))
                scorer = RewardScorer({"formula": "judge_cost_v2", "format_penalty_mode": "per_id",
                                       "format_penalty": .5, "read_call_cost": .01},
                                      max_ids=2, judge=judge,
                                      task_output_policy=TASK_TAGGED_PARTIAL_OUTPUT_POLICY)
                with redirect_stdout(io.StringIO()):
                    row.prediction, row.task_input_hash = self.agent(service).answer(
                        self.example, row.evidence, metadata=row.task_completion)
                    scorer.score(self.example, row)
                judge.score.assert_called_once_with(self.example, expected)
                self.assertEqual(row.prediction, expected)
                self.assertFalse(row.format_failure)
                self.assertEqual(row.format_violation_count, 0)
                self.assertEqual(row.format_reward, 0)
                self.assertAlmostEqual(row.reward, .99 if expected == "Paris" else -1.01)
                self.assertTrue(row.task_completion["truncated"])
                self.assertFalse(trajectory_truncated(row))
                self.assertEqual(metrics([row])["pooled"]["task_truncation_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
