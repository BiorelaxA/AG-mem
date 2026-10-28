from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import httpx

from agmem.core import Example, InfrastructureError
from agmem.judge_prompts import LOCOMO_REFINED, LONGMEMEVAL_TEMPLATES
from agmem.judges import JudgeScorer, parse_longmemeval, parse_refined, prompt_for
from agmem.prompts import reader_messages, task_messages
from agmem.services import ChatService
from agmem.store import Store


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "store")
        self.config = {"model": "agmem-27b", "model_revision": "fixture-27b",
                       "base_url": "http://127.0.0.1:8000/v1", "retries": 0}
        self.example = Example("locomo/train/0", "locomo", "train", "m", "When did Alice move?",
                               gold_answers=["May 2023"])

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def scorer(self, responses, *, config=None, judge_config=None):
        requests = []
        responses = iter(responses)
        def respond(request):
            requests.append(json.loads(request.content))
            item = next(responses)
            if isinstance(item, httpx.Response):
                return item
            return httpx.Response(200, json={"choices": [
                {"finish_reason": "stop", "message": {"content": item}}]})
        judge = JudgeScorer(config or self.config, self.store, judge_config)
        judge.service.client.close()
        judge.service.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(judge.close)
        return judge, requests

    def longmemeval(self, kind="multi-session", qid="question-1"):
        example = Example("split_longmemeval_s/train/0", "split_longmemeval_s", "train", "m",
                          "What is the answer?", gold_answers=["Reference"])
        example.judge_metadata = {"question_type": kind, "question_id": qid}
        return example

    def test_refined_payload_is_original_template_and_cached(self):
        judge, requests = self.scorer(['{"label": "CORRECT"}'])
        self.assertEqual(judge.score(self.example, "In May 2023."), 1.0)
        self.assertEqual(judge.score(self.example, "In May 2023."), 1.0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["messages"], [{"role": "user", "content": LOCOMO_REFINED.format(
            question=self.example.query, gold_answer="May 2023", generated_answer="In May 2023.")}])
        self.assertEqual(requests[0]["temperature"], 0)
        self.assertFalse(requests[0]["chat_template_kwargs"]["enable_thinking"])

    def test_longmemeval_all_types_and_abstention_use_upstream_branches(self):
        kinds = [name for name in LONGMEMEVAL_TEMPLATES if name != "abstention"]
        for kind in kinds:
            for abstention in (False, True):
                with self.subTest(kind=kind, abstention=abstention):
                    qid = "q_abs" if abstention else "q"
                    example = self.longmemeval(kind, qid)
                    prompt, parser = prompt_for(example, "gold", "response", "longmemeval_official")
                    branch = "abstention" if abstention else kind
                    self.assertEqual(prompt, LONGMEMEVAL_TEMPLATES[branch].format(example.query, "gold", "response"))
                    self.assertIs(parser, parse_longmemeval)

    def test_longmemeval_metadata_is_required_instead_of_guessing(self):
        example = self.longmemeval()
        for metadata in ({}, {"question_type": "multi-session"},
                         {"question_type": "unknown", "question_id": "q"}):
            example.judge_metadata = metadata
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                prompt_for(example, "gold", "response", "longmemeval_official")

    def test_strict_parsers_reject_ambiguous_or_explanatory_outputs(self):
        self.assertEqual(parse_refined('{"label":"WRONG"}'), 0)
        self.assertEqual(parse_refined('```json\n{"label":"CORRECT"}\n```'), 1)
        self.assertEqual(parse_longmemeval(" YES \n"), 1)
        for text in ("CORRECT", '{"label":"CORRECT","reason":"fine"}',
                     '{"label":"CORRECT","label":"WRONG"}',
                     '{"label":"INCORRECT"}', '{"label":true}',
                     '{"label":"WRONG"} {"label":"CORRECT"}',
                     'Explanation: {"label":"CORRECT"}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_refined(text)
        for text in ("yesterday", "yes and no", "yes.", "The answer is yes", ""):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_longmemeval(text)

    def test_invalid_output_retried_before_caching(self):
        judge, requests = self.scorer(['{"label":"MAYBE"}', '{"label":"CORRECT"}'],
                                     config={**self.config, "retries": 1})
        with patch("agmem.services.time.sleep"):
            self.assertEqual(judge.score(self.example, "May 2023"), 1)
        self.assertEqual(judge.score(self.example, "May 2023"), 1)
        self.assertEqual(len(requests), 2)
        self.assertEqual(len(self.store.completions.keys()), 1)
        failures = [json.loads(line) for line in (self.store.root / "service_failures.jsonl").read_text().splitlines()]
        self.assertEqual(failures[0]["role"], "judge")

    def test_judge_failure_is_not_a_zero_reward_and_is_not_cached(self):
        for response in ("yes", httpx.Response(503), httpx.Response(200, json={"choices": [
            {"finish_reason": "length", "message": {"content": '{"label":"CORRECT"}'}}]})):
            judge, requests = self.scorer([response])
            with self.subTest(response=response), self.assertRaises(InfrastructureError):
                judge.score(self.example, "candidate")
            self.assertEqual(len(requests), 1)
            self.assertEqual(len(self.store.completions.keys()), 0)

    def test_complete_reference_alternatives_are_scored_separately(self):
        self.example.gold_answers = ["A, B and C", "X, Y and Z"]
        judge, requests = self.scorer(['{"label":"WRONG"}', '{"label":"CORRECT"}'])
        self.assertEqual(judge.score(self.example, "X, Y and Z"), 1)
        self.assertEqual(len(requests), 2)
        self.assertIn("Gold answer: A, B and C\n", requests[0]["messages"][0]["content"])
        self.assertIn("Gold answer: X, Y and Z\n", requests[1]["messages"][0]["content"])

    def test_model_and_parser_protocol_identity_separate_cache(self):
        first, first_requests = self.scorer(['{"label":"CORRECT"}'])
        second, second_requests = self.scorer(['{"label":"WRONG"}'],
                                            config={**self.config, "model_revision": "new-weights"})
        third, third_requests = self.scorer(['{"label":"WRONG"}'], judge_config={"max_output_tokens": 64})
        self.assertEqual(first.score(self.example, "candidate"), 1)
        self.assertEqual(second.score(self.example, "candidate"), 0)
        self.assertEqual(third.score(self.example, "candidate"), 0)
        self.assertEqual([len(first_requests), len(second_requests), len(third_requests)], [1, 1, 1])
        self.assertNotEqual(first.cache_scope, second.cache_scope)
        self.assertNotEqual(first.cache_scope, third.cache_scope)
        self.assertEqual(first.provenance()["parser"], "strict_binary_v1")

    def test_judge_metadata_and_gold_do_not_enter_agent_messages(self):
        example = self.longmemeval("knowledge-update", "SECRET_ABSTENTION_abs")
        example.gold_answers = ["SECRET_GOLD"]
        messages = task_messages(example, []) + reader_messages(example, [], [])
        serialized = json.dumps(messages)
        self.assertNotIn("SECRET_GOLD", serialized)
        self.assertNotIn("SECRET_ABSTENTION", serialized)
        self.assertNotIn("knowledge-update", serialized)
        self.assertNotIn("judge_metadata", example.public_task())

    def test_task_and_judge_roles_cannot_share_cached_result(self):
        judge, requests = self.scorer(['{"label":"WRONG"}'])
        prompt, _ = prompt_for(self.example, "May 2023", "candidate", "locomo_refined")
        task = ChatService(self.config, self.store, "task_agent")
        task.client.close()
        task.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"label":"CORRECT"}'}}]})))
        self.addCleanup(task.client.close)
        task.complete([{"role": "user", "content": prompt}], judge.max_tokens,
                      seed=judge.seed, cache_scope=judge.cache_scope)
        self.assertEqual(judge.score(self.example, "candidate"), 0)
        self.assertEqual(len(requests), 1)

    def test_missing_protocol_and_reference_fail_before_http(self):
        judge, requests = self.scorer([])
        self.example.source = "unknown"
        with self.assertRaisesRegex(ValueError, "No Judge protocol"):
            judge.score(self.example, "prediction")
        self.example.source = "locomo"
        self.example.gold_answers = []
        with self.assertRaisesRegex(ValueError, "reference"):
            judge.score(self.example, "prediction")
        self.assertEqual(requests, [])

    def test_preflight_validation_requires_no_service_calls(self):
        judge, requests = self.scorer([])
        self.assertIsNone(judge.validate_example(self.example))
        self.assertIsNone(judge.validate_example(self.longmemeval()))
        self.assertIsNone(judge.validate_example(self.longmemeval(qid="question_abs")))
        example = self.longmemeval()
        for metadata in ({}, None, {"question_id": "q", "question_type": "unsupported"}):
            example.judge_metadata = metadata
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                judge.validate_example(example)
        self.assertEqual(requests, [])


if __name__ == "__main__":
    unittest.main()
