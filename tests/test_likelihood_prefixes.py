import json
from pathlib import Path
import tempfile
import unittest

import httpx

from agmem.core import Example, InfrastructureError, canonical
from agmem.likelihood import (ANSWER_CONTINUATION_PREFIX, LIKELIHOOD_PROTOCOL,
                              PREFIX_LIKELIHOOD_PROTOCOL, ReferenceLikelihoodScorer)
from agmem.prompts import task_input_contract, task_messages
from agmem.store import Store
from tests.test_likelihood import ScoringTokenizer, scoring_response


class HistoryScoringTokenizer(ScoringTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        if len(messages) != 1:
            return super().apply_chat_template(messages, **kwargs)
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["truncation"] is False
        self.messages.append(messages)
        content = messages[0]["content"]
        marker = [72, 80] if "EVIDENCE_B" in content else [71, 80] if "EVIDENCE_A" in content else []
        return [1, 50, 51] + marker + [2]

    def encode(self, text, *, add_special_tokens):
        if text == ANSWER_CONTINUATION_PREFIX:
            assert add_special_tokens is False
            return [110, 111]
        return super().encode(text, add_special_tokens=add_special_tokens)


class CompleteHistoryScoringTokenizer(HistoryScoringTokenizer):
    """Bind fixture token IDs to all public input, not just evidence markers."""

    def apply_chat_template(self, messages, **kwargs):
        ids = super().apply_chat_template(messages, **kwargs)
        return ids[:-1] + [1000 + ord(char) for char in canonical(messages)] + ids[-1:]


class PrefixLikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "store")
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.store.close)
        self.service = {"model": "agmem-27b", "model_revision": "fixture-weights-v1",
                        "base_url": "http://127.0.0.1/v1", "retries": 0}
        self.example = Example("fixture/train/0", "fixture", "train", "memory", "Question?",
                               observation="UNUSED_OBSERVATION", gold_answers=["alpha beta"])
        self.tokenizer = HistoryScoringTokenizer()
        self.a = {"id": "a", "raw_text": "EVIDENCE_A", "time": "2026-01-01"}
        self.b = {"id": "b", "raw_text": "EVIDENCE_B", "time": "2026-01-02"}

    def scorer(self, version="history_answer_query_v1", handler=None):
        scorer = ReferenceLikelihoodScorer(self.service, self.store, {"mode": "diagnostic"},
                                           lambda: self.tokenizer, prompt_version=version)
        scorer.client.close()
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            return handler(request) if handler else httpx.Response(200, json=scoring_response(request))
        scorer.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(scorer.close)
        return scorer, requests

    def test_prefix_gold_only_scores_repeated_evidence_and_no_future_leakage(self):
        scorer, requests = self.scorer()
        result = scorer.score_prefixes(self.example, [[], [self.a], [self.a, self.b], [self.a, self.b]])
        self.assertEqual(result["protocol"], PREFIX_LIKELIHOOD_PROTOCOL)
        self.assertEqual(result["prefix_logprobs"], [-5.0, -2.0, -3.0, -3.0])
        self.assertEqual(result["reference_token_count"], 2)
        self.assertEqual(result["references"][0]["logprobs"], result["prefix_logprobs"])
        self.assertEqual(len(requests), 3)
        for request in requests:
            self.assertEqual(request["prompt"][-4:], [110, 111, 101, 102])
        messages = [value[0]["content"] for value in self.tokenizer.messages]
        self.assertNotIn("EVIDENCE_A", messages[0])
        self.assertNotIn("EVIDENCE_B", messages[1])
        self.assertIn("EVIDENCE_B", messages[2])
        self.assertTrue(all("UNUSED_OBSERVATION" not in message for message in messages))
        self.assertTrue(all("alpha beta" not in message for message in messages))
        before = len(requests)
        self.assertEqual(scorer.score(self.example, [self.a])["delta"], 3.0)
        self.assertEqual(len(requests), before)

    def test_equal_weight_alternatives_and_prefix_target_metadata(self):
        self.example.gold_answers = ["alpha beta", "gamma"]
        scorer, requests = self.scorer()
        result = scorer.score_prefixes(self.example, [[], [self.a]])
        self.assertEqual(result["prefix_logprobs"], [-6.5, -3.5])
        self.assertEqual(result["reference_token_count"], 3)
        self.assertEqual([row["token_count"] for row in result["references"]], [2, 1])
        self.assertEqual(len(requests), 4)
        self.assertEqual(requests[0]["prompt"][-2:], requests[1]["prompt"][-2:])
        self.assertEqual(requests[2]["prompt"][-1:], requests[3]["prompt"][-1:])

    def test_tagged_prompt_versions_use_real_task_contract(self):
        for version in ("history_answer_v1", "history_answer_query_v1", "history_answer_evidence_v2"):
            with self.subTest(version=version):
                scorer, _ = self.scorer(version)
                scorer.score_prefixes(self.example, [[]])
                metadata = scorer.prefix_provenance()
                self.assertEqual(metadata["task_input_contract"], task_input_contract(version))
                self.assertEqual(metadata["assistant_prefix"], "<answer>\n")
                self.assertFalse(metadata["closing_answer_tag_scored"])
                self.assertEqual(metadata["probability_temperature"], 1.0)
                self.assertEqual(metadata["target_encoding"], "fixed_gold_ids_after_answer_prefix_v1")
                content = self.tokenizer.messages[-1][0]["content"]
                self.assertEqual("UNUSED_OBSERVATION" in content, version != "history_answer_query_v1")

    def test_evidence_prompt_prefixes_match_generation_inputs_without_future_or_gold_leakage(self):
        version = "history_answer_evidence_v2"
        self.tokenizer = CompleteHistoryScoringTokenizer()
        self.example.observation = "PUBLIC_OBSERVATION"
        self.example.options = {"A": "PUBLIC_OPTION"}
        a, b = {**self.a, "content": "SUMMARY_A"}, {**self.b, "content": "SUMMARY_B"}
        prefixes = [[], [a], [a, b], [a, b]]
        scorer, requests = self.scorer(version)
        result = scorer.score_prefixes(self.example, prefixes)
        self.assertEqual(result["prefix_logprobs"], [-5.0, -2.0, -3.0, -3.0])
        self.assertEqual(result["reference_token_count"], 2)
        self.assertEqual(len(requests), 3)
        self.assertEqual(self.tokenizer.messages,
                         [task_messages(self.example, prefix, version) for prefix in prefixes])
        texts = [messages[0]["content"] for messages in self.tokenizer.messages]
        for text in texts:
            self.assertIn("PUBLIC_OBSERVATION", text)
            self.assertIn("PUBLIC_OPTION", text)
            self.assertNotIn("alpha beta", text)
        self.assertNotIn("SUMMARY_A", texts[0])
        self.assertNotIn("EVIDENCE_A", texts[0])
        self.assertIn("SUMMARY_A", texts[1])
        self.assertIn("EVIDENCE_A", texts[1])
        self.assertNotIn("SUMMARY_B", texts[1])
        self.assertNotIn("EVIDENCE_B", texts[1])
        self.assertIn("SUMMARY_B", texts[2])
        self.assertIn("EVIDENCE_B", texts[2])
        for request in requests:
            # Tag tokens are context; fixture assigns them -999 so scoring any
            # prefix/tag positions would visibly corrupt the expected mean.
            self.assertEqual(request["prompt"][-4:], [110, 111, 101, 102])
        metadata = scorer.prefix_provenance()
        self.assertEqual(metadata["task_input_contract"]["answer_parser"], "answer_tag_partial_v1")
        self.assertEqual(metadata["assistant_prefix"], ANSWER_CONTINUATION_PREFIX)
        self.assertFalse(metadata["closing_answer_tag_scored"])

    def test_evidence_likelihood_cache_changes_with_summary_observation_and_prompt_version(self):
        self.tokenizer = CompleteHistoryScoringTokenizer()
        scorer, requests = self.scorer("history_answer_evidence_v2")
        evidence = [{**self.a, "content": "SUMMARY_A"}]
        initial = scorer.score_prefixes(self.example, [evidence])
        self.assertEqual(scorer.score_prefixes(self.example, [evidence]), initial)
        self.assertEqual(len(requests), 1)
        evidence[0]["content"] = "UPDATED_SUMMARY_A"
        scorer.score_prefixes(self.example, [evidence])
        self.assertEqual(len(requests), 2)
        self.example.observation = "UPDATED_OBSERVATION"
        scorer.score_prefixes(self.example, [evidence])
        self.assertEqual(len(requests), 3)
        previous, previous_requests = self.scorer("history_answer_query_v1")
        previous.score_prefixes(self.example, [evidence])
        self.assertEqual(len(previous_requests), 1)
        scorer.score_prefixes(self.example, [evidence])
        self.assertEqual(len(requests), 3)
        self.assertNotEqual(previous.provenance()["task_input_contract"],
                            scorer.provenance()["task_input_contract"])

    def test_legacy_provenance_is_unchanged_and_legacy_prefix_has_no_answer_tag(self):
        scorer, requests = self.scorer("legacy_v1")
        result = scorer.score_prefixes(self.example, [[], [{"marker": 1}]])
        self.assertEqual(result["prefix_logprobs"], [-5.0, -2.0])
        self.assertTrue(all(110 not in request["prompt"] for request in requests))
        self.assertEqual(scorer.provenance(), {
            "protocol": LIKELIHOOD_PROTOCOL, "mode": "diagnostic", "model": {
                "model": "agmem-27b", "model_revision": "fixture-weights-v1",
                "base_url": "http://127.0.0.1/v1"},
            "task_input_contract": task_input_contract(), "tau": 1.0,
            "reference_aggregation": "mean",
            "target_tokens": "gold continuation only; no template or special tokens",
            "probability_temperature": 1.0, "baseline": "same task with empty retrieved_evidence",
            "max_context_tokens": 65536})

    def test_missing_gold_logprob_is_failure_and_never_zero_process_score(self):
        def incomplete(request):
            payload = scoring_response(request)
            payload["choices"][0]["prompt_logprobs"][-2] = None
            return httpx.Response(200, json=payload)
        scorer, _ = self.scorer(handler=incomplete)
        with self.assertRaises(InfrastructureError):
            scorer.score_prefixes(self.example, [[]])
        self.assertEqual(len(self.store.completions.keys()), 0)

    def test_prefix_validation_and_off_mode_cannot_silently_score(self):
        scorer, requests = self.scorer()
        for bad in (None, [], [None], [{"raw_text": "bad nesting"}]):
            with self.subTest(prefixes=bad), self.assertRaises(ValueError):
                scorer.score_prefixes(self.example, bad)
        self.assertEqual(requests, [])
        off = ReferenceLikelihoodScorer({}, None)
        with self.assertRaisesRegex(ValueError, "active likelihood mode"):
            off.score_prefixes(self.example, [[]])


if __name__ == "__main__":
    unittest.main()
