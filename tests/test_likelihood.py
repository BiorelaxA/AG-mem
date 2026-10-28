import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx

from agmem.core import Example, InfrastructureError
from agmem.likelihood import LIKELIHOOD_PROTOCOL, ReferenceLikelihoodScorer
from agmem.store import Store


class ScoringTokenizer:
    all_special_ids = [1, 2]

    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["truncation"] is False
        self.messages.append(messages)
        evidence = json.loads(messages[1]["content"])["retrieved_evidence"]
        return [1, 50, 51] + ([70 + evidence[0]["marker"], 80] if evidence else []) + [2]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"alpha beta": [101, 102], "gamma": [103], "A": [104], "special": [2]}[text]


def scoring_response(request):
    """Different context/target probabilities make masking mistakes visible."""
    body = json.loads(request.content)
    prompt = body["prompt"]
    assert request.url.path == "/v1/completions"
    assert body["echo"] is True and body["max_tokens"] == 0
    assert body["prompt_logprobs"] == 0 and body["return_token_ids"] is True
    assert body["temperature"] == 1.0 and body["add_special_tokens"] is False
    improvement = 3 if 71 in prompt else 2 if 72 in prompt else 0
    values = {101: -4 + improvement, 102: -6 + improvement, 103: -8 + improvement, 104: -5 + improvement}
    positions = [None] + [{str(token): {"logprob": values.get(token, -999), "rank": 7}}
                          for token in prompt[1:]]
    # The implementation must ignore this internal generation token and normal
    # generated-token logprobs, even though the engine ends with "length".
    return {"choices": [{"prompt_token_ids": prompt, "prompt_logprobs": positions,
                          "token_ids": [999], "finish_reason": "length",
                          "logprobs": {"token_logprobs": [-12345]}}]}


class LikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "store")
        self.service = {"model": "agmem-27b", "model_revision": "fixture-weights-v1",
                        "base_url": "http://127.0.0.1/v1", "retries": 0}
        self.example = Example("fixture/train/0", "fixture", "train", "memory", "Question?",
                               observation="Current date: 2026/10/27", gold_answers=["alpha beta"])
        self.tokenizer = ScoringTokenizer()

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def scorer(self, handler=None, config=None, service=None):
        scorer = ReferenceLikelihoodScorer(service or self.service, self.store,
                                           config or {"mode": "diagnostic"}, lambda: self.tokenizer)
        scorer.client.close()
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            return (handler(request) if handler is not None
                    else httpx.Response(200, json=scoring_response(request)))
        scorer.client = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(scorer.close)
        return scorer, requests

    def test_off_does_not_construct_client_tokenizer_or_read_references(self):
        tokenizer_factory = Mock(side_effect=AssertionError("Tokenizer loaded in off mode"))
        with patch("httpx.Client", side_effect=AssertionError("Client constructed in off mode")):
            scorer = ReferenceLikelihoodScorer({}, None, {"mode": "off"}, tokenizer_factory)
            result = scorer.score(None, None)
            scorer.close()
        tokenizer_factory.assert_not_called()
        self.assertEqual(result["mode"], "off")
        for field in ("d", "delta", "logprob", "baseline_logprob", "reference_token_count"):
            self.assertIsNone(result[field])
        self.assertEqual(result["references"], [])

    def test_only_gold_tokens_scored_and_baseline_is_reused(self):
        scorer, requests = self.scorer(config={"mode": "reward", "tau": 2.0})
        result = scorer.score(self.example, [{"marker": 1}])
        self.assertEqual(result["logprob"], -2.0)
        self.assertEqual(result["baseline_logprob"], -5.0)
        self.assertEqual(result["delta"], 3.0)
        self.assertAlmostEqual(result["d"], math.tanh(1.5))
        self.assertEqual(result["reference_token_count"], 2)
        self.assertEqual(len(requests), 2)
        # Both contexts use exactly the same gold continuation, including its
        # first token. Their prompt lengths intentionally differ.
        self.assertEqual(requests[0]["prompt"][-2:], requests[1]["prompt"][-2:])
        self.assertNotEqual(len(requests[0]["prompt"]), len(requests[1]["prompt"]))
        before = json.loads(self.tokenizer.messages[0][1]["content"])
        baseline = json.loads(self.tokenizer.messages[1][1]["content"])
        self.assertEqual({k: v for k, v in before.items() if k != "retrieved_evidence"},
                         {k: v for k, v in baseline.items() if k != "retrieved_evidence"})
        self.assertEqual(baseline["retrieved_evidence"], [])
        self.assertNotIn("gold_answers", before)
        self.assertEqual(scorer.score(self.example, [{"marker": 1}]), result)
        self.assertEqual(len(requests), 2)
        changed = scorer.score(self.example, [{"marker": 2}])
        self.assertEqual(changed["delta"], 2.0)
        self.assertEqual(len(requests), 3)  # New context, already-cached baseline.

    def test_empty_memory_needs_one_request_and_has_exact_zero_delta(self):
        scorer, requests = self.scorer()
        result = scorer.score(self.example, [])
        self.assertEqual(len(requests), 1)
        self.assertEqual(result["delta"], 0.0)
        self.assertEqual(result["d"], 0.0)

    def test_complete_gold_alternatives_are_averaged_after_token_normalization(self):
        self.example.gold_answers = ["alpha beta", "gamma"]
        scorer, requests = self.scorer()
        result = scorer.score(self.example, [{"marker": 1}])
        self.assertEqual(result["logprob"], (-2.0 + -5.0) / 2)
        self.assertEqual(result["baseline_logprob"], (-5.0 + -8.0) / 2)
        self.assertEqual(result["reference_token_count"], 3)
        self.assertEqual([row["token_count"] for row in result["references"]], [2, 1])
        self.assertEqual(len(requests), 4)
        self.assertEqual(scorer.provenance()["reference_aggregation"], "mean")

    def test_choice_scores_label_and_model_revision_separates_cache(self):
        self.example.answer_mode = "choice"
        self.example.options = {"A": "alpha beta", "B": "gamma"}
        self.example.gold_answers = ["A"]
        scorer, requests = self.scorer()
        result = scorer.score(self.example, [{"marker": 1}])
        self.assertEqual(result["reference_token_count"], 1)
        self.assertEqual(requests[0]["prompt"][-1], 104)
        other, other_requests = self.scorer(service={**self.service, "model_revision": "different-weights"})
        self.assertEqual(other.score(self.example, [{"marker": 1}]), result)
        self.assertEqual(len(other_requests), 2)

    def test_bad_responses_never_become_scores_or_cached_successes(self):
        def changed_ids(payload):
            payload["choices"][0]["prompt_token_ids"] = [999]
        def missing_first_answer(payload):
            payload["choices"][0]["prompt_logprobs"][-2] = None
        def wrong_token(payload):
            payload["choices"][0]["prompt_logprobs"][-2] = {"999": {"logprob": -1}}
        def truncated_positions(payload):
            payload["choices"][0]["prompt_logprobs"].pop()
        def nonfinite(payload):
            payload["choices"][0]["prompt_logprobs"][-1]["102"]["logprob"] = "NaN"
        def positive(payload):
            payload["choices"][0]["prompt_logprobs"][-1]["102"]["logprob"] = 0.5
        for mutate in (changed_ids, missing_first_answer, wrong_token, truncated_positions, nonfinite, positive):
            with self.subTest(case=mutate.__name__):
                def respond(request):
                    payload = scoring_response(request)
                    mutate(payload)
                    return httpx.Response(200, json=payload)
                scorer, requests = self.scorer(handler=respond)
                with self.assertRaises(InfrastructureError):
                    scorer.score(self.example, [{"marker": 1}])
                self.assertEqual(len(requests), 1)
                self.assertEqual(len(self.store.completions.keys()), 0)

    def test_invalid_response_is_retried_before_caching(self):
        attempts = []
        def respond(request):
            attempts.append(1)
            payload = scoring_response(request)
            if len(attempts) == 1:
                payload["choices"][0]["prompt_logprobs"][-2] = None
            return httpx.Response(200, json=payload)
        scorer, requests = self.scorer(handler=respond, service={**self.service, "retries": 1})
        with patch("agmem.likelihood.time.sleep"):
            result = scorer.score(self.example, [{"marker": 1}])
        self.assertEqual(len(requests), 3)  # Invalid + retried memory, then baseline.
        self.assertEqual(result["delta"], 3.0)
        scorer.score(self.example, [{"marker": 1}])
        self.assertEqual(len(requests), 3)

    def test_http_error_is_infrastructure_failure(self):
        scorer, _ = self.scorer(handler=lambda request: httpx.Response(503))
        with self.assertRaises(InfrastructureError):
            scorer.score(self.example, [])
        self.assertEqual(len(self.store.completions.keys()), 0)

    def test_special_targets_and_context_overflow_are_not_silently_truncated(self):
        scorer, requests = self.scorer()
        self.example.gold_answers = ["special"]
        with self.assertRaisesRegex(ValueError, "special/template"):
            scorer.score(self.example, [])
        self.assertEqual(requests, [])
        self.example.gold_answers = ["alpha beta"]
        small, requests = self.scorer(config={"mode": "diagnostic", "max_context_tokens": 6})
        with self.assertRaisesRegex(InfrastructureError, "context exceeds"):
            small.score(self.example, [])
        self.assertEqual(requests, [])

    def test_configuration_and_provenance(self):
        for config in ({"mode": "invalid"}, {"tau": 0}, {"tau": float("nan")},
                       {"tau": True}, {"reference_aggregation": "max"}, {"max_context_tokens": 0}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                ReferenceLikelihoodScorer({}, None, config)
        scorer, _ = self.scorer()
        metadata = scorer.provenance()
        self.assertEqual(metadata["protocol"], LIKELIHOOD_PROTOCOL)
        self.assertEqual(metadata["probability_temperature"], 1.0)
        self.assertEqual(metadata["model"]["model_revision"], "fixture-weights-v1")
        self.assertIn("task_prompt_sha256", metadata["task_input_contract"])


if __name__ == "__main__":
    unittest.main()
