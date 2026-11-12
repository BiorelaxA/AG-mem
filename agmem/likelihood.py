"""Optional, frozen Task Agent reference-answer likelihood scoring.

The scorer is independent of both the binary Judge and reward composition.  In
``off`` mode it does not load a tokenizer, create a client, or make requests.

The vLLM protocol used here is documented by the v0.22.1 completion API:
https://github.com/vllm-project/vllm/blob/v0.22.1/vllm/entrypoints/openai/completion/protocol.py
``prompt_logprobs=0`` returns the observed token's probability without a large
top-k vocabulary response. ``echo=True, max_tokens=0`` requests prompt scoring;
vLLM internally permits one generation token, which is never part of our score.
"""
from __future__ import annotations

from concurrent.futures import Future
from ipaddress import ip_address
import json
import math
import os
from threading import Lock
import time
from urllib.parse import urlsplit

from .core import InfrastructureError, append_json, canonical, digest
from .policy import chat_token_ids
from .prompts import task_input_contract, task_messages, validate_task_prompt_version


LIKELIHOOD_PROTOCOL = "task_reference_mean_logprob_v1"
PREFIX_LIKELIHOOD_PROTOCOL = "task_reference_prefix_mean_logprob_v1"
ANSWER_CONTINUATION_PREFIX = "<answer>\n"


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class ReferenceLikelihoodScorer:
    """Score complete gold alternatives under the actual Task Agent prompt.

    Each gold is encoded once without special tokens and reused for memory and
    no-memory contexts. Multiple complete alternatives receive equal weight:
    first average tokens within each answer, then average across answers. This
    fixed rule never chooses a different gold to favor a particular rollout.

    ``tokenizer_factory`` is a zero-argument callable, invoked lazily only in an
    active mode. The reward composer alone decides whether ``d`` affects reward;
    ``diagnostic`` and ``reward`` perform identical scoring.
    """

    def __init__(self, service_config, store, config=None, tokenizer_factory=None, *,
                 prompt_version="legacy_v1"):
        self.service_config = dict(service_config)
        self.store = store
        self.config = dict(config or {})
        self.prompt_version = validate_task_prompt_version(prompt_version)
        self.task_contract = task_input_contract(self.prompt_version)
        self.assistant_prefix = (ANSWER_CONTINUATION_PREFIX
                                 if self.task_contract.get("answer_parser") in {
                                     "answer_tag_v1", "answer_tag_partial_v1"} else "")
        self.mode = self.config.get("mode", "off")
        if self.mode not in {"off", "diagnostic", "reward"}:
            raise ValueError("Likelihood mode must be off, diagnostic or reward")
        self.tau = self.config.get("tau", 1.0)
        if not _finite_number(self.tau) or self.tau <= 0:
            raise ValueError("Likelihood tau must be finite and positive")
        self.reference_aggregation = self.config.get("reference_aggregation", "mean")
        if self.reference_aggregation != "mean":
            raise ValueError("Likelihood reference_aggregation must be mean")
        self.max_context_tokens = self.config.get("max_context_tokens", 65536)
        if type(self.max_context_tokens) is not int or self.max_context_tokens < 2:
            raise ValueError("Likelihood max_context_tokens must be an integer >= 2")
        self.tokenizer_factory = tokenizer_factory
        self._tokenizer = None
        self._tokenizer_lock = Lock()
        self._lock = Lock()
        self._inflight = {}
        self.client = None
        self.identity = {key: service_config.get(key) for key in ("model", "model_revision", "base_url")}
        if self.mode == "off":
            return
        if any(not isinstance(value, str) or not value for value in self.identity.values()):
            raise ValueError("Likelihood service needs model, model_revision and base_url")
        if self.identity["model_revision"].startswith("REQUIRED"):
            raise ValueError("Likelihood service needs a fixed local model revision")
        if not callable(tokenizer_factory):
            raise ValueError("Active likelihood scoring requires a tokenizer_factory")
        retries = self.service_config.get("retries", 2)
        if type(retries) is not int or retries < 0:
            raise ValueError("Likelihood retries must be a nonnegative integer")
        import httpx
        endpoint = self.service_config.get("request_base_url", self.identity["base_url"])
        host = (urlsplit(endpoint).hostname or "").rstrip(".").lower()
        try:
            loopback = ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        self.client = httpx.Client(timeout=self.service_config.get("timeout_seconds", 600), trust_env=not loopback)

    def provenance(self):
        result = {"protocol": LIKELIHOOD_PROTOCOL, "mode": self.mode, "model": self.identity,
                "task_input_contract": self.task_contract, "tau": self.tau,
                "reference_aggregation": self.reference_aggregation,
                "target_tokens": "gold continuation only; no template or special tokens",
                "probability_temperature": 1.0, "baseline": "same task with empty retrieved_evidence",
                "max_context_tokens": self.max_context_tokens}
        # Keep legacy provenance exactly stable for existing training resumes.
        if self.assistant_prefix:
            result.update(assistant_prefix=self.assistant_prefix,
                          target_encoding="fixed_gold_ids_after_answer_prefix_v1")
        return result

    def prefix_provenance(self):
        """Describe the stepwise scoring protocol independently of terminal reward."""
        return {**self.provenance(), "protocol": PREFIX_LIKELIHOOD_PROTOCOL,
                "baseline": "caller-supplied ordered evidence prefixes, including empty E_0",
                "assistant_prefix": self.assistant_prefix,
                "target_encoding": "fixed_gold_ids_after_answer_prefix_v1" if self.assistant_prefix
                                   else "fixed_gold_ids_after_chat_prefix_v1",
                "closing_answer_tag_scored": False}

    def close(self):
        if self.client is not None:
            self.client.close()

    def _get_tokenizer(self):
        with self._tokenizer_lock:
            if self._tokenizer is None:
                self._tokenizer = self.tokenizer_factory()
            return self._tokenizer

    def score(self, example, evidence):
        result = {"protocol": LIKELIHOOD_PROTOCOL, "mode": self.mode,
                  "logprob": None, "baseline_logprob": None, "delta": None, "d": None,
                  "reference_token_count": None, "references": []}
        if self.mode == "off":
            return result
        prefixes = self.score_prefixes(example, [evidence, []])
        for row in prefixes["references"]:
            with_memory, baseline = row["logprobs"]
            result["references"].append({"index": row["index"], "token_count": row["token_count"],
                                         "target_token_sha256": row["target_token_sha256"],
                                         "logprob": with_memory, "baseline_logprob": baseline,
                                         "delta": with_memory - baseline})
        result["logprob"], result["baseline_logprob"] = prefixes["prefix_logprobs"]
        result["delta"] = result["logprob"] - result["baseline_logprob"]
        result["d"] = math.tanh(result["delta"] / self.tau)
        result["reference_token_count"] = prefixes["reference_token_count"]
        return result

    def score_prefixes(self, example, evidence_prefixes):
        """Return f(E) for each supplied evidence prefix, without composing reward.

        Every complete gold alternative is encoded once and reused unchanged
        across all prefixes. For answer-tag prompts, the fixed opening tag and
        newline are context, not targets. Only gold tokens are scored: neither
        the opening/closing tags nor any internal server generation contribute.
        A closing tag is unnecessary because suffixes cannot affect earlier
        token probabilities. The caller controls prefix order and computes
        successive differences; this method never sees future evidence when
        evaluating an earlier prefix.
        """
        if self.mode == "off":
            raise ValueError("Prefix likelihood scoring requires an active likelihood mode")
        if (not isinstance(evidence_prefixes, (list, tuple)) or not evidence_prefixes
                or any(not isinstance(prefix, list) for prefix in evidence_prefixes)):
            raise ValueError("Evidence prefixes must be a nonempty sequence of evidence lists")
        golds = example.gold_answers
        if (not isinstance(golds, list) or not golds
                or any(not isinstance(gold, str) or not gold.strip() for gold in golds)):
            raise ValueError("Likelihood scoring requires complete, nonempty gold answers")
        if example.answer_mode == "choice" and any(gold not in example.options for gold in golds):
            raise ValueError("Choice likelihood targets must be valid option labels")
        tokenizer = self._get_tokenizer()
        # Fast tokenizers may mutate backend padding/truncation settings during
        # encoding. Scoring workers share this tokenizer, so serialize encoding
        # only; independent HTTP requests must remain concurrent.
        with self._tokenizer_lock:
            continuation_ids = (tokenizer.encode(self.assistant_prefix, add_special_tokens=False)
                                if self.assistant_prefix else [])
            if (not isinstance(continuation_ids, list)
                    or (self.assistant_prefix and not continuation_ids)
                    or any(type(token) is not int or token < 0 for token in continuation_ids)):
                raise ValueError("Assistant continuation tokenization must produce integer IDs")
            contexts = [chat_token_ids(tokenizer, task_messages(example, evidence, self.prompt_version))
                        + continuation_ids for evidence in evidence_prefixes]
            special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
            targets = []
            for gold in golds:
                target_ids = tokenizer.encode(gold, add_special_tokens=False)
                if (not isinstance(target_ids, list) or not target_ids
                        or any(type(token) is not int or token < 0 for token in target_ids)):
                    raise ValueError("Gold answer tokenization must produce nonempty integer IDs")
                if special_ids.intersection(target_ids):
                    raise ValueError("Gold answer contains a special/template token")
                targets.append(target_ids)
        result = {"protocol": PREFIX_LIKELIHOOD_PROTOCOL,
                  "task_input_contract": self.task_contract,
                  "prefix_logprobs": [], "reference_token_count": sum(map(len, targets)),
                  "references": []}
        for index, target_ids in enumerate(targets):
            logprobs = [self._mean_logprob(context_ids, target_ids) for context_ids in contexts]
            result["references"].append({"index": index, "token_count": len(target_ids),
                                         "target_token_sha256": digest(target_ids),
                                         "logprobs": logprobs})
        result["prefix_logprobs"] = [math.fsum(row["logprobs"][index] for row in result["references"])
                                     / len(targets) for index in range(len(contexts))]
        return result

    def _mean_logprob(self, context_ids, target_ids):
        prompt_ids = context_ids + target_ids
        # echo/max_tokens=0 still permits one internal vLLM generation token.
        if len(prompt_ids) + 1 > self.max_context_tokens:
            raise InfrastructureError("Reference likelihood context exceeds the scoring service limit; no score assigned")
        request = {"model": self.identity["model"], "prompt": prompt_ids,
                   "echo": True, "max_tokens": 0, "prompt_logprobs": 0,
                   "return_token_ids": True, "add_special_tokens": False,
                   "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                   "min_p": 0.0, "repetition_penalty": 1.0,
                   "presence_penalty": 0.0, "frequency_penalty": 0.0,
                   "seed": 42, "stream": False}
        target_start = len(context_ids)
        key = digest({"role": "reference_likelihood", "identity": self.identity,
                      "protocol": LIKELIHOOD_PROTOCOL, "request": request, "target_start": target_start})
        with self._lock:
            cached = self.store.completion(key)
            if cached is not None:
                try:
                    return self._validate_cached(json.loads(cached), target_ids)
                except (TypeError, ValueError, KeyError) as exc:
                    raise InfrastructureError("Invalid reference likelihood cache; no score assigned") from exc
            future = self._inflight.get(key)
            owner = future is None
            if owner:
                future = self._inflight[key] = Future()
        if not owner:
            return future.result()
        try:
            score = self._request(request, target_start, key)
            future.set_result(score)
            return score
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    @staticmethod
    def _validate_cached(record, target_ids):
        if record["protocol"] != LIKELIHOOD_PROTOCOL or record["target_token_ids"] != target_ids:
            raise ValueError("Likelihood cache protocol/targets differ")
        values = record["token_logprobs"]
        if not isinstance(values, list) or len(values) != len(target_ids):
            raise ValueError("Likelihood cache token count differs")
        if any(not _finite_number(value) or value > 1e-6 for value in values):
            raise ValueError("Invalid likelihood cache probability")
        return math.fsum(values) / len(values)

    @staticmethod
    def _extract(payload, prompt_ids, target_start):
        choices = payload["choices"]
        if len(choices) != 1:
            raise ValueError("Expected one reference likelihood result")
        choice = choices[0]
        # Validate every ID, not only the length, to detect BOS injection,
        # truncation or retokenization by an incompatible server.
        if choice.get("prompt_token_ids") != prompt_ids:
            raise ValueError("Scoring service changed prompt token IDs")
        positions = choice.get("prompt_logprobs")
        if not isinstance(positions, list) or len(positions) != len(prompt_ids):
            raise ValueError("Scoring service omitted prompt logprobs")
        values = []
        # positions[t] is log p(prompt_ids[t] | prompt_ids[:t]); the first
        # answer token is at target_start, not target_start-1 or +1.
        for index in range(target_start, len(prompt_ids)):
            row = positions[index]
            if not isinstance(row, dict):
                raise ValueError("Missing gold-token probability")
            entry = row.get(str(prompt_ids[index]), row.get(prompt_ids[index]))
            if not isinstance(entry, dict):
                raise ValueError("Gold token absent from prompt_logprobs")
            value = entry.get("logprob")
            if not _finite_number(value) or value > 1e-6:
                raise ValueError("Nonfinite or positive gold-token logprob")
            values.append(float(value))
        return {"protocol": LIKELIHOOD_PROTOCOL, "target_token_ids": prompt_ids[target_start:],
                "token_logprobs": values}

    def _request(self, request, target_start, key):
        endpoint = self.service_config.get("request_base_url", self.identity["base_url"])
        headers = {}
        env_name = self.service_config.get("api_key_env")
        if env_name and os.environ.get(env_name):
            headers["Authorization"] = "Bearer " + os.environ[env_name]
        last_error = None
        retries = self.service_config.get("retries", 2)
        for attempt in range(retries + 1):
            response = None
            try:
                response = self.client.post(endpoint.rstrip("/") + "/completions", json=request, headers=headers)
                response.raise_for_status()
                record = self._extract(response.json(), request["prompt"], target_start)
                score = self._validate_cached(record, request["prompt"][target_start:])
                # Only a fully validated set of answer-token probabilities is cached.
                self.store.put_completion(key, canonical(record))
                return score
            except Exception as exc:
                last_error = exc
                failure = {"timestamp": time.time(), "role": "reference_likelihood", "request_key": key,
                           "attempt": attempt + 1, "status_code": response.status_code if response is not None else None,
                           "error_type": type(exc).__name__, "error": str(exc)}
                with self._lock:
                    append_json(self.store.root / "service_failures.jsonl", failure)
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 8))
        raise InfrastructureError(f"Reference likelihood service failed ({type(last_error).__name__}); no score assigned") from last_error
