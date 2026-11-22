"""Strict remote binary judging with the frozen training prompt and resumable calls."""
from copy import deepcopy
from datetime import datetime, timezone
import errno
import json
import math
import os
from pathlib import Path
from threading import Lock
from time import sleep

from .core import InfrastructureError, atomic_json, canonical, digest
from .judges import DEFAULT_PROTOCOLS, PARSER_VERSION, JudgeScorer, load_judge_templates, prompt_for
from .remote_task_runner import _complete, validate_completion
from .services import _redact_secret, _resolved_chat_config


def _recoverable_network_failure(failure):
    """Classify the transport's safe terminal metadata, never exception text."""
    if (failure.get("reason") != "response_or_transport_error"
            or failure.get("will_retry") is not False
            or failure.get("error_type") == "HTTPError"
            or (failure.get("status_code") is not None and failure["status_code"] >= 300)):
        return False
    kind = failure.get("transport_error_type") or failure.get("error_type")
    if kind in {"SSLCertVerificationError", "CertificateError"}:
        return False
    if kind in {"gaierror", "TimeoutError", "timeout", "ConnectionError", "ConnectionResetError",
                "ConnectionRefusedError", "ConnectionAbortedError", "BrokenPipeError",
                "RemoteDisconnected", "IncompleteRead", "SSLError", "SSLEOFError",
                "SSLZeroReturnError", "SSLWantReadError", "SSLWantWriteError"}:
        return True
    return kind == "OSError" and failure.get("transport_errno") in {
        errno.ECONNRESET, errno.ECONNREFUSED, errno.ECONNABORTED, errno.ETIMEDOUT,
        errno.ENETUNREACH, errno.EHOSTUNREACH, errno.ENETDOWN, errno.EPIPE, errno.ENOTCONN,
    }


class RemoteJudgeScorer(JudgeScorer):
    """Use training's question/gold/prediction rubric with a separate remote Judge.

    Unlike Task generation, a length-ended Judge response is never a label,
    even if its visible prefix happens to be valid JSON. Missing/invalid labels
    are infrastructure failures and never become a WRONG judgment.
    """

    def __init__(self, remote_service_config, judge_config, diagnostics_root,
                 max_output_tokens=2048, max_completion_tokens=16384, *,
                 network_retries=6, network_retry_delay=30.0):
        self.config = deepcopy(judge_config or {})
        self.remote = _resolved_chat_config(remote_service_config)
        if self.remote.get("request_profile") != "openai_chat":
            raise ValueError("Remote Judge requires an explicit openai_chat service")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("Remote Judge initial output budget must be a positive integer")
        if type(max_completion_tokens) is not int or max_completion_tokens < max_output_tokens:
            raise ValueError("Remote Judge maximum budget must be at least its initial output budget")
        self.max_tokens, self.max_completion_tokens = max_output_tokens, max_completion_tokens
        self.retries = self.remote.get("retries", 3)
        self.timeout = self.remote.get("timeout_seconds", 600)
        if type(self.retries) is not int or self.retries < 0:
            raise ValueError("Remote Judge retries must be a nonnegative integer")
        if (isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float))
                or not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("Remote Judge timeout must be positive and finite")
        if type(network_retries) is not int or network_retries < 0:
            raise ValueError("Remote Judge network retries must be a nonnegative integer")
        if (isinstance(network_retry_delay, bool) or not isinstance(network_retry_delay, (int, float))
                or not math.isfinite(network_retry_delay) or network_retry_delay <= 0):
            raise ValueError("Remote Judge network retry delay must be positive and finite")
        self.network_retries, self.network_retry_delay = network_retries, network_retry_delay
        self.seed = self.config.get("seed", 42)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Judge seed must be a nonnegative integer")
        self.protocols = {**DEFAULT_PROTOCOLS, **self.config.get("protocols", {})}
        if any(p not in DEFAULT_PROTOCOLS.values() for p in self.protocols.values()):
            raise ValueError("Unsupported remote Judge protocol")
        self.templates, upstream, hashes = load_judge_templates(self.config)
        self.settings = {
            "model": self.remote["model"], "model_revision": self.remote["model_revision"],
            "request_profile": "openai_chat", "send_seed": self.remote.get("send_seed", False),
            "temperature": self.remote.get("temperature", 0.0),
            "max_tokens_parameter": self.remote.get("max_tokens_parameter", "max_tokens"),
            "max_output_tokens": self.max_tokens, "seed": self.seed,
        }
        self._provenance = {
            "model": {key: self.remote[key] for key in
                      ("model", "model_revision", "base_url", "request_profile")},
            "protocols": self.protocols, "upstream": upstream, "template_sha256": hashes,
            "parser": PARSER_VERSION, "gold_aggregation": "any_complete_reference_v1",
            "input": "upstream_question_gold_prediction_v1",
            "request_options": {key: self.settings[key] for key in
                                ("request_profile", "send_seed", "temperature", "max_tokens_parameter")},
            "seed": self.seed if self.settings["send_seed"] else None,
            "max_output_tokens": self.max_tokens,
            "completion_policy": {
                "protocol": "strict_remote_judge_v1", "max_completion_tokens": self.max_completion_tokens,
                "budget_multiplier": 4, "reject_finish_reason_length": True,
                "ordinary_retries_per_budget": self.retries, "invalid_label_retries": self.retries,
            },
        }
        self.signature = digest(self._provenance)
        self.root = Path(diagnostics_root)
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = {"protocol": "remote_judge_calls_v1", "scoring_provenance": self._provenance}
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != manifest:
                raise ValueError("Remote Judge protocol changed; choose a new diagnostics/output directory")
        else:
            atomic_json(manifest_path, manifest)
        self._event_lock, self._cache_lock = Lock(), Lock()
        self._key_locks = {}
        # Scheduling changes do not change scoring identity or invalidate labels.
        self._report("remote_judge_network_settings", network_retries=self.network_retries,
                     network_retry_delay_seconds=self.network_retry_delay, maximum_delay_seconds=60)

    def provenance(self):
        return deepcopy(self._provenance)

    def _report(self, event, **fields):
        # Transport emits only local reason codes, scalar usage and sample IDs.
        event = event.replace("remote_task_", "remote_judge_")
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        with self._event_lock:
            with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(canonical(record) + "\n")
            if event not in {"remote_judge_attempt_started", "remote_judge_attempt_succeeded"}:
                print(canonical(record), flush=True)

    def _validate_cached(self, row, expected):
        if (not isinstance(row, dict) or any(row.get(k) != v for k, v in expected.items())
                or row.get("label") not in {"CORRECT", "WRONG"}):
            raise ValueError("Remote Judge cached call provenance/label mismatch")
        metadata = row.get("completion")
        budget = metadata.get("max_tokens") if isinstance(metadata, dict) else None
        allowed, current = {self.max_tokens}, self.max_tokens
        if self.settings["max_tokens_parameter"] == "max_completion_tokens":
            while current < self.max_completion_tokens:
                current = min(current * 4, self.max_completion_tokens)
                allowed.add(current)
        if (type(budget) is not int or budget not in allowed
                or metadata.get("finish_reason") == "length" or metadata.get("truncated") is not False):
            raise ValueError("Remote Judge cached call is incomplete or has an invalid budget")
        validate_completion({**metadata, "text": row["label"]}, budget)
        return float(row["label"] == "CORRECT")

    def _reference_score(self, example, index, messages, parse):
        expected = {"protocol": "remote_judge_call_v1", "judge_provenance_sha256": self.signature,
                    "sample_id": example.sample_id, "reference_index": index,
                    "input_sha256": digest(messages)}
        key = digest(expected)
        path = self.root / "calls" / f"{key}.json"
        with self._cache_lock:
            lock = self._key_locks.setdefault(key, Lock())
        with lock:
            if path.exists():
                return self._validate_cached(json.loads(path.read_text()), expected)
            budget, invalid_retries, network_recoveries = self.max_tokens, 0, 0
            network_delay = min(self.network_retry_delay, 60.0)
            while True:
                api_key = os.environ.get(self.remote["api_key_env"], "").strip()
                if not api_key:
                    raise InfrastructureError("Remote Judge API key is missing; no score assigned")
                settings = {**self.settings, "max_output_tokens": budget}
                last_failure = {}

                def report(event, **fields):
                    if event == "remote_task_attempt_failed":
                        last_failure.clear()
                        last_failure.update(fields)
                    self._report(event, **fields)

                try:
                    completion = _complete(
                        {"sample_id": f"{example.sample_id}/reference/{index}", "messages": messages},
                        settings, self.remote["base_url"], api_key, self.timeout, self.retries,
                        report, self.max_completion_tokens)
                except Exception as exc:
                    transport_type = last_failure.get("transport_error_type") or last_failure.get(
                        "error_type", type(exc).__name__)
                    if _recoverable_network_failure(last_failure) and network_recoveries < self.network_retries:
                        # _complete may have increased its budget before the network failed.
                        latest_budget = last_failure.get("max_tokens")
                        if type(latest_budget) is int and budget <= latest_budget <= self.max_completion_tokens:
                            budget = latest_budget
                        network_recoveries += 1
                        self._report("remote_judge_network_recovery", sample_id=example.sample_id,
                                     reference_index=index, recovery_round=network_recoveries,
                                     max_recovery_rounds=self.network_retries, delay_seconds=network_delay,
                                     error_type=last_failure.get("error_type"), transport_error_type=transport_type,
                                     max_tokens=budget)
                        sleep(network_delay)
                        network_delay = min(network_delay * 2, 60.0)
                        continue
                    raise InfrastructureError(
                        f"Remote Judge request failed for {example.sample_id}/reference/{index} "
                        f"(transport={transport_type}, network_recoveries={network_recoveries}); "
                        "inspect events.jsonl; no score assigned") from None
                budget = completion["max_tokens"]
                reason = "length" if completion.get("finish_reason") == "length" else None
                if reason is None:
                    try:
                        value = parse(completion["text"])
                    except (ValueError, TypeError):
                        reason = "invalid_label"
                if reason is None:
                    metadata = _redact_secret({k: v for k, v in completion.items() if k != "text"}, api_key)
                    record = {**expected, "label": "CORRECT" if value == 1.0 else "WRONG", "completion": metadata}
                    self._validate_cached(record, expected)
                    atomic_json(path, record)
                    return value
                can_grow = (reason == "length" and self.settings["max_tokens_parameter"] == "max_completion_tokens"
                            and budget < self.max_completion_tokens)
                next_budget = min(budget * 4, self.max_completion_tokens) if can_grow else budget
                will_retry = can_grow or invalid_retries < self.retries
                self._report("remote_judge_label_rejected", sample_id=example.sample_id,
                             reference_index=index, reason=reason, max_tokens=budget,
                             next_max_tokens=next_budget if will_retry else None,
                             invalid_label_retries=invalid_retries, will_retry=will_retry)
                if not will_retry:
                    raise InfrastructureError(f"Invalid remote Judge response: {reason}; no score assigned")
                invalid_retries = 0 if can_grow else invalid_retries + 1
                budget = next_budget

    def score(self, example, prediction):
        self.validate_example(example)
        if not isinstance(prediction, str):
            raise ValueError("Judge prediction must be text")
        protocol = self.protocols[example.source]
        for index, reference in enumerate(dict.fromkeys(example.gold_answers)):
            prompt, parse = prompt_for(example, reference, prediction, protocol, templates=self.templates)
            if self._reference_score(example, index, [{"role": "user", "content": prompt}], parse) == 1.0:
                return 1.0
        return 0.0

    def close(self):
        # The shared urllib transport owns and closes each request's response.
        pass
