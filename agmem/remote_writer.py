"""Remote session summaries with the same DirectoryBuilder input/output contract."""
from concurrent.futures import Future
from copy import deepcopy
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from threading import Lock

from .core import InfrastructureError, append_json, canonical, digest, file_digest
from .remote_task_runner import _complete, _endpoint, validate_completion


PROTOCOL = "strict_remote_writer_v1"


def _remote_settings(config):
    remote = deepcopy(config)
    if remote.get("request_profile") != "openai_chat":
        raise ValueError("writer_service.request_profile must be openai_chat")
    if "api_key" in remote:
        raise ValueError("Use writer_service.api_key_env instead of storing credentials")
    if "request_base_url" in remote:
        raise ValueError("Remote Writer uses base_url/base_url_env as its actual endpoint and identity")
    for field in ("model", "model_revision"):
        value = remote.get(field)
        if not isinstance(value, str) or not value.strip() or value.startswith("REQUIRED"):
            raise ValueError(f"writer_service.{field} must identify the remote model")
    for field, default in (("base_url_env", "OPENAI_BASE_URL"), ("api_key_env", "OPENAI_API_KEY")):
        remote.setdefault(field, default)
        if not isinstance(remote[field], str) or not remote[field]:
            raise ValueError(f"writer_service.{field} must name an environment variable")
    endpoint = os.environ.get(remote["base_url_env"], "").strip()
    remote["base_url"] = _endpoint(endpoint or remote.get("base_url", "https://api.openai.com/v1"))
    remote.setdefault("temperature", None)
    temperature = remote["temperature"]
    if temperature is not None and (isinstance(temperature, bool)
            or not isinstance(temperature, (int, float)) or not math.isfinite(temperature)
            or not 0 <= temperature <= 2):
        raise ValueError("writer_service.temperature must be null or a number between 0 and 2")
    remote.setdefault("send_seed", False)
    if type(remote["send_seed"]) is not bool:
        raise ValueError("writer_service.send_seed must be a boolean")
    remote.setdefault("max_tokens_parameter", "max_completion_tokens")
    if remote["max_tokens_parameter"] not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("writer_service.max_tokens_parameter is unsupported")
    for field, default, minimum in (("max_output_tokens", 2048, 1),
                                    ("max_completion_tokens", 16384, 1), ("retries", 3, 0)):
        remote.setdefault(field, default)
        if type(remote[field]) is not int or remote[field] < minimum:
            raise ValueError(f"writer_service.{field} must be an integer >= {minimum}")
    if remote["max_completion_tokens"] < remote["max_output_tokens"]:
        raise ValueError("Writer maximum completion budget must be at least its initial budget")
    remote.setdefault("timeout_seconds", 600)
    timeout = remote["timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("writer_service.timeout_seconds must be positive and finite")
    return remote


def _remote_identity(remote):
    return {**{key: remote[key] for key in ("base_url", "model", "model_revision", "request_profile")},
            "request_options": {key: remote[key] for key in
                                ("temperature", "send_seed", "max_tokens_parameter")},
            "completion_policy": {"protocol": PROTOCOL, "initial_max_tokens": remote["max_output_tokens"],
                                  "max_completion_tokens": remote["max_completion_tokens"],
                                  "budget_multiplier": 4, "require_finish_reason": "stop"}}


def _summary_tokenizer_identity(config):
    revision = config["writer_service"].get("summary_tokenizer_revision")
    if revision is not None:
        if not isinstance(revision, str) or not revision.strip() or revision.startswith("REQUIRED"):
            raise ValueError("writer_service.summary_tokenizer_revision must identify the summary tokenizer")
        return {"revision": revision}
    root = Path(config["models"]["task_path"])
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
             "vocab.json", "vocab.txt", "merges.txt", "tokenizer.model", "spiece.model")
    files = {name: file_digest(root / name) for name in names if (root / name).is_file()}
    if not any(name in files for name in ("tokenizer.json", "vocab.json", "vocab.txt", "tokenizer.model", "spiece.model")):
        raise ValueError("Remote Writer requires the original local summary tokenizer files at models.task_path")
    return {"protocol": "local_tokenizer_files_v1", "files": files}


def writer_identity(config):
    """Compute Writer provenance without requiring an API key or loading weights.

    Local identities preserve the original three-field directory-version input.
    Remote identities bind the actual provider, generation options and budget.
    """
    override = config.get("writer_service")
    if override is not None:
        if not isinstance(override, dict):
            raise ValueError("writer_service must be a mapping")
        return {**_remote_identity(_remote_settings(override)),
                "summary_tokenizer": _summary_tokenizer_identity(config)}
    from .policy import model_fingerprint
    identity = {key: config["service"][key] for key in ("base_url", "model")}
    identity["model_revision"] = model_fingerprint(config["models"]["task_path"])["fingerprint"]
    return identity


def create_writer_service(config, store):
    if config.get("writer_service") is not None:
        return RemoteWriterService(config["writer_service"], store, identity=writer_identity(config))
    from .services import ChatService
    local = {**config["service"], "model_revision": writer_identity(config)["model_revision"]}
    return ChatService(local, store, "writer")


class RemoteWriterService:
    """Require nonempty, normally completed summaries; cache only valid results."""

    def __init__(self, config, store, *, identity):
        self.config = _remote_settings(config)
        self.store, self.identity = store, deepcopy(identity)
        self._api_key()
        self._lock, self._event_lock, self._inflight = Lock(), Lock(), {}
        self.events_path = store.root / "remote_writer_events.jsonl"

    def _api_key(self):
        key = os.environ.get(self.config["api_key_env"], "").strip()
        if not key:
            raise InfrastructureError(f"Set {self.config['api_key_env']} for the remote Writer")
        return key

    def _report(self, event, **fields):
        # The transport supplies local reason codes and scalar counters only.
        # Never log exception text, headers, messages, summaries or response bodies.
        record = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "event": event.replace("remote_task_", "remote_writer_"), **fields}
        with self._event_lock:
            append_json(self.events_path, record)
            if record["event"] != "remote_writer_attempt_started":
                print(canonical(record), flush=True)

    def _allowed_budgets(self):
        budget, maximum = self.config["max_output_tokens"], self.config["max_completion_tokens"]
        values = {budget}
        if self.config["max_tokens_parameter"] == "max_completion_tokens":
            while budget < maximum:
                budget = min(budget * 4, maximum)
                values.add(budget)
        return values

    def _validate_cached(self, record, expected):
        if not isinstance(record, dict) or any(record.get(key) != value for key, value in expected.items()):
            raise InfrastructureError("Remote Writer cache provenance mismatch")
        completion = record.get("completion")
        budget = completion.get("max_tokens") if isinstance(completion, dict) else None
        if (type(budget) is not int or budget not in self._allowed_budgets()
                or completion.get("finish_reason") != "stop" or completion.get("truncated") is not False):
            raise InfrastructureError("Remote Writer cache contains an incomplete summary")
        try:
            validate_completion(completion, budget)
        except (TypeError, ValueError):
            raise InfrastructureError("Remote Writer cache contains invalid completion metadata") from None
        return completion["text"]

    def complete(self, messages, max_tokens, seed=42, cache_scope=None):
        # max_tokens is DirectoryBuilder's visible-summary limit, not the total
        # remote reasoning/completion allowance. The caller truncates with the
        # original local Writer tokenizer after this normally completed response.
        if type(max_tokens) is not int or max_tokens < 1 or type(seed) is not int or seed < 0:
            raise ValueError("Writer summary limit and seed must be valid integers")
        request_identity = {"messages": messages, "content_max_tokens": max_tokens,
                            "seed": seed if self.config["send_seed"] else None, "cache_scope": cache_scope}
        expected = {"protocol": PROTOCOL, "writer_identity_sha256": digest(self.identity),
                    "request_sha256": digest(request_identity)}
        key = digest(expected)
        with self._lock:
            cached = self.store.completion(key)
            if cached is not None:
                return self._validate_cached(cached, expected)
            future = self._inflight.get(key)
            owner = future is None
            if owner:
                future = self._inflight[key] = Future()
        if not owner:
            return self._validate_cached(future.result(), expected)
        try:
            completion = self._generate(messages, seed, key)
            record = {**expected, "completion": completion}
            text = self._validate_cached(record, expected)
            self.store.put_completion(key, record)
            self._report("remote_writer_session_completed", request_key=key,
                         content_chars=len(text), max_tokens=completion["max_tokens"])
            future.set_result(record)
            return text
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    def _generate(self, messages, seed, key):
        remote = self.config
        budget, rejected = remote["max_output_tokens"], 0
        while True:
            settings = {**{field: remote[field] for field in
                           ("model", "model_revision", "request_profile", "temperature",
                            "send_seed", "max_tokens_parameter")}, "seed": seed, "max_output_tokens": budget}
            last_failure = {}

            def report(event, **fields):
                if event == "remote_task_attempt_failed":
                    last_failure.clear()
                    last_failure.update(fields)
                self._report(event, **fields)

            try:
                completion = _complete({"sample_id": key, "messages": messages}, settings, remote["base_url"],
                                       self._api_key(), remote["timeout_seconds"], remote["retries"],
                                       report, remote["max_completion_tokens"])
            except Exception as exc:
                raise InfrastructureError(
                    f"Remote Writer request failed ({type(exc).__name__}, "
                    f"reason={last_failure.get('reason', 'request_failed')}); "
                    "inspect remote_writer_events.jsonl; no summary saved") from None
            # _complete expands empty length-ended output. Nonempty prefixes also
            # need expansion here: they must not become frozen directory facts.
            budget = completion.get("max_tokens") if isinstance(completion, dict) else None
            if type(budget) is not int or budget not in self._allowed_budgets():
                raise InfrastructureError("Remote Writer returned invalid completion budget metadata")
            try:
                validate_completion(completion, budget)
            except (TypeError, ValueError):
                raise InfrastructureError("Remote Writer returned invalid completion metadata") from None
            if completion["finish_reason"] == "stop":
                return completion
            can_grow = (completion["finish_reason"] == "length"
                        and remote["max_tokens_parameter"] == "max_completion_tokens"
                        and budget < remote["max_completion_tokens"])
            next_budget = min(budget * 4, remote["max_completion_tokens"]) if can_grow else budget
            will_retry = can_grow or rejected < remote["retries"]
            self._report("remote_writer_completion_rejected", sample_id=key,
                         reason="length" if completion["finish_reason"] == "length" else "incomplete_response",
                         max_tokens=budget, next_max_tokens=next_budget if will_retry else None,
                         will_retry=will_retry)
            if not will_retry:
                raise InfrastructureError("Remote Writer response did not finish normally; no summary saved")
            rejected = 0 if can_grow else rejected + 1
            budget = next_budget
