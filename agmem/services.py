from __future__ import annotations

import math
import os
import time
from ipaddress import ip_address
from concurrent.futures import Future
from copy import deepcopy
from threading import Lock
from urllib.parse import urlsplit

from .core import InfrastructureError, append_json, canonical, digest
from .data import ADAPTER_VERSION
from .execution import bounded_results
from .prompts import writer_messages


TASK_OUTPUT_POLICY = "score_nonempty_truncated_v1"
TASK_TAGGED_PARTIAL_OUTPUT_POLICY = "score_tagged_truncated_v1"


def extract_partial_tagged_answer(text, metadata=None):
    """Extract the first answer block, retaining an unfinished block's suffix.

    Analysis before an opening tag is never an answer. The parser deliberately
    also accepts empty answers; semantic scoring belongs to the Judge.
    """
    if not isinstance(text, str):
        raise ValueError("Task response must be text")
    start = text.find("<answer>")
    if start < 0:
        answer, status = "", "missing_open_tag"
    else:
        start += len("<answer>")
        end = text.find("</answer>", start)
        answer = text[start:end if end >= 0 else None].strip()
        status = "complete" if end >= 0 else "open_tag_only"
    if metadata is not None:
        metadata.update(answer_status=status, answer_tag_complete=status == "complete")
    return answer


def _resolved_chat_config(config):
    """Resolve endpoints without retaining credentials in a service config."""
    config = deepcopy(config)
    profile = config.get("request_profile", "local_vllm")
    if profile not in {"local_vllm", "openai_chat"}:
        raise ValueError("service.request_profile must be local_vllm or openai_chat")
    if profile == "openai_chat":
        if "api_key" in config:
            raise ValueError("Use service.api_key_env instead of storing an API key in configuration")
        config.setdefault("api_key_env", "OPENAI_API_KEY")
        if "base_url" not in config and "base_url_env" not in config and "request_base_url" not in config:
            config["base_url_env"] = "OPENAI_BASE_URL"
    endpoint_env = config.get("base_url_env")
    if endpoint_env is not None:
        if not isinstance(endpoint_env, str) or not endpoint_env:
            raise ValueError("service.base_url_env must name an environment variable")
        endpoint = os.environ.get(endpoint_env, "").strip()
        if not endpoint:
            raise ValueError(f"Set environment variable {endpoint_env} for the service endpoint")
        config["base_url"] = endpoint
    if profile == "openai_chat":
        endpoint = config.get("request_base_url", config.get("base_url", ""))
        if not isinstance(endpoint, str):
            raise ValueError("Remote service endpoint must be an HTTP(S) base URL")
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError("Remote service endpoint must be an HTTP(S) base URL without credentials, query or fragment")
        # Unlike a fingerprinted local deployment, a remote endpoint is part of
        # the model identity; changing the provider must not reuse its cache.
        config["base_url"] = endpoint.rstrip("/")
        config.pop("request_base_url", None)
        env_name = config["api_key_env"]
        if not isinstance(env_name, str) or not env_name or not os.environ.get(env_name, "").strip():
            raise ValueError("Set the remote service API key environment variable "
                             + (env_name if isinstance(env_name, str) else "named by service.api_key_env"))
        revision = config.get("model_revision")
        if not isinstance(revision, str) or not revision.strip() or revision.startswith("REQUIRED"):
            raise ValueError("Remote service.model_revision must be an explicit remote model version")
        if not isinstance(config.get("model"), str) or not config["model"].strip():
            raise ValueError("Remote service.model must be the provider's model ID")
        if type(config.get("send_seed", False)) is not bool:
            raise ValueError("service.send_seed must be a boolean")
        temperature = config.get("temperature", 0.0)
        if temperature is not None and (isinstance(temperature, bool)
                or not isinstance(temperature, (int, float)) or not math.isfinite(temperature)
                or not 0 <= temperature <= 2):
            raise ValueError("service.temperature must be null or a finite number between 0 and 2")
        if config.get("max_tokens_parameter", "max_tokens") not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("service.max_tokens_parameter must be max_tokens or max_completion_tokens")
    return config


def resolve_task_service(config):
    """Select an independent Task service; preserve the legacy local default.

    An explicit override is a complete service configuration, so remote Task
    settings cannot inherit the Writer/Judge endpoint or local model revision.
    """
    override = config.get("task_agent", {}).get("service")
    if override is not None and not isinstance(override, dict):
        raise ValueError("task_agent.service must be a mapping")
    service = deepcopy(override if override is not None else config["service"])
    if service.get("request_profile", "local_vllm") != "openai_chat":
        if override is None or "model_revision" not in service:
            from .policy import model_fingerprint
            service["model_revision"] = model_fingerprint(config["models"]["task_path"])["fingerprint"]
    return _resolved_chat_config(service)


def _redact_secret(value, secret):
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]")
    if isinstance(value, list):
        return [_redact_secret(item, secret) for item in value]
    if isinstance(value, dict):
        return {_redact_secret(key, secret): _redact_secret(item, secret) for key, item in value.items()}
    return value


class IncompleteServiceCompletion(InfrastructureError):
    """A response with missing content or an unacceptable truncated judge label."""

    def __init__(self, text, finish_reason, usage):
        self.text, self.finish_reason, self.usage = text, finish_reason, usage
        super().__init__(f"Incomplete service completion: finish_reason={finish_reason!r}, "
                         f"content_chars={len(text) if isinstance(text, str) else 0}, usage={usage}")


class ChatService:
    """Cached local vLLM or explicitly configured standard Chat Completions."""

    def __init__(self, config: dict, store, role: str):
        config = _resolved_chat_config(config)
        self.config, self.store, self.role = config, store, role
        self.identity = {key: config[key] for key in ("model", "model_revision", "base_url")}
        self.request_profile = config.get("request_profile", "local_vllm")
        if self.request_profile == "openai_chat":
            self.identity["request_profile"] = self.request_profile
        if not config["model_revision"] or config["model_revision"].startswith("REQUIRED"):
            raise ValueError("Set service.model_revision to the local checkpoint fingerprint/version")
        import httpx
        endpoint = config.get("request_base_url", config["base_url"])
        host = (urlsplit(endpoint).hostname or "").rstrip(".").lower()
        try:
            loopback = ip_address(host).is_loopback
        except ValueError:
            loopback = host == "localhost"
        # SSH/tmux may inherit a proxy whose forwarded port no longer exists.
        # Local vLLM must be contacted directly; remote endpoints retain proxies.
        self.client = httpx.Client(timeout=config.get("timeout_seconds", 600), trust_env=not loopback)
        self._lock = Lock()
        self._inflight = {}

    def complete(self, messages: list[dict], max_tokens: int, seed: int = 42, cache_scope: str | None = None,
                 *, validator=None, metadata: dict | None = None, require_stop=False,
                 answer_policy=None) -> str:
        if type(require_stop) is not bool or require_stop and self.role != "task_agent":
            raise ValueError("require_stop must be a boolean and is only supported for Task responses")
        if answer_policy is not None and (answer_policy != TASK_TAGGED_PARTIAL_OUTPUT_POLICY
                or self.role != "task_agent" or require_stop):
            raise ValueError("Partial tagged answers require the dedicated Task policy without require_stop")
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError("Completion metadata must be a per-call dictionary")
            metadata.clear()
        request = {"model": self.config["model"], "messages": messages, "temperature": 0.0,
                   "max_tokens": max_tokens, "seed": seed,
                   "chat_template_kwargs": {"enable_thinking": False}}
        if self.request_profile == "openai_chat":
            request = {"model": self.config["model"], "messages": messages,
                       self.config.get("max_tokens_parameter", "max_tokens"): max_tokens}
            if self.config.get("temperature", 0.0) is not None:
                request["temperature"] = self.config.get("temperature", 0.0)
            if self.config.get("send_seed", False):
                request["seed"] = seed
        identity = {"identity": self.identity, "role": self.role, "request": request, "cache_scope": cache_scope}
        if self.role == "task_agent":
            # Old cache values contain only text, so they cannot attest whether
            # a task answer was truncated. Keep the response metadata protocol
            # separate from Writer/Judge string caches and from older readers.
            identity["response_protocol"] = answer_policy or TASK_OUTPUT_POLICY
            if require_stop:
                identity["completion_policy"] = "require_stop_v1"
        key = digest(identity)
        # Identical evidence can occur in several rollouts. Share one request,
        # including its failure, without serializing unrelated HTTP requests.
        with self._lock:
            cached = self.store.completion(key)
            if cached is not None:
                text = self._completion_text(cached, metadata, require_stop=require_stop, answer_policy=answer_policy)
                if validator is not None:
                    try:
                        validator(text)
                    except Exception as exc:
                        raise InfrastructureError("Invalid cached service response; use a corrected scoring protocol") from exc
                return text
            future = self._inflight.get(key)
            owner = future is None
            if owner:
                future = self._inflight[key] = Future()
        if not owner:
            return self._completion_text(future.result(), metadata, require_stop=require_stop, answer_policy=answer_policy)
        try:
            record = self._request(request, key, cache_scope, max_tokens,
                                   validator=validator, require_stop=require_stop, answer_policy=answer_policy)
            future.set_result(record)
            return self._completion_text(record, metadata, require_stop=require_stop, answer_policy=answer_policy)
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    def _completion_text(self, record, metadata, *, require_stop=False, answer_policy=None):
        if self.role != "task_agent":
            return record
        partial = answer_policy == TASK_TAGGED_PARTIAL_OUTPUT_POLICY
        if (not isinstance(record, dict) or record.get("protocol") != (answer_policy or TASK_OUTPUT_POLICY)
                or not isinstance(record.get("text"), str) or (not partial and not record["text"].strip())
                or type(record.get("truncated")) is not bool
                or record["truncated"] != (record.get("finish_reason") == "length")
                or "finish_reason" not in record or "usage" not in record
                or (record["usage"] is not None and not isinstance(record["usage"], dict))
                or type(record.get("max_tokens")) is not int or record["max_tokens"] < 1):
            raise InfrastructureError("Invalid cached Task completion metadata; no task score assigned")
        if require_stop and record["finish_reason"] != "stop":
            raise InfrastructureError("Task answer did not finish normally; no task score assigned")
        if partial and record["finish_reason"] not in {"stop", "length"}:
            raise InfrastructureError("Task completion has an unexpected finish reason; no task score assigned")
        if metadata is not None:
            # In-flight requests can have many rollout consumers. Never let a
            # caller mutate the shared result or another trajectory's usage.
            metadata.update(deepcopy({key: value for key, value in record.items() if key != "text"}))
        return record["text"]

    def _request(self, request, key, cache_scope, max_tokens, *, validator=None, require_stop=False,
                 answer_policy=None):
        headers = {}
        env_name = self.config.get("api_key_env")
        api_key = os.environ.get(env_name, "") if env_name else ""
        if self.request_profile == "openai_chat" and not api_key.strip():
            raise InfrastructureError(f"Set environment variable {env_name} for the remote service API key")
        if api_key:
            headers["Authorization"] = "Bearer " + api_key
        last_error = None
        for attempt in range(self.config.get("retries", 2) + 1):
            response, payload = None, None
            try:
                # Deployment port may differ between jobs; base_url remains the
                # frozen cache identity for the same fingerprinted local model.
                endpoint = self.config.get("request_base_url", self.config["base_url"])
                response = self.client.post(endpoint.rstrip("/") + "/chat/completions", json=request, headers=headers)
                response.raise_for_status()
                payload = response.json()
                choice = payload["choices"][0]
                text = choice["message"]["content"]
                partial = answer_policy == TASK_TAGGED_PARTIAL_OUTPUT_POLICY
                if (not isinstance(text, str) or (not partial and not text.strip())
                        or (partial and choice.get("finish_reason") not in {"stop", "length"})
                        or (require_stop and choice.get("finish_reason") != "stop")
                        or (choice.get("finish_reason") == "length" and self.role not in {"writer", "task_agent"})):
                    raise IncompleteServiceCompletion(text, choice.get("finish_reason"), payload.get("usage"))
                if choice.get("finish_reason") == "length" and self.role in {"writer", "task_agent"}:
                    print(canonical({"event": f"{self.role}_output_truncated", "cache_scope": cache_scope,
                                     "max_tokens": max_tokens, "usage": payload.get("usage")}), flush=True)
                if validator is not None:
                    validator(text)
                record = text
                if self.role == "task_agent":
                    record = {"protocol": answer_policy or TASK_OUTPUT_POLICY, "text": text,
                              "finish_reason": choice.get("finish_reason"),
                              "truncated": choice.get("finish_reason") == "length",
                              "usage": payload.get("usage"), "max_tokens": max_tokens}
                    self._completion_text(record, None, require_stop=require_stop, answer_policy=answer_policy)
                self.store.put_completion(key, record)
                return record
            except Exception as exc:
                last_error = exc
                log_path = self.store.root / "service_failures.jsonl"
                error = _redact_secret(str(exc), api_key)
                record = {"timestamp": time.time(), "role": self.role, "cache_scope": cache_scope, "request_key": key,
                          "attempt": attempt + 1, "max_tokens": max_tokens,
                          "status_code": response.status_code if response is not None else None,
                          "error_type": type(exc).__name__, "error": error,
                          "response": _redact_secret(payload, api_key)}
                with self._lock:
                    append_json(log_path, record)
                print(canonical({"event": "service_failure", "role": self.role,
                                 "cache_scope": cache_scope, "attempt": attempt + 1,
                                 "max_tokens": max_tokens, "error": error,
                                 "response_log": str(log_path)}), flush=True)
                if isinstance(exc, IncompleteServiceCompletion) and self.role == "writer":
                    raise
                if attempt < self.config.get("retries", 2):
                    time.sleep(min(2 ** attempt, 8))
        if self.request_profile == "openai_chat":
            # Provider errors may echo credentials; the diagnostic file above is
            # redacted and is safer than an unfiltered exception cause.
            raise InfrastructureError(f"{self.role} service failed ({type(last_error).__name__}); no task score assigned") from None
        raise InfrastructureError(f"{self.role} service failed ({type(last_error).__name__}); no task score assigned") from last_error


class DirectoryBuilder:
    def __init__(self, store, service, config: dict, tokenizer):
        self.store, self.service, self.config, self.tokenizer = store, service, config, tokenizer
        if config["lookback_sessions"] < 3:
            raise ValueError("Writer needs at least three previous original sessions")
        identity = {"writer": service.identity, "config": config, "record_unit": ADAPTER_VERSION,
                    "storage_format": store.FORMAT, "reuse_scope": "complete_history",
                    "format_protocol": 4, "prompt": writer_messages([], {}, config["content_max_tokens"])}
        self.version = digest(identity)

    def build(self, memory_id: str, workers: int = 1) -> list[dict]:
        try:
            directory = self.store.directory(memory_id, self.version)
            if all(entry.get("writer_status") != "empty_pending" and entry["content"].strip()
                   for entry in directory):
                return directory
        except ValueError:
            pass
        references = self.store.memory(memory_id)
        entries = [None] * len(references)
        lookback = self.config["lookback_sessions"]

        def requests():
            # Only the coordinator reads histories and modifies directory indexes.
            # Raw preceding sessions are independent of their Writer summaries.
            for index, reference in enumerate(references):
                cached = self.store.directory_entry(memory_id, self.version, reference["id"])
                if cached is not None and cached.get("writer_status") != "empty_pending" and cached["content"].strip():
                    entries[index] = cached
                    continue
                target = self.store.record(reference)
                context = [self.store.record(row) for row in references[max(0, index - lookback):index]]
                messages = writer_messages(context, target, self.config["content_max_tokens"])
                yield index, reference, target, messages

        def generate(item):
            _, _, target, messages = item
            if not target["messages"]:
                return "Empty session; no recorded messages."
            try:
                return self.service.complete(messages, self.config["content_max_tokens"],
                                             cache_scope=f"{memory_id}/{self.version}")
            except IncompleteServiceCompletion as exc:
                if isinstance(exc.text, str) and exc.text.strip():
                    raise
                return ""

        for (index, reference, target, _), content in bounded_results(generate, requests(), workers):
            # Metadata and JSON escaping belong to the program. A length-limited
            # Writer response is usable as-is; never ask the model to repair it.
            content = content.strip() if isinstance(content, str) else ""
            token_ids = self.tokenizer.encode(content, add_special_tokens=False)
            if len(token_ids) > self.config["content_max_tokens"]:
                content = self.tokenizer.decode(token_ids[:self.config["content_max_tokens"]],
                                                skip_special_tokens=True, clean_up_tokenization_spaces=False)
            names = sorted({message.get("name") or message["role"] for message in target["messages"]})
            entry = {"id": reference["id"],
                     "task_session": {"task_id": None, "session_id": reference["source_session_id"]},
                     "actors_source": {"actors": names, "sources": []},
                     "time_order": {"time": reference["time"], "order": index}, "content": content}
            if not content:
                entry["writer_status"] = "empty_pending"
                print(canonical({"event": "writer_empty_pending", "memory_id": memory_id,
                                 "session_id": reference["source_session_id"], "id": reference["id"],
                                 "version": self.version}), flush=True)
            self.store.put_directory_entry(memory_id, self.version, entry)
            entries[index] = entry
        self.store.put_directory(memory_id, self.version, entries)
        return entries


class TaskAgent:
    def __init__(self, service, max_output_tokens=512, prompt_version="legacy_v1"):
        from .prompts import validate_task_prompt_version
        self.service, self.max_output_tokens = service, max_output_tokens
        self.prompt_version = validate_task_prompt_version(prompt_version)

    def answer(self, example, evidence, *, metadata: dict | None = None):
        from .prompts import task_messages
        messages = task_messages(example, evidence, self.prompt_version)
        if self.prompt_version == "legacy_v1":
            return self.service.complete(messages, self.max_output_tokens, metadata=metadata), digest(messages)
        if self.prompt_version in {"history_answer_evidence_v2", "hierarchical_evidence_v1"}:
            completion_metadata = metadata if metadata is not None else {}
            raw = self.service.complete(messages, self.max_output_tokens, metadata=completion_metadata,
                                        answer_policy=TASK_TAGGED_PARTIAL_OUTPUT_POLICY)
            answer = extract_partial_tagged_answer(raw, completion_metadata)
            completion_metadata.update(raw_text=raw, answer_parser="answer_tag_partial_v1")
            return answer, digest(messages)
        from .remote_task_runner import extract_tagged_answer
        completion_metadata = metadata if metadata is not None else {}
        raw = self.service.complete(messages, self.max_output_tokens,
                                    validator=extract_tagged_answer, metadata=completion_metadata,
                                    require_stop=True)
        if completion_metadata.get("finish_reason") != "stop" or completion_metadata.get("truncated"):
            raise InfrastructureError("Task answer did not finish normally; no task score assigned")
        try:
            answer = extract_tagged_answer(raw)
        except ValueError:
            raise InfrastructureError("Task answer has invalid answer tags; no task score assigned") from None
        completion_metadata.update(raw_text=raw, answer_parser="answer_tag_v1")
        return answer, digest(messages)
