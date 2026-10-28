from __future__ import annotations

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


class IncompleteServiceCompletion(InfrastructureError):
    """A response with missing content or an unacceptable truncated judge label."""

    def __init__(self, text, finish_reason, usage):
        self.text, self.finish_reason, self.usage = text, finish_reason, usage
        super().__init__(f"Incomplete service completion: finish_reason={finish_reason!r}, "
                         f"content_chars={len(text) if isinstance(text, str) else 0}, usage={usage}")


class ChatService:
    """OpenAI-compatible local HTTP protocol; no OpenAI SDK or remote model dependency."""

    def __init__(self, config: dict, store, role: str):
        self.config, self.store, self.role = config, store, role
        self.identity = {key: config[key] for key in ("model", "model_revision", "base_url")}
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
                 *, validator=None, metadata: dict | None = None) -> str:
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError("Completion metadata must be a per-call dictionary")
            metadata.clear()
        request = {"model": self.config["model"], "messages": messages, "temperature": 0.0,
                   "max_tokens": max_tokens, "seed": seed,
                   "chat_template_kwargs": {"enable_thinking": False}}
        identity = {"identity": self.identity, "role": self.role, "request": request, "cache_scope": cache_scope}
        if self.role == "task_agent":
            # Old cache values contain only text, so they cannot attest whether
            # a task answer was truncated. Keep the response metadata protocol
            # separate from Writer/Judge string caches and from older readers.
            identity["response_protocol"] = TASK_OUTPUT_POLICY
        key = digest(identity)
        # Identical evidence can occur in several rollouts. Share one request,
        # including its failure, without serializing unrelated HTTP requests.
        with self._lock:
            cached = self.store.completion(key)
            if cached is not None:
                text = self._completion_text(cached, metadata)
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
            return self._completion_text(future.result(), metadata)
        try:
            record = self._request(request, key, cache_scope, max_tokens, validator=validator)
            future.set_result(record)
            return self._completion_text(record, metadata)
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    def _completion_text(self, record, metadata):
        if self.role != "task_agent":
            return record
        if (not isinstance(record, dict) or record.get("protocol") != TASK_OUTPUT_POLICY
                or not isinstance(record.get("text"), str) or not record["text"].strip()
                or type(record.get("truncated")) is not bool
                or record["truncated"] != (record.get("finish_reason") == "length")
                or "finish_reason" not in record or "usage" not in record
                or (record["usage"] is not None and not isinstance(record["usage"], dict))
                or type(record.get("max_tokens")) is not int or record["max_tokens"] < 1):
            raise InfrastructureError("Invalid cached Task completion metadata; no task score assigned")
        if metadata is not None:
            # In-flight requests can have many rollout consumers. Never let a
            # caller mutate the shared result or another trajectory's usage.
            metadata.update(deepcopy({key: value for key, value in record.items() if key != "text"}))
        return record["text"]

    def _request(self, request, key, cache_scope, max_tokens, *, validator=None):
        headers = {}
        env_name = self.config.get("api_key_env")
        if env_name and os.environ.get(env_name):
            headers["Authorization"] = "Bearer " + os.environ[env_name]
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
                if (not isinstance(text, str) or not text.strip()
                        or (choice.get("finish_reason") == "length" and self.role not in {"writer", "task_agent"})):
                    raise IncompleteServiceCompletion(text, choice.get("finish_reason"), payload.get("usage"))
                if choice.get("finish_reason") == "length" and self.role in {"writer", "task_agent"}:
                    print(canonical({"event": f"{self.role}_output_truncated", "cache_scope": cache_scope,
                                     "max_tokens": max_tokens, "usage": payload.get("usage")}), flush=True)
                if validator is not None:
                    validator(text)
                record = text
                if self.role == "task_agent":
                    record = {"protocol": TASK_OUTPUT_POLICY, "text": text,
                              "finish_reason": choice.get("finish_reason"),
                              "truncated": choice.get("finish_reason") == "length",
                              "usage": payload.get("usage"), "max_tokens": max_tokens}
                    self._completion_text(record, None)
                self.store.put_completion(key, record)
                return record
            except Exception as exc:
                last_error = exc
                log_path = self.store.root / "service_failures.jsonl"
                record = {"timestamp": time.time(), "role": self.role, "cache_scope": cache_scope, "request_key": key,
                          "attempt": attempt + 1, "max_tokens": max_tokens,
                          "status_code": response.status_code if response is not None else None,
                          "error_type": type(exc).__name__, "error": str(exc), "response": payload}
                with self._lock:
                    append_json(log_path, record)
                print(canonical({"event": "service_failure", "role": self.role,
                                 "cache_scope": cache_scope, "attempt": attempt + 1,
                                 "max_tokens": max_tokens, "error": str(exc),
                                 "response_log": str(log_path)}), flush=True)
                if isinstance(exc, IncompleteServiceCompletion) and self.role == "writer":
                    raise
                if attempt < self.config.get("retries", 2):
                    time.sleep(min(2 ** attempt, 8))
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
    def __init__(self, service, max_output_tokens=512):
        self.service, self.max_output_tokens = service, max_output_tokens

    def answer(self, example, evidence, *, metadata: dict | None = None):
        from .prompts import task_messages
        messages = task_messages(example, evidence)
        return self.service.complete(messages, self.max_output_tokens, metadata=metadata), digest(messages)
