"""Bounded DeepSeek directory Writer transport; semantic validation stays local.

API reference: https://api-docs.deepseek.com/zh-cn/api/create-chat-completion/
JSON mode is not schema enforcement. The existing Writer validates every result.
"""

from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
import os
from queue import Queue
import re
from threading import Lock
import time
from urllib.parse import urlsplit

from agmem.core import InfrastructureError, digest


def schema_messages(messages, schema):
    """Preserve semantic prompts and add the full output schema as instructions."""
    result = deepcopy(messages)
    instruction = (
        "Return exactly one JSON object, with no Markdown or text outside the JSON. "
        "The JSON must satisfy the following JSON Schema, including all required fields "
        "and uniqueness constraints. Keep following the directory generation instructions.\n"
        "JSON Schema:\n" + json.dumps(schema, ensure_ascii=False, sort_keys=True)
    )
    # Extend the original system message, keeping its semantic prompt intact and
    # keeping the final user message last for provider chat-template compatibility.
    for message in result:
        if message.get("role") == "system":
            message["content"] = message["content"] + "\n\n" + instruction
            break
    else:
        result.insert(0, {"role": "system", "content": instruction})
    return result


def _positive_number(settings, key, default):
    value = settings.get(key, default)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"DeepSeek Writer {key} must be a finite positive number")
    return value


class DeepSeekWriterPool:
    """One global worker bound, reusable HTTP connections, and shared retry pacing.

    ``transport`` accepts an httpx transport (for example MockTransport). Clock
    and sleep injection keep retry tests entirely offline and deterministic.
    After a transient failure, starts are staggered by retry_spacing_seconds so
    waiting workers do not all retry at the shared cooldown boundary.
    """

    def __init__(self, endpoints, workers, settings, store, *, transport=None,
                 sleep=None, monotonic=None, wall_time=None):
        import httpx

        self.settings, self.store = dict(settings), store
        self.clients, self.slots = [], Queue()
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.monotonic
        self._wall_time = wall_time or time.time
        self._cooldown_lock = Lock()
        self._cooldown_until = self._next_start = 0.0
        self._recovering = False
        self._retries = self.settings.get("network_retries", 3)
        if type(self._retries) is not int or self._retries < 0:
            raise ValueError("network_retries must be a nonnegative integer")
        if type(workers) is not int or workers < 1:
            raise ValueError("Writer workers must be a positive integer")
        self._retry_base = _positive_number(self.settings, "retry_base_seconds", 1)
        self._retry_max = _positive_number(self.settings, "retry_max_seconds", 30)
        self._retry_spacing = _positive_number(self.settings, "retry_spacing_seconds", 0.05)
        timeout = _positive_number(self.settings, "timeout_seconds", 600)
        trust_env = self.settings.get("trust_env", True)
        if type(trust_env) is not bool:
            raise ValueError("DeepSeek Writer trust_env must be a boolean")
        model = self.settings.get("model", "deepseek-flash")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("DeepSeek Writer model must be nonempty text")
        key_env = self.settings.get("api_key_env", "DEEPSEEK_API_KEY")
        if not isinstance(key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env):
            raise ValueError("DeepSeek Writer api_key_env must be an environment variable name")
        if not isinstance(endpoints, (list, tuple)) or not endpoints:
            raise ValueError("DeepSeek Writer endpoints must be a nonempty sequence of URLs")
        validated = []
        for endpoint in endpoints:
            if not isinstance(endpoint, str):
                raise ValueError("DeepSeek Writer endpoint URLs must be strings")
            try:
                url = urlsplit(endpoint)
                port = url.port
            except ValueError:
                raise ValueError("Invalid DeepSeek Writer endpoint URL") from None
            if (url.scheme not in {"http", "https"} or not url.hostname or url.username is not None
                    or url.password is not None or "?" in endpoint or "#" in endpoint or port == 0
                    or any(character.isspace() or ord(character) < 32 for character in endpoint)):
                raise ValueError("DeepSeek Writer URLs must not contain credentials/query/fragment or whitespace")
            validated.append(endpoint.rstrip("/"))
        self._api_key = os.environ.get(key_env, "")
        if not self._api_key:
            raise InfrastructureError(f"DeepSeek Writer API key is missing; set {key_env}")
        if any(ord(character) < 33 or ord(character) > 126 for character in self._api_key):
            raise InfrastructureError("DeepSeek Writer API key contains invalid characters")
        try:
            for endpoint in validated:
                client = httpx.Client(
                    timeout=timeout, trust_env=trust_env,
                    limits=httpx.Limits(max_connections=workers, max_keepalive_connections=workers),
                    transport=transport,
                )
                self.clients.append((endpoint, client))
        except Exception:
            self.close()
            # Proxy or transport constructor errors can contain credentials.
            raise InfrastructureError("DeepSeek Writer HTTP client initialization failed") from None
        for index in range(workers):
            self.slots.put(self.clients[index % len(self.clients)])

    def close(self):
        for _, client in self.clients:
            client.close()

    def for_session(self, session):
        return DeepSeekSessionClient(self, session["bank_id"], session["parent_id"])

    def _redact(self, value):
        if isinstance(value, str):
            return value.replace(self._api_key, "[REDACTED]")
        if isinstance(value, dict):
            return {self._redact(key): self._redact(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._redact(item) for item in value]
        return value

    def _retry_after(self, response):
        value = response.headers.get("Retry-After")
        if not value:
            return 0.0
        try:
            delay = float(value)
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                delay = date.timestamp() - self._wall_time()
            except (TypeError, ValueError, OverflowError):
                return 0.0
        return max(0.0, delay) if math.isfinite(delay) else 0.0

    def _defer(self, attempt, retry_after=0.0):
        delay = max(min(self._retry_base * 2 ** min(attempt, 30), self._retry_max), retry_after)
        with self._cooldown_lock:
            self._recovering = True
            self._cooldown_until = max(self._cooldown_until, self._monotonic() + delay)
        return delay

    def _wait_for_turn(self):
        with self._cooldown_lock:
            if not self._recovering:
                return
            scheduled = max(self._monotonic(), self._cooldown_until, self._next_start)
            self._next_start = scheduled + self._retry_spacing
        while True:
            delay = scheduled - self._monotonic()
            if delay > 0:
                self._sleep(delay)
            with self._cooldown_lock:
                # A different worker may have received another 429 while this
                # worker slept. Reserve a new paced start after that cooldown.
                if self._cooldown_until <= scheduled:
                    return
                scheduled = max(self._monotonic(), self._cooldown_until, self._next_start)
                self._next_start = scheduled + self._retry_spacing


class DeepSeekSessionClient:
    def __init__(self, pool, bank_id, parent_id):
        self.pool, self.bank_id, self.parent_id = pool, bank_id, parent_id

    def _trace(self, event, **data):
        self.pool.store.trace(self.pool._redact({
            "timestamp": datetime.now(timezone.utc).isoformat(), "event": event,
            "bank_id": self.bank_id, "parent_id": self.parent_id, **data,
        }))

    def __call__(self, messages, max_tokens, schema, stage):
        import httpx

        request = {
            "model": self.pool.settings.get("model", "deepseek-flash"),
            "messages": schema_messages(messages, schema), "max_tokens": max_tokens,
            "temperature": 0, "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"}, "stream": False,
        }
        request_hash = digest(request)
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self.pool._api_key}
        self._trace("writer_request", stage=stage, request_sha256=request_hash, request=request)
        for attempt in range(self.pool._retries + 1):
            endpoint, client = self.pool.slots.get()
            status_code, retry_after, retryable = None, 0.0, True
            try:
                self.pool._wait_for_turn()
                response = client.post(endpoint + "/chat/completions", json=request, headers=headers)
                status_code = response.status_code
                if status_code != 200:
                    retryable = status_code == 429 or 500 <= status_code < 600
                    if retryable:
                        retry_after = self.pool._retry_after(response)
                    raise InfrastructureError(f"DeepSeek Writer HTTP {status_code}")
                payload = response.json()
                choice = payload["choices"][0]
                result = {"text": choice["message"].get("content"),
                          "finish_reason": choice.get("finish_reason"),
                          "usage": payload.get("usage", {}), "model": payload.get("model")}
                if result["text"] is not None and not isinstance(result["text"], str):
                    raise ValueError("Invalid DeepSeek Writer content type")
                self._trace("writer_response", stage=stage, attempt=attempt + 1,
                            request_sha256=request_hash, response=result)
                # Preserve truncation for the Writer's output-budget expansion.
                if result["finish_reason"] != "length" and not (result["text"] or "").strip():
                    raise InfrastructureError("Empty DeepSeek Writer JSON content")
                return result
            except (httpx.HTTPError, InfrastructureError, ValueError, KeyError, IndexError,
                    TypeError, AttributeError) as exc:
                will_retry = retryable and attempt < self.pool._retries
                # Even an exhausted request cools down other sessions sharing
                # this account. No response body or exception text is logged.
                delay = self.pool._defer(attempt, retry_after) if retryable else 0.0
                self._trace("writer_transport_error", stage=stage, attempt=attempt + 1,
                            request_sha256=request_hash, error_type=type(exc).__name__,
                            status_code=status_code, retryable=retryable, will_retry=will_retry,
                            retry_delay_seconds=delay)
                if not will_retry:
                    status = f"HTTP {status_code}" if status_code is not None else type(exc).__name__
                    raise InfrastructureError(
                        f"DeepSeek Writer transport failed ({status}); inspect writer_traces"
                    ) from None
            finally:
                self.pool.slots.put((endpoint, client))

    def accept(self, messages, max_tokens, schema, stage, result, status="complete"):
        self._trace("writer_validated", stage=stage, status=status,
                    response_sha256=digest(result), messages_sha256=digest(messages), max_tokens=max_tokens)

    def reject(self, messages, max_tokens, schema, stage, result, reason):
        self._trace("writer_validation_failed", stage=stage, reason=str(reason),
                    response_sha256=digest(result), messages_sha256=digest(messages), max_tokens=max_tokens)
