"""Bounded local Qwen replicas; transport errors never become fabricated summaries."""

from datetime import datetime, timezone
import os
from queue import Queue
import time
from urllib.parse import urlsplit

from agmem.core import InfrastructureError, digest


def guidance_schema(schema):
    """Project decoder constraints; the Writer still validates the full schema.

    The installed vLLM llguidance backend rejects uniqueItems during grammar
    compilation. Uniqueness is therefore enforced after generation, along with
    the other semantic checks, rather than delegated to the decoder.
    """
    if isinstance(schema, dict):
        return {key: guidance_schema(value) for key, value in schema.items() if key != "uniqueItems"}
    if isinstance(schema, list):
        return [guidance_schema(value) for value in schema]
    return schema


class WriterPool:
    def __init__(self, endpoints, workers, settings, store):
        import httpx
        self.settings, self.store = dict(settings), store
        self.clients = []
        self.slots = Queue()
        retries = self.settings.get("network_retries", 3)
        if type(retries) is not int or retries < 0:
            raise ValueError("network_retries must be a nonnegative integer")
        if type(workers) is not int or workers < 1:
            raise ValueError("Writer workers must be a positive integer")
        if isinstance(endpoints, str):
            raise ValueError("Writer endpoints must be a sequence of URLs")
        validated = []
        for endpoint in endpoints:
            if not isinstance(endpoint, str):
                raise ValueError("Writer endpoint URLs must be strings")
            try:
                url = urlsplit(endpoint)
                port = url.port
            except ValueError:
                raise ValueError("Invalid Writer endpoint URL") from None
            if (url.scheme not in {"http", "https"} or not url.hostname or url.username is not None
                    or url.password is not None or url.query or url.fragment or port == 0):
                raise ValueError("Writer URLs must not contain credentials/query/fragment")
            validated.append(endpoint.rstrip("/"))
        if not validated:
            raise ValueError("At least one local Writer endpoint is required")
        try:
            for endpoint in validated:
                client = httpx.Client(timeout=self.settings.get("timeout_seconds", 600), trust_env=False)
                self.clients.append((endpoint, client))
        except Exception:
            self.close()
            raise
        for i in range(workers):
            self.slots.put(self.clients[i % len(self.clients)])

    def close(self):
        for _, client in self.clients:
            client.close()

    def for_session(self, session):
        return SessionClient(self, session["bank_id"], session["parent_id"])


class SessionClient:
    def __init__(self, pool, bank_id, parent_id):
        self.pool, self.bank_id, self.parent_id = pool, bank_id, parent_id

    def _trace(self, event, **data):
        self.pool.store.trace({"timestamp": datetime.now(timezone.utc).isoformat(), "event": event,
                               "bank_id": self.bank_id, "parent_id": self.parent_id, **data})

    def __call__(self, messages, max_tokens, schema, stage):
        import httpx
        settings = self.pool.settings
        request = {"model": settings.get("model", "agmem-27b"), "messages": messages,
                   "max_tokens": max_tokens, "temperature": 0, "seed": settings.get("seed", 42),
                   "chat_template_kwargs": {"enable_thinking": False}}
        if settings.get("structured_outputs", True):
            request["structured_outputs"] = {"json": guidance_schema(schema)}
        request_hash = digest(request)
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(settings.get("api_key_env", "AGMEM_API_KEY"), "")
        if key:
            headers["Authorization"] = "Bearer " + key
        self._trace("writer_request", stage=stage, request_sha256=request_hash, request=request)
        for attempt in range(settings.get("network_retries", 3) + 1):
            endpoint, client = self.pool.slots.get()
            retryable, status_code = True, None
            try:
                response = client.post(endpoint + "/chat/completions", json=request, headers=headers)
                status_code = response.status_code
                if status_code != 200:
                    retryable = status_code == 429 or status_code >= 500
                    raise InfrastructureError(f"Writer HTTP {status_code}")
                payload = response.json()
                choice = payload["choices"][0]
                result = {"text": choice["message"].get("content"), "finish_reason": choice.get("finish_reason"),
                          "usage": payload.get("usage", {})}
                self._trace("writer_response", stage=stage, attempt=attempt + 1,
                            request_sha256=request_hash, response=result)
                return result
            except (httpx.HTTPError, InfrastructureError, ValueError, KeyError, IndexError, TypeError) as exc:
                will_retry = retryable and attempt < settings.get("network_retries", 3)
                self._trace("writer_transport_error", stage=stage, attempt=attempt + 1,
                            request_sha256=request_hash, error_type=type(exc).__name__,
                            status_code=status_code, retryable=retryable, will_retry=will_retry)
                if not will_retry:
                    raise InfrastructureError(f"Writer transport failed ({type(exc).__name__}); inspect writer_traces") from None
            finally:
                self.pool.slots.put((endpoint, client))
            time.sleep(min(2 ** attempt, 8))

    def accept(self, messages, max_tokens, schema, stage, result, status="complete"):
        self._trace("writer_validated", stage=stage, status=status,
                    response_sha256=digest(result), messages_sha256=digest(messages), max_tokens=max_tokens)

    def reject(self, messages, max_tokens, schema, stage, result, reason):
        self._trace("writer_validation_failed", stage=stage, reason=str(reason),
                    response_sha256=digest(result), messages_sha256=digest(messages), max_tokens=max_tokens)
