"""Bounded embedding requests and a cache independent of the build journal.

The client never truncates a source message. Long inputs are divided at exact
character boundaries, embedded in full, and pooled back into one message vector.
All text preparation and pooling options are part of the cache namespace.
"""

from concurrent.futures import Future
import json
import math
import os
from pathlib import Path
import sqlite3
from threading import BoundedSemaphore, RLock
import time
from urllib.parse import urlsplit

from ..core import InfrastructureError, canonical, digest


REPRESENTATION_VERSION = "directory-embedding.v1"
DEFAULTS = {
    "model": "agmem-embedding",
    "instruction": ("Represent this conversation passage for grouping adjacent passages about the same "
                    "local topic or information need. Preserve the speaker's intent, personal facts, "
                    "constraints and corrections."),
    "max_input_tokens": 4096,
    "token_margin": 8,
    "batch_size": 32,
    "max_batch_tokens": 16384,
    "workers": 4,
    "timeout_seconds": 120,
    "network_retries": 3,
    "expected_dimensions": 2560,
    "api_key_env": "AGMEM_API_KEY",
}


def embedding_settings(supplied=None):
    values = {**DEFAULTS, **(supplied or {})}
    unknown = values.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown embedding settings: {sorted(unknown)}")
    for key in ("max_input_tokens", "batch_size", "max_batch_tokens", "workers", "expected_dimensions"):
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError(f"Embedding {key} must be a positive integer")
    for key in ("token_margin", "network_retries"):
        if type(values[key]) is not int or values[key] < 0:
            raise ValueError(f"Embedding {key} must be a nonnegative integer")
    if (isinstance(values["timeout_seconds"], bool)
            or not isinstance(values["timeout_seconds"], (int, float))
            or not math.isfinite(values["timeout_seconds"]) or values["timeout_seconds"] <= 0):
        raise ValueError("Embedding timeout_seconds must be finite and positive")
    for key in ("model", "instruction", "api_key_env"):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f"Embedding {key} must be nonempty text")
    if values["token_margin"] >= values["max_input_tokens"]:
        raise ValueError("Embedding token_margin leaves no input capacity")
    if values["max_batch_tokens"] < values["max_input_tokens"]:
        raise ValueError("Embedding max_batch_tokens must fit one maximum-size input")
    return values


def normalized_vector(value, dimensions=None):
    if (not isinstance(value, (list, tuple)) or not value
            or dimensions is not None and len(value) != dimensions
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                   for v in value)):
        raise ValueError("Embedding response contains an invalid vector or dimension")
    norm = math.hypot(*value)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding response contains a zero or nonfinite norm")
    return [float(v) / norm for v in value]


class EmbeddingClient:
    """Thread-safe /embeddings client with same-key request coalescing.

    ``identity`` must identify the actual model weights/tokenizer used by the
    service; the served model name alone cannot distinguish weight revisions.
    The coordinator supplies that identity from local artifacts. Endpoint and
    concurrency changes intentionally do not invalidate cached vectors.
    """

    def __init__(self, base_url, tokenizer, *, cache_path, identity, settings=None, transport=None):
        import httpx
        self.settings = embedding_settings(settings)
        if not isinstance(base_url, str):
            raise ValueError("Embedding endpoint must be a URL")
        try:
            parsed = urlsplit(base_url)
            port = parsed.port
        except ValueError:
            raise ValueError("Invalid embedding endpoint URL") from None
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or port == 0
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError("Embedding endpoint must not contain credentials/query/fragment")
        if not isinstance(identity, dict) or not identity:
            raise ValueError("Provide a nonempty actual embedding model/tokenizer identity")
        self.base_url, self.tokenizer = base_url.rstrip("/"), tokenizer
        self.lock, self.tokenizer_lock = RLock(), RLock()
        self.slots = BoundedSemaphore(self.settings["workers"])
        self.pending = {}
        self.stats = {"requests": 0, "retries": 0, "cache_hits": 0, "coalesced": 0,
                      "texts": 0, "chunks": 0}
        preparation = {key: self.settings[key] for key in
                       ("model", "instruction", "max_input_tokens", "token_margin", "expected_dimensions")}
        self.namespace = digest({"version": REPRESENTATION_VERSION, "identity": identity,
                                 "preparation": preparation, "pooling": "equal_chunk_mean_then_l2"})
        self.client = transport or httpx.Client(timeout=self.settings["timeout_seconds"], trust_env=False)
        self._owns_client = transport is None
        self.closed = False
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
            self.db.execute("PRAGMA journal_mode=DELETE")
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.db.commit()
        except Exception:
            if hasattr(self, "db"):
                self.db.close()
            if self._owns_client:
                self.client.close()
            raise

    def close(self):
        with self.lock:
            if self.closed:
                return
            if self.pending:
                raise RuntimeError("Cannot close embedding client while requests are in flight")
            self.closed = True
            self.db.close()
            if self._owns_client:
                self.client.close()

    def _prepare(self, text):
        return f"Instruct: {self.settings['instruction']}\nQuery: {text}"

    def _tokens(self, text):
        with self.tokenizer_lock:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not isinstance(ids, list) or any(type(i) is not int for i in ids):
            raise TypeError("Embedding tokenizer must return a flat list of token IDs")
        return len(ids)

    def _chunks(self, text):
        """Return all source characters exactly once; only representation splits."""
        available = self.settings["max_input_tokens"] - self.settings["token_margin"]
        if self._tokens(self._prepare("")) >= available:
            raise ValueError("Embedding instruction leaves no room for text")
        if self._tokens(self._prepare(text)) <= available:
            return [text]
        chunks, start = [], 0
        while start < len(text):
            # Binary search yields a fitting prefix. Token length is not strictly
            # monotonic in characters, so this need not be the longest prefix;
            # every chosen prefix is explicitly checked and no text is dropped.
            low, high, fit = 1, len(text) - start, 0
            while low <= high:
                mid = (low + high) // 2
                if self._tokens(self._prepare(text[start:start + mid])) <= available:
                    fit, low = mid, mid + 1
                else:
                    high = mid - 1
            if not fit:
                # A boundary inside an unusual Unicode sequence may tokenize
                # less efficiently, but an impossible one-character input must
                # be reported rather than silently clipped.
                raise ValueError("Embedding budget cannot fit one source character")
            chunk = text[start:start + fit]
            if self._tokens(self._prepare(chunk)) > available:
                raise AssertionError("Embedding chunk budget invariant failed")
            chunks.append(chunk)
            start += fit
        if "".join(chunks) != text:
            raise AssertionError("Embedding chunks lost source characters")
        return chunks

    def _batches(self, prepared):
        batch, total = [], 0
        for text in prepared:
            tokens = self._tokens(text) + self.settings["token_margin"]
            if tokens > self.settings["max_input_tokens"]:
                raise ValueError("Oversized embedding input; truncation is not permitted")
            if batch and (len(batch) >= self.settings["batch_size"]
                          or total + tokens > self.settings["max_batch_tokens"]):
                yield batch
                batch, total = [], 0
            batch.append(text)
            total += tokens
        if batch:
            yield batch

    def _response_vectors(self, payload, size):
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("Embedding response is missing data")
        rows = payload["data"]
        if len(rows) != size:
            raise ValueError("Embedding response count differs from request")
        result = [None] * size
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Invalid embedding response row")
            index = row.get("index")
            if type(index) is not int or not 0 <= index < size or result[index] is not None:
                raise ValueError("Embedding response indices must cover the request exactly once")
            result[index] = normalized_vector(row.get("embedding"), self.settings["expected_dimensions"])
        return result

    def _request(self, batch):
        import httpx
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.settings["api_key_env"], "")
        if key:
            headers["Authorization"] = "Bearer " + key
        request = {"model": self.settings["model"], "input": batch, "encoding_format": "float"}
        for attempt in range(self.settings["network_retries"] + 1):
            retryable, status_code = True, None
            try:
                with self.slots:
                    with self.lock:
                        self.stats["requests"] += 1
                    response = self.client.post(self.base_url + "/embeddings", json=request, headers=headers)
                    status_code = response.status_code
                    if response.status_code != 200:
                        retryable = response.status_code == 429 or response.status_code >= 500
                        raise InfrastructureError(f"Embedding HTTP {response.status_code}")
                    return self._response_vectors(response.json(), len(batch))
            except (httpx.HTTPError, InfrastructureError, ValueError, KeyError, IndexError, TypeError) as exc:
                if not retryable or attempt == self.settings["network_retries"]:
                    status = f", HTTP {status_code}" if status_code is not None else ""
                    raise InfrastructureError(f"Embedding request failed ({type(exc).__name__}{status}); "
                                              "no vector was cached") from None
                with self.lock:
                    self.stats["retries"] += 1
                time.sleep(min(2 ** attempt, 8))

    def _encode_owned(self, owned):
        prepared, positions = [], {}
        for key, text in owned:
            chunks = self._chunks(text)
            positions[key] = (len(prepared), len(chunks))
            prepared.extend(self._prepare(chunk) for chunk in chunks)
        with self.lock:
            self.stats["texts"] += len(owned)
            self.stats["chunks"] += len(prepared)
        vectors = []
        for batch in self._batches(prepared):
            vectors.extend(self._request(batch))
        results = {}
        for key, _ in owned:
            start, count = positions[key]
            selected = vectors[start:start + count]
            pooled = [sum(values) / count for values in zip(*selected)]
            results[key] = normalized_vector(pooled, self.settings["expected_dimensions"])
        return results

    def encode(self, texts):
        if not isinstance(texts, (list, tuple)) or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Embedding inputs must be a sequence of nonempty strings")
        if not texts:
            return []
        keys = [digest({"namespace": self.namespace, "text": text}) for text in texts]
        owned, futures, cached = [], {}, {}
        # Claim this call's complete key set atomically. Owners finish their own
        # work before waiting on other calls, preventing cross-call deadlocks.
        with self.lock:
            if self.closed:
                raise RuntimeError("Embedding client is closed")
            missing = []
            for key, text in dict(zip(keys, texts)).items():
                row = self.db.execute("SELECT value FROM vectors WHERE key=?", (key,)).fetchone()
                if row:
                    try:
                        cached[key] = normalized_vector(json.loads(row[0]), self.settings["expected_dimensions"])
                    except (ValueError, TypeError):
                        raise ValueError("Invalid cached embedding; inspect or remove its separate cache") from None
                    self.stats["cache_hits"] += 1
                else:
                    missing.append((key, text))
            # Validate all existing rows before creating any promises, so a
            # corrupt cached row cannot strand a newly claimed missing key.
            for key, text in missing:
                if key in self.pending:
                    futures[key] = self.pending[key]
                    self.stats["coalesced"] += 1
                else:
                    futures[key] = self.pending[key] = Future()
                    owned.append((key, text))
        if owned:
            try:
                generated = self._encode_owned(owned)
                with self.lock:
                    with self.db:
                        self.db.executemany("INSERT OR REPLACE INTO vectors VALUES (?,?)",
                                            [(key, canonical(vector)) for key, vector in generated.items()])
                    for key, _ in owned:
                        self.pending.pop(key).set_result(generated[key])
            except BaseException as exc:
                with self.lock:
                    for key, _ in owned:
                        self.pending.pop(key).set_exception(exc)
                raise
        return [list(cached[key] if key in cached else futures[key].result()) for key in keys]
