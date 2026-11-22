"""Round-robin embedding replicas sharing the existing cache and request bound."""

from threading import Lock
from urllib.parse import urlsplit

from .embedding import EmbeddingClient, embedding_settings


def embedding_endpoints(endpoints):
    """Validate every replica before opening a client or a cache database."""
    if isinstance(endpoints, str) or not endpoints:
        raise ValueError("Provide a nonempty sequence of embedding endpoint URLs")
    result = []
    for endpoint in endpoints:
        if not isinstance(endpoint, str):
            raise ValueError("Embedding endpoint must be a URL")
        try:
            parsed = urlsplit(endpoint)
            port = parsed.port
        except ValueError:
            raise ValueError("Invalid embedding endpoint URL") from None
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or port == 0
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError("Embedding endpoint must not contain credentials/query/fragment")
        result.append(endpoint.rstrip("/"))
    if len(set(result)) != len(result):
        raise ValueError("Embedding replica URLs must be distinct")
    return tuple(result)


class RoundRobinEmbeddingTransport:
    """Route each HTTP attempt, including retries, to the next replica.

    One HTTP connection pool serves all replicas. The parent EmbeddingClient's
    semaphore limits total in-flight requests across the entire pool.
    """

    def __init__(self, endpoints, *, timeout, client=None):
        import httpx
        self.endpoints = embedding_endpoints(endpoints)
        self.lock, self.next_index, self.closed = Lock(), 0, False
        self._owns_client = client is None
        self.client = client if client is not None else httpx.Client(timeout=timeout, trust_env=False)

    def post(self, url, **kwargs):
        with self.lock:
            if self.closed:
                raise RuntimeError("Embedding transport is closed")
            endpoint = self.endpoints[self.next_index]
            self.next_index = (self.next_index + 1) % len(self.endpoints)
        return self.client.post(endpoint + "/embeddings", **kwargs)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self._owns_client:
                self.client.close()


class MultiEndpointEmbeddingClient(EmbeddingClient):
    """Keep base identity, tokenization, validation, cache and coalescing intact."""

    def __init__(self, endpoints, tokenizer, *, cache_path, identity, settings=None, transport=None):
        values = embedding_settings(settings)
        self.endpoints = embedding_endpoints(endpoints)
        self.replica_transport = RoundRobinEmbeddingTransport(
            self.endpoints, timeout=values["timeout_seconds"], client=transport)
        try:
            super().__init__(self.endpoints[0], tokenizer, cache_path=cache_path,
                             identity=identity, settings=values, transport=self.replica_transport)
        except BaseException:
            self.replica_transport.close()
            raise

    def close(self):
        # The base does not own an injected transport. Close ours explicitly,
        # but only after the base has verified there are no in-flight requests.
        with self.lock:
            super().close()
            self.replica_transport.close()
