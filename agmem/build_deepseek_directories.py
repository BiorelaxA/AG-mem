"""Fresh LongMemEval-M topic/event directories using DeepSeek and local embeddings.

The immutable legacy build engine supplies segmentation, semantic validation,
source binding and journaling. Only this dedicated worker adapts its globals;
old Qwen journals and implementation fingerprints are never modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import copy, deepcopy
import json
import os
from pathlib import Path
from threading import RLock
from urllib.parse import urlsplit

from . import build_adjacent_directories as engine
from .build_topic_event_directories import BOUNDARY_PROMPT, BOUNDARY_PROMPT_VERSION, TopicEventDirectoryWriter
from .core import InfrastructureError, file_digest
from .hierarchical.deepseek_client import DeepSeekWriterPool, schema_messages
from .hierarchical.embedding_pool import MultiEndpointEmbeddingClient, embedding_endpoints
from .hierarchical.segmentation import AdjacentSegmenter
from .hierarchical.writer import chat_token_count


DEFAULT_CONFIG = "configs/build_longmemeval_m_deepseek.yaml"
PROTOCOL = "adjacent_topic_event_deepseek_v1"
_BASE_LOAD_RECIPE = engine.load_recipe
_BASE_BOUNDED_RESULTS = engine.bounded_results
_LOCK = RLock()


def remote_endpoint(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Configure a nonempty DeepSeek base URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid DeepSeek base URL") from None
    if (parsed.scheme not in {"https", "http"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or "?" in value or "#" in value or port == 0
            or any(c.isspace() or ord(c) < 32 for c in value)):
        raise ValueError("DeepSeek base URL must not contain credentials, query, fragment or whitespace")
    return value.rstrip("/")


def _check_fresh_or_matching_root(root):
    """Only this new protocol may resume; no implicit Qwen result/cache import."""
    root = Path(root)
    if root.exists() and not root.is_dir():
        raise ValueError("DeepSeek output root must be a directory")
    if not root.exists() or not any(root.iterdir()):
        return
    manifest = root / "run_manifest.json"
    if not manifest.is_file():
        raise ValueError("Nonempty output root has no DeepSeek build manifest; choose a fresh output root")
    if json.loads(manifest.read_text()).get("protocol") != PROTOCOL:
        raise ValueError("DeepSeek construction cannot reuse or overwrite a different build protocol")


def _fail_fast_results(function, inputs, workers):
    """Commit the failing session then stop admission after exhausted API retries.

Otherwise the legacy per-session error handler would continue across the full
dataset with a bad API key or exhausted credit. Existing in-flight calls drain
through the bounded scheduler before its transport/cache is closed.
"""
    results = _BASE_BOUNDED_RESULTS(function, inputs, workers)
    try:
        for item, result in results:
            yield item, result
            if result.get("error_type") == "InfrastructureError":
                raise InfrastructureError("Directory service failed after retries; inspect writer traces, "
                                          "fix the service, and rerun to resume this DeepSeek journal")
    finally:
        close = getattr(results, "close", None)
        if close:
            close()


@contextmanager
def _deepseek_engine():
    with _LOCK:
        old = engine.PROTOCOL, engine.DATASETS, engine.load_recipe, engine.bounded_results
        engine.PROTOCOL, engine.DATASETS = PROTOCOL, {"split_longmemeval_m"}
        engine.load_recipe, engine.bounded_results = load_recipe, _fail_fast_results
        try:
            yield
        finally:
            engine.PROTOCOL, engine.DATASETS, engine.load_recipe, engine.bounded_results = old


def load_recipe(path, workers=None, *, embedding_model_path=None, output_root=None,
                allow_missing_embedding=False):
    with _deepseek_engine():
        config, identity = _BASE_LOAD_RECIPE(path, workers,
            embedding_model_path=embedding_model_path, output_root=output_root,
            allow_missing_embedding=allow_missing_embedding)
    if config.get("preserve_existing_directories", False) or config.get("reuse_root"):
        raise ValueError("This DeepSeek experiment rebuilds M from source; Qwen directory reuse is disabled")
    if any(spec["name"] != "split_longmemeval_m" for spec in config["datasets"]):
        raise ValueError("This recipe is for LongMemEval-M only")
    service = config.setdefault("service", {})
    if service.get("model", "deepseek-flash") != "deepseek-flash":
        raise ValueError("This versioned construction recipe requires model=deepseek-flash")
    service["model"] = "deepseek-flash"
    service.setdefault("api_key_env", "DEEPSEEK_API_KEY")
    service["base_url"] = remote_endpoint(os.environ.get("DEEPSEEK_BASE_URL")
                                           or service.get("base_url", "https://api.deepseek.com"))
    _check_fresh_or_matching_root(config["output_root"])
    identity = deepcopy(identity)
    identity["writer"] = {
        "provider": "deepseek", "model": service["model"],
        "model_revision": "provider-alias:deepseek-flash",
        "revision_verification": "API alias; immutable remote weight revision is not exposed",
        "model_version_documented": "DeepSeek-V4.1-Flash-1017 (docs checked 2026-11-17)",
        "base_url": service["base_url"],
    }
    identity["budget_tokenizer"] = {
        "tokenizer_id": identity["tokenizer_id"],
        "path": config["model_path"],
        "purpose": "Reference Qwen token budgets and directory compatibility; not DeepSeek billing tokens",
        "local_writer_weights_loaded": False,
        "schema_budget": "count full transport schema in reference-token context checks",
        "trace_input_tokens": "legacy Writer traces count pre-schema reference tokens; HTTP traces include the full request and provider usage",
    }
    identity["boundary_prompt_version"] = BOUNDARY_PROMPT_VERSION
    identity["prompt_hashes"]["boundary.txt"] = file_digest(BOUNDARY_PROMPT)
    identity["request_options"] = {
        "temperature": 0, "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"}, "stream": False,
        "schema_validation": "local JSON Schema and semantic validation",
        "seed": "not sent; remote API does not promise seeded reproducibility",
    }
    identity["directory_reuse_policy"] = "fresh_deepseek_root_only.v1"
    for resource in [Path(__file__), Path(__file__).with_name("build_topic_event_directories.py"),
                     Path(__file__).with_name("hierarchical") / "deepseek_client.py",
                     Path(__file__).with_name("hierarchical") / "embedding_pool.py"]:
        identity["implementation_hashes"][resource.name] = file_digest(resource)
    return config, identity


class DeepSeekDirectoryWriter(TopicEventDirectoryWriter):
    """Same source/summary prompts, with the API schema included in input budgets."""

    def _available(self, messages):
        sizes = [chat_token_count(self.tokenizer, schema_messages(messages, self._schema(kind)))
                 for kind in ("boundary", "child", "parent")]
        return self.settings["context_tokens"] - self.settings["safety_margin"] - max(sizes)


class _BoundaryServiceFailure(BaseException):
    """Internal control signal through the legacy boundary review catch-all."""

    def __init__(self, cause):
        self.cause = cause


class DeepSeekSegmenter(AdjacentSegmenter):
    """Never turn an exhausted remote API failure into a semantic fallback."""

    def __init__(self, encoder, *, settings=None, review_boundary=None, modality_policy="existing_captions"):
        def review(payload):
            try:
                return review_boundary(payload)
            except InfrastructureError as exc:
                # The legacy segmenter treats ordinary review exceptions as
                # unavailable hints. Service failure must instead stop this
                # session immediately, then reach the journal as an error.
                raise _BoundaryServiceFailure(exc) from None
        super().__init__(encoder, settings=settings,
                         review_boundary=review if review_boundary is not None else None,
                         modality_policy=modality_policy)

    def segment(self, session):
        try:
            return super().segment(session)
        except _BoundaryServiceFailure as failure:
            raise InfrastructureError(str(failure.cause)) from None


def run(args, **factories):
    config, identity = load_recipe(args.config, args.workers,
        embedding_model_path=getattr(args, "embedding_model_path", None),
        output_root=getattr(args, "output_root", None), allow_missing_embedding=args.dry_run)
    endpoints = getattr(args, "embedding_base_urls", None)
    if endpoints:
        endpoints = embedding_endpoints(endpoints)
    elif not args.dry_run:
        raise ValueError("Provide --embedding-base-urls or use the two-embedding-GPU launcher")
    planned = copy(args)
    planned.base_urls, planned.base_url = [config["service"]["base_url"]], None
    planned.embedding_base_url = endpoints[0] if endpoints else None
    if factories.get("pool_factory") is None:
        factories["pool_factory"] = DeepSeekWriterPool
    if factories.get("writer_factory") is None:
        factories["writer_factory"] = DeepSeekDirectoryWriter
    if factories.get("segmenter_factory") is None:
        factories["segmenter_factory"] = DeepSeekSegmenter
    if factories.get("embedding_factory") is None:
        def create_encoder(_base_url, tokenizer, **kwargs):
            return MultiEndpointEmbeddingClient(endpoints, tokenizer, **kwargs)
        factories["embedding_factory"] = create_encoder
    engine.announce("deepseek_directory_execution", writer=identity["writer"],
                    writer_concurrency=config["workers"], embedding_replicas=len(endpoints or ()),
                    embedding_concurrency=config["embedding"]["workers"],
                    budget_tokenizer=identity["budget_tokenizer"],
                    reuse_existing_qwen_results=False, output_root=config["output_root"])
    with _deepseek_engine():
        return engine.run(planned, **factories)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--datasets", nargs="+", choices=["split_longmemeval_m"])
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"])
    parser.add_argument("--embedding-base-urls", nargs="+")
    parser.add_argument("--embedding-model-path")
    parser.add_argument("--output-root")
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    run(argument_parser().parse_args())


if __name__ == "__main__":
    main()
