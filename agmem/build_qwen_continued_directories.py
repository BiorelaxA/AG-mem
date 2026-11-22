"""Continue the stopped DeepSeek M build with a local Qwen Writer.

Committed source directories keep their original model and version. New work
uses the same segmentation and source/summary prompts, in an independent journal.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock

from . import build_adjacent_directories as engine
from .build_topic_event_directories import BOUNDARY_PROMPT, BOUNDARY_PROMPT_VERSION, TopicEventDirectoryWriter
from .core import InfrastructureError, file_digest
from .hierarchical.segmentation import AdjacentSegmenter


DEFAULT_CONFIG = "configs/build_longmemeval_m_qwen_continue.yaml"
PROTOCOL = "adjacent_topic_event_qwen_continuation_v1"
_BASE_LOAD_RECIPE = engine.load_recipe
_BASE_BOUNDED_RESULTS = engine.bounded_results
_LOCK = RLock()


def _fail_fast_results(function, inputs, workers):
    results = _BASE_BOUNDED_RESULTS(function, inputs, workers)
    try:
        for item, result in results:
            yield item, result
            if result.get("error_type") == "InfrastructureError":
                raise InfrastructureError("Local directory service failed after retries; inspect this "
                                          "job's service logs and rerun the Qwen continuation script")
    finally:
        results.close()


@contextmanager
def _qwen_engine():
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
    with _qwen_engine():
        config, original = _BASE_LOAD_RECIPE(path, workers,
            embedding_model_path=embedding_model_path, output_root=output_root,
            allow_missing_embedding=allow_missing_embedding)
    if config.get("retry_fallback") is not False:
        raise ValueError("Continuation must preserve existing fallback results: retry_fallback=false")
    if config.get("preserve_existing_directories") or config.get("reuse_root"):
        raise ValueError("Use source_root for the explicit DeepSeek import")
    if not config.get("source_root"):
        raise ValueError("Configure the stopped DeepSeek source_root")
    if any(spec["name"] != "split_longmemeval_m" for spec in config["datasets"]):
        raise ValueError("This continuation recipe is for LongMemEval-M only")
    config["source_root"] = str(Path(config["source_root"]).resolve())
    service = config.setdefault("service", {})
    if service.get("model", "agmem-27b") != "agmem-27b" or service.get("base_url"):
        raise ValueError("Use the local agmem-27b Writer; endpoints are supplied by the launcher")
    if service.get("api_key_env", "AGMEM_API_KEY") != "AGMEM_API_KEY":
        raise ValueError("The local Writer must not use a remote provider API key")
    service["model"] = "agmem-27b"
    identity = deepcopy(original)
    identity["boundary_prompt_version"] = BOUNDARY_PROMPT_VERSION
    identity["prompt_hashes"]["boundary.txt"] = file_digest(BOUNDARY_PROMPT)
    identity["directory_reuse_policy"] = "preserve_deepseek_commits_then_local_qwen.v1"
    for resource in (Path(__file__), Path(__file__).with_name("build_topic_event_directories.py"),
                     Path(__file__).with_name("import_deepseek_directories.py")):
        identity["implementation_hashes"][resource.name] = file_digest(resource)
    return config, identity


class _BoundaryServiceFailure(BaseException):
    def __init__(self, cause):
        self.cause = cause


class QwenContinuationSegmenter(AdjacentSegmenter):
    """Service failure must stop, rather than become a semantic boundary fallback."""

    def __init__(self, encoder, *, settings=None, review_boundary=None, modality_policy="existing_captions"):
        def review(payload):
            try:
                return review_boundary(payload)
            except InfrastructureError as exc:
                raise _BoundaryServiceFailure(exc) from None
        super().__init__(encoder, settings=settings,
                         review_boundary=review if review_boundary is not None else None,
                         modality_policy=modality_policy)

    def segment(self, session):
        try:
            return super().segment(session)
        except _BoundaryServiceFailure as exc:
            raise InfrastructureError(str(exc.cause)) from None


def run(args, **factories):
    from .import_deepseek_directories import import_committed

    config, identity = load_recipe(args.config, args.workers,
        embedding_model_path=getattr(args, "embedding_model_path", None),
        output_root=getattr(args, "output_root", None), allow_missing_embedding=args.dry_run)
    source_root = getattr(args, "source_root", None) or config["source_root"]
    engine.plan(config, identity, args.datasets, args.splits)
    count_tokens = None
    if not args.dry_run:
        if not getattr(args, "import_only", False) and (
                not (args.base_urls or getattr(args, "base_url", None))
                or not getattr(args, "embedding_base_url", None)):
            raise ValueError("Provide local Writer and embedding endpoints, or use the GPU launcher")
        from .hierarchical.writer import token_count
        from .policy import load_tokenizer

        tokenizer = factories.get("tokenizer")
        if tokenizer is None:
            tokenizer = load_tokenizer(config["model_path"])
            factories["tokenizer"] = tokenizer
        count_tokens = lambda text: token_count(tokenizer, text)
    imported = import_committed(source_root, config["output_root"], identity,
                                dry_run=args.dry_run, count_tokens=count_tokens)
    engine.announce("qwen_continuation_import", **{key: imported[key] for key in (
        "version", "source_root", "output_root", "status", "source_states", "source_banks",
        "registered_sessions", "counters", "import_already_complete", "excluded_artifacts") if key in imported})
    if getattr(args, "import_only", False):
        return imported
    factories.setdefault("writer_factory", TopicEventDirectoryWriter)
    factories.setdefault("segmenter_factory", QwenContinuationSegmenter)
    with _qwen_engine():
        return engine.run(args, **factories)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source-root")
    parser.add_argument("--output-root")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--datasets", nargs="+", choices=["split_longmemeval_m"])
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"])
    urls = parser.add_mutually_exclusive_group()
    urls.add_argument("--base-url")
    urls.add_argument("--base-urls", nargs="+")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-model-path")
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--import-only", action="store_true",
                        help="Validate and import committed DeepSeek directories without GPU inference")
    return parser


def main():
    run(argument_parser().parse_args())


if __name__ == "__main__":
    main()
