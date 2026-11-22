"""Adjacent semantic segmentation, then child summaries, then source-grounded parents."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from threading import Lock
import time

from .core import InfrastructureError, canonical, digest, file_digest
from .execution import bounded_results
from .jobs import file_lock
from .pipeline import announce
from .hierarchical.adapters import input_files, iter_banks
from .hierarchical.binding import bind_directory, validate_directory
from .hierarchical.client import WriterPool
from .hierarchical.records import validate_session
from .hierarchical.storage import BuildStore


DEFAULT_CONFIG = "configs/build_adjacent_qwen_locomo_lmes.yaml"
PROTOCOL = "adjacent_parent_child_directory_v1"
PROMPT_VERSION = "directory-writer.adjacent.v1"
DATASETS = {"locomo", "split_longmemeval_s"}
EMBEDDING_MODELS = {
    "Qwen/Qwen3-Embedding-4B": ("../model/qwen3-embedding-4b", 2560),
    "Qwen/Qwen3-Embedding-0.6B": ("models/Qwen3-Embedding-0.6B", 1024),
}


def _model_identity(path, *, allow_missing=False):
    from .policy import model_fingerprint
    root = Path(path)
    try:
        fingerprint = model_fingerprint(root)
    except (FileNotFoundError, ValueError):
        if allow_missing:
            return {"ready": False, "path": str(root), "reason": "missing_or_incomplete_local_model"}, None
        raise ValueError(f"Missing or incomplete local model: {root}. For the embedding model, run "
                         "bash scripts/download_directory_embedding.sh on a network-connected node.") from None
    if not all((root / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")):
        if allow_missing:
            return {"ready": False, "path": str(root), "reason": "missing_local_tokenizer"}, None
        raise ValueError(f"Missing local tokenizer files: {root}")
    tokenizer_files = {name: value for name, value in fingerprint["files"].items()
                       if "tokenizer" in name or name in {"special_tokens_map.json", "vocab.json", "added_tokens.json"}}
    for name in ("chat_template.jinja", "merges.txt", "tokenizer.model", "spiece.model"):
        resource = root / name
        if resource.is_file():
            tokenizer_files[name] = {"bytes": resource.stat().st_size, "sha256": file_digest(resource)}
    if not tokenizer_files:
        raise ValueError(f"Missing tokenizer provenance: {root}")
    # Qwen3-Embedding's sentence-transformer pooling metadata can live under
    # 1_Pooling/. Include it even though the ordinary model fingerprint is flat.
    nested = {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*.json"))
              if p.parent != root and ".cache" not in p.relative_to(root).parts}
    revision = digest({"base": fingerprint["fingerprint"], "nested_config": nested}) if nested else fingerprint["fingerprint"]
    return {"ready": True, "model_revision": revision}, "qwen-tokenizer@" + digest(tokenizer_files)


def load_recipe(path, workers=None, *, embedding_model_path=None, output_root=None, allow_missing_embedding=False):
    import yaml
    from .hierarchical.embedding import embedding_settings
    from .hierarchical.segmentation import segmentation_settings
    from .hierarchical.writer import prompt_hashes, writer_settings

    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or config.get("protocol") != PROTOCOL:
        raise ValueError(f"Use a {PROTOCOL} recipe")
    for section in ("service", "embedding"):
        if any(key in config.get(section, {}) for key in ("api_key", "authorization")):
            raise ValueError("Pass credentials through environment variables")
    config["model_path"] = str(Path(config["model_path"]).resolve())
    config["output_root"] = str(Path(output_root or config["output_root"]).resolve())
    config["workers"] = workers if workers is not None else config.get("workers", 64)
    if type(config["workers"]) is not int or config["workers"] < 1:
        raise ValueError("workers must be a positive integer")
    if type(config.get("retry_fallback", True)) is not bool:
        raise ValueError("retry_fallback must be boolean")
    config["writer"] = writer_settings(config.get("writer"))
    config["segmentation"] = segmentation_settings(config.get("segmentation"))
    embed = deepcopy(config.get("embedding", {}))
    model_id = embed.pop("model_id", "Qwen/Qwen3-Embedding-4B")
    if model_id not in EMBEDDING_MODELS:
        raise ValueError(f"Unsupported embedding model; choose one of {sorted(EMBEDDING_MODELS)}")
    default_path, native_dimensions = EMBEDDING_MODELS[model_id]
    configured_path = embed.pop("model_path", default_path)
    local_embed = str(Path(embedding_model_path or configured_path).expanduser().resolve())
    embed.setdefault("expected_dimensions", native_dimensions)
    if embed["expected_dimensions"] != native_dimensions:
        raise ValueError(f"{model_id} requires expected_dimensions={native_dimensions}; "
                         "this pipeline uses full-size embeddings")
    config["embedding"] = {"model_id": model_id, "model_path": local_embed, **embedding_settings(embed)}
    service = config.get("service", {})
    if type(service.get("network_retries", 3)) is not int or service.get("network_retries", 3) < 0:
        raise ValueError("network_retries must be a nonnegative integer")
    if type(service.get("structured_outputs", True)) is not bool:
        raise ValueError("structured_outputs must be boolean")
    specs = config.get("datasets", [])
    if (not specs or len({s["name"] for s in specs}) != len(specs)
            or not {s["name"] for s in specs}.issubset(DATASETS)):
        raise ValueError("Configure unique LoCoMo / LongMemEval-S datasets; LongMemEval-M is excluded")
    writer_id, tokenizer_id = _model_identity(config["model_path"])
    encoder_id, encoder_tokenizer = _model_identity(local_embed, allow_missing=allow_missing_embedding)
    if encoder_id["ready"]:
        encoder_config = json.loads((Path(local_embed) / "config.json").read_text())
        if encoder_config.get("hidden_size") != native_dimensions:
            raise ValueError(f"Local embedding hidden_size does not match {model_id} ({native_dimensions}); "
                             "check model_id and model_path")
        pooling_path = Path(local_embed) / "1_Pooling" / "config.json"
        if pooling_path.is_file():
            pooling_config = json.loads(pooling_path.read_text())
            if pooling_config.get("word_embedding_dimension") != native_dimensions:
                raise ValueError("Local embedding pooling dimension does not match the selected model")
    resources = Path(__file__).with_name("hierarchical") / "resources" / "adjacent"
    prompts = {p.name: file_digest(p) for p in sorted(resources.iterdir()) if p.is_file()}
    if not prompts:
        raise ValueError("Missing adjacent-directory prompt resources")
    implementation_files = [Path(__file__), *[Path(__file__).with_name("hierarchical") / name for name in
                            ("segmentation.py", "embedding.py", "adjacent_writer.py", "records.py", "binding.py",
                             "writer.py", "client.py", "adapters.py")]]
    sources = [{"spec": spec, "files": [{"split": split, "path": str(p), "sha256": file_digest(p)}
                                       for split, p in input_files(spec)]} for spec in specs]
    # Execution concurrency, service URLs, HTTP retry counts and batch sizes may
    # change on resume; model/text transformations and segmentation thresholds may not.
    encoder_semantics = {key: value for key, value in config["embedding"].items()
                         if key not in {"model_path", "workers", "batch_size", "max_batch_tokens",
                                        "timeout_seconds", "network_retries", "api_key_env"}}
    identity = {"protocol": PROTOCOL, "datasets": sources,
                "writer": {"model": Path(config["model_path"]).name, "model_revision": writer_id["model_revision"]},
                "tokenizer_id": tokenizer_id, "writer_settings": config["writer"],
                "embedding": {**encoder_id, "tokenizer_id": encoder_tokenizer, "settings": encoder_semantics},
                "segmentation": config["segmentation"], "prompt_hashes": prompts,
                "base_resource_hashes": prompt_hashes(),
                "implementation_hashes": {p.name: file_digest(p) for p in implementation_files},
                "prompt_version": PROMPT_VERSION, "block_policy": "block.v1",
                "request_options": {"temperature": 0, "enable_thinking": False,
                                    "seed": service.get("seed", 42),
                                    "structured_outputs": service.get("structured_outputs", True)}}
    return config, identity


def plan(config, identity, datasets=None, splits=None):
    from .build_parent_child_directories import plan as source_plan
    specs, preview = source_plan(config, identity, datasets, splits)
    preview.update({"embedding": identity["embedding"], "embedding_model_path": config["embedding"]["model_path"],
                    "embedding_concurrency": config["embedding"]["workers"],
                    "embedding_batch_size": config["embedding"]["batch_size"],
                    "segmentation": config["segmentation"], "thresholds_calibrated": False,
                    "ready_for_inference": identity["embedding"]["ready"],
                    "parent_generation": "after_children_with_source_records"})
    return specs, preview


class AdjacentBuildStore(BuildStore):
    """Longer SQLite lock waits plus independently resumable boundary decisions."""

    def __init__(self, root, identity):
        super().__init__(root, identity)
        self.db.execute("PRAGMA busy_timeout=60000")
        with self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS segment_cache (key TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    def segments(self, key):
        import json
        with self.lock:
            row = self.db.execute("SELECT payload FROM segment_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_segments(self, key, value):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO segment_cache VALUES (?,?)", (key, canonical(value)))


def run(args, *, tokenizer=None, embedding_tokenizer=None, pool_factory=WriterPool,
        embedding_factory=None, segmenter_factory=None, writer_factory=None):
    from .policy import load_tokenizer
    from .hierarchical.embedding import EmbeddingClient
    from .hierarchical.segmentation import AdjacentSegmenter
    from .hierarchical.adjacent_writer import AdjacentDirectoryWriter
    from .hierarchical.writer import public_session, token_count, validate_build_result

    config, identity = load_recipe(args.config, args.workers,
        embedding_model_path=getattr(args, "embedding_model_path", None),
        output_root=getattr(args, "output_root", None), allow_missing_embedding=args.dry_run)
    specs, preview = plan(config, identity, args.datasets, args.splits)
    announce("adjacent_directory_plan", **preview)
    limit = getattr(args, "limit_sessions", None)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit_sessions must be a positive integer")
    if args.dry_run:
        return preview
    endpoints = args.base_urls or ([args.base_url] if getattr(args, "base_url", None) else None)
    embedding_url = getattr(args, "embedding_base_url", None)
    if not endpoints or len(set(endpoints)) != len(endpoints) or not embedding_url:
        raise ValueError("Provide unique --base-urls and --embedding-base-url, or use the three-GPU launcher")
    tokenizer = tokenizer or load_tokenizer(config["model_path"])
    embedding_tokenizer = embedding_tokenizer or load_tokenizer(config["embedding"]["model_path"])
    count_tokens = lambda text: token_count(tokenizer, text)
    root, counters = Path(config["output_root"]), Counter()
    key_locks, key_lock_guard = {}, Lock()
    started = time.monotonic()
    embedding_factory = embedding_factory or EmbeddingClient
    segmenter_factory = segmenter_factory or AdjacentSegmenter
    writer_factory = writer_factory or AdjacentDirectoryWriter
    with file_lock(root / ".build.lock"):
        store = AdjacentBuildStore(root, identity)
        pool, encoder = None, None
        execution = {"run_status": "running", "datasets": [s["name"] for s in specs],
                     "splits": args.splits or ["train", "validation", "test"], "limit_sessions": limit,
                     "thresholds_calibrated": False}
        try:
            pool = pool_factory(endpoints, config["workers"], config.get("service", {}), store)
            embed_settings = {k: v for k, v in config["embedding"].items() if k not in {"model_id", "model_path"}}
            encoder = embedding_factory(embedding_url, embedding_tokenizer,
                cache_path=root / "embedding_cache.sqlite3", identity=identity["embedding"], settings=embed_settings)

            def inputs():
                queued = 0
                for spec in specs:
                    modality = spec.get("modality_policy", "existing_captions")
                    for bank in iter_banks(spec, args.splits, count_tokens=count_tokens,
                                           tokenizer_id=identity["tokenizer_id"]):
                        store.add_bank(bank)
                        counters["banks_seen"] += 1
                        for session in bank["sessions"]:
                            validate_session(session)
                            store.add_session(session)
                            state, directory = store.committed(session["bank_id"], session["parent_id"])
                            if state in {"complete", "empty"} or (state == "fallback" and not config.get("retry_fallback", True)):
                                validate_directory(directory, session, count_tokens=count_tokens)
                                counters["sessions_resumed"] += 1
                                continue
                            if limit is not None and queued >= limit:
                                execution["selection_limited"] = True
                                return
                            queued += 1
                            yield spec["name"], session, modality

            def build(item):
                dataset, session, modality = item
                try:
                    cache_key = digest({"input": public_session(session, modality), "construction": identity,
                                        "modality_policy": modality})
                    with key_lock_guard:
                        key_lock = key_locks.setdefault(cache_key, Lock())
                    with key_lock:
                        result = store.cached(cache_key)
                        if result and result["status"] == "fallback" and config.get("retry_fallback", True):
                            result = None
                        cached, segment_cached = result is not None, False
                        if result is None:
                            writer = writer_factory(tokenizer, {**config["writer"], "modality_policy": modality},
                                                    pool.for_session(session))
                            segmentation = store.segments(cache_key)
                            segment_cached = segmentation is not None
                            if segmentation is None:
                                segmenter = segmenter_factory(encoder, settings=config["segmentation"],
                                                              review_boundary=writer.review_boundary,
                                                              modality_policy=modality)
                                segmentation = segmenter.segment(session)
                                if not any(segmentation.get("stats", {}).get(key, 0) for key in
                                           ("review_failures", "review_unavailable")):
                                    store.save_segments(cache_key, segmentation)
                            result = writer.build(session, segmentation["spans"])
                            result["segmentation"] = segmentation
                            if any(segmentation.get("stats", {}).get(key, 0) for key in
                                   ("review_failures", "review_unavailable")):
                                result["status"] = "fallback"
                        validate_build_result(result, session, tokenizer, config["writer"])
                        directory = bind_directory(session, result, identity["writer"], identity["tokenizer_id"],
                                                   count_tokens=count_tokens, prompt_version=PROMPT_VERSION)
                        if not cached:
                            store.cache(cache_key, result)
                    if not cached:
                        for trace in result.get("traces", []):
                            store.trace({"event": "adjacent_writer_trace", "bank_id": session["bank_id"],
                                         "parent_id": session["parent_id"], **trace})
                    segmentation = result.get("segmentation", {})
                    store.trace({"event": "adjacent_segmentation", "bank_id": session["bank_id"],
                                 "parent_id": session["parent_id"], "cache_hit": cached or segment_cached,
                                 **segmentation})
                    return {"directory": directory, "cache_hit": cached, "segment_cache_hit": segment_cached,
                            "stats": result.get("stats", {}), "segmentation_stats": segmentation.get("stats", {})}
                except Exception as exc:
                    return {"error_type": type(exc).__name__, "reason": str(exc)}

            for (dataset, session, _), result in bounded_results(build, inputs(), config["workers"]):
                if "directory" in result:
                    stats = {**result["stats"], **{"segmentation_" + k: v for k, v in result["segmentation_stats"].items()}}
                    store.commit(session, result["directory"], stats)
                    status = result["directory"]["build"]["status"]
                    counters[status] += 1
                    counters["cache_hits" if result["cache_hit"] else "new_results"] += 1
                    counters["segment_cache_hits"] += int(result["segment_cache_hit"])
                    if not result["cache_hit"]:
                        for name, value in stats.items():
                            if type(value) is int:
                                counters[name if name.startswith("segmentation_") else "writer_" + name] += value
                else:
                    status = "failed"
                    store.failed(session, result["error_type"], result["reason"])
                    counters["failed"] += 1
                counters["sessions_processed"] += 1
                error = ({"error_type": result["error_type"], "reason": result["reason"][:400]}
                         if status == "failed" else {})
                announce("adjacent_session_progress", dataset=dataset, bank_id=session["bank_id"],
                         parent_id=session["parent_id"], status=status, **error, **dict(counters))
            current = store.report()
            execution["run_status"] = ("limited" if execution.get("selection_limited") else
                "incomplete" if current["states"]["failed"] or current["states"]["pending"] or current["source_failures"]
                else "finished")
        except BaseException as exc:
            execution["run_status"] = "interrupted"
            execution["fatal_error"] = {"error_type": type(exc).__name__, "reason": str(exc)}
            raise
        finally:
            try:
                try:
                    if encoder is not None:
                        encoder.close()
                finally:
                    if pool is not None:
                        pool.close()
            finally:
                try:
                    report = store.export({**dict(counters), **execution, "elapsed_seconds": time.monotonic() - started})
                finally:
                    store.close()
        announce("adjacent_build_finished", output_root=str(root), sessions=report["sessions"],
                 states=report["states"], source_failures=report["source_failures"], run_status=execution["run_status"])
        if report["states"]["failed"] or report["source_failures"] or (report["states"]["pending"] and not execution.get("selection_limited")):
            raise InfrastructureError(f"Some sessions failed or remain pending; inspect {root / 'build_report.json'}; rerun to resume")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS))
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"])
    urls = parser.add_mutually_exclusive_group()
    urls.add_argument("--base-url")
    urls.add_argument("--base-urls", nargs="+")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-model-path")
    parser.add_argument("--output-root")
    parser.add_argument("--limit-sessions", type=int, help="Build at most this many unfinished sessions for a smoke run")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs/config without inference; report missing encoder weights")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
