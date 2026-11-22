"""Repair registered fallback sessions without scheduling unfinished histories.

Only the parent-summary budget may migrate from 200 to 400 tokens. Existing
complete directories remain unchanged; the journal records the previous recipe
so its validated segmentation caches can be reused under the relaxed budget.
"""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from threading import Lock
import time

from .build_adjacent_directories import AdjacentBuildStore, DATASETS, PROMPT_VERSION, load_recipe
from .core import InfrastructureError, atomic_json, canonical, digest, file_digest
from .execution import bounded_results
from .hierarchical.binding import bind_directory, validate_directory
from .hierarchical.client import WriterPool
from .jobs import file_lock
from .pipeline import announce


DEFAULT_CONFIG = "configs/build_adjacent_qwen_locomo_lmes_parent400.yaml"


def _connect(root, *, readonly=True):
    path = Path(root).resolve() / "build_journal.sqlite3"
    if not path.is_file():
        raise ValueError(f"Repair requires an existing build journal: {path}")
    db = sqlite3.connect(path.as_uri() + ("?mode=ro" if readonly else "?mode=rw"),
                         uri=True, timeout=60)
    return db


def prepare_identity(root, new_identity, *, apply=False):
    """Check or atomically record the sole supported relaxation; caller locks writes."""
    db = _connect(root, readonly=not apply)
    try:
        row = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if row is None:
            raise ValueError("Build journal is missing its construction identity")
        old = json.loads(row[0])
        row = db.execute("SELECT value FROM metadata WHERE key='parent_budget_migrations'").fetchone()
        history = json.loads(row[0]) if row else []
        if new_identity.get("writer_settings", {}).get("parent_content_tokens") != 400:
            raise ValueError("Fallback repair requires parent_content_tokens=400")
        if old != new_identity:
            relaxed = deepcopy(old)
            if relaxed.get("writer_settings", {}).get("parent_content_tokens") != 200:
                raise ValueError("Only a parent summary budget migration from 200 to 400 is supported")
            relaxed["writer_settings"]["parent_content_tokens"] = 400
            if relaxed != new_identity:
                raise ValueError("Construction identity changed beyond parent_content_tokens; refusing migration")
            record = {"timestamp": datetime.now(timezone.utc).isoformat(),
                      "reason": "Relax parent summary budget; preserve existing complete directories",
                      "from_identity": old, "to_identity": new_identity,
                      "repair_implementation_sha256": file_digest(Path(__file__))}
            history.append(record)
            if apply:
                with db:
                    db.execute("UPDATE metadata SET value=? WHERE key='identity'", (canonical(new_identity),))
                    db.execute("INSERT OR REPLACE INTO metadata VALUES ('parent_budget_migrations',?)",
                               (canonical(history),))
        identities = [new_identity]
        for migration in reversed(history):
            previous = migration["from_identity"]
            if previous not in identities:
                identities.append(previous)
        return identities
    finally:
        db.close()


def select_fallbacks(db, datasets, splits):
    """Select existing rows only, without registering any new source sessions."""
    banks = db.execute("SELECT bank_id,dataset,json_extract(payload,'$.query_memberships') FROM banks").fetchall()
    selected_banks = {bid: dataset for bid, dataset, memberships in banks
                      if dataset in datasets and any(q["split"] in splits
                                                     for q in json.loads(memberships or "[]"))}
    rows = db.execute("SELECT bank_id,parent_id FROM sessions WHERE status='fallback' "
                      "ORDER BY bank_id,parent_id").fetchall()
    return [(selected_banks[bid], bid, pid) for bid, pid in rows if bid in selected_banks]


def run(args, *, tokenizer=None, embedding_tokenizer=None, pool_factory=WriterPool,
        embedding_factory=None, segmenter_factory=None, writer_factory=None):
    from .policy import load_tokenizer
    from .hierarchical.embedding import EmbeddingClient
    from .hierarchical.segmentation import AdjacentSegmenter
    from .hierarchical.adjacent_writer import AdjacentDirectoryWriter, validate_spans
    from .hierarchical.writer import public_session, token_count, validate_build_result

    config, identity = load_recipe(args.config, args.workers,
        embedding_model_path=getattr(args, "embedding_model_path", None),
        output_root=getattr(args, "output_root", None), allow_missing_embedding=args.dry_run)
    datasets = args.datasets or ["split_longmemeval_s"]
    splits = args.splits or ["train", "test"]
    specs = {spec["name"]: spec for spec in config["datasets"]}
    if not set(datasets).issubset(specs):
        raise ValueError("Repair datasets must be present in the construction recipe")
    limit = getattr(args, "limit_sessions", None)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit_sessions must be a positive integer")
    root = Path(config["output_root"])
    cache_identities = prepare_identity(root, identity)
    db = _connect(root)
    try:
        selected = select_fallbacks(db, datasets, splits)
    finally:
        db.close()
    preview = {"output_root": str(root), "datasets": datasets, "splits": splits,
               "selection": "existing_fallback_only", "selected": len(selected),
               "by_dataset": dict(Counter(row[0] for row in selected)),
               "parent_content_tokens": config["writer"]["parent_content_tokens"],
               "limit_sessions": limit, "cache_identities": len(cache_identities),
               "unfinished_sessions_will_be_scheduled": False}
    announce("adjacent_repair_plan", **preview)
    if args.dry_run:
        return preview
    endpoints = args.base_urls or ([args.base_url] if getattr(args, "base_url", None) else None)
    embedding_url = getattr(args, "embedding_base_url", None)
    if not endpoints or len(set(endpoints)) != len(endpoints) or not embedding_url:
        raise ValueError("Provide unique Writer URLs and an embedding URL, or use the three-GPU launcher")
    tokenizer = tokenizer or load_tokenizer(config["model_path"])
    embedding_tokenizer = embedding_tokenizer or load_tokenizer(config["embedding"]["model_path"])
    count_tokens = lambda text: token_count(tokenizer, text)
    embedding_factory = embedding_factory or EmbeddingClient
    segmenter_factory = segmenter_factory or AdjacentSegmenter
    writer_factory = writer_factory or AdjacentDirectoryWriter
    counters, errors = Counter(), []
    key_locks, lock_guard = {}, Lock()
    started = time.monotonic()
    with file_lock(root / ".build.lock"):
        cache_identities = prepare_identity(root, identity, apply=True)
        store = AdjacentBuildStore(root, identity)
        pool, encoder = None, None
        selected = select_fallbacks(store.db, datasets, splits)
        selected_total = len(selected)
        if limit is not None:
            selected = selected[:limit]
        execution = {"kind": "fallback_repair", "datasets": datasets, "splits": splits,
                     "parent_content_tokens": 400, "selected": len(selected),
                     "selected_total": selected_total, "run_status": "running",
                     "repair_implementation_sha256": file_digest(Path(__file__))}
        try:
            if selected:
                pool = pool_factory(endpoints, config["workers"], config.get("service", {}), store)
                embed_settings = {k: v for k, v in config["embedding"].items() if k not in {"model_id", "model_path"}}
                encoder = embedding_factory(embedding_url, embedding_tokenizer,
                    cache_path=root / "embedding_cache.sqlite3", identity=identity["embedding"], settings=embed_settings)

            def inputs():
                for dataset, bid, pid in selected:
                    with store.lock:
                        snapshot, directory = store.db.execute(
                            "SELECT snapshot,directory FROM sessions WHERE bank_id=? AND parent_id=?",
                            (bid, pid)).fetchone()
                    session, directory = json.loads(snapshot), json.loads(directory)
                    validate_directory(directory, session, count_tokens=count_tokens)
                    yield dataset, session, specs[dataset].get("modality_policy", "existing_captions")

            def build(item):
                _, session, modality = item
                try:
                    keys = [digest({"input": public_session(session, modality), "construction": version,
                                    "modality_policy": modality}) for version in cache_identities]
                    cache_key = keys[0]
                    with lock_guard:
                        key_lock = key_locks.setdefault(cache_key, Lock())
                    with key_lock:
                        result = store.cached(cache_key)
                        if result is not None and result["status"] != "complete":
                            result = None
                        cached, segment_cached = result is not None, False
                        if result is None:
                            writer = writer_factory(tokenizer, {**config["writer"], "modality_policy": modality},
                                                    pool.for_session(session))
                            segmentation = None
                            for key in keys:
                                candidate = store.segments(key)
                                if candidate is not None and not any(candidate.get("stats", {}).get(k, 0)
                                        for k in ("review_failures", "review_unavailable")):
                                    validate_spans(candidate["spans"], len(session["blocks"]))
                                    segmentation, segment_cached = candidate, True
                                    break
                            if segmentation is None:
                                segmenter = segmenter_factory(encoder, settings=config["segmentation"],
                                                              review_boundary=writer.review_boundary,
                                                              modality_policy=modality)
                                segmentation = segmenter.segment(session)
                            boundary_failed = any(segmentation.get("stats", {}).get(k, 0)
                                                  for k in ("review_failures", "review_unavailable"))
                            if not boundary_failed:
                                store.save_segments(cache_key, segmentation)
                            result = writer.build(session, segmentation["spans"])
                            result["segmentation"] = segmentation
                            if boundary_failed:
                                result["status"] = "fallback"
                        validate_build_result(result, session, tokenizer, config["writer"])
                        directory = bind_directory(session, result, identity["writer"], identity["tokenizer_id"],
                                                   count_tokens=count_tokens, prompt_version=PROMPT_VERSION)
                        if not cached:
                            store.cache(cache_key, result)
                    if not cached:
                        for trace in result.get("traces", []):
                            store.trace({"event": "adjacent_repair_writer_trace", "bank_id": session["bank_id"],
                                         "parent_id": session["parent_id"], **trace})
                    return {"directory": directory, "stats": result.get("stats", {}),
                            "cache_hit": cached, "segment_cache_hit": segment_cached}
                except Exception as exc:
                    return {"error_type": type(exc).__name__, "reason": str(exc)[:1600]}

            for (dataset, session, _), result in bounded_results(build, inputs(), config["workers"]):
                status = result.get("directory", {}).get("build", {}).get("status")
                if status == "complete":
                    # A failed attempt must never erase the previously saved directory.
                    store.commit(session, result["directory"], result["stats"])
                    counters["repaired"] += 1
                elif status == "fallback":
                    counters["still_fallback"] += 1
                else:
                    counters["repair_errors"] += 1
                    errors.append({"bank_id": session["bank_id"], "parent_id": session["parent_id"], **result})
                counters["processed"] += 1
                counters["cache_hits"] += int(result.get("cache_hit", False))
                counters["segment_cache_hits"] += int(result.get("segment_cache_hit", False))
                event = {"event": "adjacent_repair_session", "dataset": dataset,
                         "bank_id": session["bank_id"], "parent_id": session["parent_id"],
                         "status": status or "error_preserved_fallback", **dict(counters)}
                if "error_type" in result:
                    event.update(result)
                store.trace({"timestamp": datetime.now(timezone.utc).isoformat(), **event})
                announce(event.pop("event"), **event)
            execution["run_status"] = ("incomplete" if counters["still_fallback"] or counters["repair_errors"]
                                       else "limited" if len(selected) < selected_total else "finished")
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
                    remaining = len(select_fallbacks(store.db, datasets, splits))
                    execution.update(dict(counters))
                    for key in ("repaired", "repair_errors", "still_fallback"):
                        execution.setdefault(key, 0)
                    execution.update(remaining_fallback=remaining, elapsed_seconds=time.monotonic() - started)
                    # Persist the repair summary before the potentially lengthy full export.
                    summary = {"execution": execution, "errors": errors, "global": store.report()}
                    atomic_json(root / "fallback_repair_report.json", summary)
                    report = store.export(execution)
                finally:
                    store.close()
        announce("adjacent_fallback_repair_finished", **execution)
        if execution["run_status"] == "incomplete":
            raise InfrastructureError(f"Some selected fallback sessions still need repair; inspect "
                                      f"{root / 'fallback_repair_report.json'}; rerun the repair command")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["split_longmemeval_s"])
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["train", "test"])
    urls = parser.add_mutually_exclusive_group()
    urls.add_argument("--base-url")
    urls.add_argument("--base-urls", nargs="+")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-model-path")
    parser.add_argument("--output-root")
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
