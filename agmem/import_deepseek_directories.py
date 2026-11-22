"""Import frozen DeepSeek directories into an independent Qwen continuation.

The source journal is opened read-only under its existing builder lock. Only
committed directories and immutable source snapshots are imported; inference
caches and Writer traces remain in the original DeepSeek run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sqlite3
import time

from .build_adjacent_directories import AdjacentBuildStore
from .core import atomic_json, canonical, digest, file_digest
from .hierarchical.binding import validate_directory
from .hierarchical.records import validate_session
from .jobs import file_lock
from .pipeline import announce


VERSION = "deepseek_qwen_continuation_import.v1"
SOURCE_PROTOCOL = "adjacent_topic_event_deepseek_v1"
TARGET_PROTOCOL = "adjacent_topic_event_qwen_continuation_v1"
DEFAULT_SOURCE = "runs/parent_child_directories/deepseek_flash_lmem_adjacent_v1"
DEFAULT_TARGET = "runs/parent_child_directories/qwen27b_lmem_after_deepseek_v1"
COMPLETE_MARKER = "deepseek_import_complete"
COMMITTED = {"complete", "empty", "fallback"}
STATES = COMMITTED | {"pending", "failed"}
EXCLUDED = ["writer_cache", "segment_cache", "embedding_cache", "writer_traces"]


@contextmanager
def source_lock(root):
    """Use the existing source lock, without creating or changing any file."""
    path = Path(root) / ".build.lock"
    with path.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Source directory builder is active: {path}; stop it before importing") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _read_only(root):
    path = Path(root).resolve() / "build_journal.sqlite3"
    if not path.is_file():
        raise ValueError(f"Missing directory journal: {path}")
    # These builds use DELETE journaling. Refuse recovery/WAL artifacts rather
    # than allowing a nominally read-only SQLite connection to create sidecars.
    if any(Path(str(path) + suffix).exists() for suffix in ("-journal", "-wal", "-shm")):
        raise ValueError(f"Directory journal has recovery/WAL sidecars: {path}; finish its owning build first")
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=60)


def _identity(db):
    row = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if row is None:
        raise ValueError("Directory journal has no construction identity")
    return json.loads(row[0])


def _datasets(identity):
    result = {}
    for item in identity.get("datasets", []):
        spec = item["spec"]
        name = spec["name"]
        if name in result:
            raise ValueError("Duplicate datasets in construction identity")
        semantics = {k: v for k, v in spec.items() if k not in {"input_root", "input_files", "path"}}
        semantics.setdefault("format", "longmemeval")
        semantics.setdefault("modality_policy", "existing_captions")
        files = {}
        for entry in item["files"]:
            split, sha = entry["split"], entry["sha256"]
            if split in files or not isinstance(sha, str) or len(sha) != 64:
                raise ValueError("Duplicate or invalid dataset file identity")
            files[split] = sha
        if not files:
            raise ValueError("Dataset identity contains no source files")
        result[name] = {"spec": semantics, "files": files}
    if not result:
        raise ValueError("Missing datasets in construction identity")
    return result


def _embedding_semantics(value):
    execution = {"model_path", "path", "workers", "batch_size", "max_batch_tokens", "base_url",
                 "base_urls", "timeout_seconds", "network_retries", "api_key_env"}
    result = {k: v for k, v in value.items() if k not in execution}
    if "settings" in result:
        result["settings"] = {k: v for k, v in result["settings"].items() if k not in execution}
    return result


def compatible_identity(source, target):
    """Permit a new Writer/transport while binding unchanged source semantics."""
    if source.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError(f"Source import requires protocol {SOURCE_PROTOCOL}")
    if target.get("protocol") != TARGET_PROTOCOL:
        raise ValueError(f"Target import requires protocol {TARGET_PROTOCOL}")
    if _datasets(source) != _datasets(target):
        raise ValueError("Source and target datasets/spec semantics or file SHA256 differ")
    for key in ("tokenizer_id", "writer_settings", "segmentation", "prompt_hashes", "prompt_version", "block_policy",
                "boundary_prompt_version", "base_resource_hashes"):
        if key not in source or key not in target or source[key] != target[key]:
            raise ValueError(f"Source and target {key} semantics differ or are missing")
    if not {"child.txt", "parent.txt", "boundary.txt"}.issubset(source["prompt_hashes"]):
        raise ValueError("Missing child/parent/boundary prompt hashes")
    if (not source.get("embedding") or not target.get("embedding")
            or _embedding_semantics(source["embedding"]) != _embedding_semantics(target["embedding"])):
        raise ValueError("Source and target embedding semantics differ or are missing")


def _verify_files(source, target):
    hashes = {}
    for identity in (source, target):
        for dataset in identity["datasets"]:
            for entry in dataset["files"]:
                filename = str(Path(entry["path"]).resolve())
                if filename not in hashes:
                    hashes[filename] = file_digest(filename)
                if hashes[filename] != entry["sha256"]:
                    raise ValueError(f"Dataset file SHA256 changed: {filename}")
    return hashes


def _journal_snapshot(root):
    stat = (root / "build_journal.sqlite3").stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _existing_target(root, identity):
    if not root.exists():
        return
    if not root.is_dir():
        raise ValueError("Continuation target must be a directory")
    journal = root / "build_journal.sqlite3"
    if journal.is_symlink():
        raise ValueError("Continuation target journal must not be a symlink")
    if journal.exists():
        db = _read_only(root)
        try:
            if _identity(db) != identity:
                raise ValueError("Target construction identity changed; choose a fresh output root")
        finally:
            db.close()
    elif any(path.name != ".build.lock" for path in root.iterdir()):
        raise ValueError("Nonempty continuation target has no construction journal")


def _validate_source_row(row, identity, count_tokens):
    bid, pid, source_hash, snapshot, status, directory, error, stats = row
    session = json.loads(snapshot)
    if (session["bank_id"], session["parent_id"], session["source_hash"]) != (bid, pid, source_hash):
        raise ValueError("Source session address/hash differs from its journal row")
    if status not in STATES or (status in COMMITTED) != (directory is not None):
        raise ValueError("Source directory presence differs from its committed status")
    parsed = json.loads(directory) if directory is not None else None
    if parsed is not None:
        validate_directory(parsed, session, count_tokens=count_tokens)
        expected = {"status": status, "writer_model": identity["writer"]["model"],
                    "writer_revision": identity["writer"]["model_revision"],
                    "prompt_version": identity["prompt_version"]}
        if (any(parsed["build"].get(k) != v for k, v in expected.items())
                or parsed["source"]["tokenizer_id"] != identity["tokenizer_id"]):
            raise ValueError("Source directory Writer/tokenizer provenance differs from its identity")
    else:
        validate_session(session, count_tokens=count_tokens)
    return session, parsed


def _import_session(store, row, source_identity, contract, count_tokens, counters):
    session, directory = _validate_source_row(row, source_identity, count_tokens)
    bid, pid, source_hash, snapshot, status, encoded, error, stats = row
    with store.db:
        existing = store.db.execute("SELECT snapshot,status,directory FROM sessions WHERE bank_id=? AND parent_id=?",
                                    (bid, pid)).fetchone()
        if existing is not None:
            if json.loads(existing[0]) != session:
                raise ValueError("Existing target snapshot differs from the imported immutable source")
            if existing[1] not in STATES or (existing[1] in COMMITTED) != (existing[2] is not None):
                raise ValueError("Existing target directory presence differs from its committed status")
            if existing[2] is not None:
                validate_directory(json.loads(existing[2]), session, count_tokens=count_tokens)
            imported = store.db.execute("SELECT 1 FROM reuse_provenance WHERE bank_id=? AND parent_id=?",
                                        (bid, pid)).fetchone()
            # Includes Qwen completions after a prior import, and imported
            # fallback records intentionally kept as accepted source work.
            if existing[1] in COMMITTED or imported is not None:
                counters["target_sessions_preserved"] += 1
                return
        target_status = status if directory is not None else "pending"
        provenance = {"version": VERSION, "source_root": contract["source_root"],
                      "source_identity_sha256": contract["source_identity_sha256"],
                      "target_identity_sha256": contract["target_identity_sha256"],
                      "bank_id": bid, "parent_id": pid, "source_hash": source_hash,
                      "source_status": status, "imported_status": target_status,
                      "directory_version": directory["directory_version"] if directory else None,
                      "original_writer": directory["build"] if directory else None,
                      "source_result_stats": json.loads(stats) if stats is not None else None,
                      "source_error": json.loads(error) if error is not None else None}
        # The exact original directory JSON, snapshot and stats are retained.
        # Failed or pending entries carry no Writer output into the new run.
        store.db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?)",
                         (bid, pid, source_hash, snapshot, target_status, encoded, None,
                          stats if directory is not None else None))
        store.db.execute("INSERT INTO reuse_provenance VALUES (?,?,?)", (bid, pid, canonical(provenance)))
        counters["imported_" + target_status] += 1


def import_committed(source_root, target_root, target_identity, *, dry_run=False, count_tokens=None):
    """Import each frozen session atomically; repeat safely after interruption.

    The caller must not already hold the target .build.lock. ``count_tokens``
    optionally supplies the verified reference tokenizer; no model is loaded
    by this module, including when planning with ``dry_run=True``.
    """
    source, target = Path(source_root).resolve(), Path(target_root).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Continuation target must be separate from its immutable source")
    report = {"version": VERSION, "source_root": str(source), "output_root": str(target),
              "status": "planned" if dry_run else "running", "excluded_artifacts": EXCLUDED,
              "timestamp": datetime.now(timezone.utc).isoformat(),
              "token_counts_rechecked": count_tokens is not None and not dry_run}
    with source_lock(source):
        db = _read_only(source)
        try:
            old_identity = _identity(db)
            compatible_identity(old_identity, target_identity)
            verified_files = _verify_files(old_identity, target_identity)
            _existing_target(target, target_identity)
            snapshot = _journal_snapshot(source)
            report["source_states"] = dict(db.execute("SELECT status,COUNT(*) FROM sessions GROUP BY status"))
            if not set(report["source_states"]).issubset(STATES):
                raise ValueError("Unsupported source session status")
            report["source_banks"] = db.execute("SELECT COUNT(*) FROM banks").fetchone()[0]
            report["registered_sessions"] = sum(report["source_states"].values())
            report["unregistered_sessions"] = "left for the continuation builder to register from datasets"
            contract = {"version": VERSION, "source_root": str(source), "source_journal": snapshot,
                        "source_identity_sha256": digest(old_identity),
                        "target_identity_sha256": digest(target_identity)}
            report.update(source_identity=old_identity, target_identity=target_identity,
                          verified_dataset_files=verified_files)
            if dry_run:
                return report
            counters = Counter()
            with file_lock(target / ".build.lock"):
                _existing_target(target, target_identity)
                store = AdjacentBuildStore(target, target_identity)
                try:
                    with store.db:
                        previous = store.db.execute("SELECT value FROM metadata WHERE key='deepseek_import'").fetchone()
                        if previous is not None and json.loads(previous[0]) != contract:
                            raise ValueError("Frozen DeepSeek import source or target contract changed")
                        complete = store.db.execute("SELECT value FROM metadata WHERE key=?", (COMPLETE_MARKER,)).fetchone()
                        if complete is not None:
                            marker = json.loads(complete[0])
                            if previous is None or marker.get("contract") != contract or marker.get("status") != "ready":
                                raise ValueError("DeepSeek import completion marker differs from its frozen contract")
                            if ((target / "deepseek_import_manifest.json").is_file()
                                    and (target / "reuse_provenance.jsonl").is_file()):
                                # Completed imports never scan/tokenize all
                                # directories again or replace Qwen's work.
                                report.update(status="ready", import_already_complete=True,
                                              counters={"target_sessions_preserved": report["registered_sessions"]},
                                              target=store.report(), contract=contract,
                                              token_counts_rechecked=False)
                                return report
                        store.db.execute("CREATE TABLE IF NOT EXISTS reuse_provenance (bank_id TEXT NOT NULL, "
                                         "parent_id TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(bank_id,parent_id))")
                        store.db.execute("INSERT OR IGNORE INTO metadata VALUES ('deepseek_import',?)", (canonical(contract),))
                    datasets = {entry["spec"]["name"] for entry in old_identity["datasets"]}
                    processed, last_progress = 0, time.monotonic()
                    announce("deepseek_import_progress", processed=0, total=report["registered_sessions"])
                    for bid, dataset, payload in db.execute("SELECT bank_id,dataset,payload FROM banks ORDER BY bank_id"):
                        bank = json.loads(payload)
                        if bank["bank_id"] != bid or bank["dataset"] != dataset or dataset not in datasets:
                            raise ValueError("Source bank address/dataset differs from its journal identity")
                        # add_bank merges source addresses already registered by
                        # the continuation builder without dropping its work.
                        store.add_bank(bank)
                        counters["banks_reused"] += 1
                        query = db.execute("SELECT bank_id,parent_id,source_hash,snapshot,status,directory,error,result_stats "
                                           "FROM sessions WHERE bank_id=? ORDER BY parent_id", (bid,))
                        for row in query:
                            _import_session(store, row, old_identity, contract, count_tokens, counters)
                            processed += 1
                            now = time.monotonic()
                            if (processed % 1000 == 0 or now - last_progress >= 10
                                    or processed == report["registered_sessions"]):
                                announce("deepseek_import_progress", processed=processed,
                                         total=report["registered_sessions"], **dict(counters))
                                last_progress = now
                    visited = sum(n for key, n in counters.items() if key.startswith("imported_"))
                    visited += counters["target_sessions_preserved"]
                    if visited != report["registered_sessions"]:
                        raise ValueError("Source contains sessions without a matching bank")
                    if _journal_snapshot(source) != snapshot:
                        raise ValueError("Source journal changed during import")
                    report.update(status="ready", counters=dict(counters), target=store.report(), contract=contract)
                    # The journal is authoritative after interruption. Export
                    # provenance atomically before publishing the ready marker.
                    with store._jsonl("reuse_provenance.jsonl") as write:
                        for (payload,) in store.db.execute("SELECT payload FROM reuse_provenance ORDER BY bank_id,parent_id"):
                            write(json.loads(payload))
                    atomic_json(target / "deepseek_import_manifest.json", report)
                    with store.db:
                        store.db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
                                         (COMPLETE_MARKER, canonical({"contract": contract, "status": "ready",
                                                                    "timestamp": report["timestamp"]})))
                finally:
                    store.close()
            return report
        finally:
            db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE)
    parser.add_argument("--target-root", default=DEFAULT_TARGET)
    parser.add_argument("--target-identity", required=True, help="JSON construction identity from the Qwen continuation recipe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = import_committed(args.source_root, args.target_root,
                              json.loads(Path(args.target_identity).read_text()), dry_run=args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
