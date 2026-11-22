#!/usr/bin/env python3
"""Package frozen LongMemEval-M data, DeepSeek state and a small code bundle.

Source files are opened read-only. Model weights, inference caches, traces and
credentials are never copied. Verified reference tokenizer assets can optionally
be included. Model-content proofs use full SHA256 hashes. No service is contacted
and no model is loaded.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tarfile
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION = "longmemeval_m_continuation_transfer.v1"
SOURCE_PROTOCOL = "adjacent_topic_event_deepseek_v1"
DEFAULT_SOURCE = "runs/parent_child_directories/deepseek_flash_lmem_adjacent_v1"
DEFAULT_WRITER = "models/qwen3.5-27b"
DEFAULT_EMBEDDING = "models/qwen3-embedding-4b"
MANIFEST_PATH = "migration/longmemeval_m_continuation.json"
SPLITS = ("train", "validation", "test")
TABLES = ("metadata", "banks", "sessions")
TOKENIZER_EXTRAS = {"chat_template.jinja", "merges.txt", "tokenizer.model", "spiece.model"}
SMALL_REPORT_LIMIT = 4 * 1024 * 1024


def announce(stage, **details):
    print(json.dumps({"stage": stage, **details}, ensure_ascii=False), flush=True)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _resolve(root, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _relative(root, path):
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ValueError(f"Archive input must be inside the repository: {path}") from None
    if not relative.parts:
        raise ValueError("Archive input must not be the repository root")
    return relative.as_posix()


def _stamp(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


@contextmanager
def _source_connection(source):
    """Never create a lock or allow SQLite to recover a source journal."""
    journal = source / "build_journal.sqlite3"
    if not journal.is_file():
        raise ValueError(f"Missing directory journal: {journal}")
    with (source / ".build.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Source directory builder is active; stop it before packaging") from None
        try:
            if any(Path(str(journal) + suffix).exists() for suffix in ("-journal", "-wal", "-shm")):
                raise ValueError("Source journal has recovery/WAL sidecars; finish its owning build first")
            before = _stamp(journal)
            db = sqlite3.connect(journal.as_uri() + "?mode=ro", uri=True, timeout=60)
            try:
                db.execute("PRAGMA query_only=ON")
                db.execute("BEGIN")
                yield db
                if _stamp(journal) != before:
                    raise ValueError("Source journal changed during packaging")
            finally:
                db.close()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _source_identity(db, source):
    row = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
    if row is None:
        raise ValueError("Source journal has no construction identity")
    identity = json.loads(row[0])
    if identity.get("protocol") != SOURCE_PROTOCOL:
        raise ValueError(f"Source requires protocol {SOURCE_PROTOCOL}")
    if json.loads((source / "run_manifest.json").read_text(encoding="utf-8")) != identity:
        raise ValueError("Source run_manifest.json differs from its journal identity")
    return identity


def _dataset_files(root, identity):
    datasets = identity.get("datasets", [])
    if len(datasets) != 1 or datasets[0].get("spec", {}).get("name") != "split_longmemeval_m":
        raise ValueError("Source must contain only split_longmemeval_m")
    files = datasets[0].get("files", [])
    if len(files) != 3 or {entry.get("split") for entry in files} != set(SPLITS):
        raise ValueError("Source identity must contain exactly train, validation and test")
    result = []
    for split in SPLITS:
        entry = next(item for item in files if item["split"] == split)
        path = _resolve(root, entry["path"])
        relative = _relative(root, path)
        if path.suffix != ".parquet" or not path.is_file():
            raise ValueError(f"Missing Parquet split: {path}")
        before = _stamp(path)
        announce("verify_dataset", split=split, path=relative, bytes=before[0])
        checksum = sha256(path)
        if checksum != entry.get("sha256") or _stamp(path) != before:
            raise ValueError(f"Dataset SHA256 changed: {path}")
        result.append({"path": relative, "sha256": checksum, "bytes": before[0], "split": split})
    if len({entry["path"] for entry in result}) != len(SPLITS):
        raise ValueError("Dataset splits must refer to three distinct Parquet files")
    return result


def tokenizer_file_paths(root):
    """Match the engine tokenizer predicate and its four extra assets."""
    return sorted(path for path in root.iterdir() if path.is_file()
                  and (path.name in TOKENIZER_EXTRAS
                       or (path.suffix == ".json" and
                           ("tokenizer" in path.name or path.name in
                            {"special_tokens_map.json", "vocab.json", "added_tokens.json"}))))


def embedding_file_paths(root):
    selected = set(tokenizer_file_paths(root))
    selected.update(path for path in root.rglob("*.json")
                    if path.is_file() and ".cache" not in path.relative_to(root).parts)
    selected.update(path for path in root.glob("*.safetensors") if path.is_file())
    return sorted(selected)


def _model_ids(writer, embedding, identity):
    # Delayed imports keep --help usable in a bare Python install.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from agmem.build_adjacent_directories import _model_identity

    _, writer_tokenizer = _model_identity(writer)
    embedding_identity, embedding_tokenizer = _model_identity(embedding)
    if writer_tokenizer != identity.get("tokenizer_id"):
        raise ValueError("Source Writer tokenizer identity differs from the selected local model")
    expected = identity.get("embedding", {})
    if (not expected.get("ready") or embedding_identity.get("model_revision") != expected.get("model_revision")
            or embedding_tokenizer != expected.get("tokenizer_id")):
        raise ValueError("Source embedding identity differs from the selected local model")
    return writer_tokenizer


def _model_proof(root, paths, label):
    files = []
    for path in paths:
        before = _stamp(path)
        relative = path.relative_to(root).as_posix()
        announce("hash_model_file", model=label, path=relative, bytes=before[0])
        checksum = sha256(path)
        if _stamp(path) != before:
            raise ValueError(f"Model file changed while hashing: {path}")
        files.append({"path": relative, "bytes": before[0], "sha256": checksum})
    return files


def _compact_journal(source_db, target):
    """Copy exact schemas/rows, retaining only the identity metadata row."""
    target.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(target)
    counts = {}
    try:
        db.execute("PRAGMA journal_mode=DELETE")
        with db:
            for table in TABLES:
                row = source_db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                                        (table,)).fetchone()
                if row is None or not row[0]:
                    raise ValueError(f"Missing source table: {table}")
                db.execute(row[0])
                query = f'SELECT * FROM "{table}"'
                if table == "metadata":
                    query += " WHERE key='identity'"
                rows = source_db.execute(query)
                columns = len(rows.description)
                statement = f'INSERT INTO "{table}" VALUES ({",".join("?" for _ in range(columns))})'
                count = 0
                while batch := rows.fetchmany(256):
                    db.executemany(statement, batch)
                    count += len(batch)
                counts[table] = count
                announce("copy_journal_table", table=table, rows=count)
        if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError("Packaged compact journal failed SQLite quick_check")
    finally:
        db.close()
    return counts


def _check_output(output, protected):
    if not output.is_absolute():
        raise ValueError("--output must be an absolute path to a new or empty directory")
    if output.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    resolved = output.resolve()
    for path in protected:
        if resolved == path or resolved in path.parents or path in resolved.parents:
            raise ValueError("Output must be separate from all source data, run and model paths")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be new or empty; refusing to overwrite")
    return resolved


def code_file_paths(root):
    """Include regular source files without following directory/file links."""
    excluded = {"__pycache__", ".pytest_cache"}
    selected = set()
    for name in ("agmem", "configs", "scripts", "tests"):
        folder = root / name
        if folder.is_symlink() or not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            relative = path.relative_to(root)
            if (path.is_file() and not path.is_symlink() and path.suffix != ".pyc"
                    and not (excluded & set(relative.parts))
                    and not any(parent.is_symlink() for parent in path.parents if parent != root)):
                selected.add(path)
    for pattern in ("pyproject.toml", "requirements*.txt", "environment.yml", "README.md"):
        selected.update(path for path in root.glob(pattern) if path.is_file() and not path.is_symlink())
    return sorted(selected)


def _archive(path, root, files):
    announce("compress_archive", path=str(path))
    with path.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz", format=tarfile.PAX_FORMAT, compresslevel=1) as bundle:
            for item in files:
                bundle.add(item, arcname=item.relative_to(root).as_posix(), recursive=False)
    announce("hash_archive", path=str(path), bytes=path.stat().st_size)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def package(*, output, repo_root=REPO_ROOT, source_root=DEFAULT_SOURCE,
            writer_model_path=DEFAULT_WRITER, embedding_model_path=DEFAULT_EMBEDDING,
            include_reference_tokenizer=False, dry_run=False):
    root = Path(repo_root).resolve()
    source = _resolve(root, source_root)
    writer = _resolve(root, writer_model_path)
    embedding = _resolve(root, embedding_model_path)
    source_relative = _relative(root, source)
    output = _check_output(Path(output), (source, writer, embedding))
    announce("inspect_source", source=str(source), dry_run=dry_run)
    with _source_connection(source) as db:
        identity = _source_identity(db, source)
        data_files = _dataset_files(root, identity)
        output = _check_output(output, (source, writer, embedding,
                                       *(root / item["path"] for item in data_files)))
        writer_tokenizer = _model_ids(writer, embedding, identity)
        writer_paths, embedding_paths = tokenizer_file_paths(writer), embedding_file_paths(embedding)
        model_stamps = {path: _stamp(path) for path in (*writer_paths, *embedding_paths)}
        code_paths = code_file_paths(root)
        reference_relative = None
        if include_reference_tokenizer:
            try:
                reference_relative = _relative(root, writer)
            except ValueError:
                # External model stores still need a portable archive location.
                reference_relative = DEFAULT_WRITER
        plan = {"version": VERSION, "status": "planned", "output": str(output),
                "source_root_relative": source_relative, "data_files": data_files,
                "source_states": dict(db.execute("SELECT status,COUNT(*) FROM sessions GROUP BY status")),
                "code_files": len(code_paths),
                "model_proof_files": {"writer_tokenizer": [p.relative_to(writer).as_posix() for p in writer_paths],
                                      "embedding": [p.relative_to(embedding).as_posix() for p in embedding_paths]},
                "model_proof_bytes": sum(value[0] for value in model_stamps.values()),
                "dry_run_skips": ["model_full_sha256", "journal_copy", "data_copy", "archive"]}
        if reference_relative is not None:
            plan["reference_tokenizer_path"] = reference_relative
            plan["reference_tokenizer_bytes"] = sum(model_stamps[path][0] for path in writer_paths)
            plan["dry_run_skips"].append("reference_tokenizer_copy")
        if dry_run:
            announce("planned", **{key: value for key, value in plan.items() if key != "version"})
            return plan
        # An exclusive marker claims the output. Failures retain staging for
        # diagnosis and cannot be mistaken for completed transfers.
        output.mkdir(parents=True, exist_ok=True)
        with (output / ".packaging-in-progress").open("x") as marker:
            marker.write(VERSION + "\n")
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=output))
        models = {
            "writer_tokenizer": {"source_model_path": str(writer),
                                 "files": _model_proof(writer, writer_paths, "writer_tokenizer"),
                                 "tokenizer_id": writer_tokenizer},
            "embedding": {"source_model_path": str(embedding),
                          "files": _model_proof(embedding, embedding_paths, "embedding"),
                          "identity": identity["embedding"]},
        }
        if (tokenizer_file_paths(writer) != writer_paths or embedding_file_paths(embedding) != embedding_paths
                or any(_stamp(path) != stamp for path, stamp in model_stamps.items())):
            raise ValueError("Model files changed during packaging")
        _model_ids(writer, embedding, identity)
        if reference_relative is not None:
            for entry in models["writer_tokenizer"]["files"]:
                target = staging / reference_relative / entry["path"]
                announce("copy_reference_tokenizer", path=entry["path"], bytes=entry["bytes"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(writer / entry["path"], target)
                if target.stat().st_size != entry["bytes"] or sha256(target) != entry["sha256"]:
                    raise ValueError(f"Copied reference tokenizer SHA256 or size differs: {target}")
        staged_source = staging / source_relative
        announce("compact_journal")
        counts = _compact_journal(db, staged_source / "build_journal.sqlite3")
        (staged_source / ".build.lock").touch(exist_ok=False)
        shutil.copy2(source / "run_manifest.json", staged_source / "run_manifest.json")
        report = source / "build_report.json"
        if report.is_file() and report.stat().st_size <= SMALL_REPORT_LIMIT:
            shutil.copy2(report, staged_source / report.name)
        for entry in data_files:
            original, target = root / entry["path"], staging / entry["path"]
            announce("copy_dataset", split=entry["split"], bytes=entry["bytes"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            if target.stat().st_size != entry["bytes"] or sha256(target) != entry["sha256"]:
                raise ValueError(f"Copied dataset SHA256 differs: {target}")
        if _source_identity(db, source) != identity:
            raise ValueError("Source identity changed during packaging")
        manifest = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
                    "source_identity": identity, "source_root_relative": source_relative,
                    "data_files": data_files, "models": models, "journal_rows": counts,
                    "excluded_artifacts": ["writer_cache", "writer_traces", "segment_cache", "embedding_cache",
                                           "model_weights", "credentials"]}
        if reference_relative is not None:
            manifest["reference_tokenizer_path"] = reference_relative
        _write_json(staging / MANIFEST_PATH, manifest)
        archives = [_archive(output / "data.tar.gz", staging,
                             sorted(path for path in staging.rglob("*") if path.is_file())),
                    _archive(output / "code.tar.gz", root, code_paths)]
        with (output / "SHA256SUMS").open("x", encoding="utf-8") as stream:
            for item in archives:
                stream.write(f"{item['sha256']}  {item['path']}\n")
        complete = {"version": VERSION, "status": "complete", "archives": archives,
                    "manifest_path": MANIFEST_PATH, "journal_rows": counts,
                    "source_states": plan["source_states"]}
        if reference_relative is not None:
            complete["reference_tokenizer_path"] = reference_relative
    # Delete only our private staging directory after both checksums exist and
    # after the final read-only source guard succeeds.
    shutil.rmtree(staging)
    (output / ".packaging-in-progress").unlink()
    _write_json(output / "COMPLETE.json", complete)
    announce("complete", **complete)
    return complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE)
    parser.add_argument("--writer-model-path", default=DEFAULT_WRITER)
    parser.add_argument("--embedding-model-path", default=DEFAULT_EMBEDDING)
    parser.add_argument("--include-reference-tokenizer", action="store_true",
                        help="Bundle verified Writer tokenizer assets for DeepSeek continuation; excludes model weights")
    parser.add_argument("--dry-run", action="store_true", help="Validate and plan without writing or hashing model weights")
    args = parser.parse_args(argv)
    try:
        package(**vars(args))
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        parser.exit(1, f"Packaging failed: {error}\n")


if __name__ == "__main__":
    main()
