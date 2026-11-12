"""Freeze complete hierarchical banks for prepared LoCoMo training queries."""

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time

from .core import Example, atomic_json, canonical, digest, file_digest, load_config
from .data import ADAPTER_VERSION
from .hierarchical_training_store import (BANK_FILE, PROTOCOL, SNAPSHOT_FILE, SNAPSHOT_FORMAT,
                                           HierarchicalTrainingStore, snapshot_version, validate_bank)
from .store import Store


DEFAULT_SOURCE = "runs/locomo_longmemeval/prepared_locomo_query9010_seed42_20261102/locomo"
DEFAULT_DIRECTORY = "runs/parent_child_directories/qwen27b_adjacent_qwen4b_v1"


@contextmanager
def journal_snapshot(directory_root, timeout=20.0):
    """SQLite online backup releases its source read lock after each 128 pages."""
    source_path = Path(directory_root).resolve() / "build_journal.sqlite3"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if timeout <= 0:
        raise ValueError("Snapshot timeout must be positive")
    with tempfile.TemporaryDirectory(prefix="agmem-freeze-", dir="/tmp") as temporary:
        started = time.monotonic()
        source = sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True, timeout=min(timeout, .5))
        target = sqlite3.connect(Path(temporary) / "snapshot.sqlite3")

        def progress(status, remaining, total):
            if time.monotonic() - started > timeout:
                raise TimeoutError("Online directory backup exceeded its time limit; retry without stopping the builder")

        try:
            try:
                source.backup(target, pages=128, progress=progress, sleep=.05)
            finally:
                source.close()
            yield target
        finally:
            target.close()


def _jsonl_rows(path, field):
    rows = {}
    with path.open() as stream:
        for line in stream:
            value = json.loads(line)
            rows[value[field]] = value
    return rows


def _match_history(memory, bank):
    """Check private source addresses and all public messages, not just pIDs."""
    records, sessions = memory["records"], bank["sessions"]
    if len(records) != len(sessions):
        raise ValueError("Prepared history omits or adds source sessions")
    maps, record_maps = {}, {}
    for row in bank["source_map"]:
        if "record_id" in row:
            address = row.get("parent_id"), row["record_id"]
            if address in record_maps and record_maps[address] != row:
                raise ValueError("Ambiguous private record source map")
            record_maps[address] = row
            continue
        pid = row.get("parent_id")
        if pid in maps and maps[pid] != row:
            raise ValueError("Ambiguous private session source map")
        maps[pid] = row
    if set(maps) != {s["parent_id"] for s in sessions}:
        raise ValueError("Private source map does not cover the complete bank")
    for pos, (legacy, session) in enumerate(zip(records, sessions)):
        source = maps[session["parent_id"]]
        sid = source.get("original_session_id")
        sid = re.sub(r"^(?:answer_)+", "", str(sid)) if sid is not None else f"session_{pos}"
        if (source.get("bank_id") != bank["bank_id"] or source.get("source_hash") != session["source_hash"]
                or source.get("session_index") != pos or legacy["order"] != pos
                or sid != legacy["source_session_id"] or legacy["time"] != session["time"]["raw"]):
            raise ValueError("Prepared session address/date differs from directory source_map")
        if session["source_kind"] != "dialogue" or session["header_records"]:
            raise ValueError("Prepared LoCoMo requires an exact dialogue snapshot")
        messages, normalized = legacy["messages"], session["records"]
        if len(messages) != len(normalized):
            raise ValueError("Prepared message count differs from directory snapshot")
        for index, (message, record) in enumerate(zip(messages, normalized)):
            if (message["role"] != record["speaker"] or message["content"] != record["text"]
                    or record["attachments"]):
                raise ValueError("Prepared message content differs from directory snapshot")
            mapped = record_maps.get((session["parent_id"], record["record_id"]), {})
            if (mapped.get("bank_id") != bank["bank_id"] or mapped.get("source_hash") != session["source_hash"]
                    or mapped.get("message_index") != index or mapped.get("session_index") != pos
                    or mapped.get("original_session_id") != source.get("original_session_id")
                    or mapped.get("original_message_id") != message.get("dia_id", message.get("id"))):
                raise ValueError("Prepared message address differs from private record source_map")
        if legacy["session_hash"] != digest({"messages": messages}):
            raise ValueError("Prepared session hash changed")
    references = [{key: value for key, value in record.items() if key != "messages"} for record in records]
    if set(record_maps) != {(s["parent_id"], r["record_id"]) for s in sessions for r in s["records"]}:
        raise ValueError("Private record source_map does not exactly cover the selected bank")
    if memory["memory_id"] != "mem_" + digest(references):
        raise ValueError("Prepared memory ID does not match its exact history")


def _load_banks(db, examples, memories, datasets):
    rows = db.execute("SELECT bank_id,dataset,payload FROM banks WHERE dataset IN (%s)" %
                      ",".join("?" for _ in datasets), datasets)
    banks, memberships = {}, {}
    for bank_id, dataset, payload in rows:
        bank = json.loads(payload)
        if bank.get("bank_id") != bank_id or bank.get("dataset") != dataset:
            raise ValueError("Journal bank identity mismatch")
        banks[bank_id] = bank
        for member in bank.get("query_memberships", []):
            sample_id = member["sample_id"]
            if sample_id in memberships:
                raise ValueError(f"Query belongs to multiple bank memberships: {sample_id}")
            memberships[sample_id] = (bank_id, member)
    query_map, memory_map = {}, {}
    for example in examples:
        if example.sample_id not in memberships:
            raise ValueError(f"Query missing from directory bank membership: {example.sample_id}")
        bank_id, member = memberships[example.sample_id]
        expected_sample = f"{example.source}/{example.split}/{member['row_index']:06d}"
        if (example.sample_id != expected_sample or member["split"] != example.split
                or banks[bank_id]["dataset"] != example.source):
            raise ValueError("Query split/source differs from directory bank membership")
        if example.memory_id in memory_map and memory_map[example.memory_id] != bank_id:
            raise ValueError("One prepared history maps to different directory banks")
        query_map[example.sample_id] = bank_id
        memory_map[example.memory_id] = bank_id
    selected = {}
    for bank_id in sorted(set(memory_map.values())):
        bank = deepcopy(banks[bank_id])
        if any(issue.get("status") == "failed" for issue in bank.get("issues", [])):
            raise ValueError(f"Selected bank has source failures: {bank_id}")
        sessions, directories = [], []
        for parent_id, source_hash, snapshot, status, directory in db.execute(
                "SELECT parent_id,source_hash,snapshot,status,directory FROM sessions WHERE bank_id=? ORDER BY parent_id",
                (bank_id,)):
            if status != "complete" or directory is None:
                raise ValueError(f"Selected session is not complete: {bank_id}/{parent_id} ({status})")
            session, directory = json.loads(snapshot), json.loads(directory)
            if session["parent_id"] != parent_id or session["source_hash"] != source_hash:
                raise ValueError("Journal session identity mismatch")
            sessions.append(session); directories.append(directory)
        bank["sessions"], bank["directories"] = sessions, directories
        bank["query_memberships"] = [m for m in bank["query_memberships"] if m["sample_id"] in query_map]
        validate_bank(bank)
        selected[bank_id] = bank
    for memory_id, bank_id in memory_map.items():
        if memory_id not in memories:
            raise ValueError(f"Prepared query references missing history: {memory_id}")
        _match_history(memories[memory_id], selected[bank_id])
    return selected, query_map, memory_map


def prepare_frozen_store(config, source_prepared_root, directory_root, *, datasets=None, backup_timeout=20):
    data = config["data"]
    datasets = list(datasets or data["sources"])
    splits = list(data["splits"])
    if (data.get("directory_protocol") != PROTOCOL or data.get("layout", "combined") != "combined"
            or datasets != ["locomo"] or data["sources"] != datasets):
        raise ValueError("This freeze entrypoint requires combined hierarchical_v1 LoCoMo data")
    if not splits or len(set(splits)) != len(splits) or not set(splits).issubset({"train", "validation", "test"}):
        raise ValueError("Select unique prepared train/validation/test splits")
    source, directory_root, target = (Path(p).expanduser().resolve() for p in
                                      (source_prepared_root, directory_root, data["prepared_root"]))
    if not (source / "manifest.json").is_file() and (source / "locomo" / "manifest.json").is_file():
        source /= "locomo"
    if target == source or source in target.parents or target == directory_root or directory_root in target.parents:
        raise ValueError("Frozen training output must be independent of source prepared/build roots")
    source_manifest = json.loads((source / "manifest.json").read_text())
    identity = deepcopy(source_manifest["prepare_identity"])
    if source_manifest.get("prepare_complete") is not True or identity["sources"] != datasets:
        raise ValueError("Original prepared LoCoMo data is not complete")
    if identity.get("adapter_version") != ADAPTER_VERSION or identity.get("storage_format") != Store.FORMAT:
        raise ValueError("Original prepared adapter/storage version is incompatible")
    if not set(splits).issubset(identity["splits"]):
        raise ValueError("Requested split is not available in the original prepared data")
    identity["splits"] = splits
    examples, source_files = [], {}
    for name in ["memory_records.jsonl", *[f"{split}_examples.jsonl" for split in splits]]:
        source_files[name] = file_digest(source / name)
    for split in splits:
        rows = _jsonl_rows(source / f"{split}_examples.jsonl", "sample_id")
        values = [Example(**row) for row in rows.values()]
        if not values or any(e.source not in datasets or e.split != split for e in values):
            raise ValueError("Prepared examples have an empty or inconsistent split/source")
        examples.extend(values)
    if len({e.sample_id for e in examples}) != len(examples):
        raise ValueError("Query IDs overlap across prepared splits")
    memories = _jsonl_rows(source / "memory_records.jsonl", "memory_id")
    provenance = {"source_prepared_root": str(source), "directory_root": str(directory_root),
                  "source_files": source_files}
    if target.exists():
        frozen = HierarchicalTrainingStore(target)
        try:
            if (frozen.snapshot["provenance"] != provenance or frozen.snapshot["prepare_identity"] != identity):
                raise ValueError("Existing frozen output has different provenance; choose a new prepared_root")
            frozen.validate_directories(splits)
            return {**json.loads((target / "hierarchical_preparation_report.json").read_text()), "reused": True}
        finally:
            frozen.close()
    with journal_snapshot(directory_root, backup_timeout) as db:
        row = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if row is None:
            raise ValueError("Directory journal has no construction identity")
        construction_identity = json.loads(row[0])
        banks, query_map, memory_map = _load_banks(db, examples, memories, datasets)
    # Recheck inputs after reading them; no source indices/manifests are ever written.
    if any(file_digest(source / name) != sha for name, sha in source_files.items()):
        raise ValueError("Original prepared artifacts changed while freezing")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".freeze-hierarchy-", dir=target.parent))
    try:
        store = Store(staging)
        try:
            for memory_id in sorted(memory_map):
                copied = store.add_memory(memories[memory_id]["records"])
                if copied != memory_id:
                    raise ValueError("Frozen history changed its original memory ID")
            for example in sorted(examples, key=lambda e: e.sample_id):
                store.add_example(example)
            with (staging / BANK_FILE).open("w") as stream:
                for bank_id in sorted(banks):
                    stream.write(canonical(banks[bank_id]) + "\n")
            frozen_manifest = {"schema_version": SNAPSHOT_FORMAT, "directory_protocol": PROTOCOL,
                "sources": datasets, "splits": splits, "prepare_identity": identity,
                "construction_identity": construction_identity, "provenance": provenance,
                "memory_to_bank": memory_map, "query_to_bank": query_map,
                "banks": {bid: {"sha256": digest(bank), "dataset": bank["dataset"],
                                 "sessions": len(bank["sessions"])} for bid, bank in banks.items()},
                "files": {name: file_digest(staging / name) for name in
                          [BANK_FILE, "memory_records.jsonl", *[f"{s}_examples.jsonl" for s in splits]]}}
            version = snapshot_version(frozen_manifest)
            frozen_manifest["directory_version"] = version
            atomic_json(staging / SNAPSHOT_FILE, frozen_manifest)
            store.metadata.update(prepare_identity=identity, prepare_complete=True,
                directory_protocol=PROTOCOL, directory_version=version, directory_empty_pending=0,
                data={"directory_protocol": PROTOCOL, "sources": datasets, "splits": splits})
            atomic_json(store.manifest_path, store.metadata)
        finally:
            store.close()
        report = {"event": "hierarchical_prepared", "prepared_root": str(target), "reused": False,
                  "directory_version": version, "banks": len(banks), "memories": len(memory_map),
                  "sessions": sum(len(b["sessions"]) for b in banks.values()),
                  "examples": dict(Counter(e.split for e in examples))}
        atomic_json(staging / "hierarchical_preparation_report.json", report)
        check = HierarchicalTrainingStore(staging)
        try:
            check.validate_directories(splits)
        finally:
            check.close()
        staging.rename(target)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_dapo_hierarchical_f1.yaml")
    parser.add_argument("--source-prepared-root", default=DEFAULT_SOURCE)
    parser.add_argument("--directory-root", default=DEFAULT_DIRECTORY)
    parser.add_argument("--datasets", nargs="+", choices=["locomo"])
    parser.add_argument("--backup-timeout", type=float, default=20)
    args = parser.parse_args()
    print(canonical(prepare_frozen_store(load_config(args.config), args.source_prepared_root,
        args.directory_root, datasets=args.datasets, backup_timeout=args.backup_timeout)), flush=True)


if __name__ == "__main__":
    main()
