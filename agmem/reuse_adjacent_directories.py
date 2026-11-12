"""Rebind existing adjacent directories to a verified query-resplit snapshot.

No models or APIs are used. Source files and the source journal are read-only;
every imported session has an atomic provenance record in the new journal.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import sqlite3

from .build_adjacent_directories import AdjacentBuildStore, load_recipe
from .core import atomic_json, canonical, digest, file_digest
from .hierarchical.adapters import _history_columns, _record
from .hierarchical.adjacent_writer import validate_spans
from .hierarchical.binding import directory_version, validate_directory
from .hierarchical.records import time_metadata, validate_session
from .hierarchical.writer import public_session, token_count
from .jobs import file_lock
from .pipeline import announce


VERSION = "adjacent_resplit_reuse.v1"
DEFAULT_CONFIG = "configs/build_adjacent_qwen_longmemeval_811.yaml"
DEFAULT_SOURCE = "runs/parent_child_directories/qwen27b_adjacent_qwen4b_v1"
DEFAULT_MAPPING = "data/processed/longmemeval_811_seed42_20261111/row_mapping.jsonl"


@contextmanager
def source_lock(root):
    """Acquire the builder's existing lock without creating anything in its root."""
    path = Path(root) / ".build.lock"
    with path.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Source directory builder is active: {path}; let it stop before reusing") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def source_connection(root):
    path = Path(root).resolve() / "build_journal.sqlite3"
    if not path.is_file():
        raise ValueError(f"Missing source directory journal: {path}")
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=60)


def compatible_identity(old, new):
    """Changing source rows is explicit; Writer/tokenizer/segmentation cannot change."""
    if old.get("protocol") != "adjacent_parent_child_directory_v1":
        raise ValueError("Reuse requires the adjacent semantic directory protocol")
    if {k: v for k, v in old.items() if k != "datasets"} != {
            k: v for k, v in new.items() if k != "datasets"}:
        raise ValueError("Source and target construction semantics differ beyond their dataset snapshots")
    # Paths, row selection and file hashes may change, but those are not the
    # only things recorded under datasets: adapter and modality policies affect
    # the actual source seen by the Writer and must remain identical.
    old_specs = {row["spec"]["name"]: row["spec"] for row in old["datasets"]}
    address_fields = {"input_root", "input_files", "path", "split", "splits"}

    def semantics(spec):
        result = {key: value for key, value in spec.items() if key not in address_fields}
        result.setdefault("format", "longmemeval")
        result.setdefault("modality_policy", "existing_captions")
        return result

    for row in new["datasets"]:
        spec = row["spec"]
        previous = old_specs.get(spec["name"])
        if previous is None or semantics(previous) != semantics(spec):
            raise ValueError(f"Dataset adapter/modality semantics changed for {spec['name']}")


def _files(identity):
    result = {}
    for item in identity["datasets"]:
        for row in item["files"]:
            result[item["spec"]["name"], row["split"]] = row
    return result


def _rows(path, wanted):
    import pyarrow as pa
    import pyarrow.parquet as pq
    result, offset = {}, 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
        for i in range(batch.num_rows):
            if offset + i in wanted:
                result[offset + i] = pa.Table.from_batches([batch.slice(i, 1)])
        offset += batch.num_rows
    if set(result) != wanted:
        raise ValueError(f"Mapping references missing rows in {path}")
    return result


def verified_mappings(path, old_identity, new_identity, datasets, splits):
    """Verify mapping, file hashes, all-column row equality, and bank addresses."""
    path = Path(path).resolve()
    manifest = json.loads((path.parent / "manifest.json").read_text())
    if manifest.get("row_mapping", {}).get("sha256") != file_digest(path):
        raise ValueError("Row mapping differs from the resplit manifest")
    mappings = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(mappings) != manifest["row_mapping"]["rows"]:
        raise ValueError("Row mapping count differs from the resplit manifest")
    selected = [m for m in mappings if m["source"] in datasets and m["new_split"] in splits]
    if not selected:
        raise ValueError("No mapped rows match the requested datasets and splits")
    old_files, new_files = _files(old_identity), _files(new_identity)
    hashes, wanted_old, wanted_new = {}, defaultdict(set), defaultdict(set)
    seen_old, seen_new = set(), set()
    for row in selected:
        source, osplit, nsplit = row["source"], row["old_split"], row["new_split"]
        oi, ni = row["old_row_index"], row["new_row_index"]
        if (type(oi) is not int or oi < 0 or type(ni) is not int or ni < 0
                or (source, osplit, oi) in seen_old or (source, nsplit, ni) in seen_new):
            raise ValueError("Duplicate or invalid source/target row address in mapping")
        seen_old.add((source, osplit, oi)); seen_new.add((source, nsplit, ni))
        for prefix, split, index, files in (("old", osplit, oi, old_files), ("new", nsplit, ni, new_files)):
            file = files.get((source, split))
            if file is None or file["sha256"] != row[f"{prefix}_file_sha256"]:
                raise ValueError("Mapped source file differs from the recorded construction identity")
            filename = str(Path(file["path"]).resolve())
            if filename not in hashes:
                hashes[filename] = file_digest(filename)
            if hashes[filename] != file["sha256"]:
                raise ValueError(f"Source parquet changed: {filename}")
            if row[f"{prefix}_bank_id"] != "bank_" + digest([source, split, file["sha256"], index])[:24]:
                raise ValueError("Mapped bank ID differs from its source address")
        wanted_old[source, osplit].add(oi)
        wanted_new[source, nsplit].add(ni)
    old_rows = {key: _rows(old_files[key]["path"], indices) for key, indices in wanted_old.items()}
    new_rows = {key: _rows(new_files[key]["path"], indices) for key, indices in wanted_new.items()}
    verified = []
    for mapping in selected:
        source = mapping["source"]
        old = old_rows[source, mapping["old_split"]][mapping["old_row_index"]]
        new = new_rows[source, mapping["new_split"]][mapping["new_row_index"]]
        if not old.equals(new, check_metadata=True):
            raise ValueError(f"Mapped original row changed: {source}/{mapping['question_id']}")
        raw = new.to_pylist()[0]
        if raw.get("question_id") != mapping["question_id"]:
            raise ValueError("Mapped question ID differs from its original row")
        sessions, _, _ = _history_columns(raw)
        if len(sessions) != mapping["source_session_count"]:
            raise ValueError("Mapped session count differs from its original row")
        verified.append((mapping, raw, new_files[source, mapping["new_split"]]["path"]))
    return verified, hashes


def verify_bank(bank, mapping, old_file, source_sessions, decoded, source_format):
    if bank["bank_id"] != mapping["old_bank_id"] or bank["dataset"] != mapping["source"]:
        raise ValueError("Source bank does not match the mapped dataset")
    memberships = bank["query_memberships"]
    if len(memberships) != 1 or any(memberships[0].get(k) != v for k, v in {
            "original_question_id": mapping["question_id"], "split": mapping["old_split"],
            "row_index": mapping["old_row_index"]}.items()):
        raise ValueError("Source bank question membership differs from the row mapping")
    if not bank["source_map"]:
        raise ValueError("Source bank has no source provenance")
    observed, unique = [], set()
    for row in bank["source_map"]:
        expected = {"file_sha256": mapping["old_file_sha256"], "split": mapping["old_split"],
                    "row_index": mapping["old_row_index"], "original_question_id": mapping["question_id"]}
        if (any(row.get(k) != v for k, v in expected.items())
                or Path(row["file"]).resolve() != Path(old_file).resolve()):
            raise ValueError("Source bank contains inconsistent original file provenance")
        if "bank_id" in row and row["bank_id"] != mapping["old_bank_id"]:
            raise ValueError("Source map bank address differs from the original bank")
        encoded = canonical(row)
        if encoded in unique:
            raise ValueError("Source map contains duplicate addresses")
        unique.add(encoded)
        if row.get("parent_id") in source_sessions:
            observed.append(encoded)
    # Source banks can already contain the provenance for a whole history when
    # only some session snapshots were committed. Validate full coverage for
    # the snapshots being imported; the other map rows are not propagated.
    histories, original_ids, _ = decoded
    expected_rows = []
    for pid, session in source_sessions.items():
        index = session["order"] - 1
        base = {"file": str(Path(old_file).resolve()), "file_sha256": mapping["old_file_sha256"],
                "split": mapping["old_split"], "row_index": mapping["old_row_index"],
                "original_question_id": mapping["question_id"], "source_format": source_format,
                "bank_id": mapping["old_bank_id"], "parent_id": pid, "session_index": index,
                "original_session_id": original_ids[index], "source_hash": session["source_hash"]}
        expected_rows.append(canonical(base))
        for message_index, (record, message) in enumerate(zip(session["records"], histories[index])):
            expected_rows.append(canonical({**base, "record_id": record["record_id"],
                                           "message_index": message_index,
                                           "original_message_id": message.get("dia_id", message.get("id"))}))
    if Counter(observed) != Counter(expected_rows):
        raise ValueError("Source map session/record addresses or complete coverage differ from the verified source")


def verify_session_source(session, decoded):
    sessions, _, dates = decoded
    order, pid = session["order"], session["parent_id"]
    if not 1 <= order <= len(sessions) or pid != f"p{order:04d}":
        raise ValueError("Source session does not match its original position")
    records = [_record(message, pid, i) for i, message in enumerate(sessions[order - 1], 1)]
    if (session["source_kind"] != "dialogue" or session["records"] != records
            or session["header_records"] or session["time"] != time_metadata(dates[order - 1])
            or session["participants"] != list(dict.fromkeys(r["speaker"] for r in records if r["speaker"]))):
        raise ValueError("Source snapshot text/metadata differs from the verified parquet history")


def rebind_session(session, directory, new_bank_id, *, count_tokens=None):
    validate_session(session, count_tokens=count_tokens)
    if directory is not None:
        validate_directory(directory, session, count_tokens=count_tokens)
    target = deepcopy(session)
    target["bank_id"] = new_bank_id
    target["source_hash"] = digest({k: v for k, v in target.items() if k != "source_hash"})
    validate_session(target, count_tokens=count_tokens)
    rebound = deepcopy(directory)
    if rebound is not None:
        rebound["bank_id"] = new_bank_id
        rebound["build"]["source_hash"] = target["source_hash"]
        rebound["source"]["snapshot_id"] = "snapshot_" + target["source_hash"]
        rebound["directory_version"] = directory_version(rebound)
        validate_directory(rebound, target, count_tokens=count_tokens)
    return target, rebound


def rebind_bank(bank, mapping, new_file, session_hashes):
    new = deepcopy(bank)
    new["bank_id"] = mapping["new_bank_id"]
    new["query_memberships"] = [{"sample_id": f"{mapping['source']}/{mapping['new_split']}/{mapping['new_row_index']:06d}",
                                  "split": mapping["new_split"], "row_index": mapping["new_row_index"],
                                  "original_question_id": mapping["question_id"]}]
    new["source_map"] = []
    for row in bank["source_map"]:
        pid = row.get("parent_id")
        if pid is not None and pid not in session_hashes:
            continue  # The regular builder will register the remaining source sessions.
        row = {**row, "bank_id": mapping["new_bank_id"], "file": str(Path(new_file).resolve()),
               "file_sha256": mapping["new_file_sha256"], "split": mapping["new_split"],
               "row_index": mapping["new_row_index"]}
        if "source_hash" in row:
            row["source_hash"] = session_hashes[pid]
        new["source_map"].append(row)
    for issue in new.get("issues", []):
        if "bank_id" in issue:
            issue["bank_id"] = mapping["new_bank_id"]
        if "source_file" in issue:
            issue["source_file"] = str(Path(new_file).resolve())
        if "row_index" in issue:
            issue["row_index"] = mapping["new_row_index"]
    return new


def _segment_identities(db, identity):
    result = [identity]
    row = db.execute("SELECT value FROM metadata WHERE key='parent_budget_migrations'").fetchone()
    for migration in reversed(json.loads(row[0]) if row else []):
        previous = migration["from_identity"]
        relaxed = deepcopy(previous)
        relaxed["writer_settings"]["parent_content_tokens"] = identity["writer_settings"]["parent_content_tokens"]
        if relaxed == identity and previous not in result:
            result.append(previous)
    return result


def _segments(db, session, identities, modality):
    public = public_session(session, modality)
    for identity in identities:
        key = digest({"input": public, "construction": identity, "modality_policy": modality})
        row = db.execute("SELECT payload FROM segment_cache WHERE key=?", (key,)).fetchone()
        if row:
            value = json.loads(row[0])
            if any(value.get("stats", {}).get(k, 0) for k in ("review_failures", "review_unavailable")):
                continue
            validate_spans(value["spans"], len(session["blocks"]))
            return key, value
    return None, None


def run(args, *, tokenizer=None):
    config, identity = load_recipe(args.config, getattr(args, "workers", None),
        embedding_model_path=getattr(args, "embedding_model_path", None),
        output_root=getattr(args, "output_root", None), allow_missing_embedding=args.dry_run)
    source, target = Path(args.reuse_root).resolve(), Path(config["output_root"]).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Reuse target must be separate from its immutable source")
    datasets, splits = args.datasets or ["split_longmemeval_s"], args.splits or ["train", "validation", "test"]
    specs = {s["name"]: s for s in config["datasets"]}
    if not set(datasets).issubset(specs):
        raise ValueError("Requested reuse datasets are absent from target recipe")
    count_tokens = (lambda text: token_count(tokenizer, text)) if tokenizer is not None else None
    report = {"version": VERSION, "source_root": str(source), "output_root": str(target),
              "datasets": datasets, "splits": splits, "status": "planned" if args.dry_run else "running",
              "mapping_sha256": file_digest(Path(args.mapping)),
              "timestamp": datetime.now(timezone.utc).isoformat()}
    counters = Counter()
    with source_lock(source):
        db = source_connection(source)
        try:
            old_identity = json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
            compatible_identity(old_identity, identity)
            mappings, hashes = verified_mappings(args.mapping, old_identity, identity, datasets, splits)
            report["selected_queries"] = len(mappings)
            old_files = _files(old_identity)
            old_cache_ids = _segment_identities(db, old_identity)
            if args.dry_run:
                for mapping, _, _ in mappings:
                    rows = db.execute("SELECT status,COUNT(*) FROM sessions WHERE bank_id=? GROUP BY status",
                                      (mapping["old_bank_id"],)).fetchall()
                    counters.update({status: n for status, n in rows})
                    counters["unregistered"] += mapping["source_session_count"] - sum(n for _, n in rows)
                report["source_states"] = dict(counters)
                announce("adjacent_directory_reuse_plan", **report)
                return report
            with file_lock(target / ".build.lock"):
                store = AdjacentBuildStore(target, identity)
                try:
                    with store.db:
                        store.db.execute("CREATE TABLE IF NOT EXISTS reuse_provenance (bank_id TEXT NOT NULL, "
                                         "parent_id TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(bank_id,parent_id))")
                        contract = {"version": VERSION, "source_root": str(source),
                                    "source_identity": digest(old_identity), "target_identity": digest(identity),
                                    "mapping_sha256": report["mapping_sha256"]}
                        previous = store.db.execute("SELECT value FROM metadata WHERE key='resplit_reuse'").fetchone()
                        if previous is not None and json.loads(previous[0]) != contract:
                            raise ValueError("Target resplit reuse provenance changed")
                        store.db.execute("INSERT OR REPLACE INTO metadata VALUES ('resplit_reuse',?)", (canonical(contract),))
                    for mapping, raw, new_file in mappings:
                        row = db.execute("SELECT payload FROM banks WHERE bank_id=?", (mapping["old_bank_id"],)).fetchone()
                        if row is None:
                            counters["unregistered"] += mapping["source_session_count"]
                            continue
                        bank = json.loads(row[0])
                        rows = db.execute("SELECT parent_id,snapshot,status,directory,error,result_stats FROM sessions "
                                          "WHERE bank_id=? ORDER BY parent_id", (mapping["old_bank_id"],)).fetchall()
                        decoded, prepared = _history_columns(raw), []
                        for pid, snapshot, status, directory, error, stats in rows:
                            old_session = json.loads(snapshot)
                            old_directory = json.loads(directory) if directory else None
                            if old_session["bank_id"] != mapping["old_bank_id"] or old_session["parent_id"] != pid:
                                raise ValueError("Source session address differs from its journal row")
                            verify_session_source(old_session, decoded)
                            if status not in {"complete", "empty", "fallback", "failed", "pending"}:
                                raise ValueError("Unsupported source session status")
                            if (status in {"complete", "empty", "fallback"}) != (old_directory is not None):
                                raise ValueError("Source directory presence differs from its session status")
                            if old_directory is not None:
                                expected = {"writer_model": old_identity["writer"]["model"],
                                            "writer_revision": old_identity["writer"]["model_revision"],
                                            "prompt_version": old_identity["prompt_version"], "status": status}
                                if (any(old_directory["build"][k] != v for k, v in expected.items())
                                        or old_directory["source"]["tokenizer_id"] != old_identity["tokenizer_id"]):
                                    raise ValueError("Source directory Writer/tokenizer provenance differs")
                            session, directory = rebind_session(old_session, old_directory, mapping["new_bank_id"],
                                                                count_tokens=count_tokens)
                            prepared.append((old_session, old_directory, session, directory, status, error, stats))
                        old_specs = {item["spec"]["name"]: item["spec"] for item in old_identity["datasets"]}
                        verify_bank(bank, mapping, old_files[mapping["source"], mapping["old_split"]]["path"],
                                    {row[0]["parent_id"]: row[0] for row in prepared}, decoded,
                                    old_specs[mapping["source"]].get("format", "longmemeval"))
                        store.add_bank(rebind_bank(bank, mapping, new_file,
                                                  {row[2]["parent_id"]: row[2]["source_hash"] for row in prepared}))
                        counters["unregistered"] += mapping["source_session_count"] - len(rows)
                        for old_session, old_directory, session, directory, status, error, stats in prepared:
                            pid, bid = session["parent_id"], session["bank_id"]
                            with store.db:
                                existing = store.db.execute("SELECT snapshot FROM sessions WHERE bank_id=? AND parent_id=?",
                                                            (bid, pid)).fetchone()
                                if existing is not None:
                                    if json.loads(existing[0]) != session:
                                        raise ValueError("Existing target source differs from verified imported source")
                                    counters["target_sessions_preserved"] += 1
                                    continue
                                modality = specs[mapping["source"]].get("modality_policy", "existing_captions")
                                old_key, segmentation = _segments(db, old_session, old_cache_ids, modality)
                                new_key = digest({"input": public_session(session, modality), "construction": identity,
                                                  "modality_policy": modality})
                                provenance = {"version": VERSION, "source_root": str(source),
                                              "source_bank_id": old_session["bank_id"], "bank_id": bid, "parent_id": pid,
                                              "source_hash_before": old_session["source_hash"],
                                              "source_hash_after": session["source_hash"],
                                              "directory_version_before": old_directory["directory_version"] if old_directory else None,
                                              "directory_version_after": directory["directory_version"] if directory else None,
                                              "source_status": status, "source_result_stats": json.loads(stats) if stats else None,
                                              "source_error": json.loads(error) if error else None,
                                              "mapping": mapping, "segment_cache_before": old_key,
                                              "segment_cache_after": new_key if segmentation is not None else None}
                                # failed/pending remain eligible for a fresh build; old error is preserved in provenance.
                                target_status = status if directory is not None else "pending"
                                store.db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?)",
                                    (bid, pid, session["source_hash"], canonical(session), target_status,
                                     canonical(directory) if directory else None, None, stats))
                                store.db.execute("INSERT INTO reuse_provenance VALUES (?,?,?)", (bid, pid, canonical(provenance)))
                                if segmentation is not None:
                                    store.db.execute("INSERT OR IGNORE INTO segment_cache VALUES (?,?)", (new_key, canonical(segmentation)))
                                    counters["segment_caches_reused"] += 1
                                counters["imported_" + target_status] += 1
                        counters["banks_reused"] += 1
                        announce("adjacent_directory_reuse_progress", bank_id=mapping["new_bank_id"], **dict(counters))
                    for path, expected in hashes.items():
                        if file_digest(path) != expected:
                            raise ValueError(f"Input parquet changed during reuse: {path}")
                    report.update(status="ready", counters=dict(counters), target=store.report())
                    atomic_json(target / "reuse_report.json", report)
                    with (target / "reuse_provenance.jsonl.tmp").open("w") as stream:
                        for (payload,) in store.db.execute("SELECT payload FROM reuse_provenance ORDER BY bank_id,parent_id"):
                            stream.write(payload + "\n")
                    (target / "reuse_provenance.jsonl.tmp").replace(target / "reuse_provenance.jsonl")
                finally:
                    store.close()
        finally:
            db.close()
    announce("adjacent_directory_reuse_finished", **report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--reuse-root", default=DEFAULT_SOURCE)
    parser.add_argument("--mapping", default=DEFAULT_MAPPING)
    parser.add_argument("--datasets", nargs="+", default=["split_longmemeval_s"])
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["train", "validation", "test"])
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--embedding-model-path")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
