"""Complete LongMemEval directories locally, preserving existing Writer summaries."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from urllib.parse import urlsplit

from .core import InfrastructureError, atomic_json, digest, file_digest
from .jobs import ensure_prepared, file_lock
from .pipeline import announce, validate_directories
from .prompts import writer_messages
from .services import ChatService, DirectoryBuilder
from .store import Store, _JsonlFile


PROTOCOL = "reused_and_local_session_directories_v1"
SOURCES = ("split_longmemeval_s", "split_longmemeval_m")


def _version(identity, writer):
    return DirectoryBuilder(SimpleNamespace(FORMAT=Store.FORMAT),
                            SimpleNamespace(identity=identity), writer, None).version


def _writer_identity(recorded):
    if recorded.get("request_profile") == "openai_chat":
        return deepcopy(recorded)
    return {key: recorded[key] for key in ("base_url", "model", "model_revision")}


def _origin_spec(root, source, label, local_identity, writer):
    """Resolve provenance without requiring credentials or changing source files."""
    path = Path(root).resolve() / source
    manifest = path / "manifest.json"
    identity = local_identity
    if manifest.exists():
        metadata = json.loads(manifest.read_text())
        if metadata.get("prepare_complete") is not True:
            raise ValueError(f"Reuse source preparation is incomplete: {path}")
        if metadata.get("prepare_identity", {}).get("sources") != [source]:
            raise ValueError(f"Reuse source dataset differs: {path}")
        recorded = metadata.get("directory_service_identity")
        if recorded:
            identity = _writer_identity(recorded)
        elif (path.parent / "remote_build_manifest.json").exists():
            remote = json.loads((path.parent / "remote_build_manifest.json").read_text())
            if remote["writer_config"] != writer:
                raise ValueError(f"Reuse Writer configuration differs: {path}")
            identity = remote["writer"]
        expected = _version(identity, writer)
        published = metadata.get("directory_version")
        if published and published != expected:
            raise ValueError(f"Reuse directory version differs from its Writer provenance: {path}")
    else:
        # An absent source can be skipped, but its future appearance must not
        # silently change an existing snapshot's provenance contract.
        return {"root": str(path), "label": label, "available": False}
    return {"root": str(path), "label": label, "available": True,
            "writer_identity": identity, "directory_version": _version(identity, writer)}


def load_recipe(path, workers=None, base_urls=None):
    import yaml
    from .policy import model_fingerprint

    recipe = yaml.safe_load(Path(path).read_text())
    saved = json.loads((Path(recipe["training_run"]) / "run.json").read_text())
    config = deepcopy(saved["config"])
    if config["writer"].get("lookback_sessions") != 3 or config["writer"].get("content_max_tokens") != 300:
        raise ValueError("The reference Writer must use three preceding raw sessions and 300 summary tokens")
    config.pop("writer_service", None)
    config["data"].pop("frozen_directory_version", None)
    config["models"]["task_path"] = str(Path(recipe.get("model_path", config["models"]["task_path"])).resolve())
    config["data"]["input_root"] = str(Path(recipe["input_root"]).resolve())
    config["data"]["prepared_root"] = str(Path(recipe["prepared_root"]).resolve())
    config["service"] = {"base_url": "http://127.0.0.1:50543/v1", "model": "agmem-27b",
                         "api_key_env": "AGMEM_API_KEY", "timeout_seconds": recipe.get("timeout_seconds", 600),
                         "retries": recipe.get("retries", 3), "request_profile": "local_vllm"}
    count = workers if workers is not None else recipe.get("writer_workers", 48)
    if type(count) is not int or count < 1:
        raise ValueError("workers must be a positive integer")
    config.setdefault("execution", {})["writer_workers"] = count
    urls = base_urls or recipe.get("base_urls", ["http://127.0.0.1:8000/v1"])
    if not isinstance(urls, list) or not urls or len(set(urls)) != len(urls):
        raise ValueError("Provide unique local --base-urls")
    for url in urls:
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Writer endpoints must be HTTP(S) URLs without credentials or query strings")
    local_identity = {key: config["service"][key] for key in ("base_url", "model")}
    local_identity["model_revision"] = model_fingerprint(config["models"]["task_path"])["fingerprint"]
    origins = {}
    for source in SOURCES:
        origins[source] = {
            "train_validation": [_origin_spec(recipe["reuse_train_root"], source, "qwen3.5-27B",
                                                local_identity, config["writer"])],
            "test": [_origin_spec(item["root"], source, item["label"], local_identity, config["writer"])
                     for item in recipe["reuse_test_roots"]],
        }
    target = Path(config["data"]["prepared_root"])
    for groups in origins.values():
        for specs in groups.values():
            for spec in specs:
                old = Path(spec["root"])
                if old == target or old in target.parents or target in old.parents:
                    raise ValueError("The completed directory root must be separate from every reuse source")
    contract = {"protocol": PROTOCOL, "input_manifest_sha256": file_digest(
                    Path(config["data"]["input_root"]) / "manifest.json"),
                "writer_config": config["writer"], "local_writer": local_identity,
                "writer_prompt_sha256": digest(writer_messages([], {}, 300)), "origins": origins,
                "selection": "keep_committed_then_first_nonempty_source_then_local",
                "provenance_file": "directory_provenance.jsonl"}
    # This is explicitly a mixed snapshot identity, never a claim that the
    # local model generated imported GPT summaries. It fits the frozen reader
    # contract; the full preimage is persisted beside each prepared dataset.
    identity = {"base_url": "http://mixed-session-directory.invalid/v1",
                "model": "mixed-session-directory", "model_revision": digest(contract)}
    return recipe, config, contract, identity, urls


def selected_jobs(datasets=None, splits=None):
    datasets, splits = datasets or list(SOURCES), splits or ["train", "validation", "test"]
    if not datasets or len(set(datasets)) != len(datasets) or not set(datasets).issubset(SOURCES):
        raise ValueError("Select unique LongMemEval S/M datasets")
    if not splits or len(set(splits)) != len(splits) or not set(splits).issubset({"train", "validation", "test"}):
        raise ValueError("Select unique train/validation/test splits")
    if ("train" in splits) != ("validation" in splits):
        raise ValueError("Select train and validation together; they share one prepared store")
    order = [("split_longmemeval_s", "test"), ("split_longmemeval_s", "train_validation"),
             ("split_longmemeval_m", "train_validation"), ("split_longmemeval_m", "test")]
    return [(source, group) for source, group in order if source in datasets
            and (group == "test" and "test" in splits or group == "train_validation" and "train" in splits)]


def build_plan(config, contract, identity, jobs, urls):
    import pyarrow.parquet as pq
    result = []
    for source, group in jobs:
        splits = ["test"] if group == "test" else ["train", "validation"]
        origins = _open_origins(contract["origins"][source][group], source)
        result.append({"source": source, "splits": splits, "group": group,
                       "queries": sum(pq.read_metadata(Path(config["data"]["input_root"]) / source /
                                                       f"{split}.parquet").num_rows for split in splits),
                       "prepared": str(Path(config["data"]["prepared_root"]) / group / source),
                       "reuse_sources": contract["origins"][source][group],
                       "source_entries": [{"root": spec["root"], "entries": len(store.directory_entries.keys())}
                                          for store, spec in origins]})
    return {"protocol": PROTOCOL, "directory_version": _version(identity, config["writer"]),
            "writer_workers": config["execution"]["writer_workers"], "endpoints": urls,
            "local_writer": contract["local_writer"], "jobs": result}


def _entry_metadata(record):
    return {"id": record["id"],
            "task_session": {"task_id": None, "session_id": record["source_session_id"]},
            "actors_source": {"actors": sorted({m.get("name") or m["role"] for m in record["messages"]}),
                              "sources": []},
            "time_order": {"time": record["time"], "order": record["order"]}}


def _check_entry(entry, record):
    if {key: value for key, value in entry.items() if key not in {"content", "writer_status"}} != _entry_metadata(record):
        raise ValueError(f"Reused directory metadata differs from original session: {record['id']}")
    if not isinstance(entry.get("content"), str):
        raise ValueError("Directory summary must be text")


class ProvenanceStore(Store):
    """Write-ahead provenance permits interruption between provenance/entry writes."""

    def __init__(self, root, local_identity, reuse_specs=()):
        super().__init__(root)
        self.local_origin = {"kind": "generated", "writer_identity": local_identity}
        self.allowed_origins = [self.local_origin, *[self.import_origin(spec) for spec in reuse_specs
                                                    if spec.get("available", True)]]
        self.provenance = _JsonlFile(self.root / "directory_provenance.jsonl", ("memory_id", "version", "id"))

    @staticmethod
    def import_origin(spec):
        return {"kind": "reused", "source_root": spec["root"], "source_version": spec["directory_version"],
                "writer_label": spec["label"], "writer_identity": spec["writer_identity"]}

    def _check_origin(self, origin):
        if origin not in self.allowed_origins:
            raise ValueError("Directory origin differs from the recorded mixed provenance contract")

    def put_with_origin(self, memory_id, version, entry, origin):
        self._check_origin(origin)
        key = (memory_id, version, entry["id"])
        old = self.provenance.get(key)
        row = {"memory_id": memory_id, "version": version, "id": entry["id"],
               "entry": entry, "entry_sha256": digest(entry), "origin": origin}
        if old and old != row and old["entry"].get("writer_status") != "empty_pending":
            raise ValueError("Committed directory provenance is immutable")
        self.provenance.put(row)
        super().put_directory_entry(memory_id, version, entry)

    def put_directory_entry(self, memory_id, version, entry):
        previous = self.provenance.get((memory_id, version, entry["id"]))
        origin = previous["origin"] if previous and previous["entry"] == entry else self.local_origin
        self.put_with_origin(memory_id, version, entry, origin)

    def recover_history(self, memory_id, version, records):
        keys = [(memory_id, version, record["id"]) for record in records]
        evidence = self.provenance.get_many(keys)
        entries = self.directory_entries.get_many(keys)
        for record, journal, row in zip(records, evidence, entries):
            if journal:
                entry = journal["entry"]
                self._check_origin(journal["origin"])
                if journal["entry_sha256"] != digest(entry):
                    raise ValueError("Directory provenance digest mismatch")
                _check_entry(entry, record)
                if row is not None and self._directory_fields(row) != entry:
                    if row.get("writer_status") != "empty_pending":
                        raise ValueError("Directory differs from its provenance journal")
                super().put_directory_entry(memory_id, version, entry)
            elif row is not None:
                raise ValueError("Directory entry has no provenance journal")

    def close(self):
        self.provenance.save_index()
        super().close()


class LocalWriterPool:
    """One completion cache, bounded concurrent slots for each identical replica."""

    def __init__(self, config, store, local_identity, mixed_identity, urls, workers):
        self.identity = mixed_identity
        self.services = [ChatService({**config["service"], **local_identity, "request_base_url": url},
                                     store, "writer") for url in urls]
        self.available = Queue()
        for slot in range(workers):
            self.available.put(self.services[slot % len(self.services)])

    def complete(self, *args, **kwargs):
        service = self.available.get()
        try:
            return service.complete(*args, **kwargs)
        finally:
            self.available.put(service)

    def close(self):
        for service in self.services:
            service.client.close()


def _open_origins(specs, source):
    result = []
    for spec in specs:
        if not spec["available"]:
            continue
        # No close(): source Store.close() would persist rebuilt indexes.
        store = Store(spec["root"])
        versions = {key[1] for key in store.directory_entries.keys()}
        if versions - {spec["directory_version"]}:
            raise ValueError(f"Reuse source contains an unverified Writer version: {spec['root']}")
        result.append((store, spec))
    return result


def import_history(target, memory_id, version, origins):
    original = target.histories.get((memory_id,))
    records = original["records"]
    target.recover_history(memory_id, version, records)
    copied = Counter()
    for origin, spec in origins:
        previous = origin.histories.get((memory_id,))
        if previous is None:
            continue
        if previous != original:
            raise ValueError(f"Matching history ID has different complete records: {memory_id}")
        keys = [(memory_id, spec["directory_version"], record["id"]) for record in records]
        for record, row in zip(records, origin.directory_entries.get_many(keys)):
            if row is None:
                continue
            entry = origin._directory_fields(row)
            _check_entry(entry, record)
            if entry.get("writer_status") == "empty_pending" or not entry["content"].strip():
                continue
            current = target.directory_entry(memory_id, version, record["id"])
            if current and current.get("writer_status") != "empty_pending" and current["content"].strip():
                continue
            target.put_with_origin(memory_id, version, entry, target.import_origin(spec))
            copied[spec["label"]] += 1
    return copied


def _build_job(base_config, contract, identity, job, urls):
    from .policy import load_tokenizer

    config = deepcopy(base_config)
    config["data"].update(layout="combined", sources=[job["source"]], splits=job["splits"],
                          prepared_root=job["prepared"])
    root = Path(job["prepared"])
    ensure_prepared(config)
    manifest_path = root / "local_build_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != contract:
        raise ValueError("Directory completion protocol changed; use a new prepared_root")
    if not manifest_path.exists():
        atomic_json(manifest_path, contract)
    atomic_json(root / "local_build_runtime.json", {"endpoints": urls,
                "writer_workers": config["execution"]["writer_workers"]})
    store = ProvenanceStore(root, contract["local_writer"], job["reuse_sources"])
    pool = None
    try:
        if sum(len(store.examples(split)) for split in job["splits"]) != job["queries"]:
            raise ValueError("Some input queries were quarantined during preparation; inspect preparation_report.json")
        version = _version(identity, config["writer"])
        if store.get_metadata("directory_version") not in (None, version):
            raise ValueError("Target directory version differs from the mixed snapshot")
        origins = _open_origins(job["reuse_sources"], job["source"])
        totals, imported = Counter(), Counter()
        histories = store.memory_ids()
        for index, memory_id in enumerate(histories):
            imported.update(import_history(store, memory_id, version, origins))
            # Lazily instantiate the HTTP clients/tokenizer only if at least one
            # session truly needs generation. A fully reused S train set does no inference.
            references = store.memory(memory_id)
            cached = [store.directory_entry(memory_id, version, ref["id"]) for ref in references]
            missing = any(entry is None or entry.get("writer_status") == "empty_pending"
                          or not entry["content"].strip() for entry in cached)
            if missing:
                if pool is None:
                    pool = LocalWriterPool(config, store, contract["local_writer"], identity, urls,
                                           config["execution"]["writer_workers"])
                    builder = DirectoryBuilder(store, pool, config["writer"], load_tokenizer(config["models"]["task_path"]))
                entries = builder.build(memory_id, workers=config["execution"]["writer_workers"])
            else:
                entries = cached
                store.put_directory(memory_id, version, entries)
            pending = sum(entry.get("writer_status") == "empty_pending" or not entry["content"].strip()
                          for entry in entries)
            totals.update(sessions=len(entries), pending=pending, complete_histories=int(not pending))
            announce("local_directory_history", source=job["source"], group=job["group"],
                     completed=index + 1, total=len(histories), **dict(totals), reused=dict(imported))
        if totals["pending"]:
            raise InfrastructureError(f"{totals['pending']} empty summaries remain; rerun to retry them")
        # Check provenance coverage/digests before publishing anything consumable.
        expected = {(mid, version, ref["id"]) for mid in histories for ref in store.memory(mid)}
        if set(store.directory_entries.keys()) != expected or set(store.provenance.keys()) != expected:
            raise ValueError("Mixed directory/provenance coverage differs from the complete input histories")
        counts = Counter()
        for memory_id in histories:
            store.recover_history(memory_id, version, store.histories.get((memory_id,))["records"])
            for ref in store.memory(memory_id):
                row = store.provenance.get((memory_id, version, ref["id"]))
                origin = row["origin"]
                counts[origin.get("writer_label", "qwen3.5-27B (new)")] += 1
        store.set_metadata("directory_service_identity", identity)
        store.set_metadata("directory_empty_pending", 0)
        store.set_metadata("directory_version", version)
        config["data"]["frozen_directory_version"] = version
        validate_directories(config, store, job["splits"])
        atomic_json(root / "completed_config.json", config)
        report = {"status": "ready", "source": job["source"], "group": job["group"],
                  "directory_version": version, "histories": len(histories), "sessions": totals["sessions"],
                  "by_writer": dict(counts), "prepared": str(root)}
        atomic_json(root / "local_build_status.json", report)
        announce("local_directory_source_complete", **report)
        return report
    finally:
        if pool is not None:
            pool.close()
        store.close()


def run(args):
    urls = getattr(args, "base_urls", None) or ([args.base_url] if getattr(args, "base_url", None) else None)
    recipe, config, contract, identity, urls = load_recipe(args.config, args.workers, urls)
    plan = build_plan(config, contract, identity, selected_jobs(args.datasets, args.splits), urls)
    announce("local_directory_plan", **plan)
    if args.dry_run:
        return plan
    root = Path(config["data"]["prepared_root"])
    results = {}
    with file_lock(root / ".local_build.lock"):
        contract_path = root / "local_build_manifest.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("Directory completion provenance changed; use a new prepared_root")
        if not contract_path.exists():
            atomic_json(contract_path, contract)
        for job in plan["jobs"]:
            path = Path(job["prepared"]) / "local_build_status.json"
            atomic_json(path, {"status": "building", "source": job["source"], "group": job["group"]})
            try:
                results[f"{job['group']}/{job['source']}"] = _build_job(config, contract, identity, job, urls)
            except Exception as exc:
                atomic_json(path, {"status": "failed", "source": job["source"], "group": job["group"],
                                  "error_type": type(exc).__name__,
                                  "resume": "Rerun the same command; completed entries and provenance are retained"})
                raise
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/build_longmemeval_qwen.yaml")
    urls = parser.add_mutually_exclusive_group()
    urls.add_argument("--base-url", help="Existing local Qwen vLLM endpoint")
    urls.add_argument("--base-urls", nargs="+", help="Identical Qwen replica endpoints")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--datasets", nargs="+", choices=SOURCES)
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"],
                        help="Select test and/or train validation; train and validation share a store")
    parser.add_argument("--dry-run", action="store_true", help="Read-only plan; no preparation, model load or requests")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
