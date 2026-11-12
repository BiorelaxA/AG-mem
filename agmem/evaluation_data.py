"""Prepare isolated test stores and reuse directories without changing provenance."""

from copy import deepcopy
from pathlib import Path
from collections import Counter
import json

from .core import canonical, digest, file_digest
from .jobs import ensure_prepared
from .pipeline import build_directories, open_prepared, validate_directories, validate_directory_version
from .store import Store


def load_existing_source(base_config, source, prepared_path):
    """Validate a completed mixed flat test store without preparing or repairing it.

    The explicit store path supports the local builder's ``test/source`` layout.
    Missing, partial, changed or unverified directories fail closed; evaluation
    must never silently call the Writer to complete these required snapshots.
    """
    from .build_local_directories import PROTOCOL, ProvenanceStore, _check_entry
    from .data import ADAPTER_VERSION, task_examples
    from .prompts import writer_messages
    from .store import _JsonlFile
    import pyarrow.parquet as pq

    if not source or source in {".", "..", "_runtime"} or Path(source).name != source:
        raise ValueError(f"Invalid evaluation source: {source}")
    root = Path(prepared_path).resolve()
    required = ("manifest.json", "local_build_manifest.json", "local_build_status.json",
                "directory_manifest.json", "directory_provenance.jsonl", "directories.jsonl",
                "memory_records.jsonl", "test_examples.jsonl")
    if any(not (root / name).is_file() for name in required):
        raise ValueError(f"Required test directories are not complete: {root}; finish the directory builder first")
    status = json.loads((root / "local_build_status.json").read_text())
    if status.get("status") != "ready" or status.get("source") != source or status.get("group") != "test":
        raise ValueError(f"Required directory builder has not published a complete test snapshot: {root} "
                         f"(status={status.get('status')}); finish the directory builder first")
    config = deepcopy(base_config)
    config["data"].update(layout="combined", sources=[source], splits=["test"], prepared_root=str(root))
    input_root = Path(config["data"]["input_root"])
    input_hash = file_digest(input_root / "manifest.json")
    expected_preparation = {"input_manifest_sha256": input_hash, "sources": [source], "splits": ["test"],
                            "limit_per_source_split": None, "adapter_version": ADAPTER_VERSION,
                            "storage_format": Store.FORMAT}
    metadata = json.loads((root / "manifest.json").read_text())
    if metadata.get("prepare_complete") is not True or metadata.get("prepare_identity") != expected_preparation:
        raise ValueError("Required prepared test store differs from the selected dataset snapshot")
    contract = json.loads((root / "local_build_manifest.json").read_text())
    if (contract.get("protocol") != PROTOCOL or contract.get("input_manifest_sha256") != input_hash
            or contract.get("writer_config") != config["writer"]
            or contract.get("writer_prompt_sha256") != digest(writer_messages([], {}, 300))):
        raise ValueError("Required directory construction differs from the original Writer prompt/configuration")
    mixed_identity = {"base_url": "http://mixed-session-directory.invalid/v1",
                      "model": "mixed-session-directory", "model_revision": digest(contract)}
    if metadata.get("directory_service_identity") != mixed_identity:
        raise ValueError("Required directory Writer identity differs from its provenance contract")
    version = metadata.get("directory_version")
    if (status.get("directory_version") != version
            or metadata.get("directory_empty_pending") != 0):
        raise ValueError(f"Required directory builder has not published a consistent test snapshot: {root}")
    config["data"]["frozen_directory_version"] = version
    # Do not close this validation-only Store: close() can persist rebuilt JSONL
    # indexes. No completion cache, directory entry or metadata is written here.
    store = open_prepared(config)
    validate_directories(config, store, ["test"])
    examples = task_examples(config, store, "test")
    count = pq.read_metadata(input_root / source / "test.parquet").num_rows
    if (not examples or len(examples) != count
            or {e.sample_id for e in examples} != {f"{source}/test/{i:06d}" for i in range(count)}
            or any(e.source != source or e.split != "test" for e in examples)):
        raise ValueError("Required test snapshot does not contain every input query exactly once")
    histories = store.memory_ids()
    if set(histories) != {e.memory_id for e in examples}:
        raise ValueError("Required test histories differ from the selected query histories")
    specs = contract.get("origins", {}).get(source, {}).get("test")
    if not isinstance(specs, list):
        raise ValueError("Required directory contract has no test-source provenance")
    allowed_origins = [{"kind": "generated", "writer_identity": contract["local_writer"]}]
    allowed_origins.extend(ProvenanceStore.import_origin(spec) for spec in specs if spec.get("available", True))
    allowed_origins = {canonical(origin) for origin in allowed_origins}
    provenance = _JsonlFile(root / "directory_provenance.jsonl", ("memory_id", "version", "id"))
    expected_keys, counts = set(), Counter()
    for memory_id in histories:
        records = store.histories.get((memory_id,))["records"]
        references = [{key: value for key, value in record.items() if key != "messages"} for record in records]
        if memory_id != "mem_" + digest(references):
            raise ValueError("Required test history identity differs from its source records")
        entries = store.directory(memory_id, version)
        for record, entry in zip(records, entries):
            if record.get("session_hash") != digest({"messages": record["messages"]}):
                raise ValueError("Required test session body differs from its recorded content hash")
            _check_entry(entry, record)
            key = memory_id, version, record["id"]
            expected_keys.add(key)
            prior = provenance.get(key)
            if (prior is None or prior.get("entry") != entry or prior.get("entry_sha256") != digest(entry)
                    or canonical(prior.get("origin")) not in allowed_origins):
                raise ValueError(f"Required directory entry lacks matching Writer provenance: {memory_id}/{record['id']}")
            counts[prior["origin"].get("writer_label", "qwen3.5-27B (new)")] += 1
    if (set(store.directory_entries.keys()) != expected_keys or set(provenance.keys()) != expected_keys
            or status.get("histories") != len(histories) or status.get("sessions") != len(expected_keys)
            or status.get("by_writer") != dict(counts)):
        raise ValueError("Required directory/provenance coverage differs from its completed build report")
    report = _report(config, store, "ready")
    report.update(required_existing=True, prepared=str(root), validated_provenance_entries=len(expected_keys),
                  by_writer=dict(counts))
    return report


def _origin_store(reuse_root, source):
    if reuse_root is None:
        return None
    root = Path(reuse_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Directory reuse root does not exist: {root}")
    path = root / source if (root / source / "manifest.json").is_file() else root
    if not (path / "manifest.json").is_file():
        return None
    origin = Store(path)
    if origin.get_metadata("prepare_complete") is not True:
        raise ValueError(f"Directory reuse store preparation is incomplete: {path}")
    identity = origin.get_metadata("prepare_identity", {})
    if identity.get("sources") != [source]:
        raise ValueError(f"Directory reuse store has a different source: {path}")
    # Source stores are read-only here. Do not call Store.close(): it may persist
    # a rebuilt byte-offset index in an otherwise unchanged training directory.
    return origin


def _current_writer_version(config):
    from types import SimpleNamespace
    from .remote_writer import writer_identity
    from .services import DirectoryBuilder

    identity = writer_identity(config)
    return DirectoryBuilder(SimpleNamespace(FORMAT=Store.FORMAT),
                            SimpleNamespace(identity=identity), config["writer"], None).version


def _report(config, store, status, reuse=None):
    reuse = reuse or store.get_metadata("evaluation_directory_reuse", {})
    histories = store.memory_ids()
    return {
        "source": config["data"]["sources"][0],
        "config": config,
        "status": status,
        "examples": len(store.examples("test")),
        "histories": len(histories),
        "sessions": sum(len(store.memory(mid)) for mid in histories),
        "reused_histories": reuse.get("reused_histories", 0),
        "reused_entries": reuse.get("reused_entries", 0),
        "missing_histories": 0 if status == "ready" else len(histories) - reuse.get("reused_histories", 0),
        "directory_version": store.get_metadata("directory_version"),
        "reuse": reuse,
    }


def prepare_source(base_config, source, prepared_root, reuse_root=None):
    """Prepare ``prepared_root/source`` for test only, without model/API calls.

    ``reuse_root`` can be a parent of per-source stores or one source store.
    Complete matching histories may reuse their original directory snapshots.
    Full coverage freezes the original Writer version, even after a model-file
    migration. Partial reuse requires the current Writer version to match; an
    explicit error prevents old summaries being relabelled as newly generated.
    The returned ``config`` is the combined, one-source test configuration.
    """
    if source in {".", "..", "_runtime"} or Path(source).name != source:
        raise ValueError(f"Invalid evaluation source: {source}")
    config = deepcopy(base_config)
    config["data"].update(layout="combined", sources=[source], splits=["test"],
                          prepared_root=str((Path(prepared_root) / source).resolve()))
    config["data"].pop("frozen_directory_version", None)
    target_path = Path(config["data"]["prepared_root"])
    origin = _origin_store(reuse_root, source)
    if origin is not None and target_path == origin.root.resolve():
        raise ValueError("Test preparation must use a separate store from the directory reuse source")
    ensure_prepared(config)
    target = open_prepared(config)
    try:
        # Idempotent reruns retain a completed snapshot, including its original
        # Writer identity. Validation rejects pending or inconsistent entries.
        version = target.get_metadata("directory_version")
        if version is not None:
            ready = deepcopy(config)
            ready["data"]["frozen_directory_version"] = version
            # Metadata/version mismatches always fail. A current-Writer build
            # with missing or empty summaries can be retried by the build stage.
            validate_directory_version(ready, target)
            try:
                validate_directories(ready, target, ["test"])
            except ValueError:
                if version != _current_writer_version(config):
                    raise
                return _report(config, target, "needs_build")
            else:
                return _report(ready, target, "ready")

        matches = {}
        incomplete = []
        source_version = origin.get_metadata("directory_version") if origin else None
        if source_version:
            frozen = deepcopy(config)
            frozen["data"]["frozen_directory_version"] = source_version
            validate_directory_version(frozen, origin)
            available = set(origin.memory_ids())
            for memory_id in target.memory_ids():
                if memory_id not in available:
                    continue
                if origin.histories.get((memory_id,)) != target.histories.get((memory_id,)):
                    raise ValueError(f"Matching history IDs have different complete records: {memory_id}")
                try:
                    entries = origin.directory(memory_id, source_version)
                except ValueError:
                    incomplete.append(memory_id)
                    continue
                if any(entry.get("writer_status") == "empty_pending" or not entry["content"].strip()
                       for entry in entries):
                    incomplete.append(memory_id)
                    continue
                matches[memory_id] = entries

        full_reuse = bool(matches) and len(matches) == len(target.memory_ids())
        if matches and not full_reuse and source_version != _current_writer_version(config):
            raise ValueError(
                "Only part of the test histories have reusable directories, and their Writer version "
                "differs from the current Writer. Use matching Writer provenance or separate "
                "evaluation snapshots; old summaries cannot be relabelled as the new version."
            )
        for memory_id, entries in matches.items():
            target.put_directory(memory_id, source_version, entries)
        reuse = {
            "source_root": str(origin.root.resolve()) if origin else None,
            "source_version": source_version,
            "reused_histories": len(matches),
            "reused_entries": sum(len(entries) for entries in matches.values()),
            "incomplete_source_histories": incomplete,
            "scope": "identical_complete_history_and_public_metadata",
        }
        target.set_metadata("evaluation_directory_reuse", reuse)
        if full_reuse:
            target.set_metadata("directory_version", source_version)
            target.set_metadata("directory_service_identity", deepcopy(origin.get_metadata("directory_service_identity")))
            target.set_metadata("directory_empty_pending", 0)
            config["data"]["frozen_directory_version"] = source_version
            validate_directories(config, target, ["test"])
        return _report(config, target, "ready" if full_reuse else "needs_build", reuse)
    finally:
        target.close()


def build_source_directories(prepared):
    """Build missing test directories with the configured Writer.

    This is the explicit model/API stage. The resulting configuration freezes
    the actual published version for all checkpoint/Task-agent evaluations.
    """
    config = deepcopy(prepared["config"])
    if prepared["status"] == "ready":
        store = open_prepared(config)
        try:
            validate_directories(config, store, ["test"])
            return _report(config, store, "ready")
        finally:
            store.close()
    if prepared["status"] != "needs_build":
        raise ValueError(f"Unknown evaluation preparation status: {prepared['status']}")
    config["data"].pop("frozen_directory_version", None)
    version = build_directories(config)
    config["data"]["frozen_directory_version"] = version
    store = open_prepared(config)
    try:
        validate_directories(config, store, ["test"])
        return _report(config, store, "ready")
    finally:
        store.close()
