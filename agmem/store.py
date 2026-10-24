from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
import json
from threading import RLock

from .core import Example, atomic_json, canonical, digest


class _JsonlFile:
    """JSONL is authoritative; the byte-offset index can always be rebuilt."""

    def __init__(self, path: Path, key_fields: tuple[str, ...]):
        self.path, self.key_fields = path, key_fields
        self.index_path = path.with_suffix(".index.json")
        self.offsets = None
        self.dirty = False
        self.stamp = None

    def _stamp(self):
        stat = self.path.stat()
        return [stat.st_size, stat.st_mtime_ns]

    def _load_index(self):
        if self.offsets is not None:
            return
        self.offsets = {}
        if not self.path.exists():
            return
        if self.index_path.exists():
            try:
                index = json.loads(self.index_path.read_text())
                if index["stamp"] == self._stamp() and index["key_fields"] == list(self.key_fields):
                    self.offsets = {tuple(key): offset for key, offset in index["offsets"]}
                    self.stamp = index["stamp"]
                    return
            except (ValueError, KeyError, TypeError):
                pass
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                try:
                    if not line.endswith(b"\n"):
                        raise ValueError("incomplete final line")
                    value = json.loads(line)
                    self.offsets[tuple(value[key] for key in self.key_fields)] = offset
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"Invalid JSONL at {self.path}, byte {offset}: {exc}") from exc
        self.dirty = True
        self.stamp = self._stamp()

    def keys(self):
        self._load_index()
        return self.offsets.keys()

    def get(self, key):
        self._load_index()
        offset = self.offsets.get(tuple(key))
        if offset is None:
            return None
        with self.path.open("rb") as stream:
            stream.seek(offset)
            value = json.loads(stream.readline())
        if tuple(value[field] for field in self.key_fields) != tuple(key):
            raise ValueError(f"JSONL index mismatch: {self.path}")
        return value

    def put(self, value):
        self._load_index()
        key = tuple(value[field] for field in self.key_fields)
        previous = self.get(key)
        if previous == value:
            return
        with self.path.open("ab") as stream:
            offset = stream.tell()
            stream.write((canonical(value) + "\n").encode("utf-8"))
        self.offsets[key] = offset
        self.dirty = True
        self.stamp = self._stamp()

    def get_many(self, keys):
        self._load_index()
        rows = []
        if not self.path.exists():
            return [None for _ in keys]
        with self.path.open("rb") as stream:
            for key in keys:
                offset = self.offsets.get(tuple(key))
                if offset is None:
                    rows.append(None)
                    continue
                stream.seek(offset)
                value = json.loads(stream.readline())
                if tuple(value[field] for field in self.key_fields) != tuple(key):
                    raise ValueError(f"JSONL index mismatch: {self.path}")
                rows.append(value)
        return rows

    def save_index(self):
        if self.dirty and self.path.exists() and self.stamp == self._stamp():
            atomic_json(self.index_path, {"stamp": self.stamp, "key_fields": list(self.key_fields),
                                         "offsets": [[list(key), offset] for key, offset in self.offsets.items()]})
            self.dirty = False


class Store:
    """Complete histories and per-session directories in JSONL, with history-only reuse."""

    FORMAT = "trajectory_jsonl_v1"
    ENTRY_FIELDS = ("id", "task_session", "actors_source", "time_order", "content")
    OPTIONAL_ENTRY_FIELDS = ("writer_status",)

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if self.root.suffix in {".sqlite", ".sqlite3", ".db"}:
            raise ValueError("Store expects a JSONL directory; rerun prepare with the current configuration")
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.metadata = json.loads(self.manifest_path.read_text()) if self.manifest_path.exists() else {"storage_format": self.FORMAT}
        if self.metadata.get("storage_format") != self.FORMAT:
            raise ValueError("Incompatible prepared storage format; use a fresh prepared_root")
        self.directory_manifest_path = self.root / "directory_manifest.json"
        self.directory_snapshots = json.loads(self.directory_manifest_path.read_text()) if self.directory_manifest_path.exists() else {}
        self.histories = _JsonlFile(self.root / "memory_records.jsonl", ("memory_id",))
        self.directory_entries = _JsonlFile(self.root / "directories.jsonl", ("memory_id", "version", "id"))
        self.completions = _JsonlFile(self.root / "completions.jsonl", ("key",))
        self._completion_lock = RLock()
        self.example_files = {}
        self._memory_cache = OrderedDict()
        self._directory_cache = OrderedDict()

    def close(self):
        for file in [self.histories, self.directory_entries, self.completions, *self.example_files.values()]:
            file.save_index()
        self._memory_cache.clear()
        self._directory_cache.clear()

    def get_metadata(self, key, default=None):
        return self.metadata.get(key, default)

    def set_metadata(self, key, value):
        self.metadata[key] = value
        atomic_json(self.manifest_path, self.metadata)

    def add_memory(self, records: list[dict]) -> str:
        # Include all sessions in their original order, with public IDs and timestamps.
        # Hashes identify complete histories; session bodies are never deduplicated separately.
        references = [{key: value for key, value in record.items() if key != "messages"} for record in records]
        memory_id = "mem_" + digest(references)
        if (memory_id,) not in self.histories.keys():
            self.histories.put({"memory_id": memory_id, "records": records})
        return memory_id

    def _example_file(self, split):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if split not in self.example_files:
            self.example_files[split] = _JsonlFile(self.root / f"{split}_examples.jsonl", ("sample_id",))
        return self.example_files[split]

    def add_example(self, example: Example):
        self._example_file(example.split).put(asdict(example))

    def examples(self, split: str) -> list[Example]:
        file = self._example_file(split)
        return [Example(**file.get(key)) for key in sorted(file.keys())]

    def memory_ids(self) -> list[str]:
        return sorted(key[0] for key in self.histories.keys())

    def _records(self, memory_id):
        if memory_id not in self._memory_cache:
            value = self.histories.get((memory_id,))
            if value is None:
                raise KeyError(memory_id)
            self._memory_cache[memory_id] = value["records"]
            if len(self._memory_cache) > 2:
                self._memory_cache.popitem(last=False)
        self._memory_cache.move_to_end(memory_id)
        return self._memory_cache[memory_id]

    def memory(self, memory_id: str) -> list[dict]:
        return [{**{key: value for key, value in record.items() if key != "messages"}, "memory_id": memory_id}
                for record in self._records(memory_id)]

    def record(self, reference: dict) -> dict:
        records = self._records(reference["memory_id"])
        position = reference["order"]
        if not 0 <= position < len(records) or records[position]["id"] != reference["id"]:
            raise KeyError(reference["id"])
        messages = records[position]["messages"]
        text = "\n".join(f"[turn {i}][{message['role']}] " + canonical({k: v for k, v in message.items() if k != "role"}) for i, message in enumerate(messages))
        return {**reference, "messages": messages, "raw_text": text}

    def completion(self, key):
        with self._completion_lock:
            row = self.completions.get((key,))
        return row["response"] if row else None

    def put_completion(self, key, response):
        with self._completion_lock:
            self.completions.put({"key": key, "response": response})

    def directory_entry(self, memory_id, version, record_id):
        row = self.directory_entries.get((memory_id, version, record_id))
        return self._directory_fields(row) if row else None

    def _directory_fields(self, row):
        return {**{key: row[key] for key in self.ENTRY_FIELDS},
                **{key: row[key] for key in self.OPTIONAL_ENTRY_FIELDS if key in row}}

    def put_directory_entry(self, memory_id, version, entry):
        status = entry.get("writer_status")
        if status is not None and (status != "empty_pending" or entry["content"] != ""):
            raise ValueError("empty_pending must mark empty content")
        previous = self.directory_entry(memory_id, version, entry["id"])
        if previous is not None and previous != entry:
            # Only an explicitly empty placeholder may be filled. Successful
            # summaries and all source metadata remain immutable in this version.
            metadata = lambda row: {key: value for key, value in row.items() if key not in {"content", "writer_status"}}
            pending = previous.get("writer_status") == "empty_pending" and previous["content"] == ""
            valid_replacement = bool(entry["content"].strip()) or entry.get("writer_status") == "empty_pending"
            if not pending or not valid_replacement or metadata(previous) != metadata(entry):
                raise ValueError("Directory entries are immutable within a history/version except pending empty content")
            self.directory_snapshots.pop(digest([memory_id, version]), None)
            atomic_json(self.directory_manifest_path, self.directory_snapshots)
            self._directory_cache.pop((memory_id, version), None)
        self.directory_entries.put({"memory_id": memory_id, "version": version,
                                    "session_id": entry["task_session"]["session_id"],
                                    "order": entry["time_order"]["order"], **entry})

    def put_directory(self, memory_id: str, version: str, entries: list[dict]):
        references = self.memory(memory_id)
        if [entry["id"] for entry in entries] != [row["id"] for row in references]:
            raise ValueError("Directory does not align with the complete history")
        for entry in entries:
            self.put_directory_entry(memory_id, version, entry)
        self.directory_snapshots[digest([memory_id, version])] = {"count": len(entries), "sha256": digest(entries)}
        atomic_json(self.directory_manifest_path, self.directory_snapshots)

    def directory(self, memory_id: str, version: str) -> list[dict]:
        cache_key = (memory_id, version)
        if cache_key in self._directory_cache:
            self._directory_cache.move_to_end(cache_key)
            return self._directory_cache[cache_key]
        state = self.directory_snapshots.get(digest([memory_id, version]))
        if state is None:
            raise ValueError(f"Directory missing for {memory_id}, version={version}; run build-directories")
        rows = self.directory_entries.get_many([(memory_id, version, row["id"]) for row in self.memory(memory_id)])
        entries = [self._directory_fields(row) if row else None for row in rows]
        if None in entries or len(entries) != state["count"] or digest(entries) != state["sha256"]:
            raise ValueError(f"Incomplete or changed directory for {memory_id}; run build-directories")
        self._directory_cache[cache_key] = entries
        if len(self._directory_cache) > 2:
            self._directory_cache.popitem(last=False)
        return entries


class DatasetStores:
    """Read selected per-source stores as one training view, retaining split boundaries."""

    FORMAT = Store.FORMAT

    def __init__(self, root, stores):
        self.root = Path(root)
        self.stores = stores
        self.owners = {}
        for source, store in stores.items():
            for memory_id in store.memory_ids():
                self.owners.setdefault(memory_id, []).append(source)
        # Task-agent responses are cached separately from offline Writer responses.
        self.runtime = Store(self.root / "_runtime")

    def close(self):
        for store in self.stores.values():
            store.close()
        self.runtime.close()

    def get_metadata(self, key, default=None):
        if key == "prepare_identity":
            return {"layout": "by_source", "datasets": {
                source: store.get_metadata(key) for source, store in self.stores.items()}}
        values = [store.get_metadata(key, default) for store in self.stores.values()]
        if key == "directory_empty_pending":
            return sum(value or 0 for value in values)
        if not values or any(value != values[0] for value in values):
            raise ValueError(f"Selected datasets have missing or inconsistent {key}; build all selected directories")
        return values[0]

    def examples(self, split):
        return sorted((example for store in self.stores.values() for example in store.examples(split)),
                      key=lambda example: example.sample_id)

    def memory_ids(self):
        return sorted(self.owners)

    def _owner(self, memory_id):
        return self.stores[self.owners[memory_id][0]]

    def memory(self, memory_id):
        return self._owner(memory_id).memory(memory_id)

    def record(self, reference):
        return self._owner(reference["memory_id"]).record(reference)

    def directory(self, memory_id, version):
        return self._owner(memory_id).directory(memory_id, version)

    def completion(self, key):
        return self.runtime.completion(key)

    def put_completion(self, key, response):
        self.runtime.put_completion(key, response)
