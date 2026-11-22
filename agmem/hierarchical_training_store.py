"""Frozen, bank-scoped hierarchy beside the existing prepared query format."""

from collections import OrderedDict
from copy import deepcopy
import json
from pathlib import Path
from threading import RLock

from .core import digest, file_digest
from .data import ADAPTER_VERSION
from .hierarchical.binding import validate_directory
from .store import Store, _JsonlFile


PROTOCOL = "hierarchical_v1"
SNAPSHOT_FORMAT = "hierarchical_training_snapshot.v1"
BANK_FILE = "hierarchical_banks.jsonl"
SNAPSHOT_FILE = "hierarchical_manifest.json"


def snapshot_version(manifest):
    return digest({key: value for key, value in manifest.items() if key != "directory_version"})


def _immutable(*args, **kwargs):
    raise TypeError("Frozen hierarchical snapshots are immutable")


class _FrozenDict(dict):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo):
        return {deepcopy(k, memo): deepcopy(v, memo) for k, v in self.items()}


class _FrozenList(list):
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = _immutable
    __iadd__ = __imul__ = _immutable

    def __deepcopy__(self, memo):
        return [deepcopy(value, memo) for value in self]


def _freeze(value):
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    return value


def validate_bank(bank):
    """Validate complete, contiguous parents and exact immutable source spans."""
    bank_id = bank.get("bank_id")
    directories, sessions = bank.get("directories"), bank.get("sessions")
    if not isinstance(bank_id, str) or not bank_id or not isinstance(directories, list) or not sessions:
        raise ValueError("Frozen bank requires directories and source sessions")
    if not isinstance(sessions, list) or len(sessions) != len(directories):
        raise ValueError("Frozen bank directory/session counts differ")
    expected = list(range(1, len(sessions) + 1))
    if [s["order"] for s in sessions] != expected or [d["order"] for d in directories] != expected:
        raise ValueError("Frozen parents must preserve the entire contiguous history")
    seen = set()
    for directory, session in zip(directories, sessions):
        if directory["bank_id"] != bank_id or session["bank_id"] != bank_id:
            raise ValueError("Cross-bank directory/session address")
        if session["parent_id"] in seen:
            raise ValueError("Duplicate parent ID in frozen bank")
        seen.add(session["parent_id"])
        if directory["build"]["status"] != "complete":
            raise ValueError("Only complete directories may enter hierarchical training")
        validate_directory(directory, session)


class HierarchicalTrainingStore(Store):
    """Independent query/cache store whose hierarchy cannot follow a live build.

    Returned bank lists are recursively read-only. DirectoryReader's existing
    deepcopy creates private ordinary lists/dicts without another store copy.
    """

    def __init__(self, root, *, memory_cache_size=2):
        root = Path(root)
        if not (root / "manifest.json").is_file() or not (root / SNAPSHOT_FILE).is_file():
            raise ValueError("Run python -m agmem.prepare_hierarchical_training before opening this store")
        super().__init__(root, memory_cache_size=memory_cache_size)
        if self.metadata.get("directory_protocol") != PROTOCOL or self.metadata.get("prepare_complete") is not True:
            raise ValueError("Prepared store is not a complete frozen hierarchy")
        self.snapshot = json.loads((root / SNAPSHOT_FILE).read_text())
        version = snapshot_version(self.snapshot)
        if (self.snapshot.get("schema_version") != SNAPSHOT_FORMAT
                or self.snapshot.get("directory_protocol") != PROTOCOL
                or self.snapshot.get("directory_version") != version
                or self.metadata.get("directory_version") != version):
            raise ValueError("Frozen hierarchical manifest/version mismatch")
        if self.metadata.get("prepare_identity") != self.snapshot.get("prepare_identity"):
            raise ValueError("Frozen prepared identity changed")
        if (self.snapshot["prepare_identity"].get("adapter_version") != ADAPTER_VERSION
                or self.snapshot["prepare_identity"].get("storage_format") != Store.FORMAT):
            raise ValueError("Frozen prepared adapter/storage version is incompatible")
        required_files = {BANK_FILE, "memory_records.jsonl", *[f"{s}_examples.jsonl" for s in self.snapshot["splits"]]}
        if set(self.snapshot["files"]) != required_files:
            raise ValueError("Frozen manifest must fingerprint every required prepared artifact")
        for name, expected in self.snapshot["files"].items():
            if Path(name).name != name or not (root / name).is_file() or file_digest(root / name) != expected:
                raise ValueError(f"Frozen prepared artifact changed: {name}")
        self.hierarchies = _JsonlFile(root / BANK_FILE, ("bank_id",))
        self._bank_cache = OrderedDict()
        self._bank_lock = RLock()
        self._validate_mappings()

    def _validate_mappings(self):
        memories = set(self.memory_ids())
        mapping = self.snapshot["memory_to_bank"]
        banks = self.snapshot["banks"]
        if set(mapping) != memories or set(mapping.values()) != set(banks):
            raise ValueError("Frozen memory-to-bank map is incomplete")
        if {key[0] for key in self.hierarchies.keys()} != set(banks):
            raise ValueError("Frozen bank artifacts do not match their manifest")
        examples = [e for split in self.snapshot["splits"] for e in self.examples(split)]
        if len({e.sample_id for e in examples}) != len(examples):
            raise ValueError("Frozen query IDs are duplicated")
        queries = self.snapshot["query_to_bank"]
        if set(queries) != {e.sample_id for e in examples}:
            raise ValueError("Frozen query-to-bank map is incomplete")
        for example in examples:
            if (example.memory_id not in mapping or queries[example.sample_id] != mapping[example.memory_id]
                    or example.source != banks[queries[example.sample_id]]["dataset"]):
                raise ValueError("Query mapping crosses memory banks")

    def hierarchical_bank(self, memory_id, version):
        if version != self.metadata["directory_version"]:
            raise ValueError("Requested hierarchy version differs from the frozen snapshot")
        bank_id = self.snapshot["memory_to_bank"].get(memory_id)
        if bank_id is None:
            raise KeyError(memory_id)
        with self._bank_lock:
            if bank_id not in self._bank_cache:
                bank = self.hierarchies.get((bank_id,))
                if bank is None or digest(bank) != self.snapshot["banks"][bank_id]["sha256"]:
                    raise ValueError(f"Frozen hierarchy content changed: {bank_id}")
                validate_bank(bank)
                self._bank_cache[bank_id] = _freeze(bank)
                if len(self._bank_cache) > self.memory_cache_size:
                    self._bank_cache.popitem(last=False)
            self._bank_cache.move_to_end(bank_id)
            bank = self._bank_cache[bank_id]
        return bank["directories"], bank["sessions"]

    def validate_directories(self, splits):
        if not splits or not set(splits).issubset(self.snapshot["splits"]):
            raise ValueError("Requested splits are absent from the frozen hierarchy")
        version = self.metadata["directory_version"]
        for memory_id in sorted({e.memory_id for split in splits for e in self.examples(split)}):
            self.hierarchical_bank(memory_id, version)
        return version

    def directory(self, memory_id, version):
        raise ValueError("Hierarchical training must use hierarchical_bank, not the flat directory API")

    def put_directory_entry(self, *args, **kwargs):
        raise ValueError("Hierarchical training directories are frozen")

    def put_directory(self, *args, **kwargs):
        raise ValueError("Hierarchical training directories are frozen")

    def add_memory(self, *args, **kwargs):
        raise ValueError("Hierarchical training source histories are frozen")

    def add_example(self, *args, **kwargs):
        raise ValueError("Hierarchical training query splits are frozen")

    def set_metadata(self, *args, **kwargs):
        raise ValueError("Hierarchical training provenance is frozen")

    def close(self):
        super().close()
        if hasattr(self, "hierarchies"):
            self.hierarchies.save_index()
        if hasattr(self, "_bank_cache"):
            self._bank_cache.clear()
