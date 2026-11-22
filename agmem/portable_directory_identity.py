"""Read-only content proofs for moving a frozen directory build to a new host.

Only a verified tokenizer/encoder receives the source identity alias. The new
Writer keeps its actual local model revision; source timestamps are not restored.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

from .core import digest, file_digest


VERSION = "longmemeval_m_continuation_transfer.v1"
TOKENIZER_EXTRAS = {"chat_template.jinja", "merges.txt", "tokenizer.model", "spiece.model"}
TOKENIZER_JSON = {"special_tokens_map.json", "vocab.json", "added_tokens.json"}


def validate_path(value):
    """Return a canonical, safe relative POSIX manifest path."""
    if (not isinstance(value, str) or not value or "\\" in value or "\0" in value
            or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError(f"Manifest path must be a safe relative path: {value!r}")
    return value


def model_files(root, *, tokenizer_only=False):
    """List exactly the files covered by tokenizer or full encoder provenance."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Missing local model directory: {root}")
    selected = set()
    for path in root.iterdir():
        if not path.is_file():
            continue
        name = path.name
        tokenizer = path.suffix == ".json" and ("tokenizer" in name or name in TOKENIZER_JSON)
        if (name in TOKENIZER_EXTRAS or tokenizer
                or (not tokenizer_only and path.suffix in {".json", ".safetensors"})):
            selected.add(name)
    if not tokenizer_only:
        selected.update(path.relative_to(root).as_posix() for path in root.rglob("*.json")
                        if path.is_file() and path.parent != root
                        and ".cache" not in path.relative_to(root).parts)
    return sorted(selected)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


def _entries(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"Missing manifest files: {label}")
    result = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid manifest file: {label}")
        path = validate_path(entry.get("path"))
        if (path in result or type(entry.get("bytes")) is not int or entry["bytes"] < 0
                or not isinstance(entry.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None):
            raise ValueError(f"Duplicate or invalid manifest file: {label}/{path}")
        result[path] = deepcopy(entry)
    return result


def _stat(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _snapshot(root, names):
    return {name: _stat(root / name) for name in names}


def _check_file(path, entry):
    before = _stat(path)
    if before[2] != entry["bytes"] or file_digest(path) != entry["sha256"]:
        raise ValueError(f"File content SHA256/size changed: {path}")
    if _stat(path) != before:
        raise ValueError(f"File changed during content verification: {path}")


class PortableIdentity:
    """Verify a transfer manifest and provide narrowly scoped identity aliases."""

    def __init__(self, manifest_path, project_root=None):
        self.project_root = Path(project_root or Path.cwd()).expanduser().resolve()
        self.manifest_path = Path(manifest_path).expanduser()
        if not self.manifest_path.is_absolute():
            self.manifest_path = self.project_root / self.manifest_path
        self.manifest_path = self.manifest_path.resolve()
        raw = self.manifest_path.read_bytes()
        self.manifest = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(self.manifest, dict) or self.manifest.get("version") != VERSION:
            raise ValueError(f"Use a {VERSION} manifest")
        self.source_root = self._project_path(self.manifest.get("source_root_relative"))
        self.source_identity = deepcopy(self.manifest.get("source_identity"))
        if not isinstance(self.source_identity, dict):
            raise ValueError("Missing manifest source construction identity")
        self._data_files = _entries(self.manifest.get("data_files"), "data_files")
        splits = [entry.get("split") for entry in self._data_files.values()]
        if (any(not isinstance(split, str) or not split for split in splits)
                or len(set(splits)) != len(splits)):
            raise ValueError("Manifest dataset splits must be unique and nonempty")
        models = self.manifest.get("models")
        if not isinstance(models, dict) or not all(isinstance(models.get(name), dict)
                                                  for name in ("writer_tokenizer", "embedding")):
            raise ValueError("Missing manifest tokenizer/embedding proof")
        self._model_files = {name: _entries(models[name].get("files"), name)
                             for name in ("writer_tokenizer", "embedding")}
        for name in self._model_files:
            if not isinstance(models[name].get("source_model_path"), str) or not models[name]["source_model_path"]:
                raise ValueError(f"Missing source model path: {name}")
        writer_tokenizer = models["writer_tokenizer"].get("tokenizer_id")
        if (not isinstance(writer_tokenizer, str) or not writer_tokenizer
                or writer_tokenizer != self.source_identity.get("tokenizer_id")):
            raise ValueError("Manifest writer tokenizer differs from source identity")
        embedding = models["embedding"].get("identity")
        if (not isinstance(embedding, dict) or embedding != self.source_identity.get("embedding")
                or embedding.get("ready") is not True
                or any(not isinstance(embedding.get(key), str) or not embedding[key]
                       for key in ("model_revision", "tokenizer_id"))):
            raise ValueError("Manifest embedding differs from ready source identity")
        self._verified = {}
        self._model_paths = None
        self.report = {"version": VERSION, "manifest_path": str(self.manifest_path),
                       "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                       "project_root": str(self.project_root), "models": {}, "datasets": []}

    def _project_path(self, relative):
        path = (self.project_root / validate_path(relative)).resolve()
        if not path.is_relative_to(self.project_root):
            raise ValueError(f"Manifest path escapes the project root: {relative}")
        return path

    def _unchanged(self, root, proof):
        names = model_files(root, tokenizer_only=proof["tokenizer_only"])
        guard_names = model_files(root)
        if (names != proof["names"] or guard_names != proof["guard_names"]
                or _snapshot(root, guard_names) != proof["snapshot"]):
            raise ValueError(f"Local model changed since content verification: {root}")

    def verify_models(self, writer_path, embedding_path, base_identity_callable):
        """Hash relevant local files once and retain the actual/local provenance."""
        paths = tuple(Path(path).expanduser().resolve() for path in (writer_path, embedding_path))
        if paths[0] == paths[1]:
            raise ValueError("Writer and embedding model paths must be distinct")
        if self._model_paths is not None:
            if paths != self._model_paths:
                raise ValueError("Portable identity is already bound to different model paths")
            for root, proof in self._verified.items():
                self._unchanged(root, proof)
            return deepcopy(self.report["models"])
        verified, reports = {}, {}
        for name, root in zip(("writer_tokenizer", "embedding"), paths):
            tokenizer_only = name == "writer_tokenizer"
            names = model_files(root, tokenizer_only=tokenizer_only)
            entries = self._model_files[name]
            if names != sorted(entries):
                missing, extra = sorted(set(entries) - set(names)), sorted(set(names) - set(entries))
                raise ValueError(f"Local {name} file set differs: missing={missing}, extra={extra}")
            # The Writer only needs tokenizer content equivalence, but its own
            # local revision must also stay stable after the base lookup.
            guard_names = model_files(root)
            before = _snapshot(root, guard_names)
            for filename in names:
                _check_file(root / filename, entries[filename])
            actual_identity, actual_tokenizer = base_identity_callable(str(root), allow_missing=False)
            if (not isinstance(actual_identity, dict) or actual_identity.get("ready") is not True
                    or not actual_identity.get("model_revision") or not actual_tokenizer):
                raise ValueError(f"Local model is not ready: {root}")
            source = self.manifest["models"][name]
            if tokenizer_only:
                effective_identity = deepcopy(actual_identity)
                effective_tokenizer = source["tokenizer_id"]
                source_identity = {"tokenizer_id": effective_tokenizer}
            else:
                source_identity = deepcopy(source["identity"])
                effective_identity = {key: source_identity[key] for key in ("ready", "model_revision")}
                effective_tokenizer = source_identity["tokenizer_id"]
            proof = {"tokenizer_only": tokenizer_only, "names": names, "guard_names": guard_names, "snapshot": before,
                     "alias": (effective_identity, effective_tokenizer)}
            self._unchanged(root, proof)
            verified[root] = proof
            content = [{key: entries[filename][key] for key in ("path", "bytes", "sha256")}
                       for filename in names]
            reports[name] = {"source_model_path": source["source_model_path"], "local_model_path": str(root),
                             "content_proof_digest": digest(content), "verified_files": len(names),
                             "verified_bytes": sum(entry["bytes"] for entry in content),
                             "actual_identity": {**deepcopy(actual_identity), "tokenizer_id": actual_tokenizer},
                             "source_identity": source_identity,
                             "effective_identity": {**deepcopy(effective_identity), "tokenizer_id": effective_tokenizer}}
        # A failed second proof must not leave the first model aliased.
        self._verified, self._model_paths = verified, paths
        self.report["models"] = reports
        return deepcopy(reports)

    def model_identity(self, path, *, allow_missing=False):
        """Replacement for the engine lookup, limited to the two proved paths."""
        root = Path(path).expanduser().resolve()
        proof = self._verified.get(root)
        if proof is None:
            raise ValueError(f"No verified portable identity for local model: {root}")
        self._unchanged(root, proof)
        return deepcopy(proof["alias"])

    @staticmethod
    def _identity_files(identity):
        datasets = identity.get("datasets")
        if not isinstance(datasets, list) or len(datasets) != 1:
            raise ValueError("Portable continuation requires exactly one dataset")
        dataset = datasets[0]
        if not isinstance(dataset, dict) or not isinstance(dataset.get("spec"), dict):
            raise ValueError("Invalid dataset construction identity")
        name = dataset.get("spec", {}).get("name")
        if not isinstance(name, str) or not name or not isinstance(dataset.get("files"), list):
            raise ValueError("Invalid dataset construction identity")
        files = {}
        for entry in dataset["files"]:
            if not isinstance(entry, dict):
                raise ValueError("Invalid dataset file identity")
            split = entry.get("split")
            if (not isinstance(split, str) or not split or split in files
                    or not isinstance(entry.get("path"), str) or not entry["path"]):
                raise ValueError("Duplicate or invalid dataset file identity")
            files[split] = entry
        return name, files

    def verify_files(self, source_identity, target_identity):
        """Verify mapped local data without opening or modifying old source paths."""
        if source_identity != self.source_identity:
            raise ValueError("Source journal identity differs from transfer manifest")
        source_name, source = self._identity_files(source_identity)
        target_name, target = self._identity_files(target_identity)
        entries = {entry["split"]: entry for entry in self._data_files.values()}
        if source_name != target_name or set(source) != set(entries) or set(target) != set(entries):
            raise ValueError("Source/target dataset splits differ from transfer manifest")
        hashes, mapping = {}, []
        for split, entry in sorted(entries.items()):
            if source[split].get("sha256") != entry["sha256"] or target[split].get("sha256") != entry["sha256"]:
                raise ValueError(f"Source/target dataset SHA256 differs from transfer manifest: {split}")
            local = self._project_path(entry["path"])
            target_path = Path(target[split]["path"]).expanduser()
            if not target_path.is_absolute():
                target_path = self.project_root / target_path
            if target_path.resolve() != local:
                raise ValueError(f"Target dataset path differs from transfer manifest: {split}")
            _check_file(local, entry)
            hashes[str(local)] = entry["sha256"]
            mapping.append({"dataset": source_name, "split": split, "source_path": source[split]["path"],
                            "target_path": str(local), "project_relative_path": entry["path"],
                            "bytes": entry["bytes"], "sha256": entry["sha256"]})
        self.report["datasets"] = mapping
        return hashes
