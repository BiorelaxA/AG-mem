"""Explicitly preserve v1 directories while building missing sessions with v2.

Source verification and atomic rebinding remain in the frozen resplit importer.
No old segmentation/Writer cache is relabeled as v2. A full mixed-policy manifest
is committed before any session import, and every reused directory retains its
source provenance and original generation status.
"""

import argparse
from collections import Counter
from contextlib import closing, contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from threading import RLock

from . import build_topic_event_directories as topic
from . import reuse_adjacent_directories as legacy
from .core import atomic_json, canonical, digest, file_digest
from .hierarchical.binding import directory_version
from .jobs import file_lock


VERSION = "topic_event_incremental_reuse.v1"
POLICY = "preserve_existing_build_missing.v1"
LEGACY_PROTOCOL = "adjacent_parent_child_directory_v1"
LEGACY_BOUNDARY_VERSION = "adjacent.boundary.v1"
DEFAULT_CONFIG = "configs/build_adjacent_qwen_longmemeval_811_incremental.yaml"
DEFAULT_SOURCE = legacy.DEFAULT_SOURCE
DEFAULT_MAPPING = legacy.DEFAULT_MAPPING
POLICY_KEY = "mixed_directory_policy"
IMPORT_STATE_KEY = "mixed_directory_import_state"
_BASE_COMPATIBLE = legacy.compatible_identity
_BASE_MAPPINGS = legacy.verified_mappings
_BASE_STORE = legacy.AdjacentBuildStore
_ADAPTATION_LOCK = RLock()


def compatible_identity(old, new, *, allow_migrated_source_identity=False):
    """Permit the boundary prompt and approved 0.78/0.55 -> 0.72/0.50 transition."""
    old_prompt = topic.BOUNDARY_PROMPT.parent.parent / "boundary.txt"
    if (old.get("protocol") != LEGACY_PROTOCOL
            or old.get("boundary_prompt_version") is not None
            or old.get("directory_reuse_policy") is not None
            or old.get("prompt_hashes", {}).get("boundary.txt") != file_digest(old_prompt)):
        raise ValueError("Source is not the frozen legacy boundary construction identity")
    if (new.get("protocol") != topic.PROTOCOL
            or new.get("boundary_prompt_version") != topic.BOUNDARY_PROMPT_VERSION
            or new.get("directory_reuse_policy") != POLICY
            or new.get("prompt_hashes", {}).get("boundary.txt") != file_digest(topic.BOUNDARY_PROMPT)
            or new.get("implementation_hashes", {}).get("build_topic_event_directories.py")
            != file_digest(Path(topic.__file__))):
        raise ValueError("Target is not the explicit topic/event incremental construction identity")
    normalized = deepcopy(new)
    normalized["protocol"] = LEGACY_PROTOCOL
    normalized.pop("boundary_prompt_version")
    normalized.pop("directory_reuse_policy")
    normalized["prompt_hashes"]["boundary.txt"] = old["prompt_hashes"]["boundary.txt"]
    normalized["implementation_hashes"].pop("build_topic_event_directories.py")
    if type(allow_migrated_source_identity) is not bool:
        raise ValueError("allow_migrated_source_identity must be boolean")
    if allow_migrated_source_identity:
        # A re-downloaded local model has a different fast fingerprint (which
        # includes mtimes). Do not assert equal weights or relabel old output:
        # imported directories keep their original Writer/tokenizer metadata,
        # and the policy records both full construction identities. Model names,
        # embedding settings, transforms, prompts and implementations stay exact.
        for section, key in (("writer", "model_revision"), ("embedding", "model_revision"),
                             ("embedding", "tokenizer_id")):
            previous, current = old.get(section, {}), normalized.get(section, {})
            if key in previous and key in current:
                current[key] = previous[key]
        if "tokenizer_id" in old and "tokenizer_id" in normalized:
            normalized["tokenizer_id"] = old["tokenizer_id"]
    old_names = {item["spec"]["name"] for item in old["datasets"]}
    retained = []
    for item in normalized["datasets"]:
        spec = item["spec"]
        if spec["name"] not in old_names:
            # M was intentionally absent from the original adjacent build. It
            # is a fresh dataset here; no source bank or old cache is reused.
            semantic = {key: value for key, value in spec.items()
                        if key not in {"input_root", "input_files", "path", "split", "splits"}}
            if semantic != {"name": "split_longmemeval_m", "format": "longmemeval_parquet",
                            "modality_policy": "existing_captions"}:
                raise ValueError("Only fresh LongMemEval-M may be added to this source construction")
        else:
            retained.append(item)
    normalized["datasets"] = retained
    old_segment, new_segment = old.get("segmentation", {}), normalized.get("segmentation", {})
    threshold_keys = ("high_threshold", "low_threshold")
    if (tuple(old_segment.get(key) for key in threshold_keys) == (0.78, 0.55)
            and tuple(new_segment.get(key) for key in threshold_keys) == (0.72, 0.50)):
        # This normalization permits preservation of old directories, not reuse
        # of their decisions under new thresholds. Their caches are never copied.
        for key in threshold_keys:
            new_segment[key] = old_segment[key]
    _BASE_COMPATIBLE(old, normalized)


def _load_recipe(*args, **kwargs):
    config, identity = topic.load_recipe(*args, **kwargs)
    if config.get("preserve_existing_directories") is not True or config.get("retry_fallback") is not False:
        raise ValueError("Incremental reuse requires preserve_existing_directories=true and retry_fallback=false")
    if identity.get("directory_reuse_policy") != POLICY:
        raise ValueError("Incremental construction identity is missing its preservation policy")
    if type(config.get("allow_migrated_source_identity", False)) is not bool:
        raise ValueError("allow_migrated_source_identity must be boolean")
    return config, identity


def _read_policy(root):
    journal = Path(root) / "build_journal.sqlite3"
    if not journal.is_file():
        return None, None, None
    db = sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        identity = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        policy = db.execute("SELECT value FROM metadata WHERE key=?", (POLICY_KEY,)).fetchone()
        state = db.execute("SELECT value FROM metadata WHERE key=?", (IMPORT_STATE_KEY,)).fetchone()
        return (json.loads(identity[0]) if identity else None,
                json.loads(policy[0]) if policy else None, json.loads(state[0]) if state else None)
    finally:
        db.close()


def validate_mixed_policy(config, identity, *, dry_run=False, datasets=None, splits=None):
    """Guard direct builds; importing is mandatory for a preservation recipe."""
    existing_identity, policy, state = _read_policy(config["output_root"])
    if existing_identity is not None and existing_identity != identity:
        raise ValueError("Directory build identity changed; use a fresh output_root")
    required = config.get("preserve_existing_directories") is True
    if policy is None:
        if required and not dry_run:
            raise ValueError("Preserving existing directories requires the explicit importer before building")
        return {"requires_import": True, "policy": POLICY} if required else None
    if (not required or config.get("retry_fallback") is not False
            or identity.get("directory_reuse_policy") != POLICY
            or policy.get("policy") != POLICY
            or policy.get("target_identity_sha256") != digest(identity)
            or policy.get("source_identity_sha256") != digest(policy.get("source_construction_identity"))):
        raise ValueError("Mixed directory preservation policy or construction identity changed")
    if policy.get("allow_migrated_source_identity", False) != config.get("allow_migrated_source_identity", False):
        raise ValueError("Mixed directory migration identity policy changed")
    if (not isinstance(state, dict) or state.get("status") != "ready"
            or state.get("target_identity_sha256") != digest(identity)):
        if dry_run:
            return {"requires_import": True, "policy": POLICY, "reason": "import_not_complete"}
        raise ValueError("The explicit importer has not completed; rerun the importer before building")
    specs = {spec["name"]: spec for spec in config["datasets"]}
    selected = datasets or list(specs)
    if not set(selected).issubset(specs):
        raise ValueError("Requested build datasets are absent from the preservation recipe")
    required_scope = {(name, split) for name in selected for split in
                      (splits or specs[name].get("splits", [specs[name].get("split", "train")]))}
    imported_scope = {tuple(pair) for pair in state.get("completed_scope", [])}
    missing = sorted(required_scope - imported_scope)
    if missing:
        if dry_run:
            return {"requires_import": True, "policy": POLICY, "reason": "import_scope_incomplete",
                    "missing_scope": missing}
        raise ValueError(f"The explicit importer has not completed the requested dataset/split scope: {missing}")
    return {"requires_import": False, "policy": POLICY,
            "source_root": policy["source_root"], "existing_fallbacks_preserved": True}


def export_rule_provenance(root):
    """Export per-session old/new rule attribution; return None for ordinary v2."""
    root = Path(root).resolve()
    identity, policy, state = _read_policy(root)
    if policy is None:
        return None
    with file_lock(root / ".build.lock"):
        identity, policy, state = _read_policy(root)
        if policy is None or policy.get("target_identity_sha256") != digest(identity):
            raise ValueError("Mixed directory provenance does not match the target construction identity")
        db = sqlite3.connect((root / "build_journal.sqlite3").as_uri() + "?mode=ro", uri=True)
        counts, temporary = Counter(), root / "directory_rule_provenance.jsonl.tmp"
        try:
            query = db.execute("SELECT s.bank_id,s.parent_id,s.status,s.directory,p.payload FROM sessions s "
                               "LEFT JOIN reuse_provenance p USING(bank_id,parent_id) ORDER BY s.bank_id,s.parent_id")
            with temporary.open("w") as stream:
                for bid, pid, status, encoded, imported in query:
                    directory = json.loads(encoded) if encoded else None
                    prior = json.loads(imported) if imported else None
                    retained = prior is not None and prior.get("directory_version_before") is not None
                    if retained:
                        if (directory is None or directory["directory_version"] != prior["directory_version_after"]
                                or status != prior["source_status"]
                                or directory_version(directory) != directory["directory_version"]):
                            raise ValueError(f"Preserved source directory was changed: {bid}/{pid}")
                        origin, rule = "reused_legacy_directory", policy["imported_boundary_rule"]
                    else:
                        origin = "built_topic_event_v2" if directory is not None else "pending_topic_event_v2"
                        rule = policy["new_boundary_rule"]
                    row = {"bank_id": bid, "parent_id": pid, "status": status, "origin": origin,
                           "boundary_prompt_version": rule["version"], "boundary_prompt_sha256": rule["sha256"],
                           "similarity_thresholds": deepcopy(rule["similarity_thresholds"]),
                           "directory_version": directory["directory_version"] if directory else None,
                           "generation_assigned": directory is not None}
                    if prior is not None:
                        row["imported_from"] = {"source_root": prior["source_root"],
                            "bank_id": prior["source_bank_id"], "parent_id": pid,
                            "directory_version": prior["directory_version_before"],
                            "source_status": prior["source_status"],
                            "source_construction_identity_sha256": policy["source_identity_sha256"]}
                    stream.write(canonical(row) + "\n")
                    counts[origin] += 1
            temporary.replace(root / "directory_rule_provenance.jsonl")
            report = {"policy": POLICY, "counts": dict(counts), "registered_sessions": sum(counts.values()),
                      "import_state": state,
                      "source_identity_sha256": policy["source_identity_sha256"],
                      "target_identity_sha256": policy["target_identity_sha256"]}
            atomic_json(root / "directory_rule_summary.json", report)
            return report
        finally:
            db.close()
            if temporary.exists():
                temporary.unlink()


@contextmanager
def _incremental_importer(args):
    """Keep all source-row validation in the v1 importer, changing policy only."""
    context = {}

    def recipe(*args, **kwargs):
        config, identity = _load_recipe(*args, **kwargs)
        context["allow_migrated_source_identity"] = config.get("allow_migrated_source_identity", False)
        return config, identity

    def check(old, new):
        migration = context.get("allow_migrated_source_identity", False)
        compatible_identity(old, new, allow_migrated_source_identity=migration)
        with closing(legacy.source_connection(args.reuse_root)) as db:
            row = db.execute("SELECT value FROM metadata WHERE key='parent_budget_migrations'").fetchone()
            history = json.loads(row[0]) if row else []
        context["policy"] = {"version": VERSION, "policy": POLICY,
            "source_root": str(Path(args.reuse_root).resolve()),
            "source_construction_identity": deepcopy(old), "source_parent_budget_migrations": history,
            "source_identity_sha256": digest(old), "target_identity_sha256": digest(new),
            "mapping_sha256": file_digest(Path(args.mapping)),
            "imported_boundary_rule": {"version": LEGACY_BOUNDARY_VERSION,
                                       "sha256": old["prompt_hashes"]["boundary.txt"],
                                       "similarity_thresholds": {key: old["segmentation"][key]
                                           for key in ("high_threshold", "low_threshold")}},
            "new_boundary_rule": {"version": topic.BOUNDARY_PROMPT_VERSION,
                                  "sha256": new["prompt_hashes"]["boundary.txt"],
                                  "similarity_thresholds": {key: new["segmentation"][key]
                                      for key in ("high_threshold", "low_threshold")}},
            "preserved_statuses": ["complete", "empty", "fallback"], "retry_fallback": False,
            "copy_segment_cache": False, "copy_writer_cache": False,
            "importer_implementation_sha256": file_digest(Path(__file__))}
        if migration:
            context["policy"].update(allow_migrated_source_identity=True,
                model_fingerprint_relation="not_asserted_equal; original directory Writer provenance retained",
                target_model_provenance={key: deepcopy(new[key]) for key in ("writer", "tokenizer_id", "embedding")})

    def mappings(*args, **kwargs):
        return _BASE_MAPPINGS(*args, **kwargs, allow_new_datasets=True)

    def store_factory(root, identity):
        policy = context["policy"]
        store = _BASE_STORE(root, identity)
        try:
            with store.db:
                old = store.db.execute("SELECT value FROM metadata WHERE key=?", (POLICY_KEY,)).fetchone()
                if old is not None and json.loads(old[0]) != policy:
                    raise ValueError("Mixed directory reuse source or policy changed")
                old_state = store.db.execute("SELECT value FROM metadata WHERE key=?", (IMPORT_STATE_KEY,)).fetchone()
                old_state = json.loads(old_state[0]) if old_state else {}
                context["completed_scope"] = (old_state.get("completed_scope", [])
                    if old_state.get("status") == "ready" and old_state.get("target_identity_sha256") == digest(identity)
                    else [])
                store.db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (POLICY_KEY, canonical(policy)))
                store.db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (IMPORT_STATE_KEY,
                    canonical({"status": "running", "target_identity_sha256": digest(identity)})))
            atomic_json(Path(root) / "mixed_directory_manifest.json", policy)
            return store
        except BaseException:
            store.close()
            raise

    replacements = {"load_recipe": recipe, "compatible_identity": check,
                    "verified_mappings": mappings,
                    "_segments": lambda *unused: (None, None), "VERSION": VERSION,
                    "AdjacentBuildStore": store_factory}
    with _ADAPTATION_LOCK:
        previous = {key: getattr(legacy, key) for key in replacements}
        for key, value in replacements.items():
            setattr(legacy, key, value)
        try:
            yield context
        finally:
            for key, value in previous.items():
                setattr(legacy, key, value)


def run(args, *, tokenizer=None):
    with _incremental_importer(args) as context:
        report = legacy.run(args, tokenizer=tokenizer)
    if not args.dry_run:
        root = Path(report["output_root"])
        with file_lock(root / ".build.lock"):
            identity, policy, _ = _read_policy(root)
            if policy is None or policy["target_identity_sha256"] != digest(identity):
                raise ValueError("Mixed directory policy changed before import completion")
            completed_scope = {tuple(pair) for pair in context.get("completed_scope", [])}
            completed_scope.update((name, split) for name in report["datasets"] for split in report["splits"])
            with closing(sqlite3.connect(root / "build_journal.sqlite3")) as db, db:
                db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (IMPORT_STATE_KEY,
                    canonical({"status": "ready", "target_identity_sha256": digest(identity),
                               "completed_scope": sorted(completed_scope)})))
        report["directory_rules"] = export_rule_provenance(report["output_root"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--reuse-root", default=DEFAULT_SOURCE)
    parser.add_argument("--mapping", default=DEFAULT_MAPPING)
    parser.add_argument("--datasets", nargs="+", default=["split_longmemeval_s"])
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"],
                        default=["train", "validation", "test"])
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--embedding-model-path")
    parser.add_argument("--dry-run", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
