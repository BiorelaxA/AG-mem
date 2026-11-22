"""Rebuild LoCoMo with topic/event boundaries and three raw context segments.

This opt-in worker keeps the legacy engine and directory schemas immutable.
Construction identity separates its summaries and segmentation caches from
all previous builds. Run it in a dedicated process, as the GPU launcher does.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

from . import build_adjacent_directories as legacy
from . import build_topic_event_directories as topic_event
from .build_topic_event_directories import (
    BOUNDARY_PROMPT, BOUNDARY_PROMPT_VERSION, _ADAPTATION_LOCK,
    _BASE_LOAD_RECIPE,
)
from .core import file_digest
from .hierarchical import segment_context_writer
from .hierarchical.segment_context_writer import (
    CHILD_PROMPT, CHILD_PROMPT_VERSION, PREVIOUS_SEGMENTS,
    SegmentContextDirectoryWriter, context_policy,
)


DEFAULT_CONFIG = "configs/build_locomo_topic_event_context3.yaml"
PROTOCOL = "adjacent_topic_event_context3_v1"
# Binding and Reader schema compatibility is independent of the new prompt.
PROMPT_VERSION = "directory-writer.adjacent.v1"
DATASETS = {"locomo"}


@contextmanager
def _context_engine():
    """Use the versioned worker without changing old source-file identities."""
    with _ADAPTATION_LOCK:
        previous = legacy.PROTOCOL, legacy.load_recipe, legacy.DATASETS
        legacy.PROTOCOL, legacy.load_recipe, legacy.DATASETS = PROTOCOL, load_recipe, DATASETS
        try:
            yield
        finally:
            legacy.PROTOCOL, legacy.load_recipe, legacy.DATASETS = previous


def load_recipe(path, workers=None, *, embedding_model_path=None, output_root=None,
                allow_missing_embedding=False):
    with _context_engine():
        config, original = _BASE_LOAD_RECIPE(
            path, workers, embedding_model_path=embedding_model_path,
            output_root=output_root, allow_missing_embedding=allow_missing_embedding)
        if config.get("protocol") != PROTOCOL or original.get("protocol") != PROTOCOL:
            raise ValueError(f"Use a {PROTOCOL} recipe")
        if {spec["name"] for spec in config["datasets"]} != DATASETS:
            raise ValueError("Segment-context rebuild currently supports LoCoMo only")
        if (type(config.get("child_context_segments")) is not int
                or config["child_context_segments"] != PREVIOUS_SEGMENTS):
            raise ValueError("child_context_segments must be 3 complete preceding segments")
        if config["writer"]["context_blocks"] != 0:
            raise ValueError("Set writer.context_blocks=0 when using complete preceding segments")
        if any(config.get(key) for key in (
                "preserve_existing_directories", "allow_migrated_source_identity",
                "reuse_root", "reuse_mapping")):
            raise ValueError("Use a fresh output_root; old directories or boundary caches cannot be imported")
        if not CHILD_PROMPT.read_text().strip() or not BOUNDARY_PROMPT.read_text().strip():
            raise ValueError("Segment-context and topic/event prompts must be nonempty")
        identity = deepcopy(original)
        identity.update(boundary_prompt_version=BOUNDARY_PROMPT_VERSION,
                        child_prompt_version=CHILD_PROMPT_VERSION,
                        child_context_policy=context_policy(),
                        directory_reuse_policy="new_protocol_root_only.v1")
        identity["prompt_hashes"]["child.txt"] = file_digest(CHILD_PROMPT)
        identity["prompt_hashes"]["boundary.txt"] = file_digest(BOUNDARY_PROMPT)
        for source in (Path(__file__), Path(segment_context_writer.__file__), Path(topic_event.__file__)):
            identity["implementation_hashes"][source.name] = file_digest(source)
        return config, identity


def _validate_output_root(root, identity):
    """Reject another protocol read-only, before the legacy store can create files."""
    root = Path(root)
    if not root.exists():
        return
    if not root.is_dir():
        raise ValueError("Directory output_root must be a directory")
    entries = {p.name for p in root.iterdir()} - {".build.lock"}
    if not entries:
        return
    manifest = root / "run_manifest.json"
    journal = root / "build_journal.sqlite3"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("Directory build identity changed; use a fresh output_root")
    if not journal.is_file():
        raise ValueError("Nonempty output_root has no matching build journal; use a fresh output_root")
    db = sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        try:
            previous = db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        except sqlite3.DatabaseError as exc:
            raise ValueError("Invalid build journal; use a fresh output_root") from exc
        if previous is None or json.loads(previous[0]) != identity:
            raise ValueError("Directory build identity changed; use a fresh output_root")
    finally:
        db.close()


def run(args, **injected_factories):
    limit = getattr(args, "limit_sessions", None)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit_sessions must be a positive integer")
    if getattr(args, "datasets", None) and set(args.datasets) != DATASETS:
        raise ValueError("Segment-context rebuild supports LoCoMo only")
    with _context_engine():
        config, identity = load_recipe(args.config, args.workers,
            embedding_model_path=getattr(args, "embedding_model_path", None),
            output_root=getattr(args, "output_root", None), allow_missing_embedding=args.dry_run)
        _validate_output_root(config["output_root"], identity)
        if args.dry_run:
            _, preview = legacy.plan(config, identity, args.datasets, args.splits)
            preview.update(child_prompt_version=CHILD_PROMPT_VERSION,
                           child_context_policy=context_policy(),
                           reuses_old_directories=False, reuses_old_boundary_cache=False)
            legacy.announce("segment_context_directory_plan", **preview)
            return preview
        if injected_factories.get("writer_factory") is None:
            injected_factories["writer_factory"] = SegmentContextDirectoryWriter
        return legacy.run(args, **injected_factories)


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS))
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"])
    urls = parser.add_mutually_exclusive_group()
    urls.add_argument("--base-url")
    urls.add_argument("--base-urls", nargs="+")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-model-path")
    parser.add_argument("--output-root")
    parser.add_argument("--limit-sessions", type=int,
                        help="Build at most this many unfinished sessions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the source/config without model requests or GPU deployment")
    return parser


def main():
    run(argument_parser().parse_args())


if __name__ == "__main__":
    main()
