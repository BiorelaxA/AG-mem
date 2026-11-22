"""Versioned topic/event boundary review with the frozen v1 build engine.

The legacy engine fingerprints its implementation files and top-level prompt
resources. Keeping those files unchanged preserves existing build journals and
their explicit resplit-reuse contract. This worker changes only the boundary
prompt, records that change in construction identity, and requires a fresh root.

Run this entry point in its own process. The scoped module adaptation is restored
even on failure; the lock serializes this module's nested/reentrant entry points,
not unrelated callers of the legacy engine in the same process.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from threading import RLock

from . import build_adjacent_directories as legacy
from .core import file_digest
from .hierarchical.adjacent_writer import AdjacentDirectoryWriter


DEFAULT_CONFIG = "configs/build_adjacent_qwen_topic_event_v2.yaml"
PROTOCOL = "adjacent_topic_event_directory_v2"
# Parent/child summary prompts and directory schema retain their v1 protocol.
# The distinct construction identity below versions the boundary policy alone.
PROMPT_VERSION = "directory-writer.adjacent.v1"
BOUNDARY_PROMPT_VERSION = "topic_event.v2"
DATASETS = legacy.DATASETS | {"split_longmemeval_m"}
BOUNDARY_PROMPT = (Path(__file__).with_name("hierarchical") / "resources" /
                   "adjacent" / "topic_event_v2" / "boundary.txt")
_BASE_LOAD_RECIPE = legacy.load_recipe
_ADAPTATION_LOCK = RLock()


@contextmanager
def _topic_event_engine():
    """Adapt only this dedicated worker's engine globals, restoring on exit."""
    with _ADAPTATION_LOCK:
        previous = legacy.PROTOCOL, legacy.load_recipe, legacy.DATASETS
        legacy.PROTOCOL = PROTOCOL
        legacy.load_recipe = load_recipe
        # Extend the v2 workflow without changing the implementation hash of
        # the legacy engine used by existing v1 build journals.
        legacy.DATASETS = DATASETS
        try:
            yield
        finally:
            legacy.PROTOCOL, legacy.load_recipe, legacy.DATASETS = previous


def load_recipe(path, workers=None, *, embedding_model_path=None, output_root=None,
                allow_missing_embedding=False):
    """Load v2 independently or within ``run``, retaining all other semantics."""
    with _topic_event_engine():
        config, original = _BASE_LOAD_RECIPE(
            path, workers, embedding_model_path=embedding_model_path,
            output_root=output_root, allow_missing_embedding=allow_missing_embedding)
        if config.get("protocol") != PROTOCOL or original.get("protocol") != PROTOCOL:
            raise ValueError(f"Use a {PROTOCOL} recipe")
        if not BOUNDARY_PROMPT.read_text().strip():
            raise ValueError("The topic/event boundary prompt is empty")
        identity = deepcopy(original)
        identity["boundary_prompt_version"] = BOUNDARY_PROMPT_VERSION
        identity["prompt_hashes"]["boundary.txt"] = file_digest(BOUNDARY_PROMPT)
        identity["implementation_hashes"][Path(__file__).name] = file_digest(Path(__file__))
        preserve = config.get("preserve_existing_directories", False)
        if type(preserve) is not bool:
            raise ValueError("preserve_existing_directories must be boolean")
        if preserve:
            if config.get("retry_fallback") is not False:
                raise ValueError("Preserving existing directories requires retry_fallback=false")
            identity["directory_reuse_policy"] = "preserve_existing_build_missing.v1"
        return config, identity


class TopicEventDirectoryWriter(AdjacentDirectoryWriter):
    """Use the new boundary policy; retain child and parent writing unchanged."""

    def __init__(self, tokenizer, settings, complete):
        super().__init__(tokenizer, settings, complete)
        self.adjacent_prompts["boundary"] = BOUNDARY_PROMPT.read_text()


def run(args, **injected_factories):
    from .reuse_topic_event_directories import export_rule_provenance, validate_mixed_policy

    if injected_factories.get("writer_factory") is None:
        injected_factories["writer_factory"] = TopicEventDirectoryWriter
    with _topic_event_engine():
        config, identity = load_recipe(args.config, args.workers,
            embedding_model_path=getattr(args, "embedding_model_path", None),
            output_root=getattr(args, "output_root", None), allow_missing_embedding=args.dry_run)
        policy = validate_mixed_policy(config, identity, dry_run=args.dry_run,
            datasets=getattr(args, "datasets", None), splits=getattr(args, "splits", None))
        if policy is not None:
            legacy.announce("topic_event_directory_preservation", **policy)
        failure = None
        try:
            return legacy.run(args, **injected_factories)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            if policy is not None and not args.dry_run:
                try:
                    export_rule_provenance(config["output_root"])
                except Exception as exc:
                    if failure is None:
                        raise
                    legacy.announce("mixed_directory_provenance_export_failed",
                                    error_type=type(exc).__name__, reason=str(exc))


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
                        help="Build at most this many unfinished sessions for a smoke run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check inputs/config without inference; report missing encoder weights")
    return parser


def main():
    run(argument_parser().parse_args())


if __name__ == "__main__":
    main()
