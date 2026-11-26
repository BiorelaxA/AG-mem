"""DeepSeek M train shard with original Parquet row numbers and bank IDs."""
from contextlib import closing, nullcontext
from copy import copy
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from . import build_adjacent_directories as engine
from . import build_deepseek_directories as deepseek
from . import build_deepseek_previous_segment as previous
from .build_portable_deepseek_directories import portable_worker
from .core import digest
from .hierarchical.longmemeval_range import range_banks


def missing_parents(spec, sha256, session_counts, output_root):
    first, last = spec["trajectory_range_1based"]
    wanted = {}
    row_numbers = {}
    for index in range(first - 1, last):
        bank = "bank_" + digest([spec["name"], "train", sha256, index])[:24]
        row_numbers[bank] = index
        wanted[index] = {f"p{n:04d}" for n in range(1, len(session_counts[index]) + 1)}
    baseline = Path(spec["exclude_committed_root"]).resolve()
    target = Path(output_root).resolve()
    if baseline == target:
        raise ValueError("Missing-session output must be separate from the merged snapshot")
    for root in (baseline, target):
        journal = root / "build_journal.sqlite3"
        if root == target and not journal.exists():
            continue
        with closing(sqlite3.connect(journal.as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("BEGIN")
            if root == baseline:
                identity = json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
                if (identity.get("protocol"), identity.get("dataset"), identity.get("split"),
                    identity.get("input_sha256")) != ("merged_directory_snapshot.v1", spec["name"], "train", sha256):
                    raise ValueError("Merged journal does not match the requested train Parquet")
            committed = db.execute("SELECT bank_id,parent_id FROM sessions WHERE directory IS NOT NULL "
                                   "AND status IN ('complete','empty','fallback')")
            for bank, parent in committed:
                wanted[row_numbers[bank]].remove(parent)
    return {index: parents for index, parents in wanted.items() if parents}


def train_banks(path, sha256, first, last, *, count_tokens, wanted=None):
    return range_banks(path, sha256, "train", first, last, count_tokens=count_tokens, wanted=wanted)


def run(args, portable=None):
    config, identity = previous.load_recipe(args.config, args.workers,
        embedding_model_path=args.embedding_model_path, output_root=args.output_root,
        allow_missing_embedding=args.dry_run)
    spec, = config["datasets"]
    if (spec["name"] != "split_longmemeval_m" or spec["splits"] != ["train"]
            or args.splits != ["train"] or args.datasets != [spec["name"]]):
        raise ValueError("The range worker only accepts LongMemEval-M train")
    first, last = spec["trajectory_range_1based"]
    source, = identity["datasets"][0]["files"]
    path = Path(source["path"])
    if portable:
        original = next(f for d in portable.source_identity["datasets"] if d["spec"]["name"] == spec["name"]
                        for f in d["files"] if f["split"] == "train")
        if source["sha256"] != original["sha256"]:
            raise ValueError("Training Parquet differs from the transferred original")
    session_counts = pq.read_table(path, columns=["haystack_session_ids"]).column(0)
    if not (1 <= first <= last <= len(session_counts)):
        raise ValueError("Trajectory range is outside the training Parquet")
    wanted = None
    if spec.get("exclude_committed_root"):
        wanted = missing_parents(spec, source["sha256"], session_counts, config["output_root"])
        engine.announce("deepseek_train_missing", trajectories=len(wanted), sessions=sum(map(len, wanted.values())),
                        remaining_rows_1based=[i + 1 for i in wanted], existing_fallbacks_preserved=True)
    engine.announce("deepseek_train_range", first_1based=first, last_1based=last,
                    trajectories=last-first+1, sessions=sum(len(row) for row in session_counts.slice(first-1,last-first+1)),
                    source_sha256=source["sha256"], output_root=config["output_root"])
    planned = copy(args)
    planned.base_urls = [config["service"]["base_url"]]
    endpoints = args.embedding_base_urls or []
    if not args.dry_run and len(endpoints) != 1:
        raise ValueError("This shard uses one embedding service")
    planned.embedding_base_url = endpoints[0] if endpoints else None

    def selected_banks(spec, splits, *, count_tokens, tokenizer_id):
        return train_banks(path, source["sha256"], first, last, count_tokens=count_tokens, wanted=wanted)

    # Run in this dedicated worker process, preserving other builders' identities.
    saved = engine.iter_banks, engine.load_recipe, engine.bounded_results
    engine.iter_banks = selected_banks
    engine.load_recipe = lambda *a, **kw: (config, identity)
    engine.bounded_results = deepseek._fail_fast_results
    try:
        return engine.run(planned, pool_factory=deepseek.DeepSeekWriterPool,
                          writer_factory=previous.PreviousSegmentWriter, segmenter_factory=deepseek.DeepSeekSegmenter)
    finally:
        engine.iter_banks, engine.load_recipe, engine.bounded_results = saved


def main():
    parser = deepseek.argument_parser()
    parser.add_argument("--migration-manifest")
    args = parser.parse_args()
    context = (portable_worker(args.migration_manifest, args.config, args.embedding_model_path)
               if args.migration_manifest else nullcontext())
    with context as portable:
        run(args, portable)


if __name__ == "__main__":
    main()
