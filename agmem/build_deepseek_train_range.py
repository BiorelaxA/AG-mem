"""DeepSeek M train shard with original Parquet row numbers and bank IDs."""
from contextlib import nullcontext
from copy import copy
from pathlib import Path

import pyarrow.parquet as pq

from . import build_adjacent_directories as engine
from . import build_deepseek_directories as deepseek
from . import build_deepseek_previous_segment as previous
from .build_portable_deepseek_directories import portable_worker
from .core import digest
from .hierarchical.adapters import _source, _normalize_history


def train_banks(path, sha256, first, last, *, count_tokens):
    parquet = pq.ParquetFile(path)
    start = 0
    for group in range(parquet.num_row_groups):
        end = start + parquet.metadata.row_group(group).num_rows
        if start >= last:
            break
        if end >= first:
            index = start
            for batch in parquet.iter_batches(batch_size=2, row_groups=[group]):
                left, right = max(0, first - 1 - index), min(len(batch), last - index)
                if right > left:
                    for offset, row in enumerate(batch.slice(left, right - left).to_pylist(), index + left):
                        source = _source(path, sha256, "train", offset, row, "longmemeval_parquet")
                        name = "split_longmemeval_m"
                        bank_id = "bank_" + digest([name, "train", sha256, offset])[:24]
                        yield _normalize_history(name, row, source, bank_id, count_tokens)
                index += len(batch)
                if index >= last:
                    break
        start = end


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
        return train_banks(path, source["sha256"], first, last, count_tokens=count_tokens)

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
