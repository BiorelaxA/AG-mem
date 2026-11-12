"""Expand query-held-out validation and reuse already built memory directories.

Run with ``python -m agmem.validation_split``. No models are loaded or called.
Existing input and prepared directories are read-only; outputs must be new paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path
import random
import re
import shutil
import tempfile

from .core import atomic_json, canonical, file_digest, load_config
from .data import ADAPTER_VERSION, adapt_row, decode_list
from .pipeline import dataset_configs, validate_directories
from .store import Store


VERSION = "validation_query_fraction_v2"
DEFAULT_INPUT = "data/processed/query_dedup_val10_plus1pct_seed42_20261031"
DEFAULT_PREPARED = "runs/locomo_longmemeval/prepared_val10_plus1pct_seed42_20261031"


def support_count(row):
    values = decode_list(row.get("answer_session_ids"), "answer_session_ids")
    return len({re.sub(r"^(?:answer_)+", "", str(value)) for value in values})


def select_additions(rows, *, excluded_queries=(), fraction="0.01", seed=42,
                     mode="add", validation_queries=0, target_queries=None):
    """Sample unique query strings, balanced across annotated support counts.

    The denominator counts distinct training questions, never sessions or rollout
    trajectories. Repeated questions are ineligible so no twin remains in train.
    """
    fraction = Decimal(str(fraction))
    if not fraction.is_finite() or not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    if mode not in {"add", "target"}:
        raise ValueError("mode must be add or target")
    if type(validation_queries) is not int or validation_queries < 0:
        raise ValueError("validation_queries must be a nonnegative integer")
    if target_queries is not None and (mode != "target" or type(target_queries) is not int or target_queries < 0):
        raise ValueError("An explicit nonnegative target_queries requires target mode")
    counts = Counter(row["question"] for row in rows)
    fraction_size = int((Decimal(len(counts)) * fraction).to_integral_value(rounding=ROUND_CEILING))
    target = (validation_queries + fraction_size if mode == "add"
              else fraction_size if target_queries is None else target_queries)
    size = max(0, target - validation_queries)
    if 0 < size < 2:
        raise ValueError("The requested addition needs at least two queries to include both strata")
    excluded = set(excluded_queries)
    pools = {"single": [], "multi": []}
    for index, row in enumerate(rows):
        query = row["question"]
        if counts[query] != 1 or query in excluded:
            continue
        count = support_count(row)
        if count:
            pools["single" if count == 1 else "multi"].append((query, index))
    # An odd addition gives the extra question to multi-session coverage.
    quotas = {"single": size // 2, "multi": size - size // 2}
    rng = random.Random(seed)
    selected = []
    for kind in ("single", "multi"):
        pool = sorted(pools[kind])
        if len(pool) < quotas[kind]:
            raise ValueError(f"Not enough eligible {kind}-session questions: {len(pool)} < {quotas[kind]}")
        selected.extend(index for _, index in rng.sample(pool, quotas[kind]))
    return sorted(selected), {"train_rows_before": len(rows), "train_queries_before": len(counts),
                              "fraction": str(fraction), "added_queries": size,
                              "mode": mode, "validation_queries_before": validation_queries,
                              "requested_validation_queries": target,
                              "explicit_target_queries": target_queries,
                              "fraction_basis": "distinct_train_queries_before_holdout",
                              "added_strata": quotas,
                              "eligible_strata": {kind: len(pool) for kind, pool in pools.items()}}


def _scalar_rows(path):
    import pyarrow.parquet as pq
    file = pq.ParquetFile(path)
    fields = ("question", "answer", "question_id", "question_type", "question_date", "answer_session_ids")
    return file.read(columns=[field for field in fields if field in file.schema_arrow.names]).to_pylist()


def _write_parquet(path, inputs):
    """Stream original Arrow rows without converting large histories to Python."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    schema = pq.ParquetFile(inputs[0][0]).schema_arrow
    with pq.ParquetWriter(path, schema, compression="zstd") as writer:
        for source, wanted in inputs:
            file = pq.ParquetFile(source)
            if not file.schema_arrow.equals(schema, check_metadata=True):
                raise ValueError(f"Source schemas differ: {source}")
            offset = 0
            for batch in file.iter_batches(batch_size=8):
                indices = [i for i in range(batch.num_rows) if offset + i in wanted]
                if indices:
                    writer.write_batch(batch.take(pa.array(indices, type=pa.int64())))
                offset += batch.num_rows


def _verify_parquet(path, inputs):
    """Compare every output field and schema against its original ordered rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    output = pq.ParquetFile(path)
    expected_schema = pq.ParquetFile(inputs[0][0]).schema_arrow
    if not output.schema_arrow.equals(expected_schema, check_metadata=True):
        raise ValueError(f"Output schema differs: {path}")
    actual = iter(output.iter_batches(batch_size=1))
    count = 0
    for source, wanted in inputs:
        offset = 0
        for batch in pq.ParquetFile(source).iter_batches(batch_size=8):
            for i in range(batch.num_rows):
                if offset + i not in wanted:
                    continue
                got = next(actual, None)
                if got is None or not pa.Table.from_batches([got]).equals(
                        pa.Table.from_batches([batch.slice(i, 1)]), check_metadata=True):
                    raise ValueError(f"Output row {count} differs from its original: {path}")
                count += 1
            offset += batch.num_rows
    if next(actual, None) is not None:
        raise ValueError(f"Unexpected additional rows: {path}")
    return count


def _reuse_prepared(config, source, target, new_manifest_hash, mappings, scalar, *, version=VERSION):
    old_root = Path(config["data"]["prepared_root"]) / source
    old = Store(old_root)
    metadata = deepcopy(old.metadata)
    identity = metadata.get("prepare_identity", {})
    expected = {"input_manifest_sha256": file_digest(Path(config["data"]["input_root"]) / "manifest.json"),
                "sources": [source], "splits": ["train", "validation"], "limit_per_source_split": None,
                "adapter_version": ADAPTER_VERSION, "storage_format": Store.FORMAT}
    if not metadata.get("prepare_complete") or identity != expected:
        raise ValueError(f"Existing prepared provenance differs: {old_root}")
    child = dict(dataset_configs(config))[source]
    validate_directories(child, old, ["train", "validation"])
    examples = {split: {e.sample_id: e for e in old.examples(split)} for split in ("train", "validation")}
    target.mkdir(parents=True)
    # Copies have independent inodes: no runtime may append into an older run's store.
    for name in ("memory_records.jsonl", "memory_records.index.json", "directories.jsonl",
                 "directories.index.json", "directory_manifest.json"):
        path = old_root / name
        if path.exists():
            shutil.copy2(path, target / name)
    metadata["prepare_identity"] = {**expected, "input_manifest_sha256": new_manifest_hash}
    metadata["validation_split_origin"] = {"prepared_root": str(old_root.resolve()), "version": version}
    atomic_json(target / "manifest.json", metadata)
    counts = {}
    for split, mapping in mappings.items():
        with (target / f"{split}_examples.jsonl").open("w") as stream:
            for index, (parent_split, parent_index) in enumerate(mapping):
                prior = examples[parent_split][f"{source}/{parent_split}/{parent_index:06d}"]
                raw = scalar[parent_split][parent_index]
                example = adapt_row(raw, source, split, index, memory_id=prior.memory_id)
                if example.query != prior.query or example.gold_answers != prior.gold_answers:
                    raise ValueError(f"Prepared task differs from source row: {prior.sample_id}")
                stream.write(canonical(asdict(example)) + "\n")
        counts[split] = len(mapping)
    atomic_json(target / "preparation_report.json", {"status": "complete", "identity": metadata["prepare_identity"],
                "counts": {f"{source}/{split}/valid": n for split, n in counts.items()},
                "reused_directory_version": metadata["directory_version"],
                "origin": metadata["validation_split_origin"]})
    # Deliberately do not close old: its index caches must remain untouched.
    return counts


def expand_validation(config, output_data, output_prepared, *, fraction="0.01", seed=42,
                      mode="add", target_counts=None):
    root = Path(config["data"]["input_root"]).resolve()
    output_data, output_prepared = Path(output_data).resolve(), Path(output_prepared).resolve()
    if config["data"].get("layout") != "by_source" or config["data"]["splits"] != ["train", "validation"]:
        raise ValueError("Expansion requires by_source prepared train/validation data")
    if output_data == output_prepared or output_data in output_prepared.parents or output_prepared in output_data.parents:
        raise ValueError("Output data and prepared roots must be separate directories")
    if output_data.exists() or output_prepared.exists():
        raise ValueError("Use fresh output roots; existing splits and prepared stores are never overwritten")
    manifest = json.loads((root / "manifest.json").read_text())
    entries = {entry["source"]: entry for entry in manifest["sources"]}
    sources = config["data"]["sources"]
    target_counts = {} if target_counts is None else dict(target_counts)
    if set(target_counts) - set(sources):
        raise ValueError("Explicit validation targets must belong to configured sources")
    blocked = set()
    scalar = {}
    # Keep held-out queries disjoint from every existing validation/test source.
    for source, entry in entries.items():
        for split in ("validation", "test"):
            path = root / source / f"{split}.parquet"
            if file_digest(path) != entry["output_files"][split]["sha256"]:
                raise ValueError(f"Changed source data: {path}")
            rows = _scalar_rows(path)
            blocked.update(row["question"] for row in rows)
            if source in sources:
                scalar[source, split] = rows
        if source in sources:
            path = root / source / "train.parquet"
            if file_digest(path) != entry["output_files"]["train"]["sha256"]:
                raise ValueError(f"Changed source data: {path}")
            scalar[source, "train"] = _scalar_rows(path)
    train_counts = Counter(row["question"] for source in sources for row in scalar[source, "train"])
    excluded = blocked | {query for query, n in train_counts.items() if n != 1}
    selections = {}
    for source in sources:
        selections[source] = select_additions(scalar[source, "train"], excluded_queries=excluded,
            fraction=fraction, seed=seed, mode=mode,
            validation_queries=len({row["question"] for row in scalar[source, "validation"]}),
            target_queries=target_counts.get(source))
    output_data.parent.mkdir(parents=True, exist_ok=True)
    output_prepared.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".validation-split-", dir=output_data.parent) as data_tmp, \
            tempfile.TemporaryDirectory(prefix=".validation-prepared-", dir=output_prepared.parent) as prepared_tmp:
        stage, prepared = Path(data_tmp), Path(prepared_tmp)
        new_manifest = {"status": "complete", "version": VERSION, "seed": seed,
                        "parent_manifest": str(root / "manifest.json"),
                        "parent_manifest_sha256": file_digest(root / "manifest.json"),
                        "policy": {"operation": ("add_ceil_fraction_of_distinct_train_queries" if mode == "add"
                                                 else "expand_to_target_validation_queries"),
                                   "mode": mode, "explicit_targets": target_counts,
                                   "fraction_basis": "distinct_train_queries_before_holdout",
                                   "fraction": str(fraction), "preserve_existing_validation_prefix": True,
                                   "single_multi_sampling": "equal_counts_extra_multi_if_odd",
                                   "support_definition": "distinct normalized answer_session_ids; not minimum required reads",
                                   "query_match": "exact_original_string",
                                   "scope": "configured training sources only; other source snapshots are unchanged"},
                        "sources": []}
        mappings = {}
        added_manifest = []
        row_manifests = {split: [] for split in ("train", "validation")}
        for source in sources:
            added, report = selections[source]
            keep = sorted(set(range(len(scalar[source, "train"]))) - set(added))
            old_val = list(range(len(scalar[source, "validation"])))
            mappings[source] = {"train": [("train", i) for i in keep],
                                "validation": [("validation", i) for i in old_val] + [("train", i) for i in added]}
            destination = stage / source
            destination.mkdir()
            files = {}
            for split, inputs in {
                    "train": [(root / source / "train.parquet", set(keep))],
                    "validation": [(root / source / "validation.parquet", set(old_val)),
                                   (root / source / "train.parquet", set(added))]}.items():
                path = destination / f"{split}.parquet"
                _write_parquet(path, inputs)
                n = _verify_parquet(path, inputs)
                files[split] = {"rows": n, "bytes": path.stat().st_size, "sha256": file_digest(path)}
                for index, (parent_split, parent_index) in enumerate(mappings[source][split]):
                    raw = scalar[source, parent_split][parent_index]
                    row_manifests[split].append({"source": source, "split": split, "output_row_index": index,
                        "parent_split": parent_split, "parent_row_index": parent_index,
                        "parent_path": str(root / source / f"{parent_split}.parquet"),
                        "question": raw["question"], "question_id": raw.get("question_id"),
                        "support_session_count": support_count(raw)})
            path = destination / "test.parquet"
            shutil.copy2(root / source / "test.parquet", path)
            files["test"] = {"rows": len(scalar[source, "test"]), "bytes": path.stat().st_size,
                             "sha256": file_digest(path)}
            if files["test"]["sha256"] != entries[source]["output_files"]["test"]["sha256"]:
                raise ValueError("Test copy changed")
            val_rows = [scalar[source, ps][i] for ps, i in mappings[source]["validation"]]
            kinds = Counter("single" if support_count(row) == 1 else "multi" for row in val_rows)
            new_manifest["sources"].append({"source": source, **report, "train_rows": len(keep),
                "train_queries": len({scalar[source, "train"][i]["question"] for i in keep}),
                "validation_rows": len(val_rows), "validation_strata": dict(kinds),
                "added_parent_train_rows": added, "output_files": files})
            added_manifest.extend(row for row in row_manifests["validation"]
                                  if row["source"] == source and row["parent_split"] == "train")
        train_queries = {row["question"] for row in row_manifests["train"]}
        val_queries = {row["question"] for row in row_manifests["validation"]}
        test_queries = {row["question"] for source in sources for row in scalar[source, "test"]}
        if train_queries & val_queries or val_queries & test_queries:
            raise ValueError("Train/validation/test query overlap")
        new_manifest["verification"] = {"all_fields_and_schema_match_parent_rows": True,
            "test_files_byte_identical": True, "train_validation_query_overlap": 0,
            "validation_test_query_overlap": 0, "existing_validation_prefix_preserved": True}
        atomic_json(stage / "manifest.json", new_manifest)
        for name, rows in {"row_mapping_train.jsonl": row_manifests["train"],
                           "row_mapping_validation.jsonl": row_manifests["validation"],
                           "added_validation_manifest.jsonl": added_manifest}.items():
            (stage / name).write_text("".join(canonical(row) + "\n" for row in rows))
        for source in sources:
            _reuse_prepared(config, source, prepared / source, file_digest(stage / "manifest.json"),
                           mappings[source], {split: scalar[source, split] for split in ("train", "validation")})
        # Both products have passed source/content checks before becoming visible.
        stage.rename(output_data)
        prepared.rename(output_prepared)
    return new_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_dapo_max1_no_call_cost_mixed_v2.yaml")
    parser.add_argument("--output-data", default=DEFAULT_INPUT)
    parser.add_argument("--output-prepared", default=DEFAULT_PREPARED)
    parser.add_argument("--fraction", default="0.01")
    parser.add_argument("--mode", choices=("add", "target"), default="add",
                        help="Add a fraction, or make total validation reach that fraction of current training queries")
    parser.add_argument("--target-size", action="append", default=[], metavar="SOURCE=N",
                        help="Override one source's total validation target (requires --mode target)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    targets = {}
    for value in args.target_size:
        source, sep, count = value.partition("=")
        if not source or not sep or not count.isdigit() or source in targets:
            parser.error("--target-size must be a unique SOURCE=nonnegative_integer")
        targets[source] = int(count)
    report = expand_validation(load_config(args.config), args.output_data, args.output_prepared,
                               fraction=args.fraction, seed=args.seed, mode=args.mode, target_counts=targets)
    print(json.dumps({"data_root": args.output_data, "prepared_root": args.output_prepared,
                      "sources": [{k: row[k] for k in ("source", "train_rows", "train_queries",
                        "validation_rows", "validation_strata", "added_queries", "added_strata")}
                                  for row in report["sources"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
