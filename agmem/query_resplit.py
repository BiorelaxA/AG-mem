"""Merge LoCoMo train/validation and repartition by exact query, reusing Writer data."""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_CEILING
import json
from pathlib import Path
import random
import shutil
import tempfile

from .core import atomic_json, canonical, file_digest, load_config
from .validation_split import _reuse_prepared, _scalar_rows, _verify_parquet, _write_parquet


VERSION = "merged_exact_query_split_v1"
DEFAULT_DATA = "data/processed/locomo_query9010_seed42_20261102"
DEFAULT_PREPARED = "runs/locomo_longmemeval/prepared_locomo_query9010_seed42_20261102"


def split_queries(rows, *, fraction="0.1", seed=42):
    """Return validation queries; duplicate rows never influence the denominator."""
    fraction = Decimal(str(fraction))
    if not fraction.is_finite() or not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    queries = sorted({row["question"] for row in rows})
    count = int((len(queries) * fraction).to_integral_value(rounding=ROUND_CEILING))
    if not 0 < count < len(queries):
        raise ValueError("Both query partitions must be nonempty")
    random.Random(seed).shuffle(queries)
    return set(queries[:count])


def resplit_locomo(config, output_data, output_prepared, *, fraction="0.1", seed=42):
    data = config["data"]
    if data.get("layout") != "by_source" or data["splits"] != ["train", "validation"]:
        raise ValueError("Resplitting requires by_source train/validation data")
    sources = data["sources"]
    if "locomo" not in sources:
        raise ValueError("Configured sources must include locomo")
    root = Path(data["input_root"]).resolve()
    output_data, output_prepared = Path(output_data).resolve(), Path(output_prepared).resolve()
    if output_data == output_prepared or output_data in output_prepared.parents or output_prepared in output_data.parents:
        raise ValueError("Output roots must be separate directories")
    if output_data.exists() or output_prepared.exists():
        raise ValueError("Use fresh output roots; existing data is never overwritten")
    parent = root / "manifest.json"
    entries = {row["source"]: row for row in json.loads(parent.read_text())["sources"]}
    scalar = {}
    for source in sources:
        for split in ("train", "validation", "test"):
            path = root / source / f"{split}.parquet"
            if file_digest(path) != entries[source]["output_files"][split]["sha256"]:
                raise ValueError(f"Changed source data: {path}")
            scalar[source, split] = _scalar_rows(path)
    held_out = split_queries(scalar["locomo", "train"] + scalar["locomo", "validation"],
                             fraction=fraction, seed=seed)
    mappings = {}
    for source in sources:
        mappings[source] = {"train": [], "validation": []}
        for old_split in ("train", "validation"):
            for index, row in enumerate(scalar[source, old_split]):
                split = ("validation" if row["question"] in held_out else "train") if source == "locomo" else old_split
                mappings[source][split].append((old_split, index))
    queries = {split: {scalar[source, old_split][i]["question"]
                      for source in sources for old_split, i in mappings[source][split]}
               for split in ("train", "validation")}
    test_queries = {row["question"] for source in sources for row in scalar[source, "test"]}
    if queries["train"] & queries["validation"] or (queries["train"] | queries["validation"]) & test_queries:
        raise ValueError("Train/validation/test query overlap")
    report = {"status": "complete", "version": VERSION, "seed": seed,
              "parent_manifest": str(parent), "parent_manifest_sha256": file_digest(parent),
              "policy": {"operation": "merge_train_validation_then_resplit_locomo",
                         "fraction": str(fraction), "fraction_basis": "distinct_merged_queries",
                         "rounding": "ceil_validation_query_count", "query_match": "exact_original_string",
                         "sampling": "sorted_unique_queries_then_seeded_shuffle",
                         "duplicate_queries": "keep_all_rows_in_same_partition",
                         "other_configured_sources": "preserve_all_splits"}, "sources": []}
    output_data.parent.mkdir(parents=True, exist_ok=True)
    output_prepared.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".query-resplit-", dir=output_data.parent) as temporary_data, \
            tempfile.TemporaryDirectory(prefix=".query-prepared-", dir=output_prepared.parent) as temporary_prepared:
        stage, prepared = Path(temporary_data), Path(temporary_prepared)
        row_mapping = []
        for source in sources:
            destination = stage / source
            destination.mkdir()
            files, counts = {}, {}
            for split in ("train", "validation", "test"):
                path = destination / f"{split}.parquet"
                if split == "test" or source != "locomo":
                    shutil.copy2(root / source / path.name, path)
                    n = len(scalar[source, split])
                    if file_digest(path) != entries[source]["output_files"][split]["sha256"]:
                        raise ValueError(f"Copied data differs: {path}")
                else:
                    inputs = [(root / source / f"{old_split}.parquet",
                               {i for origin, i in mappings[source][split] if origin == old_split})
                              for old_split in ("train", "validation")]
                    _write_parquet(path, inputs)
                    n = _verify_parquet(path, inputs)
                files[split] = {"rows": n, "bytes": path.stat().st_size, "sha256": file_digest(path)}
                if split != "test":
                    mapping = mappings[source][split]
                    assert n == len(mapping)
                    counts[split + "_rows"] = n
                    counts[split + "_queries"] = len({scalar[source, origin][i]["question"] for origin, i in mapping})
                    for index, (origin, i) in enumerate(mapping):
                        row_mapping.append({"source": source, "split": split, "output_row_index": index,
                                            "parent_split": origin, "parent_row_index": i,
                                            "question": scalar[source, origin][i]["question"]})
            report["sources"].append({"source": source, **counts, "output_files": files})
        report["verification"] = {"all_fields_and_schema_match_parent_rows": True,
                                  "test_files_byte_identical": True, "other_sources_byte_identical": True,
                                  "train_validation_query_overlap": 0, "train_test_query_overlap": 0,
                                  "validation_test_query_overlap": 0}
        atomic_json(stage / "manifest.json", report)
        (stage / "row_mapping.jsonl").write_text("".join(canonical(row) + "\n" for row in row_mapping))
        for source in sources:
            _reuse_prepared(config, source, prepared / source, file_digest(stage / "manifest.json"),
                            mappings[source], {split: scalar[source, split] for split in ("train", "validation")},
                            version=VERSION)
        stage.rename(output_data)
        prepared.rename(output_prepared)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_dapo_reader_loop_v3_val100.yaml")
    parser.add_argument("--output-data", default=DEFAULT_DATA)
    parser.add_argument("--output-prepared", default=DEFAULT_PREPARED)
    parser.add_argument("--fraction", default="0.1", help="Fraction of merged distinct queries held out")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = resplit_locomo(load_config(args.config), args.output_data, args.output_prepared,
                            fraction=args.fraction, seed=args.seed)
    print(json.dumps({"data_root": args.output_data, "prepared_root": args.output_prepared,
                      "sources": [{k: v for k, v in row.items() if k != "output_files"}
                                  for row in report["sources"]]}, indent=2))


if __name__ == "__main__":
    main()
