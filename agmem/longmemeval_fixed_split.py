"""Keep the original LongMemEval test set and expand validation from training.

S/M use one seeded question-ID assignment. Test parquet files are copied exactly;
the row/bank mapping supports directory reuse. Reruns verify without overwriting.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import tempfile

from .core import atomic_json, canonical, digest, file_digest
from .longmemeval_resplit import DEFAULT_INPUT, SOURCES, SPLITS, _read_scalars, _stratified_sample
from .validation_split import _verify_parquet, _write_parquet

VERSION = "longmemeval_preserve_test_expand_validation_v1"
DEFAULT_OUTPUT = "data/processed/longmemeval_315_35_150_seed42_20261113"


def plan_fixed_assignments(rows_by_source, *, validation_rows=35, seed=42):
    if type(validation_rows) is not int or validation_rows < 1:
        raise ValueError("validation_rows must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if set(rows_by_source) != set(SOURCES):
        raise ValueError("Provide both LongMemEval-S and LongMemEval-M")
    canonical_rows, original_ids = None, None
    for source in SOURCES:
        splits = rows_by_source[source]
        if set(splits) != set(SPLITS):
            raise ValueError(f"Missing original splits: {source}")
        by_id, queries, ids = {}, set(), {}
        for split in SPLITS:
            ids[split] = set()
            for row in splits[split]:
                if any(not isinstance(row.get(key), str) or not row[key].strip()
                       for key in ("question_id", "question", "question_type")):
                    raise ValueError(f"Invalid question identity in {source}/{split}")
                qid = row["question_id"]
                query = " ".join(row["question"].split()).casefold()
                if qid in by_id or query in queries:
                    raise ValueError(f"Duplicate question ID/query in {source}")
                by_id[qid] = row
                queries.add(query)
                ids[split].add(qid)
        if canonical_rows is None:
            canonical_rows, original_ids = by_id, ids
        else:
            if ids != original_ids:
                raise ValueError("Original S/M split memberships must match by question_id")
            for qid, row in by_id.items():
                if any(row[key] != canonical_rows[qid][key] for key in ("question", "question_type")):
                    raise ValueError(f"S/M question/type mismatch: {qid}")
    added = validation_rows - len(original_ids["validation"])
    if not 0 <= added < len(original_ids["train"]):
        raise ValueError("Validation may expand from training while retaining a nonempty training set")
    candidates = [canonical_rows[qid] for qid in sorted(original_ids["train"])]
    moved, quotas = _stratified_sample(candidates, added, random.Random(seed))
    assignments = {qid: split for split in SPLITS for qid in original_ids[split]}
    assignments.update({qid: "validation" for qid in moved})
    target = {"train": len(original_ids["train"]) - added, "validation": validation_rows,
              "test": len(original_ids["test"])}
    if Counter(assignments.values()) != Counter(target):
        raise AssertionError("Invalid fixed-test assignment counts")
    allocation = {"counts_before": {s: len(original_ids[s]) for s in SPLITS}, "counts_after": target,
                  "moved_from_train_to_validation": added,
                  "added_validation_question_ids": sorted(moved),
                  "added_validation_type_quotas": quotas}
    return assignments, allocation


def _indices(scalars, assignments):
    return {
        "train": [("train", i) for i, row in enumerate(scalars["train"])
                  if assignments[row["question_id"]] == "train"],
        "validation": [("validation", i) for i in range(len(scalars["validation"]))] +
                      [("train", i) for i, row in enumerate(scalars["train"])
                       if assignments[row["question_id"]] == "validation"],
        "test": [("test", i) for i in range(len(scalars["test"]))],
    }


def _mapping_rows(source, indices, scalar, hashes, new_hashes):
    for split in SPLITS:
        for new_index, (old_split, old_index) in enumerate(indices[split]):
            row = scalar[old_split][old_index]
            yield {"source": source, "question_id": row["question_id"], "question_type": row["question_type"],
                   "old_split": old_split, "old_row_index": old_index,
                   "new_split": split, "new_row_index": new_index,
                   "old_sample_id": f"{source}/{old_split}/{old_index:06d}",
                   "new_sample_id": f"{source}/{split}/{new_index:06d}",
                   "old_file_sha256": hashes[old_split], "new_file_sha256": new_hashes[split],
                   "source_session_count": row.get("source_session_count"),
                   "old_bank_id": "bank_" + digest([source, old_split, hashes[old_split], old_index])[:24],
                   "new_bank_id": "bank_" + digest([source, split, new_hashes[split], new_index])[:24]}


def _verify_existing(output, plan, scalar, hashes, assignments):
    import pyarrow.parquet as pq
    manifest = output / "manifest.json"
    if not manifest.is_file():
        raise ValueError("Existing output is not a completed split snapshot; use a fresh output root")
    report = json.loads(manifest.read_text())
    for key in ("version", "seed", "validation_rows", "parent_manifest", "parent_manifest_sha256",
                "output_root", "allocation", "policy"):
        if report.get(key) != plan[key]:
            raise ValueError(f"Existing split snapshot differs in {key}; refusing to overwrite")
    if report.get("status") != "complete":
        raise ValueError("Existing split snapshot is incomplete")
    sources = {item["source"]: item for item in report["sources"]}
    if set(sources) != set(SOURCES):
        raise ValueError("Existing split snapshot has different sources")
    mappings = []
    for source in SOURCES:
        if sources[source]["counts"] != plan["allocation"]["counts_after"]:
            raise ValueError("Existing split snapshot has different counts")
        output_hashes = {}
        for split in SPLITS:
            original = sources[source]["input_files"][split]
            published = sources[source]["output_files"][split]
            path = output / source / f"{split}.parquet"
            if original["sha256"] != hashes[source][split]:
                raise ValueError("Existing split input provenance differs")
            if (not path.is_file() or file_digest(path) != published["sha256"]
                    or pq.read_metadata(path).num_rows != plan["allocation"]["counts_after"][split]
                    or published["rows"] != plan["allocation"]["counts_after"][split]):
                raise ValueError(f"Existing split file changed or is missing: {path}")
            output_hashes[split] = published["sha256"]
        if output_hashes["test"] != hashes[source]["test"]:
            raise ValueError("Existing test file is not an exact original copy")
        mappings.extend(_mapping_rows(source, _indices(scalar[source], assignments),
                                      scalar[source], hashes[source], output_hashes))
    mapping_path = output / "row_mapping.jsonl"
    if (not mapping_path.is_file() or file_digest(mapping_path) != report["row_mapping"]["sha256"]
            or len(mappings) != report["row_mapping"]["rows"]
            or [json.loads(line) for line in mapping_path.read_text().splitlines() if line.strip()] != mappings):
        raise ValueError("Existing row mapping differs from its source rows")
    return report


def resplit_fixed_test(input_root=DEFAULT_INPUT, output_root=DEFAULT_OUTPUT, *, validation_rows=35,
                       seed=42, dry_run=False):
    root, output = Path(input_root).resolve(), Path(output_root).resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Use an output root separate from the original dataset")
    parent_path = root / "manifest.json"
    parent = json.loads(parent_path.read_text())
    source_entries = {item["source"]: item for item in parent.get("sources", [])}
    scalar, files, hashes = {}, {}, {}
    for source in SOURCES:
        scalar[source], files[source], hashes[source] = {}, {}, {}
        for split in SPLITS:
            path = root / source / f"{split}.parquet"
            sha = file_digest(path)
            expected = source_entries.get(source, {}).get("output_files", {}).get(split, {}).get("sha256")
            if expected is not None and sha != expected:
                raise ValueError(f"Source differs from its manifest: {path}")
            files[source][split], hashes[source][split] = path, sha
            scalar[source][split] = _read_scalars(path)
    assignments, allocation = plan_fixed_assignments(scalar, validation_rows=validation_rows, seed=seed)
    report = {"status": "planned" if dry_run else "complete", "version": VERSION, "seed": seed,
              "validation_rows": validation_rows, "parent_manifest": str(parent_path),
              "parent_manifest_sha256": file_digest(parent_path), "output_root": str(output),
              "allocation": allocation, "policy": {
                  "selection_pool": "original train only", "preserve_original_test_bytes_and_order": True,
                  "preserve_existing_validation_prefix": True, "preserve_remaining_train_order": True,
                  "selection_unit": "shared question_id with normalized-query duplicate validation",
                  "sampling": "question_type largest-remainder quotas; sorted IDs; random.Random(seed)",
                  "cross_variant_assignment": "same question_id always has same split in S/M",
                  "output_order": "original train filtered; original validation then held-out train rows; original test",
                  "directory_status_used_for_selection": False, "preserve_original_columns_and_values": True},
              "sources": []}
    if output.exists():
        return _verify_existing(output, report, scalar, hashes, assignments)
    if dry_run:
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".longmemeval-fixed-test-", dir=output.parent) as temporary:
        stage = Path(temporary)
        mappings = []
        for source in SOURCES:
            destination = stage / source
            destination.mkdir()
            indices = _indices(scalar[source], assignments)
            outputs, type_counts = {}, {}
            for split in SPLITS:
                origins = ["validation", "train"] if split == "validation" else [split]
                inputs = [(files[source][old], {index for origin, index in indices[split] if origin == old})
                          for old in origins if any(origin == old for origin, _ in indices[split])]
                path = destination / f"{split}.parquet"
                if split == "test":
                    shutil.copy2(files[source][split], path)
                else:
                    _write_parquet(path, inputs)
                count = _verify_parquet(path, inputs)
                if count != allocation["counts_after"][split]:
                    raise AssertionError(f"Incorrect output count for {source}/{split}")
                sha = file_digest(path)
                if split == "test" and sha != hashes[source][split]:
                    raise AssertionError("Test parquet copy changed bytes")
                outputs[split] = {"path": str(output / source / path.name), "sha256": sha, "rows": count}
                type_counts[split] = dict(sorted(Counter(scalar[source][old][index]["question_type"]
                                                       for old, index in indices[split]).items()))
            mappings.extend(_mapping_rows(source, indices, scalar[source], hashes[source],
                                          {split: item["sha256"] for split, item in outputs.items()}))
            report["sources"].append({"source": source, "counts": dict(allocation["counts_after"]),
                                      "question_type_counts": type_counts,
                                      "input_files": {split: {"path": str(files[source][split]),
                                                              "sha256": hashes[source][split],
                                                              "rows": len(scalar[source][split])}
                                                      for split in SPLITS}, "output_files": outputs})
        for source in SOURCES:
            for split in SPLITS:
                if file_digest(files[source][split]) != hashes[source][split]:
                    raise ValueError(f"Source changed during resplit: {source}/{split}")
        if file_digest(parent_path) != report["parent_manifest_sha256"]:
            raise ValueError("Parent manifest changed during resplit")
        with (stage / "row_mapping.jsonl").open("w", encoding="utf-8") as stream:
            for row in mappings:
                stream.write(canonical(row) + "\n")
        report["row_mapping"] = {"file": "row_mapping.jsonl", "rows": len(mappings),
                                 "sha256": file_digest(stage / "row_mapping.jsonl")}
        report["verification"] = {"all_output_fields_equal_original_rows": True,
                                  "original_files_unchanged": True, "original_test_bytes_preserved": True,
                                  "old_validation_prefix_preserved": True, "every_original_row_used_once": True,
                                  "cross_variant_assignment_aligned": True, "train_validation_query_overlap": 0,
                                  "train_test_query_overlap": 0, "validation_test_query_overlap": 0}
        atomic_json(stage / "manifest.json", report)
        stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=DEFAULT_INPUT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-rows", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = resplit_fixed_test(args.input_root, args.output_root, validation_rows=args.validation_rows,
                                seed=args.seed, dry_run=args.dry_run)
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
