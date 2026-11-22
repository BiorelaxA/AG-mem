"""Expand LongMemEval train/validation from test while retaining existing rows.

S and M share one question assignment. This operation writes only a fresh dataset
snapshot and a provenance mapping; existing directories and model runs are untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile

from .core import atomic_json, canonical, digest, file_digest
from .validation_split import _verify_parquet, _write_parquet


VERSION = "longmemeval_preserve_train_validation_811_v1"
SOURCES = ("split_longmemeval_s", "split_longmemeval_m")
SPLITS = ("train", "validation", "test")
DEFAULT_INPUT = "data/processed/query_dedup_val10_seed42_20261018"
DEFAULT_OUTPUT = "data/processed/longmemeval_811_seed42_20261111"


def _stratified_sample(rows, count, rng):
    """Largest-remainder type quotas, then seeded sampling of sorted question IDs."""
    if not 0 <= count <= len(rows):
        raise ValueError("Stratified sample count is outside the candidate pool")
    if not rows:
        return set(), {}
    strata = defaultdict(list)
    for row in rows:
        strata[row["question_type"]].append(row["question_id"])
    quotas = {kind: len(ids) * count // len(rows) for kind, ids in strata.items()}
    order = sorted(strata, key=lambda k: (-(len(strata[k]) * count % len(rows)), k))
    for kind in order[:count - sum(quotas.values())]:
        quotas[kind] += 1
    selected = set()
    for kind in sorted(strata):
        selected.update(rng.sample(sorted(strata[kind]), quotas[kind]))
    return selected, dict(sorted(quotas.items()))


def plan_assignments(rows_by_source, *, seed=42):
    """Preserve every old train/validation question and align S/M by question ID."""
    if set(rows_by_source) != set(SOURCES):
        raise ValueError("Provide both LongMemEval-S and LongMemEval-M")
    canonical_rows, old_sets = None, None
    for source in SOURCES:
        rows = rows_by_source[source]
        if set(rows) != set(SPLITS):
            raise ValueError(f"Missing train/validation/test split for {source}")
        by_id, by_query, split_ids = {}, {}, {}
        for split in SPLITS:
            split_ids[split] = set()
            for row in rows[split]:
                for field in ("question_id", "question", "question_type"):
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        raise ValueError(f"Missing or invalid {field} in {source}/{split}")
                qid, query = row["question_id"], row["question"]
                if qid in by_id or query in by_query:
                    raise ValueError(f"Duplicate question ID or exact query in {source}; cannot split safely")
                by_id[qid], by_query[query] = row, qid
                split_ids[split].add(qid)
        if canonical_rows is None:
            canonical_rows, old_sets = by_id, split_ids
        else:
            if split_ids != old_sets:
                raise ValueError("Existing S/M question assignments differ; preserving both requires an explicit policy")
            for qid, row in by_id.items():
                if any(row[k] != canonical_rows[qid][k] for k in ("question", "question_type")):
                    raise ValueError(f"Question ID {qid} has inconsistent question/type across S/M")
    total = len(canonical_rows)
    if not total or total % 10:
        raise ValueError("Exact 8:1:1 requires a positive number of unique questions divisible by ten")
    target = {"train": total * 8 // 10, "validation": total // 10, "test": total // 10}
    additions = {split: target[split] - len(old_sets[split]) for split in ("train", "validation")}
    if any(n < 0 for n in additions.values()):
        raise ValueError("The requested ratio would move an existing train/validation question")
    if sum(additions.values()) + target["test"] != len(old_sets["test"]):
        raise ValueError("Old test pool cannot satisfy the requested split sizes")
    rng = random.Random(seed)
    test_rows = [canonical_rows[qid] for qid in sorted(old_sets["test"])]
    retained_test, test_quotas = _stratified_sample(test_rows, target["test"], rng)
    remaining = [row for row in test_rows if row["question_id"] not in retained_test]
    added_validation, val_quotas = _stratified_sample(remaining, additions["validation"], rng)
    added_train = old_sets["test"] - retained_test - added_validation
    assignments = {qid: split for split in ("train", "validation") for qid in old_sets[split]}
    assignments.update({qid: "test" for qid in retained_test})
    assignments.update({qid: "validation" for qid in added_validation})
    assignments.update({qid: "train" for qid in added_train})
    if Counter(assignments.values()) != Counter(target):
        raise AssertionError("Invalid target assignment counts")
    report = {"counts_before": {s: len(old_sets[s]) for s in SPLITS}, "counts_after": target,
              "moved_from_test": additions, "retained_test_type_quotas": test_quotas,
              "added_validation_type_quotas": val_quotas,
              "added_train_type_counts": dict(sorted(Counter(canonical_rows[q]["question_type"]
                                                              for q in added_train).items()))}
    return assignments, report


def _read_scalars(path):
    import pyarrow.parquet as pq
    columns = ["question_id", "question", "question_type"]
    if "haystack_session_ids" in pq.ParquetFile(path).schema_arrow.names:
        columns.append("haystack_session_ids")
    rows = pq.read_table(path, columns=columns).to_pylist()
    for row in rows:
        ids = row.pop("haystack_session_ids", None)
        if isinstance(ids, str):
            ids = json.loads(ids)
        if ids is not None:
            if not isinstance(ids, list):
                raise ValueError("haystack_session_ids must be an array")
            row["source_session_count"] = len(ids)
    return rows


def resplit_longmemeval(input_root=DEFAULT_INPUT, output_root=DEFAULT_OUTPUT, *, seed=42, dry_run=False):
    root, output = Path(input_root).resolve(), Path(output_root).resolve()
    if output.exists() or root == output or root in output.parents or output in root.parents:
        raise ValueError("Use a fresh output root separate from the original dataset")
    manifest_path = root / "manifest.json"
    parent = json.loads(manifest_path.read_text())
    source_entries = {entry["source"]: entry for entry in parent.get("sources", [])}
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
    assignments, allocation = plan_assignments(scalar, seed=seed)
    report = {"status": "planned" if dry_run else "complete", "version": VERSION, "seed": seed,
              "parent_manifest": str(manifest_path), "parent_manifest_sha256": file_digest(manifest_path),
              "output_root": str(output), "allocation": allocation,
              "policy": {"ratio": {"train": 8, "validation": 1, "test": 1},
                         "preserve_existing_train_prefix": True, "preserve_existing_validation_prefix": True,
                         "selection_pool": "old test only", "selection_unit": "shared question_id with exact-query validation",
                         "sampling": "question_type largest-remainder quotas; sorted IDs; random.Random(seed)",
                         "selection_order": ["retain new test", "add validation", "remaining add train"],
                         "cross_variant_assignment": "same question_id always has same split in S/M",
                         "output_order": "old train/validation prefix, then additions in old-test row order",
                         "directory_status_used_for_selection": False,
                         "preserve_original_columns_and_values": True}, "sources": []}
    if dry_run:
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".longmemeval-811-", dir=output.parent) as temporary:
        stage = Path(temporary)
        mappings = []
        for source in SOURCES:
            destination = stage / source
            destination.mkdir()
            indices = {split: [] for split in SPLITS}
            for split in ("train", "validation"):
                indices[split].extend((split, i) for i in range(len(scalar[source][split])))
            for i, row in enumerate(scalar[source]["test"]):
                indices[assignments[row["question_id"]]].append(("test", i))
            outputs, type_counts = {}, {}
            for split in SPLITS:
                inputs = [(files[source][old], {i for origin, i in indices[split] if origin == old})
                          for old in SPLITS if any(origin == old for origin, _ in indices[split])]
                path = destination / f"{split}.parquet"
                _write_parquet(path, inputs)
                count = _verify_parquet(path, inputs)
                if count != allocation["counts_after"][split]:
                    raise AssertionError(f"Incorrect output row count for {source}/{split}")
                sha = file_digest(path)
                outputs[split] = {"path": str(output / source / path.name), "sha256": sha, "rows": count}
                type_counts[split] = dict(sorted(Counter(scalar[source][old][i]["question_type"]
                                                        for old, i in indices[split]).items()))
                for new_index, (old_split, old_index) in enumerate(indices[split]):
                    row = scalar[source][old_split][old_index]
                    mappings.append({"source": source, "question_id": row["question_id"],
                                     "question_type": row["question_type"],
                                     "old_split": old_split, "old_row_index": old_index,
                                     "new_split": split, "new_row_index": new_index,
                                     "old_file_sha256": hashes[source][old_split], "new_file_sha256": sha,
                                     "source_session_count": row.get("source_session_count"),
                                     "old_bank_id": "bank_" + digest([source, old_split, hashes[source][old_split], old_index])[:24],
                                     "new_bank_id": "bank_" + digest([source, split, sha, new_index])[:24]})
            report["sources"].append({"source": source, "counts": deepcopy(allocation["counts_after"]),
                                      "question_type_counts": type_counts,
                                      "input_files": {s: {"path": str(files[source][s]), "sha256": hashes[source][s],
                                                           "rows": len(scalar[source][s])} for s in SPLITS},
                                      "output_files": outputs})
        # A second hash check rejects a concurrently edited source snapshot.
        for source in SOURCES:
            for split in SPLITS:
                if file_digest(files[source][split]) != hashes[source][split]:
                    raise ValueError(f"Source changed during resplit: {source}/{split}")
        with (stage / "row_mapping.jsonl").open("w", encoding="utf-8") as stream:
            for row in mappings:
                stream.write(canonical(row) + "\n")
        report["row_mapping"] = {"file": "row_mapping.jsonl", "rows": len(mappings),
                                 "sha256": file_digest(stage / "row_mapping.jsonl")}
        report["verification"] = {"all_output_fields_equal_original_rows": True,
                                  "original_files_unchanged": True, "old_train_validation_prefix_preserved": True,
                                  "every_original_row_used_once": True, "cross_variant_assignment_aligned": True,
                                  "train_validation_query_overlap": 0, "train_test_query_overlap": 0,
                                  "validation_test_query_overlap": 0}
        atomic_json(stage / "manifest.json", report)
        stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=DEFAULT_INPUT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = resplit_longmemeval(args.input_root, args.output_root, seed=args.seed, dry_run=args.dry_run)
    print(canonical(report), flush=True)


if __name__ == "__main__":
    main()
