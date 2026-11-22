"""Compare the current three splits with QA arrays read from official LoCoMo JSON."""
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "data/processed/query_dedup_val10_seed42_20261018"


def main():
    source = json.loads((HERE / "official_locomo_qa.json").read_text())
    official = {
        f"question_{conversation['sample_id']}_{index}": qa
        for conversation in source["data"]
        for index, qa in enumerate(conversation["qa"])
    }
    local = {}
    split_counts = {}
    for split in ("train", "validation", "test"):
        rows = pq.read_table(
            DATA / "locomo" / f"{split}.parquet",
            columns=["question_id", "question", "answer", "answer_session_ids"],
        ).to_pylist()
        split_counts[split] = len(rows)
        for row in rows:
            assert row["question_id"] not in local
            local[row["question_id"]] = dict(row, split=split)

    removed = {
        row["question_id"]: row
        for line in (DATA / "removed_train_manifest.jsonl").read_text().splitlines()
        if (row := json.loads(line))["source"] == "locomo"
    }
    query_mismatches, answer_mismatches, evidence_mismatches = [], [], []
    malformed_official_evidence = []
    compound_official_evidence = []
    for qid, row in local.items():
        if qid not in official:
            continue
        qa = official[qid]
        if row["question"] != qa["question"]:
            query_mismatches.append(qid)
        # Local parquet stores numeric official answers as strings.
        if row["answer"] != str(qa.get("answer", qa.get("adversarial_answer"))):
            answer_mismatches.append(qid)
        conv = qid.split("_")[1]
        sessions = set()
        for evidence in qa["evidence"]:
            matches = list(re.finditer(r"\bD(\d+):\d+\b", evidence))
            if matches:
                sessions.update(f"{conv}_session_{match.group(1)}" for match in matches)
                if len(matches) > 1:
                    compound_official_evidence.append({"question_id": qid, "evidence": evidence})
            else:
                malformed_official_evidence.append({"question_id": qid, "evidence": evidence})
        if set(row["answer_session_ids"]) != sessions:
            evidence_mismatches.append({"question_id": qid, "official": sorted(sessions), "local": sorted(row["answer_session_ids"])})

    missing = {qid: qa for qid, qa in official.items() if qid not in local}
    missing_non_adversarial = []
    for qid, qa in missing.items():
        if qa["category"] == 5:
            continue
        matching_test_ids = [
            test_id for test_id, row in local.items()
            if row["split"] == "test" and row["question"] == qa["question"]
        ]
        missing_non_adversarial.append(dict(
            qa, question_id=qid,
            documented_local_removal=removed.get(qid),
            same_query_in_current_test=matching_test_ids,
        ))

    report = {
        "official_source": source["source_url"],
        "retrieved_date": source["retrieved_date"],
        "scope": "All official QA entries by conversation ID and original QA-array index; full history text not compared",
        "comparison_rules": "Exact query and answer strings, except str() for official numeric answers; evidence session sets derived from official dialogue evidence IDs",
        "official_rows": len(official),
        "official_category_counts": dict(Counter(q["category"] for q in official.values())),
        "local_split_counts": split_counts,
        "local_rows": len(local),
        "local_category_counts": dict(Counter(official[qid]["category"] for qid in local if qid in official)),
        "extra_local_ids": sorted(set(local) - set(official)),
        "query_mismatches": query_mismatches,
        "answer_mismatches": answer_mismatches,
        "evidence_session_mismatches": evidence_mismatches,
        "malformed_official_evidence_not_mappable_to_session": malformed_official_evidence,
        "compound_official_evidence_strings": compound_official_evidence,
        "missing_category_counts": dict(Counter(qa["category"] for qa in missing.values())),
        "missing_non_adversarial": missing_non_adversarial,
        "missing_adversarial_ids": [qid for qid, qa in missing.items() if qa["category"] == 5],
    }
    (HERE / "locomo_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"missing_adversarial_ids", "missing_non_adversarial", "evidence_session_mismatches"}}, ensure_ascii=False, indent=2))
    print("evidence_session_mismatches:", len(evidence_mismatches))
    print("missing_non_adversarial:", [x["question_id"] for x in missing_non_adversarial])


if __name__ == "__main__":
    main()
