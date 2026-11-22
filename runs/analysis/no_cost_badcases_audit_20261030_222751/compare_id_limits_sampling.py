"""Read-only audit: compare query-matched candidates at identical policy versions."""
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BASE = ROOT / "runs/locomo_longmemeval/experiments"
PREFIX = "locomo_s_dapo_mixed_v2_micro2_parallel4_reader_ids_v2_format_count"
PATHS = {
    "max4_cost": BASE / PREFIX / "train/candidate_metrics.jsonl",
    "max2_cost": BASE / (PREFIX + "_max2") / "train/candidate_metrics.jsonl",
    "max1_cost": BASE / (PREFIX + "_max1") / "train/candidate_metrics.jsonl",
    "max1_no_cost": OUT / "candidate_metrics.jsonl",
}


def aggregate(rows, include_sources=True):
    if not rows:
        return {"groups": 0}
    n = len(rows)
    acc = sum(r["judge_accuracy"] for r in rows) / n
    result = {
        "groups": n,
        "rollouts": n * 8,
        "correct_rollouts": round(acc * n * 8),
        "judge_accuracy": acc,
        "outcome_reward": 2 * acc - 1,
        "all_wrong": sum(r["judge_all_wrong"] for r in rows),
        "all_correct": sum(r["judge_all_correct"] for r in rows),
        "mixed": sum(r["judge_mixed"] for r in rows),
        "accepted": sum(r["accepted"] for r in rows),
        "selected": sum(r["selected"] for r in rows),
        "read_calls": sum(r["read_calls"] for r in rows) / n,
        "repeated_read_rate": sum(r["repeated_read_rate"] for r in rows) / n,
        "format_failure_rate": sum(r["format_failure_rate"] for r in rows) / n,
        "reward": sum(r["reward"] for r in rows) / n,
    }
    result["all_wrong_rate"] = result["all_wrong"] / n
    result["accepted_rate"] = result["accepted"] / n
    if include_sources:
        by_source = defaultdict(list)
        for row in rows:
            by_source[row["sample_id"].split("/")[0]].append(row)
        result["by_source"] = {s: aggregate(rs, False) for s, rs in by_source.items()}
    return result


runs = {name: [json.loads(line) for line in p.read_text().splitlines()] for name, p in PATHS.items()}
report = {
    "notes": [
        "Current max1_no_cost is the frozen 316-candidate snapshot, not the live file.",
        "Matched groups join identical sample_id and behavior_policy_version; candidate indices alone are not comparable after different refill counts.",
        "All groups have eight rollouts. Candidate means equally weight each query group.",
        "Policy version 0 has common base weights; later policies have independent training histories, so matching controls questions but not learned weights.",
        "Historical max4 has no behavior-policy-32 candidates and no validation32.",
        "All-wrong means eight Judge-zero trajectories; accepted can differ from mixed due to overlong masking.",
    ],
    "paths": {k: str(v) for k, v in PATHS.items()},
    "all_candidates_by_policy": {},
    "matched_candidates": {},
}
indexes = {}
for name, rows in runs.items():
    by_policy = defaultdict(list)
    for row in rows:
        by_policy[row["behavior_policy_version"]].append(row)
    report["all_candidates_by_policy"][name] = {str(p): aggregate(rs) for p, rs in by_policy.items()}
    indexes[name] = {}
    for policy, rs in by_policy.items():
        by_sample = {r["sample_id"]: r for r in rs}
        assert len(by_sample) == len(rs), (name, policy, "duplicate query IDs")
        indexes[name][policy] = by_sample

cohorts = {
    "all_four": list(runs),
    "max2_vs_current_max1": ["max2_cost", "max1_no_cost"],
    "max1_cost_vs_no_cost": ["max1_cost", "max1_no_cost"],
}
for cohort, names in cohorts.items():
    report["matched_candidates"][cohort] = {}
    for policy in (0, 16, 32, 48):
        joined = set.intersection(*(set(indexes[n].get(policy, {})) for n in names))
        if not joined:
            continue
        ids = sorted(joined)
        summaries = {n: aggregate([indexes[n][policy][s] for s in ids]) for n in names}
        per_query = [{"sample_id": s, **{n: indexes[n][policy][s]["judge_accuracy"] for n in names}} for s in ids]
        report["matched_candidates"][cohort][str(policy)] = {
            "matched_groups": len(ids),
            "sample_ids": ids,
            "summaries": summaries,
            "per_query_judge_accuracy": per_query,
        }

target = OUT / "id_limit_sampling_comparison.json"
target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
for cohort, policies in report["matched_candidates"].items():
    print(cohort)
    for policy, item in policies.items():
        print("policy", policy, "matched", item["matched_groups"])
        for name, s in item["summaries"].items():
            print(name, "accuracy", round(s["judge_accuracy"], 6), "outcome", round(s["outcome_reward"], 6), "all_wrong", s["all_wrong"], "mixed", s["mixed"])
print(target)
