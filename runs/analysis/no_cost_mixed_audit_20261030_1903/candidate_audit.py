"""Snapshot audit of no-cost/mixed-Judge candidates; no model calls."""
import json
from collections import Counter
from pathlib import Path


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SNAPSHOT = json.loads((OUT / "snapshot.json").read_text())
RUN = ROOT / SNAPSHOT["run"]
OLD = ROOT / "runs/locomo_longmemeval/experiments/locomo_s_dapo_mixed_v2_micro2_parallel4_reader_ids_v2_format_count_max1/train"


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_candidates(root, metrics):
    rows = []
    for metric in metrics:
        index = metric["candidate_id"]
        doc = json.loads((root / "candidate_rollouts" / f"candidate-{index:08d}.json").read_text())
        assert len(doc["rollouts"]) == 8
        assert doc["sampling"]["sample_id"] == metric["sample_id"]
        for row in doc["rollouts"]:
            steps = row["reader_steps"]
            ids = [key for step in steps for key in step.get("new_ids", [])]
            calls = sum(bool(s.get("read_call_consumed")) for s in steps)
            fresh = sum(bool(s.get("new_ids")) for s in steps)
            assert len(ids) == len(set(ids)) == fresh  # max1 protocol
            rows.append({
                "candidate": index, "sample_id": row["sample_id"], "rollout": row["rollout_index"],
                "policy": metric["behavior_policy_version"], "selected": metric["selected"],
                "judge": row["judge_score"], "reward": row["reward"],
                "common_no_cost_reward": row["reward"] + row["read_cost"],
                "read_cost": row["read_cost"], "calls": calls, "fresh_calls": fresh,
                "first_stop": bool(steps[0].get("action") and steps[0]["action"]["stop"]),
                "repeated": any(s.get("already_read_ids") for s in steps),
                "format_count": row["format_violation_count"], "format_failure": row["format_failure"],
                "ids": ids, "answer": row["task_agent_output"],
                "actions": [s.get("action") for s in steps],
            })
    return rows


def summary(rows, metrics):
    n = len(rows)
    return {
        "groups": len(metrics), "rollouts": n,
        "group_counts": {key: sum(m[key] for m in metrics) for key in
                         ("accepted", "selected", "judge_all_correct", "judge_all_wrong", "judge_mixed",
                          "zero_reward_variance", "cost_format_only_signal", "no_unmasked_signal")},
        "correct": sum(r["judge"] for r in rows),
        "judge_accuracy": sum(r["judge"] for r in rows) / n if n else None,
        "reward": sum(r["reward"] for r in rows) / n if n else None,
        "common_no_cost_reward": sum(r["common_no_cost_reward"] for r in rows) / n if n else None,
        "mean_calls": sum(r["calls"] for r in rows) / n if n else None,
        "call_histogram": dict(sorted(Counter(r["calls"] for r in rows).items())),
        "fresh_call_histogram": dict(sorted(Counter(r["fresh_calls"] for r in rows).items())),
        "first_stop": sum(r["first_stop"] for r in rows),
        "empty_evidence": sum(r["fresh_calls"] == 0 for r in rows),
        "multi_new_evidence": sum(r["fresh_calls"] >= 2 for r in rows),
        "correct_multi_new_evidence": sum(r["fresh_calls"] >= 2 and r["judge"] == 1 for r in rows),
        "repeated_rollouts": sum(r["repeated"] for r in rows),
        "format_failed_rollouts": sum(r["format_failure"] for r in rows),
        "format_violations": sum(r["format_count"] for r in rows),
        "outcome_fresh_histograms": {
            str(j): dict(sorted(Counter(r["fresh_calls"] for r in rows if r["judge"] == j).items())) for j in (0, 1)
        },
    }


def main():
    metrics = jsonl(OUT / "candidate_metrics.jsonl")
    old_metrics = jsonl(OLD / "candidate_metrics.jsonl")[:len(metrics)]
    assert [m["sample_id"] for m in old_metrics] == [m["sample_id"] for m in metrics]
    rows, old_rows = read_candidates(RUN, metrics), read_candidates(OLD, old_metrics)
    assert all(r["read_cost"] == 0 for r in rows)
    result = {"captured_at": SNAPSHOT["captured_at"], "source_run": str(RUN),
              "definition": "Counts from completed candidate files in frozen metrics snapshot; later batch is partial. Fresh calls require new IDs, not just a consumed read attempt.",
              "policies": {}, "matched_comparisons": {}}
    for policy in sorted({m["behavior_policy_version"] for m in metrics}):
        subset = [m for m in metrics if m["behavior_policy_version"] == policy]
        new = [r for r in rows if r["policy"] == policy]
        old = [r for r in old_rows if r["policy"] == policy]
        old_subset = [m for m in old_metrics if m["behavior_policy_version"] == policy]
        result["policies"][str(policy)] = summary(new, subset)
        assert [(r["candidate"], r["rollout"]) for r in new] == [(r["candidate"], r["rollout"]) for r in old]
        result["matched_comparisons"][str(policy)] = {
            "old": summary(old, old_subset),
            "same_actions": sum(a["actions"] == b["actions"] for a, b in zip(new, old)),
            "same_evidence_ids": sum(a["ids"] == b["ids"] for a, b in zip(new, old)),
            "same_judge": sum(a["judge"] == b["judge"] for a, b in zip(new, old)),
            "same_answer": sum(a["answer"] == b["answer"] for a, b in zip(new, old)),
        }
    (OUT / "candidate_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (OUT / "candidate_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
