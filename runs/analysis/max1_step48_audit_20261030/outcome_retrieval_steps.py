"""Read-only aggregation of max1 candidate outcomes and retrieval depth."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/locomo_longmemeval/experiments/locomo_s_dapo_mixed_v2_micro2_parallel4_reader_ids_v2_format_count_max1/train"
OUT = Path(__file__).resolve().parent


def summarize(rows):
    result = {"rollouts": len(rows)}
    for judge in (1, 0):
        subset = [r for r in rows if r["judge"] == judge]
        result["correct" if judge else "incorrect"] = {
            "rollouts": len(subset),
            "read_call_histogram": dict(sorted(Counter(r["calls"] for r in subset).items())),
            "new_evidence_call_histogram": dict(sorted(Counter(r["fresh_calls"] for r in subset).items())),
            "mean_read_calls": sum(r["calls"] for r in subset) / len(subset) if subset else None,
            "mean_new_evidence_calls": sum(r["fresh_calls"] for r in subset) / len(subset) if subset else None,
            "multi_calls": sum(r["calls"] >= 2 for r in subset),
            "multi_new_evidence_calls": sum(r["fresh_calls"] >= 2 for r in subset),
        }
    return result


def main():
    rows = []
    metrics = {d["candidate_id"]: d for d in map(json.loads, (RUN / "candidate_metrics.jsonl").read_text().splitlines())}
    for path in sorted((RUN / "candidate_rollouts").glob("candidate-*.json")):
        document = json.loads(path.read_text())
        sampling = document["sampling"]
        candidate = sampling["candidate_id"]
        assert sampling["behavior_policy_version"] == metrics[candidate]["behavior_policy_version"]
        assert len(document["rollouts"]) == 8
        for rollout in document["rollouts"]:
            steps = rollout["reader_steps"]
            calls = sum(bool(s.get("read_call_consumed")) for s in steps)
            fresh_calls = sum(bool(s.get("new_ids")) for s in steps)
            new_ids = [i for s in steps for i in s.get("new_ids", [])]
            assert len(new_ids) == len(set(new_ids)) == fresh_calls
            assert abs(rollout["raw_read_cost"] - .01 * calls) < 1e-8
            assert rollout["judge_score"] in (0, 1)
            rows.append({
                "candidate": candidate, "rollout": rollout["rollout_index"],
                "policy": sampling["behavior_policy_version"], "sample": rollout["sample_id"],
                "selected": sampling["selected"], "judge": int(rollout["judge_score"]),
                "calls": calls, "fresh_calls": fresh_calls,
            })
    previous = {(r["candidate"], r["rollout"]): r for r in json.loads((OUT / "candidate_rows.json").read_text())}
    assert len(previous) == len(rows)
    for r in rows:
        p = previous[(r["candidate"], r["rollout"])]
        assert (r["calls"], r["fresh_calls"], r["judge"], r["policy"]) == (p["calls"], p["fresh"], p["J"], p["policy"])
    report = {
        "source": str(RUN),
        "definitions": {
            "population": "All recorded candidate rollouts, including filtered groups.",
            "correctness": "Recorded binary judge_score, not answer F1.",
            "read_calls": "Calls consuming retrieval budget; includes invalid IDs/repeated reads; excludes stop decisions.",
            "new_evidence_calls": "Calls with nonempty new_ids; max1 means one distinct successful ID per such call.",
            "policy": "Number of optimizer steps completed before this candidate batch was sampled.",
            "selected": "Buffered for update; policy48 selected groups were not trained because sampling budget exhausted.",
        },
        "groups": len(metrics), "rollouts": len(rows),
        "policies": {},
        "post_first_16_updates": summarize([r for r in rows if r["policy"] >= 16]),
    }
    for policy in sorted({r["policy"] for r in rows}):
        subset = [r for r in rows if r["policy"] == policy]
        report["policies"][policy] = {
            "groups": len({r["candidate"] for r in subset}),
            "all_candidates": summarize(subset),
            "selected_candidates": summarize([r for r in subset if r["selected"]]),
        }
    (OUT / "outcome_retrieval_steps.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
