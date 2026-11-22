"""Audit actual optimizer groups, support-session labels and cost counterfactuals.

Only reads the old max1 experiment and data. Does not run models or rescore
answers. Outputs a separate analysis JSON next to this script.
"""
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/locomo_longmemeval/experiments/locomo_s_dapo_mixed_v2_micro2_parallel4_reader_ids_v2_format_count_max1/train"


def normalized(rewards):
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / (std + 1e-6) for r in rewards]


def summarize(groups):
    rows = [r for g in groups for r in g["rows"] if not r["masked"]]
    positives = [r for r in rows if r["advantage"] > 0]
    return {
        "groups": len(groups),
        "sources": dict(Counter(g["source"] for g in groups)),
        "support_session_count_histogram": dict(sorted(Counter(g["support_count"] for g in groups).items())),
        "groups_with_single_session_judge_success": sum(g["single_session_success"] for g in groups),
        "groups_with_single_read_judge_success": sum(g["single_read_success"] for g in groups),
        "groups_with_multi_session_judge_success": sum(g["multi_session_success"] for g in groups),
        "groups_with_success_only_after_multiple_sessions": sum(g["multi_session_success"] and not g["single_session_success"] for g in groups),
        "unanswerable_id_groups": sum(g["unanswerable_id"] for g in groups),
        "trainable_rollouts": len(rows), "positive_rollouts": len(positives),
        "positive_unique_session_histogram": dict(sorted(Counter(r["unique_sessions"] for r in positives).items())),
        "positive_call_histogram": dict(sorted(Counter(r["calls"] for r in positives).items())),
        "correct_rollouts": sum(r["judge"] == 1 for r in rows),
        "groups_with_equal_call_count": sum(len({r["calls"] for r in g["rows"]}) == 1 for g in groups),
        "cost_counterfactual": {
            "advantage_sign_changes": sum((r["advantage"] > 0) != (r["no_cost_advantage"] > 0) for r in rows),
            "mean_absolute_advantage_change": sum(abs(r["advantage"] - r["no_cost_advantage"]) for r in rows) / len(rows) if rows else None,
            "max_absolute_advantage_change": max((abs(r["advantage"] - r["no_cost_advantage"]) for r in rows), default=None),
        },
    }


def main():
    run_config = json.loads((RUN / "run.json").read_text())["config"]
    raw_root, prepared_root = (Path(run_config["data"][key]) for key in ("input_root", "prepared_root"))
    metadata = {}
    for source in run_config["data"]["sources"]:
        file = pq.ParquetFile(raw_root / source / "train.parquet")
        columns = ["question_id", "question", "answer", "answer_session_ids"]
        if "question_type" in file.schema_arrow.names:
            columns.append("question_type")
        raw = file.read(columns=columns).to_pylist()
        prepared = {r["sample_id"]: r for r in map(json.loads, (prepared_root / source / "train_examples.jsonl").read_text().splitlines())}
        for i, row in enumerate(raw):
            sid = f"{source}/train/{i:06d}"
            assert prepared[sid]["query"] == row["question"]
            assert prepared[sid]["gold_answers"] == [row["answer"]]
            metadata[sid] = {**row, "source": source, "support_count": len(set(row["answer_session_ids"]))}

    train = [json.loads(line) for line in (RUN / "train_metrics.jsonl").read_text().splitlines()]
    groups = []
    for metric in train:
        step = metric["step"]
        group = json.load(gzip.open(RUN / "groups" / f"group-{step - 1:08d}.json.gz", "rt"))
        trajectories = group["rollouts"]
        sid = metric["sample_id"]
        assert len(trajectories) == 8 and all(r["sample_id"] == sid for r in trajectories)
        assert group["group_id"].endswith(f"/update-{step}")
        record = {"step": step, "sample_id": sid, **metadata[sid], "rows": []}
        old_adv = normalized([r["reward"] for r in trajectories])
        no_cost_adv = normalized([r["reward"] + r["read_cost"] for r in trajectories])
        for i, row in enumerate(trajectories):
            assert abs(row["advantage"] - old_adv[i]) < 1e-10
            assert abs(row["reward"] - (2 * row["judge_score"] - 1 - .01 * row["n_read_calls"] + row["format_reward"])) < 1e-8
            masked = any(d["completion"].get("truncated", False) for d in row["decisions"])
            record["rows"].append({
                "rollout": i, "judge": row["judge_score"], "advantage": row["advantage"],
                "no_cost_advantage": no_cost_adv[i], "calls": row["n_read_calls"],
                "unique_sessions": len({e["id"] for e in row["evidence"]}),
                "fresh_calls": sum(bool((d.get("tool_result") or {}).get("new_ids")) for d in row["decisions"]),
                "reward": row["reward"], "format_count": row["format_violation_count"],
                "masked": masked, "evidence_ids": [e["id"] for e in row["evidence"]],
                "answer": row["prediction"],
            })
        successes = [r for r in record["rows"] if r["judge"] == 1 and not r["masked"]]
        record.update(single_session_success=any(r["unique_sessions"] == 1 for r in successes),
                      single_read_success=any(r["calls"] == 1 and r["unique_sessions"] == 1 for r in successes),
                      multi_session_success=any(r["unique_sessions"] >= 2 for r in successes),
                      unanswerable_id=str(record["question_id"]).endswith("_abs"))
        groups.append(record)
    assert len(groups) == 48 and [g["step"] for g in groups] == list(range(1, 49))

    candidates = [json.loads(line) for line in (RUN / "candidate_metrics.jsonl").read_text().splitlines()]
    # Policy48 collected an incomplete batch; none of it was updated.
    actual_batch_candidates = [c for c in candidates if c["behavior_policy_version"] < 48]
    candidate_stats = {}
    for name, use in (("support_1", lambda n: n == 1), ("support_ge2", lambda n: n >= 2)):
        subset = [c for c in actual_batch_candidates if use(metadata[c["sample_id"]]["support_count"])]
        candidate_stats[name] = {"candidate_groups": len(subset),
                                 **{key: sum(c[key] for c in subset) for key in ("judge_all_correct", "judge_all_wrong", "judge_mixed", "accepted", "selected")}}
    result = {
        "source_run": str(RUN),
        "definitions": {
            "scope": "48 completed optimizer updates, complete8pergroup, exclude Reader-truncated trajectories from trainable success/advantage counts",
            "support_count": "Distinct annotated answer_session_ids; NOT proven minimum necessary evidence count",
            "single_session_success": "At least one unmasked rollout with exactly one unique evidence session was scored J=1 by the existing Judge; NOT independent correctness verification",
            "cost_counterfactual": "Only remove .01 read cost on fixed historical trajectories and recompute full-group advantage; NOT a retraining causal experiment",
            "candidate_comparison": "Only policies0/16/32 supplying completed optimizer batches; excludes policy48 incomplete batch",
        },
        "overall": summarize(groups),
        "by_batch": {f"steps_{start}_{start+15}": summarize([g for g in groups if start <= g["step"] < start + 16]) for start in (1, 17, 33)},
        "by_source": {s: summarize([g for g in groups if g["source"] == s]) for s in run_config["data"]["sources"]},
        "candidate_comparison": candidate_stats,
        "groups": groups,
    }
    output = Path(__file__).with_suffix(".json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "groups"}, ensure_ascii=False, indent=2))
    print("Multi-annotated updated questions:")
    for g in groups:
        if g["support_count"] >= 2:
            print(json.dumps({k: v for k, v in g.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
