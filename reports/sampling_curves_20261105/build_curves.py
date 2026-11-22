"""Export full-candidate sampling curves without touching the training run."""
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/locomo_longmemeval/experiments/locomo_dapo_reader_loop_v3_query9010_format05/train"
OUT = Path(__file__).resolve().parent


def read_jsonl(path):
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # An append-only live log may end with an incomplete line.
                if line != path.read_text().splitlines()[-1]:
                    raise
    return rows


candidates = read_jsonl(RUN / "candidate_metrics.jsonl")
updates = read_jsonl(RUN / "train_metrics.jsonl")
snapshots = read_jsonl(RUN / "sampling_metrics.jsonl")
by_batch = defaultdict(list)
for row in candidates:
    by_batch[row["batch_id"]].append(row)
last_snapshot = {row["batch_id"]: row for row in snapshots}
rows = []
for batch_id, candidate_rows in sorted(by_batch.items()):
    policy = candidate_rows[0]["behavior_policy_version"]
    assert all(row["behavior_policy_version"] == policy for row in candidate_rows)
    rollout_rows, selected_rows = [], []
    for candidate in candidate_rows:
        path = RUN / "candidate_rollouts" / f"candidate-{candidate['candidate_id']:08d}.json"
        rollouts = json.loads(path.read_text())["rollouts"]
        assert len(rollouts) == 8
        rollout_rows.extend(rollouts)
        if candidate["selected"]:
            selected_rows.extend(rollouts)
    mean = lambda key, data: sum(r[key] for r in data) / len(data)
    batch_updates = [r for r in updates if r["batch_id"] == batch_id]
    row = {
        "sampling_policy_step": policy,
        "batch_id": batch_id,
        "training_updates": ",".join(str(r["step"]) for r in batch_updates),
        "candidate_groups": len(candidate_rows),
        "candidate_rollouts": len(rollout_rows),
        "correct_rollouts": sum(r["judge_score"] == 1 for r in rollout_rows),
        "judge_accuracy": mean("judge_score", rollout_rows),
        "reward": mean("reward", rollout_rows),
        "outcome_reward": mean("outcome_reward", rollout_rows),
        "format_reward": mean("format_reward", rollout_rows),
        "read_cost": mean("read_cost", rollout_rows),
        "selected_groups": len(selected_rows) // 8,
        "selected_rollouts": len(selected_rows),
        "selected_judge_accuracy": mean("judge_score", selected_rows),
        "selected_reward": mean("reward", selected_rows),
        "tensorboard_candidate_index": last_snapshot[batch_id]["candidate_index"],
    }
    snapshot = last_snapshot[batch_id]
    assert snapshot["candidate_groups"] == row["candidate_groups"]
    for key in ["judge_accuracy", "reward", "format_reward"]:
        assert abs(snapshot[key] - row[key]) < 1e-10
    if len(batch_updates) == 2:
        for key in ["judge_accuracy", "reward"]:
            weighted = sum(r[key] * r["n"] for r in batch_updates) / sum(r["n"] for r in batch_updates)
            assert abs(weighted - row["selected_" + key]) < 1e-10
    rows.append(row)

with (OUT / "sampling_by_policy_step.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
metadata = {
    "observed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "source_run": str(RUN),
    "aggregation": "All candidate rollouts, including rejected and accepted-but-unselected groups; one final aggregate per sampling batch.",
    "comparison": "Selected line is the 16-group batch mean, equal to the rollout-weighted mean of its two 8-group optimizer updates.",
    "limitations": [
        "Only even policy steps generated independent sampling batches in this run.",
        "Queries differ between batches; this is not fixed-set evaluation.",
        "TensorBoard sampling tags use cumulative candidate_index rather than optimizer step.",
        "No smoothing or repeated averaging of cumulative sampling snapshots.",
    ],
    "rows": rows,
}
(OUT / "sampling_by_policy_step.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

font = FontProperties(fname="/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
x = [r["sampling_policy_step"] for r in rows]
fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.8), sharex=True)
fig.subplots_adjust(left=0.10, right=0.97, top=0.85, bottom=0.13, hspace=0.20)
fig.suptitle("包含被过滤组的采样曲线", fontproperties=font, fontsize=19, y=0.975)
fig.text(0.5, 0.92, "按采样模型 step 对齐 · 每轮采样供两次更新 · 不同轮次使用不同 query · 未平滑", ha="center", fontproperties=font, fontsize=10, color="#536170")
for ax, key, selected_key, factor, ylabel in [
    (axes[0], "judge_accuracy", "selected_judge_accuracy", 100, "Judge 正确率（%）"),
    (axes[1], "reward", "selected_reward", 1, "平均 reward"),
]:
    ax.plot(x, [r[key] * factor for r in rows], color="#1675b8", linewidth=2.5, marker="o", markersize=6, label="全部候选（包含被过滤组）", zorder=3)
    ax.plot(x, [r[selected_key] * factor for r in rows], color="#cc7a32", linewidth=1.7, marker="s", markersize=4, linestyle="--", alpha=0.85, label="筛选后训练数据（每轮16组均值）", zorder=2)
    ax.grid(axis="y", color="#dce2e8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, fontproperties=font, fontsize=11)
    ax.set_xlim(-0.5, max(x) + 0.8)
    for r in rows:
        val = r[key] * factor
        label = f"{val:.1f}" if factor == 100 else f"{val:+.3f}"
        ax.annotate(label, (r["sampling_policy_step"], val), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8, color="#145983")
axes[0].set_ylim(30, 80)
axes[0].legend(prop=font, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.20))
axes[1].axhline(0, color="#8292a0", linewidth=0.8)
axes[1].set_ylim(-0.34, 0.39)
axes[1].set_xticks(x)
axes[1].set_xlabel("生成这批候选数据的模型 step", fontproperties=font, fontsize=11)
fig.text(0.10, 0.035, f"共 {len(candidates)} 组 / {sum(r['candidate_rollouts'] for r in rows)} 条轨迹；policy 18 的数据用于更新 step 19–20。", fontproperties=font, fontsize=9, color="#536170")
fig.savefig(OUT / "sampling_curves.png", dpi=180, facecolor="white")
fig.savefig(OUT / "sampling_curves.svg", facecolor="white")
plt.close(fig)
print(json.dumps({"output_dir": str(OUT), "candidate_groups": len(candidates), "rows": rows}, ensure_ascii=False))
