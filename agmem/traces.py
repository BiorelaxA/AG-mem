"""Readable Reader/Task Agent outputs, without prompts or retrieved history."""

from pathlib import Path
import gzip
import json

from .core import atomic_json, canonical


def compact_group(group: dict) -> dict:
    # Use an explicit allowlist: full decisions also contain prompts, token IDs,
    # behavior logprobs and retrieved session text, none of which belongs here.
    return {
        "group_id": group["group_id"],
        "directory_version": group["directory_version"],
        "rollouts": [
            {
                "rollout_index": index,
                "sample_id": row["sample_id"],
                "termination_reason": row["termination_reason"],
                "reader_steps": [
                    {
                        "step": step + 1,
                        "reader_output": decision["completion"]["text"],
                        "action": decision["action"],
                    }
                    for step, decision in enumerate(row["decisions"])
                ],
                "task_agent_output": row["prediction"],
                "outcome_f1": row["outcome"],
                "reward": row["reward"],
                # Historical groups keep their recorded reward unchanged.
                **{key: row[key] for key in ("read_cost", "format_failure", "format_reward") if key in row},
            }
            for index, row in enumerate(group["rollouts"])
        ],
    }


def save_training_group(run_dir, step, group_id, directory_version, trajectories):
    """Keep the full training audit and write a separate, readable output trace."""
    run_dir = Path(run_dir)
    group = {"group_id": group_id, "directory_version": directory_version,
             "rollouts": [row.to_dict() for row in trajectories]}
    full_dir = run_dir / "groups"
    full_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(full_dir / f"group-{step:08d}.json.gz", "wt", encoding="utf-8") as stream:
        stream.write(canonical(group))
    atomic_json(run_dir / "rollouts" / f"group-{step:08d}.json", compact_group(group))


def export_rollouts(run_dir):
    """Export existing groups without loading a model or modifying original logs."""
    run_dir = Path(run_dir)
    paths = sorted((run_dir / "groups").glob("group-*.json.gz"))
    if not paths:
        raise ValueError(f"No saved training groups in {run_dir / 'groups'}")
    rollout_count = 0
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            group = json.load(stream)
        summary = compact_group(group)
        atomic_json(run_dir / "rollouts" / path.name.removesuffix(".gz"), summary)
        rollout_count += len(summary["rollouts"])
    return {"output_dir": str(run_dir / "rollouts"), "groups": len(paths), "rollouts": rollout_count}
