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
                        **({"format_violations": decision["format_violations"]}
                           if decision.get("format_violations") else {}),
                        **({
                            "id_repairs": decision["tool_result"]["id_repairs"],
                            "invalid_ids": decision["tool_result"]["invalid_ids"],
                            "selected_ids": decision["tool_result"]["selected_ids"],
                            "returned_ids": [r["id"] for r in decision["tool_result"]["records"]],
                            "tool_error": decision["tool_result"].get("error"),
                        } if "id_repairs" in (decision.get("tool_result") or {}) else {}),
                        **({
                            "ids_truncated": True,
                            "selected_ids": decision["tool_result"]["selected_ids"],
                            "returned_ids": [record["id"] for record in decision["tool_result"]["records"]],
                        } if (decision.get("tool_result") or {}).get("ids_truncated") else {}),
                        **({
                            "already_read_ids": decision["tool_result"]["already_read_ids"],
                            "new_ids": decision["tool_result"]["new_ids"],
                            "no_new_evidence": decision["tool_result"]["no_new_evidence"],
                            "read_call_consumed": decision["tool_result"]["read_call_consumed"],
                            "read_calls_used": decision["tool_result"]["read_calls_used"],
                            "read_calls_remaining": decision["tool_result"]["read_calls_remaining"],
                            "tool_error": decision["tool_result"].get("error"),
                        } if "already_read_ids" in (decision.get("tool_result") or {}) else {}),
                    }
                    for step, decision in enumerate(row["decisions"])
                ],
                "task_agent_output": row["prediction"],
                "task_completion": row.get("task_completion", {}),
                "outcome_f1": row.get("answer_f1") if row.get("answer_f1") is not None else row["outcome"],
                **{key: row[key] for key in ("judge_score", "auxiliary_reward", "likelihood", "advantage") if key in row},
                "reward": row["reward"],
                # Historical groups keep their recorded reward unchanged.
                **{key: row[key] for key in ("read_cost", "raw_read_cost", "outcome_reward", "format_failure", "format_reward") if key in row},
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
