"""Retain recent resumable checkpoints and validation-best checkpoints independently."""

import json
import math
from pathlib import Path
import shutil

from .core import atomic_json


def committed_checkpoints(run_dir):
    return sorted((path for path in Path(run_dir).glob("checkpoint-*")
                   if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()),
                  key=lambda path: int(path.name.removeprefix("checkpoint-")))


def validation_scores(report):
    # Equal weight per dataset, irrespective of its number of validation queries.
    macro = report["source_macro"]
    scores = {"best_reward": macro["reward"],
              "best_outcome": macro["judge_accuracy"] if "judge_accuracy" in macro else macro["outcome_f1"]}
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in scores.values()):
        raise ValueError("Checkpoint selection requires finite validation scores")
    return scores


def select_best_checkpoints(run_dir):
    """Recover selection from committed checkpoints, even after an interrupted save.

    Validation metrics travel inside their checkpoint's atomic directory rename.
    Uncommitted .building directories and standalone evaluation logs never qualify.
    Exact ties keep the earlier checkpoint.
    """
    best = {}
    for path in committed_checkpoints(run_dir):
        metrics_path = path / "validation_metrics.json"
        if not metrics_path.exists():
            continue
        if not (path / "trainer.pt").is_file() or not (path / "model/config.json").is_file():
            raise ValueError(f"Incomplete committed checkpoint: {path}")
        report = json.loads(metrics_path.read_text())
        for name, value in validation_scores(report).items():
            if name not in best or value > best[name]["value"]:
                best[name] = {"checkpoint": path.name, "step": int(path.name.removeprefix("checkpoint-")),
                              "value": value, "scope": "validation/source_macro"}
    return best


def retain_checkpoints(run_dir, keep_last):
    run_dir = Path(run_dir)
    best = select_best_checkpoints(run_dir)
    # Publish selection before deleting anything. Recovery recomputes it from
    # checkpoint-local metrics instead of trusting potentially stale pointers.
    atomic_json(run_dir / "best_checkpoints.json", best)
    paths = committed_checkpoints(run_dir)
    protected = {entry["checkpoint"] for entry in best.values()}
    if keep_last and keep_last > 0:
        for path in paths[:-keep_last]:
            if path.name not in protected:
                shutil.rmtree(path)
    return best


def best_checkpoint(run_dir, kind="best_reward"):
    if kind not in {"best_reward", "best_outcome"}:
        raise ValueError(f"Unknown checkpoint selection: {kind}")
    selected = select_best_checkpoints(run_dir).get(kind)
    return Path(run_dir) / selected["checkpoint"] if selected else None
