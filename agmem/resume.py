"""Explicit, audited continuation under a changed retrieval batch limit."""

import json
from pathlib import Path


def validate_resume_config(previous, current, *, fork_retrieval=False):
    keys = ["reader", "retrieval", "reward", "training"]
    if fork_retrieval:
        keys += ["seed", "data", "models", "writer", "memory"]
    for key in keys:
        old, new = previous[key], current[key]
        ignored = {"tensorboard"} if key == "training" else set()
        if fork_retrieval and key == "training":
            ignored.add("output_dir")
        if fork_retrieval and key == "retrieval":
            ignored.add("max_ids_per_call")
        if ignored:
            old = {k: v for k, v in old.items() if k not in ignored}
            new = {k: v for k, v in new.items() if k not in ignored}
        if old != new:
            raise ValueError(f"Resume configuration changed: {key}")


def resume_metadata(config, checkpoint, *, fork_retrieval=False):
    destination = Path(config["training"]["output_dir"])
    source = Path(checkpoint).resolve().parent
    if fork_retrieval:
        if source == destination.resolve() or (destination.exists() and any(destination.iterdir())):
            raise ValueError("Retrieval-rule fork requires a fresh, separate training output_dir")
        path = source / "run.json"
    else:
        path = destination / "run.json"
    previous = json.loads(path.read_text())
    validate_resume_config(previous["config"], config, fork_retrieval=fork_retrieval)
    return previous
