"""Explicit, audited continuation under changed retrieval limits or ID protocol."""

import json
from pathlib import Path


def validate_resume_config(previous, current, *, fork_retrieval=False, fork_id_protocol=False):
    if fork_retrieval and fork_id_protocol:
        raise ValueError("Choose only one resume fork mode")
    if fork_id_protocol and current["retrieval"].get("id_protocol") != "repair_v2":
        raise ValueError("ID protocol fork requires repair_v2")
    keys = ["reader", "retrieval", "reward", "training"]
    if fork_retrieval or fork_id_protocol:
        keys += ["seed", "data", "models", "writer", "memory"]
    for key in keys:
        old, new = previous[key], current[key]
        # Evaluation cadence does not change training updates or the data cursor.
        ignored = {"tensorboard", "validation_every"} if key == "training" else set()
        if (fork_retrieval or fork_id_protocol) and key == "training":
            ignored.add("output_dir")
        if fork_retrieval and key == "retrieval":
            ignored.add("max_ids_per_call")
        if fork_id_protocol and key == "retrieval":
            ignored.add("id_protocol")
        if ignored:
            old = {k: v for k, v in old.items() if k not in ignored}
            new = {k: v for k, v in new.items() if k not in ignored}
        if old != new:
            raise ValueError(f"Resume configuration changed: {key}")


def resume_metadata(config, checkpoint, *, fork_retrieval=False, fork_id_protocol=False):
    destination = Path(config["training"]["output_dir"])
    source = Path(checkpoint).resolve().parent
    if fork_retrieval or fork_id_protocol:
        if source == destination.resolve() or (destination.exists() and any(destination.iterdir())):
            raise ValueError("Retrieval-rule fork requires a fresh, separate training output_dir")
        path = source / "run.json"
    else:
        path = destination / "run.json"
    previous = json.loads(path.read_text())
    validate_resume_config(previous["config"], config, fork_retrieval=fork_retrieval, fork_id_protocol=fork_id_protocol)
    return previous
