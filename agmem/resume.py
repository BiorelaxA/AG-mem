"""Explicit, audited continuation under changed retrieval limits or ID protocol."""

import json
from pathlib import Path


def validate_resume_config(previous, current, *, fork_retrieval=False, fork_id_protocol=False):
    if fork_retrieval and fork_id_protocol:
        raise ValueError("Choose only one resume fork mode")
    if fork_id_protocol and current["retrieval"].get("id_protocol") != "repair_v2":
        raise ValueError("ID protocol fork requires repair_v2")
    keys = ["reader", "retrieval", "reward", "training"]
    if current.get("training", {}).get("algorithm") == "dapo":
        keys += ["seed", "data", "models", "writer", "memory", "task_agent"]
        from .execution import sampling_protocol
        if sampling_protocol(previous) != sampling_protocol(current):
            raise ValueError("Resume configuration changed: sampling protocol; use a new experiment")
    if fork_retrieval or fork_id_protocol:
        keys += ["seed", "data", "models", "writer", "memory"]
    for key in keys:
        old, new = previous[key], current[key]
        if key == "reader":
            # Older experiments predate explicit Reader prompt versioning.
            old = {"prompt_version": "legacy_v1", **old}
            new = {"prompt_version": "legacy_v1", **new}
        # Evaluation cadence does not change training updates or the data cursor.
        ignored = {"tensorboard", "validation_every", "keep_last_checkpoints", "checkpoint_retention"} if key == "training" else set()
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


def validate_reader_input_contract(previous, current_config, *, fork_retrieval=False, fork_id_protocol=False):
    """Keep stored rollout prompts compatible with the next sampling batch."""
    from .prompts import configured_reader_input_contract
    expected = configured_reader_input_contract(previous["config"])
    recorded = previous.get("reader_input_contract")
    if recorded is None:
        if expected["prompt_version"] != "legacy_v1":
            raise ValueError("Missing reader_input_contract provenance for versioned Reader prompt")
        recorded = expected
    if recorded != expected:
        raise ValueError("Resume Reader input contract differs from the recorded prompt or state schema")
    current = configured_reader_input_contract(current_config)
    if current["prompt_version"] != expected["prompt_version"]:
        raise ValueError("Resume Reader prompt_version changed; start a new experiment")
    # Existing GRPO forks explicitly allow max_ids or id_protocol to change.
    # Their old contract is still checked against the old configuration above.
    if not (fork_retrieval or fork_id_protocol) and recorded != current:
        raise ValueError("Resume Reader input contract changed; start a new experiment")


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
    validate_reader_input_contract(previous, config, fork_retrieval=fork_retrieval, fork_id_protocol=fork_id_protocol)
    return previous
