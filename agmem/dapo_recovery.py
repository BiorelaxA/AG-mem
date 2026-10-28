"""Archive artifacts beyond a committed DAPO checkpoint before continuing a run.

The caller must hold the run's exclusive lock. Optimizer/checkpoint contents are
never edited, and discarded logs/artifacts remain available under resume_orphans.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import uuid

from .checkpoints import select_best_checkpoints


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"Recovery requires a nonnegative integer {name}")
    return value


def _reconcile_jsonl(path, field, limit, *, inclusive):
    """Plan a rewrite without mutating files; only a broken final record is recoverable."""
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    populated = [index for index, line in enumerate(lines) if line.strip()]
    last = populated[-1] if populated else -1
    retained = []
    for index, line in enumerate(lines):
        if not line.strip():
            retained.append(line)
            continue
        try:
            value = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            if index == last:
                continue
            raise ValueError(f"Corrupt JSONL record before the tail: {path}:{index + 1}") from exc
        if not isinstance(value, dict) or field not in value:
            raise ValueError(f"Missing {field} in recovery log: {path}:{index + 1}")
        position = _integer(value[field], f"{path.name}:{index + 1}:{field}")
        if position < limit or (inclusive and position == limit):
            retained.append(line if line.endswith(b"\n") else line + b"\n")
    updated = b"".join(retained)
    return updated if updated != original else None


def _indexed_artifacts(root, directory, prefix, suffix, limit, *, inclusive):
    parent = root / directory if directory else root
    if not parent.exists():
        return []
    result = []
    for path in parent.glob(prefix + "*" + suffix):
        name = path.name
        middle = name[len(prefix):len(name) - len(suffix)] if suffix else name[len(prefix):]
        if middle.isdigit() and (int(middle) > limit if inclusive else int(middle) >= limit):
            result.append(path)
    return result


def _atomic_bytes(path, content):
    temporary = path.with_name(path.name + f".resume-{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def reconcile_run(run_dir, state):
    """Retain committed updates and their saved sampling buffer, archive the rest.

    Group trace filenames use the *pre-update* zero-based step. Candidate IDs use
    a separate monotonically increasing cursor and can be ahead of optimizer
    steps because a checkpoint may contain unconsumed, already sampled groups.
    Returns the archive path (or None for an already reconciled run).
    """
    root = Path(run_dir)
    step = _integer(state["step"], "step")
    candidate_index = _integer(state["dapo_state"]["sampler"].get("candidate_index", 0), "candidate_index")
    rewrites = []
    for filename, field, limit, inclusive in (
        ("train_metrics.jsonl", "step", step, True),
        ("sampling_metrics.jsonl", "candidate_index", candidate_index, True),
        ("candidate_metrics.jsonl", "candidate_id", candidate_index, False),
    ):
        path = root / filename
        if path.is_file():
            content = _reconcile_jsonl(path, field, limit, inclusive=inclusive)
            if content is not None:
                rewrites.append((path, content))

    moves = []
    for directory, prefix, suffix, limit, inclusive in (
        ("groups", "group-", ".json.gz", step, False),
        ("rollouts", "group-", ".json", step, False),
        ("candidate_rollouts", "candidate-", ".json", candidate_index, False),
        ("validation", "step-", "", step, True),
        ("", "checkpoint-", "", step, True),
    ):
        moves.extend(_indexed_artifacts(root, directory, prefix, suffix, limit, inclusive=inclusive))
    moves.extend(root.glob("checkpoint-*.building"))
    archive = None
    archived = []

    def archive_item(path, *, move):
        nonlocal archive
        if archive is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            archive = root / "resume_orphans" / f"{stamp}-{uuid.uuid4().hex[:8]}"
            archive.mkdir(parents=True)
        relative = path.relative_to(root)
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if move:
            path.rename(destination)
        else:
            shutil.copy2(path, destination)
        archived.append(str(relative))

    # Every JSONL was checked before the first mutation, so middle corruption
    # cannot cause a partially truncated run. Existing artifacts are preserved.
    for path in sorted(set(moves)):
        archive_item(path, move=True)
    for path, content in rewrites:
        archive_item(path, move=False)
        _atomic_bytes(path, content)

    index = root / "best_checkpoints.json"
    best = select_best_checkpoints(root)
    encoded = (json.dumps(best, ensure_ascii=False, indent=2) + "\n").encode()
    previous = None
    if index.is_file():
        try:
            previous = json.loads(index.read_bytes())
        except (ValueError, UnicodeDecodeError):
            pass
        if previous != best:
            archive_item(index, move=False)
            _atomic_bytes(index, encoded)
    elif best:
        _atomic_bytes(index, encoded)

    result = {"step": step, "candidate_index": candidate_index,
              "archive": str(archive) if archive else None, "archived_artifacts": archived}
    if archive:
        _atomic_bytes(archive / "recovery.json", (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode())
    return result
