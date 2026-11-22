"""Retain committed checkpoints using the configured validation selection policy."""

import json
import math
import os
from pathlib import Path
import shutil
import uuid

from .core import atomic_json


OVERWRITE_RETENTION_POLICY = "best_and_latest_overwrite"


def committed_checkpoints(run_dir):
    return sorted((path for path in Path(run_dir).glob("checkpoint-*")
                   if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()),
                  key=lambda path: int(path.name.removeprefix("checkpoint-")))


def validation_scores(report):
    # Equal weight per dataset, irrespective of its number of validation queries.
    macro = report["source_macro"]
    primary = report.get("primary_outcome_metric")
    if primary is not None and primary not in {"answer_f1", "outcome_f1", "judge_accuracy"}:
        raise ValueError("Unknown primary_outcome_metric for checkpoint selection")
    if primary is None:
        primary = "judge_accuracy" if macro.get("judge_accuracy") is not None else "outcome_f1"
    scores = {"best_reward": macro["reward"],
              "best_outcome": macro[primary]}
    if report.get("primary_outcome_metric") is not None:
        scores["best_primary"] = macro[primary]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in scores.values()):
        raise ValueError("Checkpoint selection requires finite validation scores")
    return scores


def select_best_checkpoints(run_dir):
    """Recover selection from committed checkpoints, even after an interrupted save.

    Validation metrics travel inside their checkpoint's atomic directory rename.
    Uncommitted .building directories, pending .pruning directories, and
    standalone evaluation logs never qualify.
    Exact ties keep the earlier checkpoint.
    """
    return _select_best_checkpoints(committed_checkpoints(run_dir))


def _require_complete_checkpoint(path):
    if not (path / "trainer.pt").is_file() or not (path / "model/config.json").is_file():
        raise ValueError(f"Incomplete committed checkpoint: {path}")


def _select_best_checkpoints(paths):
    best = {}
    for path in paths:
        metrics_path = path / "validation_metrics.json"
        if not metrics_path.exists():
            continue
        _require_complete_checkpoint(path)
        report = json.loads(metrics_path.read_text())
        for name, value in validation_scores(report).items():
            if name not in best or value > best[name]["value"]:
                best[name] = {"checkpoint": path.name, "step": int(path.name.removeprefix("checkpoint-")),
                              "value": value, "scope": "validation/source_macro"}
    return best


def _prune_obsolete_checkpoints(run_dir, paths):
    # Resume interrupted cleanup without ever treating a partly deleted
    # directory as a committed checkpoint. Only our exact staging names qualify.
    for pending in sorted(run_dir.glob("checkpoint-*.pruning")):
        step = pending.name.removeprefix("checkpoint-").removesuffix(".pruning")
        if step.isascii() and step.isdigit() and not pending.is_symlink() and pending.is_dir():
            shutil.rmtree(pending)
    for path in paths:
        pending = path.with_name(path.name + ".pruning")
        path.rename(pending)
        shutil.rmtree(pending)


def _own_checkpoint_name(name):
    step = name.removeprefix("checkpoint-")
    return name.startswith("checkpoint-") and step.isascii() and step.isdigit()


def _validate_checkpoint_aliases(run_dir):
    for name in ("best", "latest"):
        alias = run_dir / name
        if alias.is_symlink():
            # Only a direct, relative pointer to one of our payload names is
            # ours to replace. A missing old target can be repaired on recovery.
            target = os.readlink(alias)
            if not _own_checkpoint_name(target):
                raise ValueError(f"Refusing to replace unknown checkpoint alias: {alias}")
            if (run_dir / target).is_symlink():
                raise ValueError(f"Checkpoint alias target must not be a symlink: {alias}")
        elif alias.exists():
            raise ValueError(f"Checkpoint alias name is already occupied: {alias}")


def _overwrite_checkpoint_state(run_dir):
    """Check everything that could affect deletion before changing any files."""
    paths = sorted((path for path in run_dir.glob("checkpoint-*")
                    if _own_checkpoint_name(path.name)),
                   key=lambda path: int(path.name.removeprefix("checkpoint-")))
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Committed checkpoint must not be a symlink: {path}")
        _require_complete_checkpoint(path)
    best = _select_best_checkpoints(paths)
    _validate_checkpoint_aliases(run_dir)
    return paths, best


def _primary_checkpoint(paths, best):
    primary = best.get("best_primary", best.get("best_outcome"))
    if primary is not None:
        return next(path for path in paths if path.name == primary["checkpoint"])
    return paths[-1] if paths else None


def _publish_checkpoint_aliases(run_dir, primary, latest):
    for name, checkpoint in (("best", primary), ("latest", latest)):
        alias = run_dir / name
        if checkpoint is None:
            if alias.is_symlink():
                alias.unlink()
            continue
        if alias.is_symlink() and os.readlink(alias) == checkpoint.name:
            continue
        temporary = run_dir / f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            temporary.symlink_to(checkpoint.name, target_is_directory=True)
            temporary.replace(alias)
        finally:
            temporary.unlink(missing_ok=True)


def refresh_checkpoint_aliases(run_dir):
    """Repair overwrite-mode best/latest links without copying any payload.

    Committed numeric payloads remain the recovery source of truth. The return
    value is the recomputed selection; this helper does not write its manifest.
    """
    run_dir = Path(run_dir)
    paths, best = _overwrite_checkpoint_state(run_dir)
    _publish_checkpoint_aliases(run_dir, _primary_checkpoint(paths, best),
                                paths[-1] if paths else None)
    return best


def _cleanup_overwrite_temporaries(run_dir):
    removed = []
    for suffix in (".building", ".pruning"):
        for path in sorted(run_dir.glob(f"checkpoint-*{suffix}")):
            if (_own_checkpoint_name(path.name.removesuffix(suffix))
                    and not path.is_symlink() and path.is_dir()):
                shutil.rmtree(path)
                removed.append(path)
    return removed


def cleanup_overwrite_temporaries(run_dir):
    """Discard only our numeric partial directories after an interrupted save.

    This is for overwrite-mode recovery or immediately before a new save, never
    while another writer is saving a checkpoint into the same run directory.
    """
    run_dir = Path(run_dir)
    _overwrite_checkpoint_state(run_dir)
    return _cleanup_overwrite_temporaries(run_dir)


def prepare_checkpoint_overwrite(run_dir, step):
    """Free the old latest slot while preserving a complete validation best.

    Call before creating the new checkpoint's .building directory. Publishing
    the surviving selection and both aliases precedes any deletion, so a save
    failure always leaves a complete recoverable checkpoint. With no validation
    history, protect the latest complete checkpoint instead. A normal save then
    needs space for at most two full payloads, including the one being written.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("Checkpoint overwrite requires a nonnegative integer step")
    run_dir = Path(run_dir)
    paths, best = _overwrite_checkpoint_state(run_dir)
    if paths and step <= int(paths[-1].name.removeprefix("checkpoint-")):
        raise ValueError("Checkpoint overwrite requires a strictly increasing checkpoint step")
    primary = _primary_checkpoint(paths, best)
    retained = [primary] if primary is not None else []
    best = _select_best_checkpoints(retained)
    atomic_json(run_dir / "best_checkpoints.json", best)
    _publish_checkpoint_aliases(run_dir, primary, primary)
    _prune_obsolete_checkpoints(run_dir, [path for path in paths if path != primary])
    _cleanup_overwrite_temporaries(run_dir)
    return best


def retain_checkpoints(run_dir, keep_last, *, policy="legacy"):
    """Prune only after the caller has atomically committed a complete checkpoint.

    ``legacy`` independently protects every validation best and the last
    ``keep_last`` checkpoints. ``best_and_latest`` ignores ``keep_last`` and
    retains at most two: the latest committed checkpoint and the validation
    primary best (or outcome best for reports without an explicit primary).
    The best and latest may be the same checkpoint. With no validation scores,
    only the latest is retained. Exact ties keep the earlier checkpoint.

    Selection is published before deletion, and its diagnostic best-reward
    entry describes only retained checkpoints. A new save therefore needs
    temporary space alongside the existing checkpoints; pruning must not be
    called before that save commits. Uncommitted .building directories are
    never selected or removed here. In ``best_and_latest``, obsolete directories
    are first atomically renamed to .pruning; interrupted deletion is excluded
    from recovery selection and retried on the next successful retention call.
    ``best_and_latest_overwrite`` uses the same final selection and publishes
    fixed best/latest symlinks. Its caller must first prepare the overwrite to
    free the previous latest slot before writing a replacement payload.
    """
    if policy not in {"legacy", "best_and_latest", OVERWRITE_RETENTION_POLICY}:
        raise ValueError(f"Unknown checkpoint retention policy: {policy}")
    run_dir = Path(run_dir)
    overwrite = policy == OVERWRITE_RETENTION_POLICY
    if overwrite:
        paths, best = _overwrite_checkpoint_state(run_dir)
    else:
        paths = committed_checkpoints(run_dir)
        best = _select_best_checkpoints(paths)
    if policy in {"best_and_latest", OVERWRITE_RETENTION_POLICY}:
        protected = {paths[-1].name} if paths else set()
        primary = best.get("best_primary", best.get("best_outcome"))
        if primary is not None:
            protected.add(primary["checkpoint"])
        retained = [path for path in paths if path.name in protected]
        for path in retained:
            _require_complete_checkpoint(path)
        # Recompute all pointers over the final retained set so recovery and
        # best_checkpoint() produce the same manifest after pruning.
        best = _select_best_checkpoints(retained)
        obsolete = [path for path in paths if path.name not in protected]
    else:
        protected = {entry["checkpoint"] for entry in best.values()}
        obsolete = ([path for path in paths[:-keep_last] if path.name not in protected]
                    if keep_last and keep_last > 0 else [])
    # Publish selection before deleting anything. Recovery recomputes it from
    # checkpoint-local metrics instead of trusting potentially stale pointers.
    atomic_json(run_dir / "best_checkpoints.json", best)
    if overwrite:
        _publish_checkpoint_aliases(run_dir, _primary_checkpoint(retained, best),
                                    retained[-1] if retained else None)
    if policy in {"best_and_latest", OVERWRITE_RETENTION_POLICY}:
        _prune_obsolete_checkpoints(run_dir, obsolete)
    else:
        for path in obsolete:
            shutil.rmtree(path)
    return best


def best_checkpoint(run_dir, kind=None):
    if kind not in {None, "best_reward", "best_outcome", "best_primary"}:
        raise ValueError(f"Unknown checkpoint selection: {kind}")
    best = select_best_checkpoints(run_dir)
    if kind is None:
        kind = "best_primary" if "best_primary" in best else "best_reward"
    selected = best.get(kind)
    return Path(run_dir) / selected["checkpoint"] if selected else None
