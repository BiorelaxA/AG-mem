from __future__ import annotations

from pathlib import Path
import json


def summary_writer_class():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError("TensorBoard logging requires: python -m pip install 'tensorboard>=2.20,<3'") from exc
    return SummaryWriter


def _write_scalars(writer, prefix, values, step):
    for name, value in values.items():
        if name != "step" and isinstance(value, (int, float)):
            writer.add_scalar(f"{prefix}/{name}", value, global_step=step)


def write_train_metrics(writer, report):
    _write_scalars(writer, "train", report, report["step"])
    writer.flush()


def write_validation_metrics(writer, report, step):
    _write_scalars(writer, "validation", report["pooled"], step)
    _write_scalars(writer, "validation/source_macro", report["source_macro"], step)
    for source, values in report["by_source"].items():
        _write_scalars(writer, f"validation/by_source/{source}", values, step)
    writer.flush()


def export_tensorboard(run_dir):
    """Rebuild event scalars from existing metrics without loading a model."""
    run_dir = Path(run_dir)
    metrics_path = run_dir / "train_metrics.jsonl"
    if not metrics_path.is_file():
        raise ValueError(f"Training metrics do not exist: {metrics_path}")
    log_dir = run_dir / "tensorboard"
    # Purge old scalars when repeating an export, while keeping authoritative JSONL untouched.
    writer = summary_writer_class()(log_dir=str(log_dir), purge_step=0)
    train_steps, validation_points = 0, 0
    try:
        with metrics_path.open() as stream:
            for line in stream:
                if line.strip():
                    write_train_metrics(writer, json.loads(line))
                    train_steps += 1
        for path in sorted((run_dir / "validation").glob("step-*/metrics.json")):
            step = int(path.parent.name.removeprefix("step-"))
            write_validation_metrics(writer, json.loads(path.read_text()), step)
            validation_points += 1
    finally:
        writer.close()
    return {"tensorboard_dir": str(log_dir), "train_steps": train_steps, "validation_points": validation_points}
