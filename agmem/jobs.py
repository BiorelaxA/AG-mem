"""Two independent Slurm workflows sharing immutable per-source directories."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

from .core import atomic_json, file_digest, load_config
from .checkpoints import best_checkpoint
from .data import ADAPTER_VERSION
from .pipeline import (announce, build_directories, dataset_configs, import_prepared,
                       open_prepared, prepare)
from .store import Store


@contextmanager
def file_lock(path, shared=False, blocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(stream, operation)
        except BlockingIOError:
            raise RuntimeError(f"Another job holds {path}; leave its artifacts intact and retry later") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def source_config(config, source, test=False):
    selected = deepcopy(config)
    selected["data"]["sources"] = [source]
    if test:
        selected["data"]["prepared_root"] = config["jobs"]["test_prepared_root"]
        selected["data"]["splits"] = ["test"]
    return next(dataset_configs(selected))[1]


def ensure_prepared(config):
    """Avoid reparsing completed datasets; reject mismatched cached provenance."""
    path = Path(config["data"]["prepared_root"]) / "manifest.json"
    if path.exists():
        metadata = json.loads(path.read_text())
        expected = {"input_manifest_sha256": file_digest(Path(config["data"]["input_root"]) / "manifest.json"),
                    "adapter_version": ADAPTER_VERSION, "storage_format": Store.FORMAT,
                    "sources": config["data"]["sources"], "splits": config["data"]["splits"],
                    "limit_per_source_split": None}
        if metadata.get("prepare_identity") not in (None, expected):
            raise ValueError(f"Prepared provenance differs: {path}; do not overwrite existing histories")
        if metadata.get("prepare_complete"):
            announce("preparation_reused", root=str(path.parent))
            return
    prepare(config)


def reuse_test_directories(train_config, test_config):
    """Copy summaries only for exactly identical complete histories and metadata.

    Caller holds the source lock. No test examples or labels enter the train store.
    """
    origin, target = open_prepared(train_config), open_prepared(test_config)
    try:
        source_ids = set(origin.memory_ids())
        version = origin.get_metadata("directory_version")
        if version is None:
            return 0
        copied = 0
        for memory_id in target.memory_ids():
            if memory_id not in source_ids:
                continue
            if origin.histories.get((memory_id,)) != target.histories.get((memory_id,)):
                raise ValueError(f"History hash collision or changed records: {memory_id}")
            for entry in origin.directory(memory_id, version):
                if entry.get("writer_status") == "empty_pending" or not entry["content"].strip():
                    continue
                old = target.directory_entry(memory_id, version, entry["id"])
                if old is None or old.get("writer_status") == "empty_pending":
                    target.put_directory_entry(memory_id, version, entry)
                    copied += 1
        announce("test_directories_reused", copied=copied, version=version)
        return copied
    finally:
        target.close()
        origin.close()


def build_source(config, source, test=False):
    child = source_config(config, source, test)
    root = Path(child["data"]["prepared_root"])
    legacy = Path(config["jobs"]["legacy_prepared_root"])
    if not test and not (root / "manifest.json").exists() and (legacy / "manifest.json").exists():
        # Legacy import is optional when that old run is active. Prepare afresh
        # instead of reading mutable JSONL from a running legacy writer.
        with ExitStack() as stack:
            try:
                stack.enter_context(file_lock(legacy.parent / ".job.lock", shared=True))
            except RuntimeError:
                announce("legacy_import_skipped", source=source, reason="legacy job active")
            else:
                subset = deepcopy(config)
                subset["data"]["sources"] = [source]
                import_prepared(subset, legacy)
    ensure_prepared(child)
    if test:
        reuse_test_directories(source_config(config, source), child)
    build_directories(child)


def run_logged(args, log):
    announce("stage_command", command=args, log=str(log))
    with Path(log).open("a") as stream:
        subprocess.run([sys.executable, "-u", "-m", "agmem", *args], stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def latest_checkpoint(train_dir):
    paths = sorted(p for p in Path(train_dir).glob("checkpoint-*")
                   if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit())
    if not paths:
        return None
    latest = paths[-1]
    if not (latest / "trainer.pt").is_file() or not (latest / "model/config.json").is_file():
        raise ValueError(f"Incomplete committed checkpoint: {latest}")
    return latest


def train_and_test(config, config_path, job_dir):
    sources = config["data"]["sources"]
    if sources != ["locomo", "split_longmemeval_s"]:
        raise ValueError("Pilot train/evaluate job must select LoCoMo and LongMemEval-S")
    train_dir = Path(config["training"]["output_dir"])
    # Hold selected source locks throughout train and test so directory snapshots
    # and empty_pending replacements cannot change underneath Reader rollouts.
    with ExitStack() as stack:
        stack.enter_context(file_lock(train_dir.parent / ".job.lock"))
        for source in sorted(sources):
            stack.enter_context(file_lock(Path(config["data"]["prepared_root"]) / ".locks" / f"{source}.lock"))
        if (train_dir / "run.json").exists():
            previous = json.loads((train_dir / "run.json").read_text())["config"]
            for section in ("data", "models", "service", "writer", "reader", "retrieval", "reward", "training"):
                ignored = {"request_base_url"} if section == "service" else set()
                before = {k: v for k, v in previous[section].items() if k not in ignored}
                after = {k: v for k, v in config[section].items() if k not in ignored}
                if before != after:
                    raise ValueError(f"Existing experiment differs in {section}; preserve it and choose a separate experiment")
        # Once a run has started, freeze its directories. Do not repair placeholders
        # behind an existing checkpoint; rebuilding must be an explicit new run.
        if not (train_dir / "run.json").exists():
            for source in sources:
                build_source(config, source)
        else:
            from .pipeline import validate_directories
            store = open_prepared(config)
            try:
                validate_directories(config, store, ["train", "validation"])
            finally:
                store.close()
        checkpoint = latest_checkpoint(train_dir)
        if not (train_dir / "train.complete").exists():
            if checkpoint is None and train_dir.exists():
                train_dir.rename(job_dir / "train_without_checkpoint")
            elif checkpoint is not None:
                for partial in train_dir.glob("checkpoint-*.building"):
                    partial.rename(job_dir / partial.name)
            args = ["--config", str(config_path), "train"]
            if checkpoint:
                args += ["--resume", str(checkpoint)]
            run_logged(args, job_dir / "train.log")
            (train_dir / "train.complete").write_text(time.strftime("%Y-%m-%dT%H:%M:%S") + "\n")
        checkpoint = latest_checkpoint(train_dir)
        if checkpoint is None:
            raise ValueError("Training finished without a committed checkpoint")
        selected = best_checkpoint(train_dir)
        checkpoint = selected or checkpoint
        announce("test_checkpoint_selected", checkpoint=str(checkpoint),
                 selection="validation/source_macro/reward" if selected else "latest_no_validation_checkpoint")
        # Build test only after training, so new test histories never delay the
        # first GRPO step. Test stores are separate from the fixed train split.
        for source in sources:
            build_source(config, source, test=True)
        test = deepcopy(config)
        test["data"].update(prepared_root=config["jobs"]["test_prepared_root"], splits=["test"])
        import yaml
        test_path = job_dir / "test_config.yaml"
        test_path.write_text(yaml.safe_dump(test, allow_unicode=True, sort_keys=False))
        output = train_dir.parent / "evaluation" / checkpoint.name / "test"
        if not (output / "metrics.json").exists():
            # A previous interrupted evaluation is retained; restart into the
            # canonical output directory without appending duplicate episodes.
            if output.exists():
                output.rename(job_dir / "incomplete_test")
            run_logged(["--config", str(test_path), "evaluate", "--checkpoint", str(checkpoint / "model"),
                        "--split", "test", "--output", str(output)], job_dir / "test.log")
        atomic_json(train_dir.parent / "evaluation.complete.json",
                    {"checkpoint": str(checkpoint), "metrics": str(output / "metrics.json"), "sources": sources})
        announce("train_evaluation_finished", checkpoint=str(checkpoint), metrics=str(output / "metrics.json"))


def build_all(config):
    pending = list(config["data"]["sources"])
    while pending:
        deferred = []
        for source in pending:
            with ExitStack() as stack:
                try:
                    stack.enter_context(file_lock(Path(config["data"]["prepared_root"]) / ".locks" / f"{source}.lock"))
                except RuntimeError:
                    deferred.append(source)
                    announce("source_deferred", source=source, reason="owned by another job")
                    continue
                build_source(config, source)
                build_source(config, source, test=True)
        pending = deferred
        if pending:
            announce("waiting_for_source_locks", sources=pending)
            time.sleep(30)
    announce("all_directories_finished", sources=config["data"]["sources"])


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["directories", "train-eval"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--job-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    job_dir = Path(args.job_dir)
    if args.mode == "directories":
        with file_lock(Path(config["data"]["prepared_root"]) / ".all_directories.lock"):
            build_all(config)
    else:
        train_and_test(config, Path(args.config), job_dir)


if __name__ == "__main__":
    main()
