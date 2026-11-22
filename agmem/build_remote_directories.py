"""Build test-session directories through a remote Writer, without loading weights."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

from .core import atomic_json, digest, file_digest
from .evaluation_data import prepare_source, build_source_directories, _current_writer_version
from .jobs import file_lock
from .pipeline import announce
from .prompts import writer_messages


def load_recipe(path, workers=None):
    import yaml

    recipe = yaml.safe_load(Path(path).read_text())
    saved = json.loads((Path(recipe["training_run"]) / "run.json").read_text())
    config = deepcopy(saved["config"])
    # Preserve the entire saved Writer configuration and exact prompt. Only its
    # serving model changes; the original tokenizer still limits summary length.
    if config["writer"].get("lookback_sessions") != 3:
        raise ValueError("The reference Writer must use the previous three raw sessions")
    if config["writer"].get("content_max_tokens") != 300:
        raise ValueError("The reference Writer must use a 300-token summary limit")
    service = deepcopy(recipe["writer_service"])
    if service.get("request_profile") != "openai_chat":
        raise ValueError("This builder requires writer_service.request_profile=openai_chat")
    config["writer_service"] = service
    config["data"].update(input_root=str(Path(recipe["input_root"]).resolve()),
                          prepared_root=str(Path(recipe["prepared_root"]).resolve()),
                          splits=["test"])
    config["data"].pop("frozen_directory_version", None)
    parallel = workers if workers is not None else recipe.get("writer_workers", 8)
    if type(parallel) is not int or parallel < 1:
        raise ValueError("writer_workers must be a positive integer")
    config.setdefault("execution", {})["writer_workers"] = parallel
    sources = recipe["datasets"]
    if (not isinstance(sources, list) or not sources or len(set(sources)) != len(sources)
            or any(source not in {"split_longmemeval_s", "split_longmemeval_m"} for source in sources)):
        raise ValueError("Select unique LongMemEval-S/M test datasets")
    config["data"]["sources"] = list(sources)
    return recipe, config


def build_plan(recipe, config, sources):
    import pyarrow.parquet as pq
    from .remote_writer import writer_identity

    root = Path(config["data"]["input_root"])
    if not (root / "manifest.json").is_file():
        raise ValueError(f"Missing processed-data manifest: {root}")
    datasets = []
    for source in sources:
        path = root / source / "test.parquet"
        datasets.append({"source": source, "queries": pq.read_metadata(path).num_rows,
                         "input": str(path),
                         "prepared": str(Path(config["data"]["prepared_root"]) / source)})
    return {"writer": writer_identity(config), "writer_config": config["writer"],
            "summary_tokenizer": config["models"]["task_path"],
            "writer_workers": config["execution"]["writer_workers"],
            "prompt_sha256": digest(writer_messages([], {}, config["writer"]["content_max_tokens"])),
            "datasets": datasets, "requires_gpu": False}


def run(args):
    recipe, config = load_recipe(args.config, args.workers)
    sources = args.datasets or recipe["datasets"]
    if not sources or len(set(sources)) != len(sources) or not set(sources).issubset(recipe["datasets"]):
        raise ValueError("--datasets must select unique configured datasets")
    plan = build_plan(recipe, config, sources)
    announce("remote_directory_plan", **plan)
    if args.dry_run:
        return plan
    # Check credentials before preparing the much larger M histories. Values
    # never enter any saved configuration, directory, event or exception.
    key_env = config["writer_service"].get("api_key_env", "OPENAI_API_KEY")
    if not os.environ.get(key_env):
        raise ValueError(f"Set {key_env} on the networked node before building directories")
    from .policy import load_tokenizer
    load_tokenizer(config["models"]["task_path"])
    root = Path(config["data"]["prepared_root"])
    expected_version = _current_writer_version(config)
    manifest = {"protocol": "remote_test_directories_v1",
                "input_root": config["data"]["input_root"],
                "input_manifest_sha256": file_digest(Path(config["data"]["input_root"]) / "manifest.json"),
                "datasets": recipe["datasets"], "split": "test",
                "writer": plan["writer"], "writer_config": config["writer"],
                "summary_tokenizer": plan["summary_tokenizer"],
                "directory_version": expected_version, "prompt_sha256": plan["prompt_sha256"]}
    with file_lock(root / ".remote_build_manifest.lock"):
        path = root / "remote_build_manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Remote Writer protocol changed; select a new prepared_root")
        if not path.exists():
            atomic_json(path, manifest)
    results = {}
    # S finishes first and is immediately usable, even while M is still building.
    # These are the same locks used by checkpoint evaluation.
    for source in sources:
        with file_lock(root / ".evaluation.lock", shared=True):
            with file_lock(root / ".evaluation_locks" / f"{source}.lock"):
                status_path = root / source / "remote_build_status.json"
                atomic_json(status_path, {"status": "preparing", "source": source})
                try:
                    prepared = prepare_source(config, source, root)
                    if (prepared["directory_version"] is not None
                            and prepared["directory_version"] != expected_version):
                        raise ValueError("Existing directories use a different Writer; select a new prepared_root")
                    atomic_json(status_path, {"status": "building", "source": source,
                                             "queries": prepared["examples"], "sessions": prepared["sessions"],
                                             "directory_version": expected_version})
                    result = build_source_directories(prepared)
                    if result["directory_version"] != expected_version:
                        raise ValueError("Published Writer version differs from the requested remote Writer")
                    atomic_json(root / source / "remote_prepared_config.json", result["config"])
                    summary = {key: result[key] for key in
                               ("source", "status", "examples", "histories", "sessions", "directory_version")}
                    atomic_json(status_path, summary)
                    announce("remote_directory_source_complete", **summary)
                    results[source] = summary
                except Exception as exc:
                    atomic_json(status_path, {"status": "failed", "source": source,
                                             "error_type": type(exc).__name__,
                                             "resume": "Rerun the same command; completed entries are retained"})
                    raise
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/build_longmemeval_gpt5mini.yaml")
    parser.add_argument("--datasets", nargs="+", choices=["split_longmemeval_s", "split_longmemeval_m"])
    parser.add_argument("--workers", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/configuration without API calls or writes")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
