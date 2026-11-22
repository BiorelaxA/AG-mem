"""Build versioned session/topic directories for LoCoMo and LongMemEval S/M."""

import argparse
from collections import Counter
from pathlib import Path
from threading import Lock
import time

from .core import digest, file_digest, InfrastructureError
from .execution import bounded_results
from .jobs import file_lock
from .pipeline import announce
from .hierarchical.adapters import input_files, iter_banks
from .hierarchical.binding import bind_directory, validate_directory
from .hierarchical.client import WriterPool
from .hierarchical.records import validate_session
from .hierarchical.storage import BuildStore


DEFAULT_CONFIG = "configs/build_parent_child_qwen.yaml"


def load_recipe(path, workers=None):
    import yaml
    from .policy import model_fingerprint
    from .hierarchical.writer import prompt_hashes, writer_settings

    config = yaml.safe_load(Path(path).read_text())
    if config.get("protocol") != "parent_child_directory_v1":
        raise ValueError("Use a parent_child_directory_v1 recipe; flat-summary caches are incompatible")
    if any(key in config.get("service", {}) for key in ("api_key", "authorization")):
        raise ValueError("Pass Writer credentials through an environment variable")
    config["model_path"] = str(Path(config["model_path"]).resolve())
    config["output_root"] = str(Path(config["output_root"]).resolve())
    config["workers"] = workers if workers is not None else config.get("workers", 48)
    if type(config["workers"]) is not int or config["workers"] < 1:
        raise ValueError("workers must be a positive integer")
    if type(config.get("retry_fallback", True)) is not bool:
        raise ValueError("retry_fallback must be boolean")
    config["writer"] = writer_settings(config.get("writer"))
    service = config.get("service", {})
    if type(service.get("network_retries", 3)) is not int or service.get("network_retries", 3) < 0:
        raise ValueError("network_retries must be a nonnegative integer")
    if type(service.get("structured_outputs", True)) is not bool:
        raise ValueError("structured_outputs must be boolean")
    specs = config["datasets"]
    if not specs or len({s["name"] for s in specs}) != len(specs):
        raise ValueError("Configure unique dataset names")
    fingerprint = model_fingerprint(config["model_path"])
    tokenizer_files = {name: info for name, info in fingerprint["files"].items()
                       if "tokenizer" in name or name in {"special_tokens_map.json", "vocab.json", "added_tokens.json"}}
    # Chat packaging changes the actual context budget and model input even
    # when the tokenizer JSON files and weights remain unchanged.
    for name in ("chat_template.jinja", "merges.txt", "tokenizer.model", "spiece.model"):
        resource = Path(config["model_path"]) / name
        if resource.is_file():
            tokenizer_files[name] = {"bytes": resource.stat().st_size, "sha256": file_digest(resource)}
    if not tokenizer_files:
        raise ValueError("Missing local tokenizer provenance")
    tokenizer_id = "qwen-tokenizer@" + digest(tokenizer_files)
    sources = []
    for spec in specs:
        files = input_files(spec)
        sources.append({"spec": spec, "files": [{"split": split, "path": str(p), "sha256": file_digest(p)}
                                                for split, p in files]})
    identity = {"protocol": config["protocol"], "datasets": sources,
                "writer": {"model": Path(config["model_path"]).name,
                           "model_revision": fingerprint["fingerprint"]},
                "tokenizer_id": tokenizer_id, "writer_settings": config["writer"],
                "prompt_hashes": prompt_hashes(), "block_policy": "block.v1",
                "request_options": {"temperature": 0, "enable_thinking": False,
                                    "seed": config.get("service", {}).get("seed", 42),
                                    "structured_outputs": config.get("service", {}).get("structured_outputs", True)}}
    return config, identity


def plan(config, identity, datasets=None, splits=None):
    names = datasets or [s["name"] for s in config["datasets"]]
    if not names or len(set(names)) != len(names) or not set(names).issubset(s["name"] for s in config["datasets"]):
        raise ValueError("Select unique configured datasets")
    selected, items = [], []
    for spec in config["datasets"]:
        if spec["name"] not in names:
            continue
        selected.append(spec)
        for split, path in input_files(spec, splits):
            if path.suffix == ".parquet":
                import pyarrow.parquet as pq
                n = pq.read_metadata(path).num_rows
            else:
                n = None
            items.append({"dataset": spec["name"], "split": split, "input": str(path), "input_rows": n,
                          "format": spec.get("format", "auto")})
    return selected, {"protocol": config["protocol"], "output_root": config["output_root"],
                      "writer": identity["writer"], "writer_workers": config["workers"],
                      "inputs": items, "writer_settings": config["writer"],
                      "reuses_old_flat_summaries": False, "tokenizer_id": identity["tokenizer_id"]}


def run(args, *, tokenizer=None, pool_factory=WriterPool):
    from .policy import load_tokenizer
    from .hierarchical.writer import DirectoryWriter, public_session, token_count, validate_build_result

    config, identity = load_recipe(args.config, args.workers)
    specs, preview = plan(config, identity, args.datasets, args.splits)
    announce("parent_child_directory_plan", **preview)
    if args.dry_run:
        return preview
    endpoints = args.base_urls or ([args.base_url] if args.base_url else config.get("base_urls"))
    if not endpoints or len(set(endpoints)) != len(endpoints):
        raise ValueError("Provide unique --base-urls, or use the three-GPU shell launcher")
    tokenizer = tokenizer or load_tokenizer(config["model_path"])
    def count_tokens(text):
        return token_count(tokenizer, text)

    root = Path(config["output_root"])
    counters, key_locks, key_lock_guard = Counter(), {}, Lock()
    started = time.monotonic()
    with file_lock(root / ".build.lock"):
        store = BuildStore(root, identity)
        pool = None
        execution = {"run_status": "running", "datasets": [s["name"] for s in specs],
                     "splits": args.splits or ["train", "validation", "test"]}
        try:
            pool = pool_factory(endpoints, config["workers"], config.get("service", {}), store)
            def inputs():
                for spec in specs:
                    modality = spec.get("modality_policy", "existing_captions")
                    for bank in iter_banks(spec, args.splits, count_tokens=count_tokens,
                                           tokenizer_id=identity["tokenizer_id"]):
                        store.add_bank(bank)
                        counters["banks_seen"] += 1
                        for session in bank["sessions"]:
                            validate_session(session, count_tokens=count_tokens)
                            store.add_session(session)
                            state, directory = store.committed(session["bank_id"], session["parent_id"])
                            if state in {"complete", "empty"} or (state == "fallback" and not config.get("retry_fallback", True)):
                                validate_directory(directory, session, count_tokens=count_tokens)
                                counters["sessions_resumed"] += 1
                                continue
                            yield spec["name"], session, modality

            def build(item):
                dataset, session, modality = item
                try:
                    cache_key = digest({"input": public_session(session, modality),
                                        "writer": identity["writer"], "settings": config["writer"],
                                        "prompt_hashes": identity["prompt_hashes"], "tokenizer": identity["tokenizer_id"],
                                        "request_options": identity["request_options"], "modality_policy": modality})
                    with key_lock_guard:
                        lock = key_locks.setdefault(cache_key, Lock())
                    with lock:
                        settings = {**config["writer"], "modality_policy": modality}
                        result = store.cached(cache_key)
                        if result and result["status"] == "fallback" and config.get("retry_fallback", True):
                            result = None
                        cached = result is not None
                        if result is None:
                            writer = DirectoryWriter(tokenizer, settings, pool.for_session(session))
                            result = writer.build(session)
                        # Cache validation includes global coverage and field budgets,
                        # independently of source addresses that are rebound per bank.
                        validate_build_result(result, session, tokenizer, settings)
                        directory = bind_directory(session, result, identity["writer"], identity["tokenizer_id"],
                                                   count_tokens=count_tokens)
                        if not cached:
                            store.cache(cache_key, result)
                    for trace in result.get("traces", []) if not cached else []:
                        store.trace({"event": "writer_semantic_trace", "bank_id": session["bank_id"],
                                     "parent_id": session["parent_id"], **trace})
                    return {"directory": directory, "cache_hit": cached, "stats": result.get("stats", {})}
                except Exception as exc:
                    # Continue other sessions; do not convert infrastructure/source
                    # failures into fabricated summaries or successful directories.
                    return {"error_type": type(exc).__name__, "reason": str(exc)}

            for (dataset, session, _), result in bounded_results(build, inputs(), config["workers"]):
                if "directory" in result:
                    store.commit(session, result["directory"], result["stats"])
                    status = result["directory"]["build"]["status"]
                    counters[status] += 1
                    counters["cache_hits" if result["cache_hit"] else "new_results"] += 1
                    if not result["cache_hit"]:
                        for name, value in result["stats"].items():
                            if type(value) is int:
                                counters["writer_" + name] += value
                else:
                    status = "failed"
                    store.failed(session, result["error_type"], result["reason"])
                    counters["failed"] += 1
                counters["sessions_processed"] += 1
                announce("parent_child_session_progress", dataset=dataset, bank_id=session["bank_id"],
                         parent_id=session["parent_id"], status=status, **dict(counters))
            current = store.report()
            execution["run_status"] = ("incomplete" if current["states"]["failed"] or current["states"]["pending"]
                                       or current["source_failures"] else "finished")
        except BaseException as exc:
            execution["run_status"] = "interrupted"
            execution["fatal_error"] = {"error_type": type(exc).__name__, "reason": str(exc)}
            raise
        finally:
            try:
                try:
                    if pool is not None:
                        pool.close()
                finally:
                    report = store.export({**dict(counters), **execution,
                                           "elapsed_seconds": time.monotonic() - started})
            finally:
                store.close()
        announce("parent_child_build_finished", output_root=str(root), sessions=report["sessions"],
                 states=report["states"], source_failures=report["source_failures"])
        if report["states"]["failed"] or report["states"]["pending"] or report["source_failures"]:
            raise InfrastructureError(f"Some sessions failed or remain pending; inspect {root / 'build_report.json'} "
                                      "and retry_failed.json. Completed sessions are resumable.")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--datasets", nargs="+", choices=["locomo", "split_longmemeval_s", "split_longmemeval_m"])
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"])
    urls = parser.add_mutually_exclusive_group()
    urls.add_argument("--base-url")
    urls.add_argument("--base-urls", nargs="+")
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and identities without writing or inference")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
