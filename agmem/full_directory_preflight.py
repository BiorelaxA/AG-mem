"""Read-only CPU checks for complete initial full-directory Reader inputs."""

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median

from .core import CapacityError
from .full_directory_prompts import full_directory_reader_messages
from .hierarchical.reader import DirectoryReader
from .policy import chat_token_ids, load_tokenizer


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _summary(rows):
    lengths = sorted(row["prompt_tokens"] for row in rows)
    if not lengths:
        return {"queries": 0, "min": None, "median": None, "p95": None, "max": None,
                "max_sample_id": None, "overflow_count": 0, "overflow_sample_ids": []}
    largest = max(rows, key=lambda row: row["prompt_tokens"])
    return {"queries": len(rows), "min": lengths[0], "median": median(lengths),
            "p95": lengths[max(0, math.ceil(0.95 * len(lengths)) - 1)], "max": lengths[-1],
            "max_sample_id": largest["sample_id"],
            "overflow_count": sum(row["overflow"] for row in rows),
            "overflow_sample_ids": [row["sample_id"] for row in rows if row["overflow"]]}


def initial_context_report(config, store, examples_by_split):
    """Count every public question with its complete directory and output reserve.

    Frozen-input integrity is validated by the launcher's existing preflight.
    Here each memory's public view is projected once, using the same projections
    as DirectoryReader.render. Source text and private scoring fields never enter
    the view. Only a tokenizer is loaded; neither model weights nor services are
    used, and the store is never closed or changed by this function.

    The report proves token capacity for the *initial* input, not GPU memory
    capacity or the size of later inputs containing retrieved evidence/history.
    An overflow raises CapacityError with the complete report in ``exc.report``.
    """
    if (config.get("data", {}).get("directory_protocol") != "hierarchical_v1"
            or config.get("reader", {}).get("prompt_version") != "hierarchical_full_v1"
            or config.get("reward", {}).get("formula") != "full_directory_orm_v1"):
        raise ValueError("Initial full-directory preflight requires full_directory_orm_v1")
    retrieval, settings = config["retrieval"], config["reader"]
    if type(retrieval.get("max_ids_per_call")) is not int or retrieval["max_ids_per_call"] != 2:
        raise ValueError("Full-directory preflight requires max_ids_per_call=2")
    max_calls = _positive_integer(retrieval["max_read_calls"], "retrieval.max_read_calls")
    reserve = _positive_integer(settings["max_new_tokens_per_decision"],
                                "reader.max_new_tokens_per_decision")
    model_path = Path(config["models"]["reader_path"]).expanduser()
    model_config = json.loads((model_path / "config.json").read_text())
    native_limit = model_config.get("text_config", model_config).get("max_position_embeddings")
    if native_limit is not None:
        _positive_integer(native_limit, "Reader native max_position_embeddings")
    limit = settings.get("max_context_tokens")
    if limit is None:
        limit = native_limit
    limit = _positive_integer(limit, "Reader context limit")
    tokenizer = load_tokenizer(str(model_path))
    version = store.get_metadata("directory_version")
    views, rows = {}, []
    by_source, by_split = defaultdict(list), defaultdict(list)
    for split, examples in examples_by_split.items():
        by_split[split]  # Preserve explicitly requested empty splits in the report.
        for example in examples:
            if example.memory_id not in views:
                directories, _ = store.hierarchical_bank(example.memory_id, version)
                ordered = sorted(directories, key=lambda directory: directory["order"])
                views[example.memory_id] = {
                    "parents": [DirectoryReader._parent_view(parent) for parent in ordered],
                    "child_directories": [
                        {"parent_id": parent["id"],
                         "children": [DirectoryReader._child_view(child) for child in parent["children"]]}
                        for parent in ordered],
                }
            messages = full_directory_reader_messages(
                example, views[example.memory_id], [], max_ids=2, max_calls=max_calls)
            length = len(chat_token_ids(tokenizer, messages))
            row = {"sample_id": example.sample_id, "prompt_tokens": length,
                   "overflow": length + reserve > limit}
            rows.append(row)
            by_source[example.source].append(row)
            by_split[split].append(row)
    report = {"protocol": "full_directory_initial_context_v1",
              "scope": "initial_reader_prompt_with_empty_evidence",
              "reader_prompt_version": "hierarchical_full_v1",
              "directory_protocol": "hierarchical_v1", "directory_version": version,
              "native_context_limit": native_limit, "context_limit": limit,
              "output_token_reserve": reserve, "memories": len(views),
              "percentile_method": "nearest_rank", "gpu_memory_checked": False,
              **_summary(rows),
              "by_source": {source: _summary(values) for source, values in sorted(by_source.items())},
              "by_split": {split: _summary(values) for split, values in by_split.items()}}
    if report["overflow_count"]:
        examples = ", ".join(report["overflow_sample_ids"][:5])
        error = CapacityError(
            f"{report['overflow_count']} initial full-directory Reader inputs exceed context limit={limit} "
            f"with output reserve={reserve}; max prompt={report['max']}; samples: {examples}. "
            "No truncation performed")
        error.report = report
        raise error
    return report
