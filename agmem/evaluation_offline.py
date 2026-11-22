"""Export exact saved Task inputs; judge returned answers without a remote API.

Only public Task messages cross the network boundary. Reader trajectories, gold
answers, reward computation and the frozen training Judge stay on the GPU host.
"""
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from .core import atomic_json, digest
from .prompts import task_input_contract, task_messages


def _prompt_version(config):
    return config.get("task_agent", {}).get("prompt_version", "legacy_v1")


def _reprompt_source(examples, retrieval, target, dataset, contract):
    """Bind a new answer run to completed retrieval and its original test data."""
    source = Path(retrieval.root).resolve().parent
    target = Path(target).resolve()
    manifest, completed = _completed_journal(retrieval.root)
    same_prompt_pair = target == source and manifest.get("task_input_contract") == contract
    if not same_prompt_pair and (target == source or source in target.parents or target in source.parents):
        raise ValueError("Re-prompted answers require a separate output directory from source retrieval")
    if (completed.signature != retrieval.signature
            or [example.sample_id for example in examples] != completed.sample_ids
            or any(example.source != dataset or example.split != "test" for example in examples)
            or manifest.get("examples_sha256") != digest([asdict(example) for example in examples])):
        raise ValueError("Re-prompted evaluation questions/gold/dataset differ from source retrieval")
    for kind in ("local", "remote"):
        path = target / "answers" / kind / "manifest.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if (saved.get("task_input_contract") != contract
                    or saved.get("retrieval_manifest_sha256") != retrieval.signature):
                raise ValueError("Existing answers use another Task prompt; choose a new output directory")
    return completed


def _answer_and_metadata(completion, prompt_version):
    metadata = deepcopy({key: value for key, value in completion.items() if key != "text"})
    text = completion["text"]
    if prompt_version == "history_answer_v1":
        from .remote_task_runner import extract_tagged_answer
        if completion.get("finish_reason") != "stop" or completion.get("truncated") is not False:
            raise ValueError("Tagged Task answers require a complete stop response")
        prediction = extract_tagged_answer(text)
        metadata.update(raw_text=text, answer_parser="answer_tag_v1")
    else:
        prediction = text
    return prediction, metadata


def _completed_journal(root):
    from .evaluate_checkpoints import EpisodeJournal
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    done = json.loads((root / "complete.json").read_text())
    if done != {"manifest_sha256": digest(manifest), "n": len(manifest["sample_ids"])}:
        raise ValueError(f"Incomplete/stale source evaluation: {root}")
    journal = EpisodeJournal(root, manifest)
    if any(not journal.path(sample_id).is_file() for sample_id in journal.sample_ids):
        raise ValueError(f"Missing completed evaluation rows: {root}")
    return manifest, journal


def _request_settings(config, remote):
    if remote.get("request_profile") != "openai_chat":
        raise ValueError("Offline Task export requires the openai_chat request profile")
    # TaskAgent.complete uses its service's default seed (42), independently
    # of the Reader sampling seed. Defaults match ChatService exactly.
    return {
        "model": remote["model"], "model_revision": remote["model_revision"],
        "request_profile": "openai_chat", "send_seed": remote.get("send_seed", False),
        "temperature": remote.get("temperature", 0.0),
        "max_tokens_parameter": remote.get("max_tokens_parameter", "max_tokens"),
        "max_output_tokens": config["task_agent"]["max_output_tokens"], "seed": 42,
    }


def export_task_requests(config, examples, retrieval, target, remote, checkpoint, dataset):
    """Publish immutable public Task inputs for the selected answering prompt.

    Legacy messages are checked against the saved Qwen Task inputs. A new
    history-answer run instead binds directly to completed Reader evidence and
    the original test examples, allowing export before local answers exist.
    No remote endpoint, API credentials or model requests are needed here.
    Caller owns the evaluation dataset/target locks.
    """
    from .pipeline import announce
    from .remote_task_runner import load_bundle, validate_bundle
    target = Path(target)
    prompt_version = _prompt_version(config)
    contract = task_input_contract(prompt_version)
    reprompt = prompt_version == "history_answer_v1"
    sample_ids = [example.sample_id for example in examples]
    if reprompt:
        _reprompt_source(examples, retrieval, target, dataset, contract)
        local = None
        # A fresh evaluation can retrieve and answer with the tagged prompt in
        # one output tree. This is not permission to relabel legacy answers:
        # the retrieval contract above must already match, and local inputs are
        # checked below before exporting exactly the same evidence/messages.
        if target.resolve() == Path(retrieval.root).resolve().parent:
            local_manifest, local = _completed_journal(target / "answers" / "local")
            if (sample_ids != local.sample_ids
                    or local_manifest.get("retrieval_manifest_sha256") != retrieval.signature
                    or local_manifest.get("task_input_contract") != contract
                    or local_manifest.get("task_max_output_tokens") != config["task_agent"]["max_output_tokens"]):
                raise ValueError("Local evaluation and exported Task inputs have different provenance")
    else:
        local_manifest, local = _completed_journal(target / "answers" / "local")
        _, completed_retrieval = _completed_journal(retrieval.root)
        if (sample_ids != local.sample_ids or sample_ids != retrieval.sample_ids
                or completed_retrieval.signature != retrieval.signature
                or local_manifest.get("retrieval_manifest_sha256") != retrieval.signature
                or local_manifest.get("task_input_contract") != contract
                or local_manifest.get("task_max_output_tokens") != config["task_agent"]["max_output_tokens"]):
            raise ValueError("Local evaluation and exported Task inputs have different provenance")
    requests = []
    for example in examples:
        row = retrieval.read(example.sample_id)
        answer = local.read(example.sample_id) if local is not None else None
        if (row is None or row.get("memory_id") != example.memory_id
                or local is not None and (answer is None
                    or answer.get("memory_id") != example.memory_id
                    or row["evidence"] != answer["evidence"])):
            raise ValueError(f"Local/retrieval evidence mismatch: {example.sample_id}")
        messages = task_messages(example, row["evidence"], prompt_version=prompt_version)
        input_hash = digest(messages)
        if answer is not None and answer.get("task_input_hash") != input_hash:
            raise ValueError(f"Saved local Task input hash mismatch: {example.sample_id}")
        requests.append({"sample_id": example.sample_id, "task_input_hash": input_hash,
                         "messages": messages})
    bundle = {"manifest": {
        "protocol": "checkpoint_task_requests_v2" if reprompt else "checkpoint_task_requests_v1",
        **({"source_kind": "retrieval_reprompt"} if reprompt else {}), "sample_ids": sample_ids,
        "checkpoint": checkpoint, "dataset": dataset,
        "retrieval_manifest_sha256": retrieval.signature,
        "source_task_manifest_sha256": local.signature if local is not None and not reprompt else None,
        **({"paired_local_task_manifest_sha256": local.signature} if reprompt and local is not None else {}),
        "task_input_contract": contract, "requests_sha256": digest(requests),
        "request_settings": _request_settings(config, remote),
    }, "requests": requests}
    validate_bundle(bundle)
    path = target / "offline" / "requests.json.gz"
    if path.exists():
        if load_bundle(path) != bundle:
            raise ValueError(f"Exported Task protocol changed; choose a new output directory: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(bundle, stream, ensure_ascii=False, allow_nan=False)
        temporary.replace(path)
    announce("test_remote_inputs_exported", checkpoint=checkpoint, dataset=dataset,
             n=len(requests), path=str(path.resolve()),
             request_manifest_sha256=digest(bundle["manifest"]))
    return path


class SavedTaskAnswers:
    """TaskAgent-compatible answer reader. It has no HTTP client or credentials."""

    def __init__(self, manifest, records, prompt_version="legacy_v1"):
        self.records = records
        task_input_contract(prompt_version)
        self.prompt_version = prompt_version
        self.service = SimpleNamespace(
            identity=deepcopy(manifest["task"]["identity"]),
            config=deepcopy(manifest["task"]["request_options"]),
            client=SimpleNamespace(close=lambda: None))

    def answer(self, example, evidence, *, metadata=None):
        record = self.records[example.sample_id]
        input_hash = digest(task_messages(example, evidence, prompt_version=self.prompt_version))
        if input_hash != record["task_input_hash"]:
            raise ValueError(f"Returned answer does not match Task input: {example.sample_id}")
        prediction, completion_metadata = _answer_and_metadata(record["completion"], self.prompt_version)
        if metadata is not None:
            metadata.clear()
            metadata.update(completion_metadata)
        return prediction, input_hash


def judge_remote_answers(config, store, examples, retrieval, target, bundle_path,
                         responses, expected_scoring, workers=2, *, reward_scorer=None):
    """Validate all returned inputs first, then resume the usual fixed local Judge."""
    from .evaluate_checkpoints import EpisodeJournal, score_evidence
    from .remote_task_runner import load_bundle, load_response_records
    target = Path(target)
    bundle = load_bundle(bundle_path)
    manifest = bundle["manifest"]
    prompt_version = _prompt_version(config)
    contract = task_input_contract(prompt_version)
    sample_ids = [example.sample_id for example in examples]
    if (manifest["sample_ids"] != sample_ids
            or manifest["retrieval_manifest_sha256"] != retrieval.signature
            or manifest["task_input_contract"] != contract
            or manifest["request_settings"]["max_output_tokens"] != config["task_agent"]["max_output_tokens"]):
        raise ValueError("Returned Task bundle belongs to another evaluation protocol")
    if prompt_version == "history_answer_v1":
        _reprompt_source(examples, retrieval, target, manifest["dataset"], contract)
    response_manifest, records = load_response_records(bundle, responses, require_complete=True)
    # Validate all evidence before making any Judge requests, even on a resumed
    # import whose existing answers would otherwise skip TaskAgent.answer().
    for example, request in zip(examples, bundle["requests"]):
        row = retrieval.read(example.sample_id)
        if row is None or row.get("memory_id") != example.memory_id:
            raise ValueError(f"Missing/mismatched retrieval: {example.sample_id}")
        if digest(task_messages(example, row["evidence"], prompt_version=prompt_version)) != request["task_input_hash"]:
            raise ValueError(f"Returned Task evidence mismatch: {example.sample_id}")
    output = target / "answers" / "remote"
    if (output / "manifest.json").exists():
        old_manifest = json.loads((output / "manifest.json").read_text())
        if prompt_version == "legacy_v1" and old_manifest.get("task") != response_manifest["task"]:
            # A failed online attempt may have created an empty v1 answer
            # journal. Its identity can adopt the documented retry policy only
            # before ANY scored episode exists. Existing scores never change.
            old_task = deepcopy(response_manifest["task"])
            retry_policy = old_task.get("request_options", {}).pop("empty_answer_retry", None)
            if (retry_policy is not None and old_manifest.get("task") == old_task
                    and old_manifest.get("sample_ids") == sample_ids
                    and old_manifest.get("retrieval_manifest_sha256") == retrieval.signature
                    and not any((output / "episodes").glob("*.json.gz"))
                    and not (output / "complete.json").exists()
                    and not (output / "metrics.json").exists()):
                atomic_json(target / "offline" / "legacy_remote_answer_manifest.json", old_manifest)
                old_manifest["task"] = deepcopy(response_manifest["task"])
                atomic_json(output / "manifest.json", old_manifest)
        if (old_manifest.get("sample_ids") != sample_ids
                or old_manifest.get("retrieval_manifest_sha256") != retrieval.signature
                or old_manifest.get("task_input_contract") != contract
                or old_manifest.get("task") != response_manifest["task"]):
            raise ValueError("Existing remote evaluation has different Task provenance; use a new output root")
        previous = EpisodeJournal(output, old_manifest)
        for sample_id in sample_ids:
            old = previous.read(sample_id)
            record = records[sample_id]
            prediction, completion_metadata = _answer_and_metadata(record["completion"], prompt_version)
            if old is not None and (
                    old["prediction"] != prediction
                    or old["task_input_hash"] != record["task_input_hash"]
                    or old["task_completion"] != completion_metadata):
                raise ValueError(f"Already-scored remote answer differs from imported response: {sample_id}")
    return score_evidence(config, store, examples, retrieval, output, expected_scoring,
                          workers=workers, answer_agent=SavedTaskAnswers(response_manifest, records,
                                                                         prompt_version=prompt_version),
                          reward_scorer=reward_scorer)
