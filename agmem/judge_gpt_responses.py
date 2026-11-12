"""Judge saved GPT Task answers with GPT-5-mini and the frozen training prompt.

Run on a networked CPU host. The original Reader, Task answers and local Qwen
Judge results are read-only; remote Judge results have a separate output root.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path

from .core import Example, atomic_json, digest
from .evaluate_checkpoints import EpisodeJournal, load_suite, validate_selections, write_summary
from .evaluation_offline import judge_remote_answers
from .jobs import file_lock
from .judges import load_judge_templates
from .pipeline import announce
from .prompts import task_input_contract, task_messages
from .remote_task_runner import load_bundle, load_response_records
from .rewards import RewardScorer


def load_inputs(suite, label, source, responses_root=None):
    """Validate the complete pairing before making any paid Judge requests."""
    source_target = Path(suite["output_root"]) / label / source
    retrieval_target = Path(suite.get("retrieval_reuse_root", suite["output_root"])) / label / source
    path = retrieval_target / "retrieval"
    manifest = json.loads((path / "manifest.json").read_text())
    marker = json.loads((path / "complete.json").read_text())
    if marker != {"manifest_sha256": digest(manifest), "n": len(manifest["sample_ids"])}:
        raise ValueError("Reader retrieval is incomplete or has a stale completion marker")
    retrieval = EpisodeJournal(path, manifest)
    prepared_path = (Path(suite["required_prepared_sources"][source]) if "required_prepared_sources" in suite
                     else Path(suite["prepared_root"]) / source)
    example_path = prepared_path / "test_examples.jsonl"
    with example_path.open(encoding="utf-8") as stream:
        all_examples = [Example(**json.loads(line)) for line in stream if line.strip()]
    by_id = {example.sample_id: example for example in all_examples}
    if len(by_id) != len(all_examples):
        raise ValueError("Duplicate prepared test example IDs")
    examples = [by_id[sample_id] for sample_id in retrieval.sample_ids]
    if (any(e.source != source or e.split != "test" for e in examples)
            or digest([asdict(e) for e in examples]) != manifest["examples_sha256"]):
        raise ValueError("Prepared test questions/gold answers differ from the original evaluation")
    bundle_path = source_target / "offline" / "requests.json.gz"
    bundle = load_bundle(bundle_path)
    bundle_manifest = bundle["manifest"]
    if (bundle_manifest["checkpoint"] != label or bundle_manifest["dataset"] != source
            or bundle_manifest["sample_ids"] != retrieval.sample_ids
            or bundle_manifest["retrieval_manifest_sha256"] != retrieval.signature):
        raise ValueError("GPT Task inputs do not belong to the selected checkpoint/dataset")
    prompt_version = suite.get("task_prompt_version", "legacy_v1")
    if bundle_manifest["task_input_contract"] != task_input_contract(prompt_version):
        raise ValueError("GPT Task prompt differs from the selected evaluation")
    if bundle_manifest["protocol"] == "checkpoint_task_requests_v2":
        if (prompt_version != "history_answer_v1"
                or bundle_manifest.get("source_kind") != "retrieval_reprompt"
                or bundle_manifest["source_task_manifest_sha256"] is not None):
            raise ValueError("Invalid re-answer source provenance")
        paired = bundle_manifest.get("paired_local_task_manifest_sha256")
        if paired is not None:
            from .evaluation_offline import _completed_journal
            local_manifest, local = _completed_journal(source_target / "answers/local")
            if (paired != local.signature or local.sample_ids != retrieval.sample_ids
                    or local_manifest.get("retrieval_manifest_sha256") != retrieval.signature
                    or local_manifest.get("task_input_contract") != bundle_manifest["task_input_contract"]):
                raise ValueError("Paired local Task provenance differs from the exported GPT inputs")
    else:
        local_manifest = json.loads((source_target / "answers/local/manifest.json").read_text())
        if bundle_manifest["source_task_manifest_sha256"] != digest(local_manifest):
            raise ValueError("GPT Task inputs do not belong to the selected checkpoint/dataset")
    responses = (Path(responses_root) / label / source if responses_root
                 else source_target / "offline" / "responses")
    response_manifest, records = load_response_records(bundle, responses, require_complete=True)
    for example in examples:
        value = retrieval.read(example.sample_id)
        if (value is None or value.get("memory_id") != example.memory_id
                or digest(task_messages(example, value["evidence"], prompt_version=prompt_version))
                != records[example.sample_id]["task_input_hash"]):
            raise ValueError(f"Saved evidence does not match the GPT answer: {example.sample_id}")
    counts = Counter(row["completion"]["finish_reason"] for row in records.values())
    return {
        "label": label, "source": source, "examples": examples, "retrieval": retrieval,
        "bundle_path": bundle_path, "responses": responses,
        "response_manifest_sha256": digest(response_manifest),
        "responses_sha256": digest(records),
        "n": len(records), "finish_reasons": dict(counts),
        "truncated": sum(row["completion"]["truncated"] for row in records.values()),
        "empty": sum(not row["completion"]["text"].strip() for row in records.values()),
    }


def output_root(suite, override=None):
    source = Path(suite["output_root"]).resolve()
    output = Path(override).resolve() if override else source.parent / "gpt_judge_results"
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("GPT Judge output must be separate from the original evaluation results")
    if suite.get("retrieval_reuse_root"):
        original = Path(suite["retrieval_reuse_root"]).resolve()
        if output == original or original in output.parents or output in original.parents:
            raise ValueError("GPT Judge output must be separate from the original Reader evaluation")
    return output


def run(args):
    suite, config, saved = load_suite(args.config)
    datasets = args.datasets or suite["datasets"]
    labels = args.checkpoints or list(suite["checkpoints"])
    validate_selections(suite, datasets, labels, ["remote"])
    if type(args.workers) is not int or args.workers < 1:
        raise ValueError("--workers must be a positive integer")
    if (isinstance(args.timeout, bool) or not isinstance(args.timeout, (int, float))
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        raise ValueError("--timeout must be positive and finite")
    if type(args.retries) is not int or args.retries < 0:
        raise ValueError("--retries must be a nonnegative integer")
    network_retries = getattr(args, "network_retries", 6)
    network_retry_delay = getattr(args, "network_retry_delay", 30.0)
    if type(network_retries) is not int or network_retries < 0:
        raise ValueError("--network-retries must be a nonnegative integer")
    if (isinstance(network_retry_delay, bool) or not isinstance(network_retry_delay, (int, float))
            or not math.isfinite(network_retry_delay) or network_retry_delay <= 0):
        raise ValueError("--network-retry-delay must be positive and finite")
    if config["reward"].get("likelihood", {}).get("mode", "off") != "off":
        raise ValueError("The CPU-only remote Judge requires the evaluation likelihood mode to be off")
    _, _, hashes = load_judge_templates(config["reward"]["judge"])
    if hashes != saved["scoring_protocol"]["judge"]["template_sha256"]:
        raise ValueError("GPT Judge prompt differs from the saved training prompt")
    output = output_root(suite, args.output_root)
    inputs = [load_inputs(suite, label, source, args.responses_root) for label in labels for source in datasets]
    for item in inputs:
        announce("gpt_judge_inputs_verified", checkpoint=item["label"], dataset=item["source"],
                 n=item["n"], finish_reasons=item["finish_reasons"],
                 truncated=item["truncated"], empty=item["empty"],
                 judge_template_sha256=hashes)
    if args.check_only:
        return inputs

    from .remote_judge import RemoteJudgeScorer
    remote = deepcopy(suite["remote_task"])
    remote.update(model="gpt-5-mini", model_revision="gpt-5-mini",
                  timeout_seconds=args.timeout, retries=args.retries)
    for item in inputs:
        target = output / item["label"] / item["source"]
        with file_lock(target / ".eval.lock"), ExitStack() as cleanup:
            judge = RemoteJudgeScorer(remote, config["reward"]["judge"], target / "judge_calls",
                                      network_retries=network_retries,
                                      network_retry_delay=network_retry_delay)
            cleanup.callback(judge.close)
            scorer = RewardScorer(config["reward"], judge=judge,
                                  max_ids=config["retrieval"]["max_ids_per_call"])
            source_manifest = {
                "protocol": "gpt_judge_saved_answers_v1", "checkpoint": item["label"],
                "dataset": item["source"], "response_manifest_sha256": item["response_manifest_sha256"],
                "responses_sha256": item["responses_sha256"], "n": item["n"],
                "judge": judge.provenance(),
                "training_judge_template_sha256": hashes,
            }
            source_path = target / "source_manifest.json"
            if source_path.exists() and json.loads(source_path.read_text()) != source_manifest:
                raise ValueError("GPT Judge inputs/protocol changed; choose a fresh --output-root")
            atomic_json(source_path, source_manifest)
            judge_remote_answers(config, None, item["examples"], item["retrieval"], target,
                                 item["bundle_path"], item["responses"], scorer.provenance(),
                                 workers=args.workers, reward_scorer=scorer)
            write_summary(output)
    announce("gpt_judge_finished", summary=str(output / "summary.json"))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_locomo_longmemeval_checkpoints.yaml")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--responses-root", help="Optional imported response tree organized as checkpoint/dataset")
    parser.add_argument("--output-root", help="Separate result root; default: gpt_judge_results beside results")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--network-retries", type=int, default=6,
                        help="Extra recovery rounds after exhausted network retries; 0 disables recovery")
    parser.add_argument("--network-retry-delay", type=float, default=30,
                        help="Initial network recovery delay in seconds; grows to at most 60")
    parser.add_argument("--check-only", action="store_true", help="Validate complete answers and frozen prompts without API calls")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
