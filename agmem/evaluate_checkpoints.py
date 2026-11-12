"""Resumable checkpoint x dataset x answering-model evaluation.

The Reader retrieves once per checkpoint/dataset. Answering models consume the
same immutable evidence. Writer, Judge, retrieval limits and prompts come from
the saved training run, not from the current training YAML.
"""
from contextlib import ExitStack, closing
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import gc
import gzip
import json
from datetime import datetime, timezone
import os

from .core import atomic_json, digest, trajectory_from_dict
from .jobs import file_lock
from .pipeline import (announce, directory_version, make_environment,
                       make_reward_scorer, make_task_agent, metrics,
                       open_prepared, validate_directories)
from .prompts import configured_reader_input_contract, task_input_contract


EVALUATION_DEFAULTS = {
    "reader_query_concurrency": 4,
    "reader_batch_size": 4,
    "reader_batch_max_tokens": 16384,
    "local_scoring_workers": 2,
    "remote_scoring_workers": 4,
}


def evaluation_settings(suite):
    """Evaluation scheduling is independent of training group/batch settings."""
    configured = suite.get("evaluation_execution", {})
    if not isinstance(configured, dict):
        raise ValueError("evaluation_execution must be a mapping")
    unknown = configured.keys() - EVALUATION_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown evaluation execution settings: {sorted(unknown)}")
    settings = {**EVALUATION_DEFAULTS, **configured}
    for name, value in settings.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"evaluation_execution.{name} must be a positive integer")
    return settings


class EpisodeJournal:
    """One atomic compressed file per example; partial files are never completed rows."""

    def __init__(self, root, manifest):
        self.root = Path(root)
        path = self.root / "manifest.json"
        self.sample_ids = manifest["sample_ids"]
        if not self.sample_ids or len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Evaluation requires nonempty unique sample IDs")
        self.indices = {sample_id: i for i, sample_id in enumerate(self.sample_ids)}
        self.signature = digest(manifest)
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError(f"Evaluation protocol changed; choose a new output directory: {self.root}")
        else:
            unexpected = [p for p in self.root.iterdir() if p.name != ".eval.lock"] if self.root.exists() else []
            if unexpected:
                raise ValueError(f"Evaluation artifacts have no manifest: {self.root}")
            atomic_json(path, manifest)
        (self.root / "episodes").mkdir(exist_ok=True)
        expected = {f"{i:08d}.json.gz" for i in range(len(self.sample_ids))}
        if any(p.name not in expected for p in (self.root / "episodes").glob("*.json.gz")):
            raise ValueError(f"Unknown episode file in {self.root}")

    def path(self, sample_id):
        return self.root / "episodes" / f"{self.indices[sample_id]:08d}.json.gz"

    def read(self, sample_id):
        path = self.path(sample_id)
        if not path.exists():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            record = json.load(stream)
        if (record.get("manifest_sha256") != self.signature
                or record.get("sample_id") != sample_id
                or record.get("trajectory", {}).get("sample_id") != sample_id):
            raise ValueError(f"Episode provenance mismatch: {path}")
        return record["trajectory"]

    def write(self, row):
        path = self.path(row.sample_id)
        temporary = path.with_suffix(path.suffix + ".tmp")
        record = {"manifest_sha256": self.signature, "sample_id": row.sample_id,
                  "trajectory": row.to_dict()}
        with gzip.open(temporary, "wt", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, allow_nan=False)
        temporary.replace(path)

    def finish(self):
        if any(not self.path(sample_id).is_file() for sample_id in self.sample_ids):
            raise ValueError("Cannot mark an incomplete evaluation complete")
        atomic_json(self.root / "complete.json", {"manifest_sha256": self.signature, "n": len(self.sample_ids)})


def load_suite(path):
    import yaml
    suite = yaml.safe_load(Path(path).read_text())
    evaluation_settings(suite)
    saved = json.loads((Path(suite["training_run"]) / "run.json").read_text())
    config = deepcopy(saved["config"])
    if config["retrieval"]["max_ids_per_call"] != 1:
        raise ValueError("This evaluation requires the trained one-ID-per-READ protocol")
    if config["reward"].get("formula") != "judge_cost_v2":
        raise ValueError("Expected the training run's LLM-as-judge reward protocol")
    if configured_reader_input_contract(config) != saved["reader_input_contract"]:
        raise ValueError("Current Reader prompt differs from the selected training run")
    if task_input_contract() != saved["task_input_contract"]:
        raise ValueError("Current Task prompt differs from the selected training run")
    # An explicit evaluation override permits re-answering frozen Reader evidence
    # without relabelling old answers or changing the training protocol.
    prompt_version = suite.get("task_prompt_version", "legacy_v1")
    if prompt_version != "legacy_v1":
        task_input_contract(prompt_version)  # Validate before creating any output.
        config["task_agent"]["prompt_version"] = prompt_version
    if "task_max_output_tokens" in suite:
        budget = suite["task_max_output_tokens"]
        if type(budget) is not int or budget < 1:
            raise ValueError("task_max_output_tokens must be a positive integer")
        config["task_agent"]["max_output_tokens"] = budget
    if suite.get("retrieval_reuse_root"):
        source_root = Path(suite["retrieval_reuse_root"]).resolve()
        output_root = Path(suite["output_root"]).resolve()
        if (source_root == output_root or source_root in output_root.parents
                or output_root in source_root.parents):
            raise ValueError("Re-answer output_root must be separate from retrieval_reuse_root")
    config["data"]["input_root"] = str(Path(suite["input_root"]).resolve())
    config["service"]["request_base_url"] = suite["local_base_url"]
    judge = config["reward"].setdefault("judge", {})
    judge["template_file"] = str(Path(suite["judge_template_file"]).resolve())
    judge["expected_template_sha256"] = saved["scoring_protocol"]["judge"]["template_sha256"]
    config["task_agent"].pop("service", None)
    return suite, config, saved


def checkpoint_model(suite, label):
    checkpoint = Path(suite["checkpoints"][label])
    if not checkpoint.is_absolute():
        checkpoint = Path(suite["training_run"]) / checkpoint
    checkpoint = checkpoint.resolve()
    expected_step = suite.get("checkpoint_steps", {}).get(label)
    if expected_step is not None:
        if (type(expected_step) is not int or expected_step < 0
                or checkpoint.name != f"checkpoint-{expected_step:08d}"
                or checkpoint.parent != Path(suite["training_run"]).resolve()
                or not (checkpoint / "trainer.pt").is_file()):
            raise ValueError(f"Checkpoint does not match the required training run/step: {label}")
    model = checkpoint / "model"
    weights = list(model.glob("*.safetensors"))
    if not (model / "config.json").is_file() or not weights or any(p.stat().st_size == 0 for p in weights):
        raise ValueError(f"Missing checkpoint model: {model}")
    index = model / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text()).get("weight_map", {}).values())
        if (not shards or any(not isinstance(name, str) or Path(name).name != name
                             or not (model / name).is_file() or (model / name).stat().st_size == 0
                             for name in shards)):
            raise ValueError(f"Incomplete checkpoint weight shards: {model}")
    return model.resolve()


def validate_selections(suite, datasets, labels, task_agents):
    for name, values, allowed in (("datasets", datasets, suite["datasets"]),
                                   ("checkpoints", labels, suite["checkpoints"]),
                                   ("task_agents", task_agents, {"local", "remote"})):
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) for value in values)
                or len(set(values)) != len(values) or not set(values).issubset(allowed)):
            raise ValueError(f"Select nonempty, unique, configured {name}")
    for source in datasets:
        if not source or source in {".", "..", "_runtime"} or Path(source).name != source:
            raise ValueError(f"Invalid evaluation source: {source}")
    if "required_prepared_sources" in suite:
        required = suite["required_prepared_sources"]
        if (not isinstance(required, dict) or not set(datasets).issubset(required)
                or any(not isinstance(required[source], str) or not required[source] for source in datasets)):
            raise ValueError("required_prepared_sources must provide an existing store path for every selected dataset")


def select_examples(config, store, limit=None):
    from .data import task_examples
    examples = task_examples(config, store, "test")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        examples = examples[:limit]
    if not examples:
        raise ValueError("Empty test split")
    return examples


def retrieval_manifest(config, store, model, examples, limit=None):
    from .policy import model_fingerprint
    version = directory_version(store)
    snapshots = {}
    for memory_id in sorted({example.memory_id for example in examples}):
        entries = store.directory(memory_id, version)
        snapshots[memory_id] = {"count": len(entries), "sha256": digest(entries)}
    return {
        "protocol": "checkpoint_test_retrieval_v1", "split": "test", "limit": limit,
        "sample_ids": [e.sample_id for e in examples],
        "examples_sha256": digest([asdict(e) for e in examples]),
        "checkpoint_model": str(model), "reader": model_fingerprint(model),
        "prepare_identity": store.get_metadata("prepare_identity"),
        "directory_version": version, "directory_snapshots": snapshots,
        "directory_service_identity": store.get_metadata("directory_service_identity"),
        "reader_config": {**config["reader"], "rollout_temperature": 0.0},
        "reader_input_contract": configured_reader_input_contract(config),
        "task_input_contract": task_input_contract(config["task_agent"].get("prompt_version", "legacy_v1")),
        "retrieval": config["retrieval"], "seed": config["seed"],
        "shared_evidence_capacity": {
            "tokenizer": model_fingerprint(config["models"]["task_path"]),
            "max_context_tokens": config["task_agent"].get("max_context_tokens"),
            "max_output_tokens": config["task_agent"]["max_output_tokens"],
            "scope": "Both answering models receive identical evidence; capacity uses the training Qwen tokenizer.",
        },
    }


def iter_retrieved(config, store, policy, examples, journal, settings=None):
    """Yield saved/new evidence on the main thread while scoring runs in workers."""
    from .evaluation_retrieval import iter_retrievals
    settings = settings or evaluation_settings({})
    validate_directories(config, store, ["test"])
    pending, completed = [], 0
    for example in examples:
        value = journal.read(example.sample_id)
        if value is None:
            pending.append(example)
        else:
            completed += 1
            yield example, trajectory_from_dict(value)
    if pending:
        if policy is None:
            raise ValueError("Missing Reader policy for unfinished retrievals")
        env = make_environment(config, store, policy)
        old_temperature = policy.temperature
        policy.temperature = 0.0
        try:
            with closing(iter_retrievals(
                    env, policy, pending, seed=config["seed"],
                    query_concurrency=settings["reader_query_concurrency"],
                    batch_size=settings["reader_batch_size"],
                    max_batch_tokens=settings["reader_batch_max_tokens"])) as generated:
                for example, row in generated:
                    journal.write(row)
                    completed += 1
                    announce("test_retrieval_progress", sample_id=example.sample_id,
                             completed=completed, total=len(examples), calls=row.n_read_calls)
                    yield example, row
        finally:
            policy.temperature = old_temperature
    journal.finish()


def retrieve(config, store, policy, examples, journal, settings=None):
    for _ in iter_retrieved(config, store, policy, examples, journal, settings):
        pass


def task_provenance(service):
    # Only nonsecret, request-affecting fields enter reports. Credentials stay in env.
    fields = ("request_profile", "send_seed", "temperature", "max_tokens_parameter", "empty_answer_retry",
              "answer_parser", "answer_completion_policy")
    return {"identity": service.identity,
            "request_options": {key: service.config[key] for key in fields if key in service.config}}


def compact_for_metrics(row):
    """Retain metric semantics without holding every raw history/token array in RAM."""
    row.evidence = [{"id": item["id"]} for item in row.evidence]
    for step in row.decisions:
        step.messages = []
        step.completion.text = ""
        step.completion.prompt_ids = range(len(step.completion.prompt_ids))
        step.completion.token_ids = range(len(step.completion.token_ids))
        step.completion.old_logprobs = []
        if step.tool_result and "records" in step.tool_result:
            step.tool_result["records"] = [{"id": item["id"]} for item in step.tool_result["records"]]
    return row


def validate_evaluation_scoring(config, actual, expected):
    """Keep the frozen Judge/reward while allowing the explicit tagged Task.

    Even a disabled likelihood scorer records its Task prompt. An evaluation
    that explicitly uses history_answer_v1 must record the new prompt without
    claiming that it is the legacy training prompt. Only its three derived
    prompt fields may change, and only while likelihood remains off. All other
    fields (including the Judge, model identity and reward formula) stay exact.
    This compatibility is confined to evaluation; training resume is unchanged.
    """
    if expected is None or actual == expected:
        return
    compatible = deepcopy(expected)
    previous = compatible.get("likelihood")
    current = actual.get("likelihood")
    if (config.get("task_agent", {}).get("prompt_version", "legacy_v1") == "history_answer_v1"
            and config.get("reward", {}).get("likelihood", {}).get("mode", "off") == "off"
            and isinstance(previous, dict) and previous.get("mode") == "off"
            and isinstance(current, dict) and current.get("mode") == "off"
            and previous.get("task_input_contract") == task_input_contract("legacy_v1")
            and "assistant_prefix" not in previous and "target_encoding" not in previous):
        from .likelihood import ANSWER_CONTINUATION_PREFIX
        previous.update(task_input_contract=task_input_contract("history_answer_v1"),
                        assistant_prefix=ANSWER_CONTINUATION_PREFIX,
                        target_encoding="fixed_gold_ids_after_answer_prefix_v1")
    if actual != compatible:
        raise ValueError("Judge/reward provenance differs from the saved training run")


def score_evidence(config, store, examples, retrieval, output, expected_scoring=None,
                   workers=1, retrieval_rows=None, *, answer_agent=None, reward_scorer=None):
    """Run concurrent Task + fixed Judge on cached or freshly retrieved evidence."""
    from .evaluation_scoring import score_pending
    with ExitStack() as cleanup:
        if retrieval_rows is not None and hasattr(retrieval_rows, "close"):
            cleanup.callback(retrieval_rows.close)
        agent = answer_agent if answer_agent is not None else make_task_agent(config, store)
        cleanup.callback(agent.service.client.close)
        scorer = reward_scorer if reward_scorer is not None else make_reward_scorer(config, store)
        cleanup.callback(scorer.close)
        provenance = scorer.provenance()
        validate_evaluation_scoring(config, provenance, expected_scoring)
        manifest = {
            "protocol": "checkpoint_test_answers_v1", "split": "test",
            "sample_ids": [e.sample_id for e in examples],
            "retrieval_manifest_sha256": retrieval.signature,
            "task": task_provenance(agent.service),
            "task_max_output_tokens": config["task_agent"]["max_output_tokens"],
            "task_input_contract": task_input_contract(config["task_agent"].get("prompt_version", "legacy_v1")),
            "scoring_protocol": provenance, "reward_config": config["reward"],
        }
        journal = EpisodeJournal(output, manifest)
        rows, completed = [None] * len(examples), 0
        for index, example, row, is_new in score_pending(
                examples, retrieval, journal, agent, scorer, workers=workers,
                retrieval_rows=retrieval_rows):
            completed += 1
            if is_new:
                announce("test_answer_progress", sample_id=example.sample_id,
                         completed=completed, total=len(examples), judge_accuracy=row.judge_score)
            rows[index] = compact_for_metrics(row)
        report = metrics(rows)
        report.update(task=manifest["task"], task_input_contract=manifest["task_input_contract"],
                      reader_input_contract=configured_reader_input_contract(config),
                      scoring_protocol=provenance, retrieval_manifest_sha256=retrieval.signature,
                      evaluation_manifest_sha256=journal.signature,
                      split="test", complete=True)
        categories = {}
        for example, row in zip(examples, rows):
            kind = example.judge_metadata.get("question_type")
            if kind:
                if "_abs" in example.judge_metadata.get("question_id", ""):
                    kind = "abstention"
                categories.setdefault(kind, []).append(row)
        if categories:
            report["by_question_type"] = {name: metrics(values)["pooled"] for name, values in categories.items()}
        atomic_json(Path(output) / "metrics.json", report)
        journal.finish()
        return report


def write_summary(root):
    root = Path(root)
    # Independent dataset workers may finish at the same time. Serialize both
    # the snapshot scan and publication of the shared JSON/CSV summary.
    with file_lock(root / ".summary.lock", blocking=True):
        return _write_summary_unlocked(root)


def _write_summary_unlocked(root):
    rows = []
    for path in sorted(root.glob("*/*/answers/*/metrics.json")):
        done = path.parent / "complete.json"
        if not done.exists():
            continue
        report = json.loads(path.read_text())
        complete = json.loads(done.read_text())
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Completed evaluation has no manifest: {path}")
        manifest = json.loads(manifest_path.read_text())
        sample_ids = manifest.get("sample_ids", [])
        if (complete["manifest_sha256"] != report["evaluation_manifest_sha256"]
                or digest(manifest) != complete["manifest_sha256"]
                or not sample_ids or len(set(sample_ids)) != len(sample_ids)
                or complete.get("n") != len(sample_ids)
                or report["pooled"]["n"] != len(sample_ids)
                or not report.get("complete")
                or any(not (path.parent / "episodes" / f"{i:08d}.json.gz").is_file()
                       for i in range(len(sample_ids)))):
            raise ValueError(f"Incomplete/stale metrics: {path}")
        checkpoint, dataset, _, task = path.relative_to(root).parts[:-1]
        pooled = report["pooled"]
        rows.append({"checkpoint": checkpoint, "dataset": dataset, "task_agent": task,
                     **{key: pooled[key] for key in ("n", "judge_accuracy", "answer_f1", "reward",
                                                    "read_calls", "format_failure_rate", "format_reward")},
                     "metrics": str(path.resolve())})
    atomic_json(root / "summary.json", {"results": rows})
    fields = ["checkpoint", "dataset", "task_agent", "n", "judge_accuracy", "answer_f1",
              "reward", "read_calls", "format_failure_rate", "format_reward", "metrics"]
    temporary = root / "summary.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(root / "summary.csv")
    return rows


def run_suite(args):
    suite, base, saved = load_suite(args.config)
    datasets = args.datasets or suite["datasets"]
    labels = args.checkpoints or list(suite["checkpoints"])
    task_agents = args.task_agents or suite["task_agents"]
    validate_selections(suite, datasets, labels, task_agents)
    if args.stage == "summarize":
        # No Store access; write_summary owns the only lock this stage needs.
        return _run_suite(args, suite, base, saved)
    root = Path(suite["prepared_root"])
    with ExitStack() as locks:
        # Retain compatibility with an older worker holding the global lock
        # exclusively, while new workers can operate on disjoint dataset stores.
        locks.enter_context(file_lock(root / ".evaluation.lock", shared=True))
        for source in sorted(datasets):
            locks.enter_context(file_lock(root / ".evaluation_locks" / f"{source}.lock"))
        return _run_suite(args, suite, base, saved)


def _run_suite(args, suite, base, saved):
    from .evaluation_data import prepare_source, build_source_directories, load_existing_source
    settings = evaluation_settings(suite)
    if args.local_base_url:
        base["service"]["request_base_url"] = args.local_base_url
    if args.remote_model:
        suite["remote_task"].update(model=args.remote_model, model_revision=args.remote_model)
    datasets = args.datasets or suite["datasets"]
    labels = args.checkpoints or list(suite["checkpoints"])
    task_agents = args.task_agents or suite["task_agents"]
    validate_selections(suite, datasets, labels, task_agents)
    root = Path(suite["output_root"])
    if args.limit is not None:
        root = root / f"smoke-limit-{args.limit}"
    if args.stage == "summarize":
        print(json.dumps(write_summary(root), ensure_ascii=False, indent=2))
        return
    reuse_root = suite.get("retrieval_reuse_root")
    if reuse_root and (args.stage not in {"score", "export-remote", "judge-remote"} or args.limit is not None):
        raise ValueError("Frozen evidence re-answering supports full score/export-remote/judge-remote stages only")
    announce("test_execution_settings", **settings, pipeline=args.stage == "all")
    if "required_prepared_sources" in suite and args.stage == "prepare":
        # evaluate_gpu runs this CPU stage before deploying Qwen or the Reader.
        # Constructing provenance makes no model requests; fail here if a real
        # Judge/reward protocol change would otherwise stop the scored run.
        scorer = make_reward_scorer(base, None)
        try:
            validate_evaluation_scoring(base, scorer.provenance(), saved["scoring_protocol"])
        finally:
            scorer.close()
        announce("test_scoring_protocol_verified")
    if args.stage in {"all", "score"} and "remote" in task_agents:
        from .services import resolve_task_service
        remote_config = deepcopy(base)
        remote_config["task_agent"]["service"] = suite["remote_task"]
        resolve_task_service(remote_config)  # Fail before expensive work when credentials are missing.
    prepared = {}
    for source in datasets:
        if "required_prepared_sources" in suite:
            prepared[source] = load_existing_source(base, source, suite["required_prepared_sources"][source])
        else:
            prepared[source] = prepare_source(base, source, suite["prepared_root"],
                                              suite.get("reuse_roots", {}).get(source))
            if args.stage in {"all", "build"}:
                prepared[source] = build_source_directories(prepared[source])
        announce("test_data_status", **{k: v for k, v in prepared[source].items() if k != "config"})
    if args.stage in {"prepare", "build"}:
        return
    for source, result in prepared.items():
        if result["status"] != "ready":
            raise ValueError(f"Test directories need building for {source}; run --stage build")
    for label in labels:
        model = checkpoint_model(suite, label)
        policy = None
        try:
            for source, result in prepared.items():
                config = result["config"]
                store = open_prepared(config)
                try:
                    validate_directories(config, store, ["test"])
                    examples = select_examples(config, store, args.limit)
                    target = root / label / source
                    with file_lock(target / ".eval.lock"):
                        if reuse_root:
                            from .evaluation_offline import _completed_journal
                            retrieval_config = deepcopy(config)
                            retrieval_config["task_agent"] = deepcopy(saved["config"]["task_agent"])
                            expected_retrieval = retrieval_manifest(
                                retrieval_config, store, model, examples, args.limit)
                            old_manifest, retrieval = _completed_journal(
                                Path(reuse_root) / label / source / "retrieval")
                            if old_manifest != expected_retrieval:
                                raise ValueError("Frozen Reader evidence differs from the selected evaluation")
                        else:
                            retrieval = EpisodeJournal(target / "retrieval",
                                                       retrieval_manifest(config, store, model, examples, args.limit))
                        # Scheduling is recorded separately so completed serial episodes
                        # remain reusable. Greedy batching can have floating-point effects;
                        # each new decision also records its actual sampling_batch_size.
                        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                        atomic_json(target / "executions" / f"{stamp}-{os.getpid()}.json", {
                            "scheduler": "cross_query_greedy_v1", "settings": settings,
                            "stage": args.stage,
                            "task_agents": ["remote"] if args.stage == "judge-remote" else task_agents,
                            "retrieval_manifest_sha256": retrieval.signature,
                            "pipeline_first_task": task_agents[0] if args.stage == "all" else None,
                        })
                        evidence_stream = None
                        if args.stage in {"export-remote", "judge-remote"}:
                            from .evaluation_offline import export_task_requests, judge_remote_answers
                            bundle_path = export_task_requests(
                                config, examples, retrieval, target, suite["remote_task"], label, source)
                            if args.stage == "judge-remote":
                                responses_root = getattr(args, "responses_root", None)
                                responses = (Path(responses_root) / label / source if responses_root
                                             else target / "offline" / "responses")
                                judge_remote_answers(
                                    config, store, examples, retrieval, target, bundle_path, responses,
                                    saved["scoring_protocol"], workers=settings["local_scoring_workers"])
                                write_summary(root)
                        if args.stage in {"all", "retrieve"}:
                            if any(retrieval.read(e.sample_id) is None for e in examples):
                                if policy is None:
                                    from .policy import HFPolicy, load_model, load_tokenizer
                                    policy = HFPolicy(load_model(model, config["reader"]["device"]),
                                                      load_tokenizer(model), config["reader"], temperature=0.0)
                                    policy.policy_version = int(model.parent.name.removeprefix("checkpoint-"))
                            if args.stage == "retrieve":
                                retrieve(config, store, policy, examples, retrieval, settings)
                            else:
                                evidence_stream = iter_retrieved(
                                    config, store, policy, examples, retrieval, settings)
                        if args.stage in {"all", "score"}:
                            for task in task_agents:
                                answer_config = deepcopy(config)
                                if task == "remote":
                                    answer_config["task_agent"]["service"] = deepcopy(suite["remote_task"])
                                score_evidence(answer_config, store, examples, retrieval,
                                               target / "answers" / task, saved["scoring_protocol"],
                                               workers=settings[f"{task}_scoring_workers"],
                                               retrieval_rows=evidence_stream)
                                evidence_stream = None
                                write_summary(root)
                                if task == "local" and suite.get("export_remote_inputs", False):
                                    from .evaluation_offline import export_task_requests
                                    export_task_requests(config, examples, retrieval, target,
                                                         suite["remote_task"], label, source)
                finally:
                    store.close()
        finally:
            if policy is not None:
                del policy
                gc.collect()
                import torch
                torch.cuda.empty_cache()
    announce("test_matrix_finished", stage=args.stage, summary=str(root / "summary.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_locomo_longmemeval_checkpoints.yaml")
    parser.add_argument("--stage", choices=["all", "prepare", "build", "retrieve", "score", "summarize",
                                          "export-remote", "judge-remote"], default="all")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--checkpoints", nargs="+")
    parser.add_argument("--task-agents", nargs="+", choices=["local", "remote"])
    parser.add_argument("--local-base-url")
    parser.add_argument("--remote-model")
    parser.add_argument("--responses-root", help="Returned GPT response tree, organized as checkpoint/dataset")
    parser.add_argument("--limit", type=int, help="Separate smoke-test output; never mixed with full test metrics")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    run_suite(args)


if __name__ == "__main__":
    main()
