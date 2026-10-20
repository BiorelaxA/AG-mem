from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import json
import random
import shutil
import time

from .core import CapacityError, InfrastructureError, append_json, atomic_json, canonical, digest, file_digest, seeded_int
from .data import ADAPTER_VERSION, InvalidSample, adapt_row, clean_history, decode_list, parquet_rows
from .env import MemoryEnvironment
from .rewards import advantages, score_trajectory
from .store import Store
from .traces import save_training_group


def announce(event, **values):
    print(canonical({"event": event, **values}), flush=True)


def prepare(config, limit=None, verify=True):
    root = Path(config["data"]["input_root"])
    output = Path(config["data"]["prepared_root"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    identity = {"input_manifest_sha256": file_digest(root / "manifest.json"), "sources": config["data"]["sources"],
                "splits": config["data"]["splits"], "limit_per_source_split": limit, "adapter_version": ADAPTER_VERSION,
                "storage_format": Store.FORMAT}
    store = Store(output)
    previous = store.get_metadata("prepare_identity")
    if previous and previous != identity:
        store.close()
        raise ValueError("Prepared directory belongs to a different input/adapter/limit; use a fresh prepared_root")
    store.set_metadata("prepare_identity", identity)
    store.set_metadata("prepare_complete", False)
    # These fingerprints were verified against every original column during split materialization.
    # Hashes may accelerate repeat-history parsing only when current Parquet hashes are verified.
    history_keys = {}
    if verify:
        for split in ("train", "validation"):
            path = root / f"{split}_manifest.jsonl"
            if path.exists():
                with path.open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        history_keys[row["source"], split, row["output_row_index"]] = row["ordered_history_sha256"]
    cache = {}
    counts, issues = Counter(), []
    started = time.monotonic()
    try:
        for source, split, index, row in parquet_rows(root, config["data"]["sources"], config["data"]["splits"], limit, verify):
            counts[f"{source}/{split}/input"] += 1
            try:
                example = adapt_row(row, source, split, index)
                key = history_keys.get((source, split, index))
                if key:
                    # Dates and public source IDs can differ even when full message content agrees.
                    import re
                    ids = row.get("haystack_session_ids")
                    ids = decode_list(ids, "haystack_session_ids") if ids is not None else None
                    ids = [re.sub(r"^(?:answer_)+", "", str(value)) for value in ids] if ids is not None else None
                    key = digest([key, row.get("haystack_dates"), ids])
                memory_id = cache.get(key) if key else None
                if memory_id is None:
                    memory_id = store.add_memory(clean_history(row))
                    if key:
                        cache[key] = memory_id
                example.memory_id = memory_id
                store.add_example(example)
                counts[f"{source}/{split}/valid"] += 1
                if source == "personamem_v2_128k" and None in decode_list(row["incorrect_answers"], "incorrect_answers"):
                    issues.append({"sample_id": example.sample_id, "action": "ignored_null_distractor_padding"})
            except InvalidSample as exc:
                counts[f"{source}/{split}/quarantined"] += 1
                issues.append({"source": source, "split": split, "output_row_index": index, "question_id": row.get("question_id"), "action": "quarantined", "reason": str(exc)})
            if sum(value for key_, value in counts.items() if key_.endswith("/input")) % 100 == 0:
                announce("prepare_progress", source=source, split=split, row=index)
        report = {"status": "complete", "identity": identity, "counts": dict(counts), "issues": issues,
                  "memories": len(store.memory_ids()),
                  "sessions": sum(len(store.memory(memory_id)) for memory_id in store.memory_ids()),
                  "elapsed_seconds": time.monotonic() - started}
        atomic_json(output / "preparation_report.json", report)
        store.set_metadata("prepare_complete", True)
        announce("prepared", **{key: report[key] for key in ("memories", "sessions", "elapsed_seconds")}, counts=dict(counts))
        return report
    finally:
        store.close()


def open_prepared(config):
    store = Store(config["data"]["prepared_root"])
    if store.get_metadata("prepare_complete") is not True:
        store.close()
        raise ValueError("Run prepare successfully before using this store")
    identity = store.get_metadata("prepare_identity")
    if (identity["adapter_version"] != ADAPTER_VERSION or identity.get("storage_format") != Store.FORMAT
            or any(identity[key] != config["data"][key] for key in ("sources", "splits"))):
        store.close()
        raise ValueError("Prepared adapter/source selection differs from configuration; use a freshly prepared store")
    return store


def directory_version(store):
    version = store.get_metadata("directory_version")
    if version is None:
        raise ValueError("Run build-directories first")
    return version


def build_directories(config, limit=None):
    from .policy import load_tokenizer, model_fingerprint
    from .services import ChatService, DirectoryBuilder
    store = open_prepared(config)
    service_config = dict(config["service"])
    service_config["model_revision"] = model_fingerprint(config["models"]["task_path"])["fingerprint"]
    try:
        builder = DirectoryBuilder(store, ChatService(service_config, store, "writer"), config["writer"], load_tokenizer(config["models"]["task_path"]))
        for i, memory_id in enumerate(store.memory_ids()):
            if limit is not None and i >= limit:
                break
            entries = builder.build(memory_id)
            announce("directory_built", index=i, memory_id=memory_id, sessions=len(entries), version=builder.version)
        store.set_metadata("directory_version", builder.version)
        store.set_metadata("directory_service_identity", service_config)
        return builder.version
    finally:
        store.close()


def make_environment(config, store, policy):
    from types import SimpleNamespace
    from .policy import chat_token_ids, load_tokenizer, model_fingerprint
    from .services import DirectoryBuilder
    identity = {"base_url": config["service"]["base_url"], "model": config["service"]["model"],
                "model_revision": model_fingerprint(config["models"]["task_path"])["fingerprint"]}
    expected_version = DirectoryBuilder(store, SimpleNamespace(identity=identity), config["writer"], None).version
    if directory_version(store) != expected_version:
        raise ValueError("Writer model/prompt/config changed; rebuild directories before training/evaluation")
    task_tokenizer = load_tokenizer(config["models"]["task_path"])
    def capacity(reader_input, task_input):
        policy.check_capacity(reader_input)
        limit = config["task_agent"].get("max_context_tokens")
        if limit is not None:
            tokens = chat_token_ids(task_tokenizer, task_input)
            if len(tokens) + config["task_agent"]["max_output_tokens"] > limit:
                raise CapacityError("Task-agent context would overflow")
    return MemoryEnvironment(store, directory_version(store), config["retrieval"]["max_ids_per_call"], config["retrieval"]["max_read_calls"], capacity)


def make_task_agent(config, store):
    from .policy import model_fingerprint
    from .services import ChatService, TaskAgent
    service_config = dict(config["service"])
    service_config["model_revision"] = model_fingerprint(config["models"]["task_path"])["fingerprint"]
    return TaskAgent(ChatService(service_config, store, "task_agent"), config["task_agent"]["max_output_tokens"])


def scored_rollout(config, env, policy, agent, example, seed):
    trajectory = env.run(example, policy, seed, policy.policy_version)
    trajectory.prediction, trajectory.task_input_hash = agent.answer(example, trajectory.evidence)
    return score_trajectory(example, trajectory, config["reward"])


def collect_group(config, env, policy, agent, example, step):
    group = []
    for rollout_index in range(8):
        seed = seeded_int(config["seed"], example.sample_id, policy.policy_version, step, rollout_index) % (2 ** 31)
        group.append(scored_rollout(config, env, policy, agent, example, seed))
    for trajectory, advantage in zip(group, advantages([row.reward for row in group], config["training"]["advantage_epsilon"])):
        trajectory.advantage = advantage
    return group


def metrics(rows):
    if not rows:
        raise ValueError("Empty evaluation split")
    def summarize(values):
        n = len(values)
        calls = sorted(row.n_read_calls for row in values)
        return {"n": n, "outcome_f1": sum(row.outcome for row in values) / n,
                "reward": sum(row.reward for row in values) / n, "read_calls": sum(calls) / n,
                "read_cost": sum(row.read_cost for row in values) / n,
                "format_reward": sum(row.format_reward for row in values) / n,
                "format_failure_rate": sum(row.format_failure for row in values) / n,
                "p95_read_calls": calls[min(n - 1, int(0.95 * n))],
                "natural_stop_rate": sum(row.termination_reason == "stop" for row in values) / n,
                "protocol_failure_rate": sum(row.termination_reason == "protocol_failure" for row in values) / n,
                "unique_ids_read": sum(len(row.evidence) for row in values) / n,
                "reader_generated_tokens": sum(sum(len(step.completion.token_ids) for step in row.decisions) for row in values) / n,
                "reader_prompt_tokens": sum(sum(len(step.completion.prompt_ids) for step in row.decisions) for row in values) / n,
                "capacity_rejections": sum(sum(bool(step.tool_result and step.tool_result.get("error", {}).get("code") == "context_capacity_exceeded") for step in row.decisions) for row in values) / n,
                "forced_termination_rate": sum(row.termination_reason != "stop" for row in values) / n}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.sample_id.split("/")[0]].append(row)
    by_source = {source: summarize(values) for source, values in grouped.items()}
    return {"pooled": summarize(rows), "by_source": by_source,
            "source_macro": {key: sum(values[key] for values in by_source.values()) / len(by_source)
                             for key in ("outcome_f1", "reward", "read_calls", "read_cost", "format_reward", "format_failure_rate")},
            "metric_scope": "This split's QA label/text F1, call cost and format penalty, not official benchmark metrics."}


def evaluate_with_policy(config, store, policy, split, output, limit=None):
    env, agent = make_environment(config, store, policy), make_task_agent(config, store)
    examples = store.examples(split)
    if limit is not None:
        examples = examples[:limit]
    rows = []
    temperature = policy.temperature
    policy.temperature = 0.0
    try:
        for example in examples:
            started = time.monotonic()
            row = scored_rollout(config, env, policy, agent, example, config["seed"])
            rows.append(row)
            append_json(Path(output) / "episodes.jsonl", {**row.to_dict(), "elapsed_seconds": time.monotonic() - started})
            announce("evaluation_progress", split=split, sample_id=example.sample_id, f1=row.outcome, calls=row.n_read_calls)
    finally:
        policy.temperature = temperature
    report = metrics(rows)
    atomic_json(Path(output) / "metrics.json", report)
    return report


def save_checkpoint(config, policy, optimizer, run_dir, step, epoch, next_index):
    import torch
    destination = Path(run_dir) / f"checkpoint-{step:08d}"
    temporary = destination.with_name(destination.name + ".building")
    temporary.mkdir(parents=True, exist_ok=False)
    policy.model.save_pretrained(temporary / "model", safe_serialization=True)
    policy.tokenizer.save_pretrained(temporary / "model")
    torch.save({"optimizer": optimizer.state_dict(), "step": step, "epoch": epoch, "next_index": next_index,
                "policy_version": policy.policy_version, "config": config}, temporary / "trainer.pt")
    temporary.rename(destination)
    keep = config["training"]["keep_last_checkpoints"]
    if keep and keep > 0:
        old = sorted(path for path in Path(run_dir).glob("checkpoint-*") if path.is_dir() and path.name.split("-")[-1].isdigit())
        for path in old[:-keep]:
            shutil.rmtree(path)
    return destination


def train(config, max_groups=None, resume=None):
    import torch
    from .grpo import CPUAdamW, GRPOUpdater
    from .monitoring import summary_writer_class, write_train_metrics, write_validation_metrics
    from .policy import HFPolicy, load_model, load_tokenizer, model_fingerprint
    if not torch.cuda.is_available() and config["reader"]["device"].startswith("cuda"):
        raise RuntimeError("CUDA is unavailable; unit tests can run on CPU but real model training needs the allocated GPU")
    run_dir = Path(config["training"]["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "run.json").exists() and resume is None:
        raise ValueError("Existing training run; supply --resume or choose a new output_dir")
    writer_class = summary_writer_class() if config["training"].get("tensorboard", True) else None
    writer = None
    store = open_prepared(config)
    try:
        model_path = str(Path(resume) / "model") if resume else config["models"]["reader_path"]
        tokenizer = load_tokenizer(model_path)
        model = load_model(model_path, config["reader"]["device"], checkpointing=True)
        model.requires_grad_(True)
        policy = HFPolicy(model, tokenizer, config["reader"])
        optimizer = CPUAdamW(model, config["training"]["learning_rate"], config["training"]["weight_decay"])
        reference = load_model(config["models"]["reader_path"], "cpu").requires_grad_(False) if config["training"]["kl_coefficient"] else None
        updater = GRPOUpdater(policy, optimizer, config["training"], reference)
        env, agent = make_environment(config, store, policy), make_task_agent(config, store)
        metadata = {"config": config, "reader": model_fingerprint(config["models"]["reader_path"]),
                    "task": model_fingerprint(config["models"]["task_path"]), "directory_version": directory_version(store),
                    "prepare_identity": store.get_metadata("prepare_identity"),
                    "reader_trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
                    "parameterization": "full", "optimizer": "CPUAdamW_fp32_master_and_moments"}
        step, start_epoch, next_index = 0, 0, 0
        if resume:
            previous = json.loads((run_dir / "run.json").read_text())
            for key in ("reader", "task", "directory_version", "prepare_identity"):
                if previous[key] != metadata[key]:
                    raise ValueError(f"Resume provenance mismatch: {key}")
            state = torch.load(Path(resume) / "trainer.pt", map_location="cpu", weights_only=True)
            for key in ("reader", "retrieval", "reward", "training"):
                # TensorBoard changes monitoring only; old checkpoints may lack this option.
                ignored = {"tensorboard"} if key == "training" else set()
                previous_config = {name: value for name, value in state["config"][key].items() if name not in ignored}
                current_config = {name: value for name, value in config[key].items() if name not in ignored}
                if previous_config != current_config:
                    raise ValueError(f"Resume configuration changed: {key}")
            optimizer.load_state_dict(state["optimizer"])
            step, start_epoch, next_index = state["step"], state["epoch"], state["next_index"]
            policy.policy_version = state["policy_version"]
            del state
        atomic_json(run_dir / "run.json", metadata)
        if writer_class:
            log_dir = run_dir / "tensorboard"
            writer = writer_class(log_dir=str(log_dir), purge_step=step + 1 if resume else None)
            announce("tensorboard_started", log_dir=str(log_dir), next_step=step + 1)
        examples = store.examples("train")
        if not examples:
            raise ValueError("No valid training examples")
        completed = 0
        for epoch in range(start_epoch, config["training"]["epochs"]):
            ordered = list(examples)
            random.Random(config["seed"] + epoch).shuffle(ordered)
            for index, example in enumerate(ordered):
                if epoch == start_epoch and index < next_index:
                    continue
                started = time.monotonic()
                for attempt in range(config["training"]["group_infrastructure_retries"] + 1):
                    try:
                        group = collect_group(config, env, policy, agent, example, step)
                        break
                    except InfrastructureError as exc:
                        append_json(run_dir / "infrastructure_failures.jsonl", {"sample_id": example.sample_id, "step": step, "attempt": attempt, "error": str(exc)})
                        if attempt == config["training"]["group_infrastructure_retries"]:
                            raise
                save_training_group(run_dir, step,
                                    f"{example.sample_id}/policy-{policy.policy_version}/step-{step}",
                                    env.version, group)
                update = updater.update(group)
                step += 1
                completed += 1
                report = {"step": step, "epoch": epoch, "sample_id": example.sample_id, "policy_version": policy.policy_version,
                          "reward": sum(row.reward for row in group) / 8, "outcome_f1": sum(row.outcome for row in group) / 8,
                          "read_calls": sum(row.n_read_calls for row in group) / 8,
                          "elapsed_seconds": time.monotonic() - started,
                          "reader_generated_tokens": sum(sum(len(decision.completion.token_ids) for decision in row.decisions) for row in group),
                          "zero_variance_group": len({row.reward for row in group}) == 1, **update}
                report.update({"read_cost": config["reward"]["read_call_cost"] * report["read_calls"],
                               "format_reward": sum(row.format_reward for row in group) / 8,
                               "format_failure_rate": sum(row.format_failure for row in group) / 8,
                               "reward_std": (sum((row.reward - report["reward"]) ** 2 for row in group) / 8) ** 0.5,
                               "reward_min": min(row.reward for row in group), "reward_max": max(row.reward for row in group),
                               "protocol_failure_rate": sum(row.termination_reason == "protocol_failure" for row in group) / 8,
                               "natural_stop_rate": sum(row.termination_reason == "stop" for row in group) / 8,
                               "unique_ids_read": sum(len(row.evidence) for row in group) / 8})
                append_json(run_dir / "train_metrics.jsonl", report)
                announce("train_step", **report)
                if writer:
                    write_train_metrics(writer, report)
                every = config["training"]["validation_every"]
                if every and step % every == 0:
                    validation = evaluate_with_policy(config, store, policy, "validation", run_dir / "validation" / f"step-{step:08d}")
                    if writer:
                        write_validation_metrics(writer, validation, step)
                every = config["training"]["checkpoint_every"]
                if (every and step % every == 0) or (max_groups is not None and completed >= max_groups):
                    save_checkpoint(config, policy, optimizer, run_dir, step, epoch, index + 1)
                if max_groups is not None and completed >= max_groups:
                    return
            next_index = 0
        if not (run_dir / f"checkpoint-{step:08d}").exists():
            save_checkpoint(config, policy, optimizer, run_dir, step, config["training"]["epochs"], 0)
    finally:
        try:
            if writer:
                writer.close()
        finally:
            store.close()
