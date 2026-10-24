from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import random
import shutil
import time

from .core import CapacityError, InfrastructureError, append_json, atomic_json, canonical, digest, file_digest, seeded_int
from .checkpoints import retain_checkpoints, validation_scores
from .data import ADAPTER_VERSION, InvalidSample, adapt_row, clean_history, decode_list, parquet_rows
from .env import MemoryEnvironment
from .execution import execution_settings
from .rewards import advantages, score_trajectory
from .store import DatasetStores, Store
from .traces import save_training_group


def announce(event, **values):
    print(canonical({"event": event, **values}), flush=True)


def dataset_configs(config, sources=None):
    """Stable configured source order; each source owns its histories and both splits."""
    configured = config["data"]["sources"]
    if not configured or len(set(configured)) != len(configured):
        raise ValueError("data.sources must be nonempty and unique")
    selected = configured if sources is None else sources
    if not selected or not set(selected).issubset(configured):
        raise ValueError("Requested sources must be a nonempty subset of data.sources")
    for source in configured:
        if source not in selected:
            continue
        if source in {".", "..", "_runtime"} or Path(source).name != source:
            raise ValueError(f"Invalid dataset directory name: {source}")
        child = deepcopy(config)
        child["data"].update(layout="combined", sources=[source],
                             prepared_root=str(Path(config["data"]["prepared_root"]) / source))
        yield source, child


def prepare(config, limit=None, verify=True, sources=None):
    if config["data"].get("layout", "combined") == "by_source":
        return {source: prepare(child, limit, verify) for source, child in dataset_configs(config, sources)}
    if sources is not None:
        raise ValueError("--sources requires data.layout=by_source")
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
    if config["data"].get("layout", "combined") == "by_source":
        stores = {}
        try:
            for source, child in dataset_configs(config):
                stores[source] = open_prepared(child)
            return DatasetStores(config["data"]["prepared_root"], stores)
        except Exception:
            for store in stores.values():
                store.close()
            raise
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


def build_directories(config, limit=None, sources=None):
    if config["data"].get("layout", "combined") == "by_source":
        versions = {}
        for source, child in dataset_configs(config, sources):
            announce("dataset_directories_started", source=source)
            versions[source] = build_directories(child, limit)
            announce("dataset_directories_finished", source=source, version=versions[source])
        return versions
    if sources is not None:
        raise ValueError("--sources requires data.layout=by_source")
    from .policy import load_tokenizer, model_fingerprint
    from .services import ChatService, DirectoryBuilder
    store = open_prepared(config)
    service_config = dict(config["service"])
    service_config["model_revision"] = model_fingerprint(config["models"]["task_path"])["fingerprint"]
    try:
        builder = DirectoryBuilder(store, ChatService(service_config, store, "writer"), config["writer"], load_tokenizer(config["models"]["task_path"]))
        workers = execution_settings(config)["writer_workers"]
        announce("directory_execution", writer_workers=workers, version=builder.version)
        pending_total = 0
        for i, memory_id in enumerate(store.memory_ids()):
            if limit is not None and i >= limit:
                break
            entries = builder.build(memory_id, workers=workers)
            pending = sum(entry.get("writer_status") == "empty_pending" for entry in entries)
            pending_total += pending
            announce("directory_built", index=i, memory_id=memory_id, sessions=len(entries),
                     empty_pending=pending, version=builder.version)
        store.set_metadata("directory_version", builder.version)
        store.set_metadata("directory_service_identity", {k: v for k, v in service_config.items() if k != "request_base_url"})
        store.set_metadata("directory_empty_pending", pending_total)
        announce("directories_finished", empty_pending=pending_total, version=builder.version)
        return builder.version
    finally:
        store.close()


def validate_directories(config, store, splits):
    """Fail before loading training weights if a selected history is not ready."""
    from types import SimpleNamespace
    from .policy import model_fingerprint
    from .services import DirectoryBuilder
    identity = {key: config["service"][key] for key in ("base_url", "model")}
    identity["model_revision"] = model_fingerprint(config["models"]["task_path"])["fingerprint"]
    expected = DirectoryBuilder(store, SimpleNamespace(identity=identity), config["writer"], None).version
    if directory_version(store) != expected:
        raise ValueError("Selected directory versions differ from current Writer; build selected datasets first")
    # Check each source, including shared histories, so incomplete source artifacts
    # cannot be hidden by another source's published directory.
    children = store.stores.values() if isinstance(store, DatasetStores) else [store]
    for child in children:
        for memory_id in sorted({e.memory_id for split in splits for e in child.examples(split)}):
            child.directory(memory_id, expected)


def import_prepared(config, source_root):
    """Copy selected source artifacts out of a legacy combined store, without inference."""
    if config["data"].get("layout") != "by_source":
        raise ValueError("import-prepared requires data.layout=by_source")
    source_root = Path(source_root).resolve()
    if not (source_root / "manifest.json").is_file():
        raise ValueError("Missing source prepared manifest")
    origin = Store(source_root)
    try:
        identity = origin.get_metadata("prepare_identity")
        if (not origin.get_metadata("prepare_complete") or not identity
                or identity["adapter_version"] != ADAPTER_VERSION
                or identity["storage_format"] != Store.FORMAT
                or identity["input_manifest_sha256"] != file_digest(Path(config["data"]["input_root"]) / "manifest.json")):
            raise ValueError("Source prepared data is incomplete or has different input/adapter provenance")
        if not set(config["data"]["splits"]).issubset(identity["splits"]):
            raise ValueError("Source prepared data lacks requested splits")
        examples = [e for split in config["data"]["splits"] for e in origin.examples(split)]
        directory_rows = defaultdict(list)
        for key in origin.directory_entries.keys():
            directory_rows[key[0]].append(key)
        reports = {}
        for source, child in dataset_configs(config):
            if source not in identity["sources"]:
                raise ValueError(f"Source prepared data lacks {source}")
            target_root = Path(child["data"]["prepared_root"]).resolve()
            if target_root == source_root or source_root.is_relative_to(target_root):
                raise ValueError("Import destination overlaps source store")
            target = Store(target_root)
            try:
                child_identity = {**identity, "sources": [source], "splits": child["data"]["splits"]}
                if target.get_metadata("prepare_complete"):
                    if target.get_metadata("prepare_identity") != child_identity:
                        raise ValueError(f"Existing prepared provenance differs for {source}")
                    announce("dataset_import_reused", source=source)
                    continue
                if target.get_metadata("prepare_identity") not in (None, child_identity):
                    raise ValueError(f"Incomplete destination provenance differs for {source}")
                target.set_metadata("prepare_identity", child_identity)
                target.set_metadata("prepare_complete", False)
                selected = [e for e in examples if e.source == source]
                memories = sorted({e.memory_id for e in selected})
                n_entries = n_sessions = 0
                for memory_id in memories:
                    row = origin.histories.get((memory_id,))
                    if target.add_memory(row["records"]) != memory_id:
                        raise ValueError("Imported history identity changed")
                    n_sessions += len(row["records"])
                    for key in directory_rows[memory_id]:
                        entry = origin.directory_entries.get(key)
                        target.put_directory_entry(memory_id, entry["version"], origin._directory_fields(entry))
                        n_entries += 1
                    versions = {key[1] for key in directory_rows[memory_id]}
                    for version in versions:
                        if digest([memory_id, version]) in origin.directory_snapshots:
                            target.put_directory(memory_id, version, origin.directory(memory_id, version))
                for example in selected:
                    target.add_example(example)
                # A partially finished legacy build may not have a global version.
                # build-directories will determine the current version and reuse entries.
                for key in ("directory_version", "directory_service_identity"):
                    value = origin.get_metadata(key)
                    if value is not None:
                        target.set_metadata(key, value)
                report = {"status": "complete", "identity": child_identity, "imported_from": str(source_root),
                          "memories": len(memories), "sessions": n_sessions, "imported_directory_entries": n_entries,
                          "counts": dict(Counter(f"{e.source}/{e.split}/valid" for e in selected))}
                atomic_json(target_root / "preparation_report.json", report)
                target.set_metadata("prepare_complete", True)
                reports[source] = report
                announce("dataset_imported", source=source, memories=len(memories), sessions=n_sessions,
                         directory_entries=n_entries)
            finally:
                target.close()
        return reports
    finally:
        origin.close()


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
    execution = execution_settings(config)
    version = policy.policy_version
    seeds = [seeded_int(config["seed"], example.sample_id, version, step, i) % (2 ** 31) for i in range(8)]
    batch_size = execution["reader_batch_size"]

    def score(trajectory):
        trajectory.prediction, trajectory.task_input_hash = agent.answer(example, trajectory.evidence)
        return score_trajectory(example, trajectory, config["reward"])

    # Only this thread owns the Reader model. Workers access the frozen HTTP
    # Task Agent; no gradients or policy changes occur until all futures finish.
    group, pending = [], []
    with ThreadPoolExecutor(max_workers=execution["task_agent_workers"]) as pool:
        for offset in range(0, 8, batch_size):
            if batch_size == 1:
                rows = [env.run(example, policy, seeds[offset], version)]
            else:
                rows = env.run_batch(example, policy, seeds[offset:offset + batch_size], version,
                                     max_batch_tokens=execution["reader_batch_max_tokens"])
            for row in rows:
                if execution["task_agent_workers"] == 1:
                    group.append(score(row))
                else:
                    pending.append(pool.submit(score, row))
        # Preserve rollout_index ordering even if scoring completes out of order.
        group.extend(future.result() for future in pending)
    if len(group) != 8 or policy.policy_version != version or any(row.policy_version != version for row in group):
        raise ValueError("GRPO collection must finish eight rollouts under one unchanged policy version")
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
                "outcome_reward": sum(row.outcome_reward for row in values) / n,
                "reward": sum(row.reward for row in values) / n, "read_calls": sum(calls) / n,
                "read_cost": sum(row.read_cost for row in values) / n,
                "raw_read_cost": sum(row.raw_read_cost for row in values) / n,
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
                             for key in ("outcome_f1", "outcome_reward", "reward", "read_calls", "read_cost", "raw_read_cost", "format_reward", "format_failure_rate")},
            "metric_scope": "This split's QA label/text F1, call cost and format penalty, not official benchmark metrics."}


def evaluate_with_policy(config, store, policy, split, output, limit=None):
    validate_directories(config, store, [split])
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


def save_checkpoint(config, policy, optimizer, run_dir, step, epoch, next_index, validation=None):
    import torch
    if validation is not None:
        validation_scores(validation)
    destination = Path(run_dir) / f"checkpoint-{step:08d}"
    temporary = destination.with_name(destination.name + ".building")
    temporary.mkdir(parents=True, exist_ok=False)
    policy.model.save_pretrained(temporary / "model", safe_serialization=True)
    policy.tokenizer.save_pretrained(temporary / "model")
    torch.save({"optimizer": optimizer.state_dict(), "step": step, "epoch": epoch, "next_index": next_index,
                "policy_version": policy.policy_version, "config": config}, temporary / "trainer.pt")
    if validation is not None:
        atomic_json(temporary / "validation_metrics.json", validation)
    temporary.rename(destination)
    best = retain_checkpoints(run_dir, config["training"]["keep_last_checkpoints"])
    announce("checkpoint_saved", step=step, path=str(destination), best=best)
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
    if resume and (run_dir / "run.json").exists():
        previous_reward = json.loads((run_dir / "run.json").read_text())["config"]["reward"]
        if previous_reward != config["reward"]:
            raise ValueError("Resume reward configuration changed; start a separate experiment")
    writer_class = summary_writer_class() if config["training"].get("tensorboard", True) else None
    writer = None
    store = open_prepared(config)
    try:
        validate_directories(config, store, ["train", "validation"] if config["training"]["validation_every"] else ["train"])
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
        announce("training_execution", **execution_settings(config))
        step, start_epoch, next_index = 0, 0, 0
        if resume:
            previous = json.loads((run_dir / "run.json").read_text())
            for key in ("reader", "task", "directory_version", "prepare_identity"):
                if previous[key] != metadata[key]:
                    raise ValueError(f"Resume provenance mismatch: {key}")
            state = torch.load(Path(resume) / "trainer.pt", map_location="cpu", weights_only=True)
            if execution_settings(state["config"]) != execution_settings(config):
                announce("resume_execution_changed", previous=execution_settings(state["config"]),
                         current=execution_settings(config), note="Sampling batches/RNG streams may change; committed optimizer state is preserved")
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
                report.update({"read_cost": sum(row.read_cost for row in group) / 8,
                               "raw_read_cost": sum(row.raw_read_cost for row in group) / 8,
                               "outcome_reward": sum(row.outcome_reward for row in group) / 8,
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
                validation = None
                if every and step % every == 0:
                    validation = evaluate_with_policy(config, store, policy, "validation", run_dir / "validation" / f"step-{step:08d}")
                    if writer:
                        write_validation_metrics(writer, validation, step)
                every = config["training"]["checkpoint_every"]
                if validation is not None or (every and step % every == 0) or (max_groups is not None and completed >= max_groups):
                    save_checkpoint(config, policy, optimizer, run_dir, step, epoch, index + 1, validation=validation)
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
