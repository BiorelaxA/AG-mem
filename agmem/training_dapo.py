"""Separate DAPO training orchestration with resumable frozen sampling batches."""
from __future__ import annotations

from pathlib import Path
import json
import time

from .core import InfrastructureError, append_json, atomic_json, trajectory_from_dict
from .dapo_sampling import DynamicGroupSampler, validate_dapo_config


def run_training_loop(config, store, policy, optimizer, updater, env, agent, scorer,
                      examples, run_dir, writer=None, *, state=None, max_groups=None):
    """Keep optimizer steps separate from candidates, generation rounds and batches."""
    from .monitoring import write_train_metrics, write_validation_metrics, write_sampling_metrics
    from .pipeline import announce, collect_group, evaluate_with_policy, metrics, save_checkpoint
    from .traces import compact_group, save_training_group

    state = state or {"step": 0, "dapo_state": {"sampler": {}, "active": None}}
    step, completed = state["step"], 0
    dapo_state = state["dapo_state"]
    mini = config["training"].get("mini_batch_groups", 1)
    counters = dict(dapo_state.get("counters", {
        "trained_groups": step * mini,
        "update_rollouts": step * mini * config["training"]["group_size"],
        "training_tokens": 0,
    }))
    if step and "counters" not in dapo_state:
        # Older checkpoints stored no totals. The committed metrics, when
        # present, provide the exact token count without counting orphan steps.
        metrics_path = run_dir / "train_metrics.jsonl"
        if metrics_path.is_file():
            with metrics_path.open() as stream:
                for line in stream:
                    if line.strip():
                        report = json.loads(line)
                        if report["step"] <= step:
                            counters["training_tokens"] += report.get("training_tokens", 0)
    sampler = DynamicGroupSampler(config, examples, **dapo_state["sampler"])
    active = dapo_state.get("active")
    groups = [[trajectory_from_dict(row) for row in group] for group in active["groups"]] if active else None
    if max_groups is not None and max_groups < 1:
        raise ValueError("max_groups must be positive (counts actual optimizer updates)")

    def checkpoint_state():
        saved = None
        if active and updater.consumed_groups < len(groups):
            saved = {**active, "consumed_groups": updater.consumed_groups,
                     "groups": [[row.to_dict() for row in group] for group in groups]}
        return {"sampler": sampler.state_dict(), "active": saved, "counters": dict(counters)}

    def save(validation=None):
        return save_checkpoint(config, policy, optimizer, run_dir, step, sampler.epoch, sampler.next_index,
                               validation=validation, dapo_state=checkpoint_state())

    def collect(example, candidate_index):
        for attempt in range(config["training"].get("group_infrastructure_retries", 1) + 1):
            try:
                return collect_group(config, env, policy, agent, example, candidate_index, scorer=scorer)
            except InfrastructureError as exc:
                append_json(run_dir / "infrastructure_failures.jsonl", {
                    "sample_id": example.sample_id, "candidate_id": candidate_index, "step": step,
                    "attempt": attempt, "error": str(exc)})
                if attempt == config["training"].get("group_infrastructure_retries", 1):
                    raise

    def candidate(group, report):
        append_json(run_dir / "candidate_metrics.jsonl", report)
        # Retain rejected answers/actions too: reporting only accepted groups
        # would conceal the all-correct/all-wrong behavior we want to measure.
        trace = {"group_id": f"{report['batch_id']}/candidate-{report['candidate_id']}",
                 "directory_version": env.version, "rollouts": [row.to_dict() for row in group]}
        atomic_json(run_dir / "candidate_rollouts" / f"candidate-{report['candidate_id']:08d}.json",
                    {**compact_group(trace), "sampling": report})
        announce("dapo_candidate", **report)

    def sampling_round(report):
        report = {**report, "step": step, "candidate_index": sampler.candidate_index}
        append_json(run_dir / "sampling_metrics.jsonl", report)
        announce("dapo_sampling", **report)
        if writer:
            write_sampling_metrics(writer, report)

    while True:
        if active is None:
            started = time.monotonic()
            groups, sampling = sampler.collect(collect, policy.policy_version,
                                                record_candidate=candidate, record_round=sampling_round)
            if not groups:
                announce("dapo_dataset_exhausted", **sampling)
                if step == 0:
                    raise RuntimeError("No complete DAPO batch was collected; inspect candidate_metrics.jsonl")
                break
            active = {"batch_id": sampling["batch_id"], "behavior_version": policy.policy_version,
                      "consumed_groups": 0, "sampling": sampling,
                      "sampling_seconds": time.monotonic() - started}
        updater.begin_batch(groups, active["batch_id"], consumed_groups=active["consumed_groups"],
                            behavior_version=active["behavior_version"])
        while updater.consumed_groups < len(groups):
            update_groups = groups[updater.consumed_groups:updater.consumed_groups + mini]
            group = [row for query_group in update_groups for row in query_group]
            started = time.monotonic()
            # File indices remain optimizer-step indices for checkpoint recovery.
            # Each rollout retains its sample_id and its own query's advantage.
            save_training_group(run_dir, step,
                                f"{active['batch_id']}/{group[0].sample_id}/update-{step + 1}", env.version, group)
            update = updater.update(group) if mini == 1 else updater.update_groups(update_groups)
            step += 1
            completed += 1
            counters["trained_groups"] += len(update_groups)
            counters["update_rollouts"] += len(group)
            counters["training_tokens"] += update["training_tokens"]
            report = {**metrics(group)["pooled"], **update, "step": step, "epoch": sampler.epoch,
                      "sample_id": group[0].sample_id, "sample_ids": [rows[0].sample_id for rows in update_groups],
                      "policy_version": policy.policy_version,
                      "behavior_policy_version": active["behavior_version"], "batch_id": active["batch_id"],
                      "batch_group_index": updater.consumed_groups,
                      "groups_per_update": len(update_groups), "rollouts_per_update": len(group),
                      **{f"cumulative_{key}": value for key, value in counters.items()},
                      "sampling_seconds": active["sampling_seconds"], "update_seconds": time.monotonic() - started,
                      "reward_std": (sum((row.reward - sum(r.reward for r in group) / len(group)) ** 2
                                          for row in group) / len(group)) ** 0.5,
                      "reward_min": min(row.reward for row in group), "reward_max": max(row.reward for row in group)}
            if all(row.likelihood.get("d") is not None for row in group):
                report.update({"likelihood_d": sum(row.likelihood["d"] for row in group) / len(group),
                               "likelihood_delta": sum(row.likelihood["delta"] for row in group) / len(group)})
            append_json(run_dir / "train_metrics.jsonl", report)
            announce("train_step", **report)
            if writer:
                write_train_metrics(writer, report)
            validation = None
            every = config["training"]["validation_every"]
            if every and step % every == 0:
                validation = evaluate_with_policy(config, store, policy, "validation",
                                                   run_dir / "validation" / f"step-{step:08d}")
                if writer:
                    write_validation_metrics(writer, validation, step)
            every = config["training"]["checkpoint_every"]
            stop = max_groups is not None and completed >= max_groups
            if validation is not None or (every and step % every == 0) or stop:
                save(validation)
            if stop:
                return
        updater.end_batch()
        active, groups = None, None
    if not (run_dir / f"checkpoint-{step:08d}").exists():
        save()


def train_dapo(config, max_groups=None, resume=None, *, validate_before_training=False):
    validate_dapo_config(config)
    if validate_before_training:
        raise ValueError("DAPO uses the evaluate command for baseline evaluation; resume preserves the fixed protocol")
    from .runtime_memory import configure_cuda_allocator, memory_settings
    configure_cuda_allocator()
    import torch
    from .dapo import DAPOUpdater
    from .data import task_examples
    from .grpo import CPUAdamW
    from .monitoring import summary_writer_class, replay_tensorboard
    from .pipeline import (announce, directory_version, make_environment,
                           make_reward_scorer, make_task_agent, open_prepared, validate_directories)
    from .policy import HFPolicy, load_model, load_tokenizer, model_fingerprint
    from .prompts import task_input_contract
    from .resume import resume_metadata, validate_resume_config
    from .execution import execution_settings

    if not torch.cuda.is_available() and config["reader"]["device"].startswith("cuda"):
        raise RuntimeError("DAPO training requires the allocated GPUs")
    run_dir = Path(config["training"]["output_dir"])
    if (run_dir / "run.json").exists() and resume is None:
        raise ValueError("Existing training run; use --resume or a new DAPO output_dir")
    previous = resume_metadata(config, resume) if resume else None
    run_dir.mkdir(parents=True, exist_ok=True)
    store, writer, scorer, agent = open_prepared(config), None, None, None
    try:
        splits = ["train", "validation"] if config["training"]["validation_every"] else ["train"]
        validate_directories(config, store, splits)
        examples = task_examples(config, store, "train")
        validation_examples = task_examples(config, store, "validation") if "validation" in splits else []
        scorer = make_reward_scorer(config, store)
        # Validate rubric metadata before loading the large Reader.
        for example in examples + validation_examples:
            scorer.judge.validate_example(example)
        runtime_memory = memory_settings()
        implementation = "agmem_dapo_v2" if config["reward"]["formula"] == "judge_cost_v2" else "agmem_dapo_v1"
        metadata = {"config": config, "algorithm": "dapo", "implementation": implementation,
                    "reader": model_fingerprint(config["models"]["reader_path"]),
                    "task": model_fingerprint(config["models"]["task_path"]),
                    "directory_version": directory_version(store),
                    "directory_service_identity": store.get_metadata("directory_service_identity"),
                    "prepare_identity": store.get_metadata("prepare_identity"),
                    "task_input_contract": task_input_contract(), "scoring_protocol": scorer.provenance(),
                    "runtime_memory": runtime_memory, "parameterization": "full",
                    "optimizer": "CPUAdamW_fp32_master_and_moments"}
        state = None
        if previous:
            for key in ("algorithm", "implementation", "reader", "task", "directory_version", "prepare_identity",
                        "task_input_contract", "scoring_protocol"):
                if previous.get(key) != metadata[key]:
                    raise ValueError(f"DAPO resume provenance mismatch: {key}")
            state = torch.load(Path(resume) / "trainer.pt", map_location="cpu", weights_only=True)
            validate_resume_config(state["config"], config)
            if "dapo_state" not in state:
                raise ValueError("A GRPO checkpoint cannot resume a DAPO experiment")
        model_path = str(Path(resume) / "model") if resume else config["models"]["reader_path"]
        model = load_model(model_path, config["reader"]["device"], checkpointing=True)
        model.requires_grad_(True)
        policy = HFPolicy(model, load_tokenizer(model_path), config["reader"])
        optimizer = CPUAdamW(model, config["training"]["learning_rate"], config["training"]["weight_decay"])
        if state:
            optimizer.load_state_dict(state.pop("optimizer"))
            policy.policy_version = state["policy_version"]
            from .dapo_recovery import reconcile_run
            recovery = reconcile_run(run_dir, state)
            announce("dapo_resume_reconciled", recovery=recovery)
        updater = DAPOUpdater(policy, optimizer, config["training"], runtime_memory=runtime_memory)
        env, agent = make_environment(config, store, policy), make_task_agent(config, store)
        metadata["reader_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        atomic_json(run_dir / "run.json", metadata)
        announce("training_execution", algorithm="dapo", **execution_settings(config))
        announce("training_memory", **runtime_memory)
        next_step = state["step"] + 1 if state else 1
        if config["training"].get("tensorboard", True):
            log_dir = run_dir / "tensorboard"
            writer = summary_writer_class()(log_dir=str(log_dir), purge_step=0 if state else None)
            if state and (run_dir / "train_metrics.jsonl").exists():
                # Sampling uses a candidate-count axis, unlike training's update
                # axis. A single purge_step would erase committed sampling points.
                # Rebuild both axes from reconciled authoritative logs instead.
                replay_tensorboard(writer, run_dir)
            announce("tensorboard_started", log_dir=str(log_dir), next_step=next_step,
                     entropy="Full-vocabulary entropy in nats, completion-token mean before each optimizer update")
        run_training_loop(config, store, policy, optimizer, updater, env, agent, scorer, examples,
                          run_dir, writer, state=state, max_groups=max_groups)
    finally:
        if writer:
            writer.close()
        if scorer:
            scorer.close()
        if agent:
            agent.service.client.close()
        store.close()
