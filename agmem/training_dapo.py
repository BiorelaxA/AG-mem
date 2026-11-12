"""Separate DAPO training orchestration with resumable frozen sampling batches."""
from __future__ import annotations

from pathlib import Path
import json
import time

from .core import InfrastructureError, append_json, atomic_json, trajectory_from_dict
from .dapo_sampling import DynamicGroupSampler, validate_dapo_config


def run_training_loop(config, store, policy, optimizer, updater, env, agent, scorer,
                      examples, run_dir, writer=None, *, state=None, max_groups=None, process_scorer=None):
    """Keep optimizer steps separate from candidates, generation rounds and batches."""
    from .monitoring import write_train_metrics, write_validation_metrics, write_sampling_metrics
    from .pipeline import announce, collect_group, evaluate_with_policy, metrics, save_checkpoint
    from .traces import compact_group, save_training_group
    from .execution import execution_settings
    from .process_advantages import decision_advantage, process_reward_settings
    from .dapo import trajectory_truncated
    from .process_rewards import process_metrics

    process_enabled = process_reward_settings(config["training"])["enabled"]
    if process_enabled and process_scorer is None:
        raise ValueError("Enabled process training requires a frozen attribution scorer")
    examples_by_id = {example.sample_id: example for example in examples}

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
    parallel_sampling = execution_settings(config)["sampling_group_concurrency"] > 1
    pipeline_progress = {}
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

    def record_failure(example, candidate_index, attempt, exc):
        append_json(run_dir / "infrastructure_failures.jsonl", {
            "sample_id": example.sample_id, "candidate_id": candidate_index, "step": step,
            "attempt": attempt, "error": str(exc)})

    def collect(example, candidate_index):
        for attempt in range(config["training"].get("group_infrastructure_retries", 1) + 1):
            try:
                return collect_group(config, env, policy, agent, example, candidate_index, scorer=scorer)
            except InfrastructureError as exc:
                record_failure(example, candidate_index, attempt, exc)
                if attempt == config["training"].get("group_infrastructure_retries", 1):
                    raise

    def pipeline_window(report):
        pipeline_progress.clear()
        pipeline_progress.update(report)
        announce("dapo_sampling_pipeline", step=step, behavior_policy_version=policy.policy_version,
                 progress=report)

    def collect_candidates(candidates):
        from .rollout_pipeline import collect_candidate_groups
        pipeline_progress.clear()
        return collect_candidate_groups(config, env, policy, agent, scorer, candidates,
                                        on_failure=record_failure, on_progress=pipeline_window)

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
        if parallel_sampling:
            # Window counters are cumulative within this generation round,
            # whereas ordinary sampling metrics cover the full refill batch.
            report.update({f"pipeline_round_{key}": value for key, value in pipeline_progress.items()
                           if isinstance(value, (int, float))})
        append_json(run_dir / "sampling_metrics.jsonl", report)
        announce("dapo_sampling", **report)
        if writer:
            write_sampling_metrics(writer, report)

    while True:
        if active is None:
            started = time.monotonic()
            groups, sampling = sampler.collect(collect, policy.policy_version,
                                                record_candidate=candidate, record_round=sampling_round,
                                                collect_candidates=collect_candidates if parallel_sampling else None)
            if not groups:
                announce("dapo_dataset_exhausted", **sampling)
                if step == 0 and not counters.get("skipped_zero_signal_batches", 0):
                    raise RuntimeError("No complete DAPO batch was collected; inspect candidate_metrics.jsonl")
                break
            if process_scorer is not None:
                attribution_started = time.monotonic()
                announce("process_attribution_started", step=step, groups=len(groups),
                         behavior_policy_version=policy.policy_version)
                for query_group in groups:
                    process_scorer.score_group(examples_by_id[query_group[0].sample_id], query_group, policy,
                        on_progress=lambda progress: announce("process_attribution_progress", step=step, **progress))
            active = {"batch_id": sampling["batch_id"], "behavior_version": policy.policy_version,
                      "consumed_groups": 0, "sampling": sampling,
                      "sampling_seconds": time.monotonic() - started,
                      **({"process_scoring_seconds": time.monotonic() - attribution_started}
                         if process_scorer is not None else {})}
        updater.begin_batch(groups, active["batch_id"], consumed_groups=active["consumed_groups"],
                            behavior_version=active["behavior_version"])
        while updater.consumed_groups < len(groups):
            update_groups = groups[updater.consumed_groups:updater.consumed_groups + mini]
            group = [row for query_group in update_groups for row in query_group]
            started = time.monotonic()
            # File indices remain optimizer-step indices for checkpoint recovery.
            # Each rollout retains its sample_id and its own query's advantage.
            has_signal = any(decision_advantage(row, index, enabled=process_enabled) != 0
                for row in group
                if not (config["training"].get("overlong_filter", True) and trajectory_truncated(row))
                for index in range(len(row.decisions)))
            trace_root = run_dir if has_signal else run_dir / "zero_signal_batches" / active["batch_id"]
            save_training_group(trace_root, step,
                                f"{active['batch_id']}/{group[0].sample_id}/update-{step + 1}", env.version, group)
            update = updater.update(group) if mini == 1 else updater.update_groups(update_groups)
            if update.get("optimizer_step_skipped", False):
                # These groups passed the correctness filter. If even the
                # combined step advantages vanish, preserve their audit but do
                # not apply Adam momentum or count a fictitious optimizer step.
                counters["skipped_zero_signal_batches"] = counters.get("skipped_zero_signal_batches", 0) + 1
                counters["zero_signal_groups"] = counters.get("zero_signal_groups", 0) + len(update_groups)
                skipped = {**update, "step": step, "batch_id": active["batch_id"],
                           "skip_index": counters["skipped_zero_signal_batches"],
                           "candidate_index": sampler.candidate_index,
                           "policy_version": policy.policy_version,
                           "sample_ids": [rows[0].sample_id for rows in update_groups],
                           "trace_root": str(trace_root), "elapsed_seconds": time.monotonic() - started}
                append_json(run_dir / "skipped_updates.jsonl", skipped)
                announce("dapo_zero_signal_batch", **skipped)
                if writer:
                    writer.add_scalar("sampling/skipped_zero_signal_batches",
                                      counters["skipped_zero_signal_batches"], sampler.candidate_index)
                    writer.flush()
                continue
            step += 1
            completed += 1
            counters["trained_groups"] += len(update_groups)
            counters["update_rollouts"] += len(group)
            counters["training_tokens"] += update["training_tokens"]
            report = {**metrics(group)["pooled"], **update,
                      **process_metrics(group, overlong_filter=config["training"].get("overlong_filter", True)),
                      **({"process_scoring_seconds": active["process_scoring_seconds"]}
                         if "process_scoring_seconds" in active else {}), "step": step, "epoch": sampler.epoch,
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
    from .prompts import configured_reader_input_contract, task_input_contract
    from .resume import resume_metadata, validate_reader_input_contract, validate_resume_config
    from .execution import execution_settings, sampling_protocol

    if not torch.cuda.is_available() and config["reader"]["device"].startswith("cuda"):
        raise RuntimeError("DAPO training requires the allocated GPUs")
    run_dir = Path(config["training"]["output_dir"])
    if (run_dir / "run.json").exists() and resume is None:
        raise ValueError("Existing training run; use --resume or a new DAPO output_dir")
    previous = resume_metadata(config, resume) if resume else None
    run_dir.mkdir(parents=True, exist_ok=True)
    store, writer, scorer, agent = open_prepared(config), None, None, None
    process_scorer = None
    try:
        splits = ["train", "validation"] if config["training"]["validation_every"] else ["train"]
        validate_directories(config, store, splits)
        examples = task_examples(config, store, "train")
        validation_examples = task_examples(config, store, "validation") if "validation" in splits else []
        scorer = make_reward_scorer(config, store)
        from .process_rewards import make_process_scorer
        process_scorer = make_process_scorer(config, store)
        # Validate rubric metadata before loading the large Reader.
        if scorer.judge is not None:
            for example in examples + validation_examples:
                scorer.judge.validate_example(example)
        runtime_memory = memory_settings()
        implementation = "agmem_dapo_v2" if config["reward"]["formula"] == "judge_cost_v2" else "agmem_dapo_v1"
        if config["reward"]["formula"] == "f1_cost_v1":
            implementation = "agmem_dapo_hierarchical_f1_v1"
        if process_scorer is not None:
            implementation = "agmem_dapo_process_v1"
        metadata = {"config": config, "algorithm": "dapo", "implementation": implementation,
                    "sampling_protocol": sampling_protocol(config),
                    "reader": model_fingerprint(config["models"]["reader_path"]),
                    "task": model_fingerprint(config["models"]["task_path"]),
                    "directory_version": directory_version(store),
                    "directory_service_identity": store.get_metadata("directory_service_identity"),
                    "prepare_identity": store.get_metadata("prepare_identity"),
                    "task_input_contract": task_input_contract(config.get("task_agent", {}).get("prompt_version", "legacy_v1")),
                    "scoring_protocol": scorer.provenance(),
                    "reader_input_contract": configured_reader_input_contract(config),
                    "runtime_memory": runtime_memory, "parameterization": "full",
                    "optimizer": "CPUAdamW_fp32_master_and_moments"}
        if config["data"].get("directory_protocol") == "hierarchical_v1":
            metadata["directory_service_identity"] = store.snapshot["construction_identity"].get("writer")
            metadata["directory_construction_identity"] = store.snapshot["construction_identity"]
        if process_scorer is not None:
            metadata["process_scoring_protocol"] = process_scorer.provenance()
        state = None
        if previous:
            previous_protocol = previous.get("sampling_protocol", sampling_protocol(previous["config"]))
            if previous_protocol != metadata["sampling_protocol"]:
                raise ValueError("DAPO resume provenance mismatch: sampling_protocol")
            for key in ("algorithm", "implementation", "reader", "task", "directory_version", "prepare_identity",
                        "task_input_contract", "scoring_protocol"):
                if previous.get(key) != metadata[key]:
                    raise ValueError(f"DAPO resume provenance mismatch: {key}")
            if previous.get("process_scoring_protocol") != metadata.get("process_scoring_protocol"):
                raise ValueError("DAPO resume provenance mismatch: process_scoring_protocol")
            state = torch.load(Path(resume) / "trainer.pt", map_location="cpu", weights_only=True)
            validate_resume_config(state["config"], config)
            validate_reader_input_contract(state, config)
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
                          run_dir, writer, state=state, max_groups=max_groups, process_scorer=process_scorer)
    finally:
        if writer:
            writer.close()
        if scorer:
            scorer.close()
        if process_scorer:
            process_scorer.close()
        if agent:
            agent.service.client.close()
        store.close()
