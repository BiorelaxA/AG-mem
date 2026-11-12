"""Ordered, bounded cross-query Reader generation and HTTP scoring.

Each fixed window is drained before admitting the next candidates. Within a
window Reader decisions follow a deterministic round-robin schedule, independent
of successful HTTP completion order (which must not affect batched generation's
RNG seed). Infrastructure retries keep individual seeds but may change batch
composition. Only HTTP Task/Judge work runs in the worker pool.
"""
from collections import deque
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from itertools import islice
from queue import Empty, SimpleQueue
import time

from .core import CapacityError, InfrastructureError, seeded_int
from .execution import execution_settings
from .rewards import advantages, score_trajectory


@dataclass(eq=False)
class _Group:
    example: object
    candidate_id: int
    seeds: list[int]
    attempt: int = 0
    episodes: dict = field(default_factory=dict)
    decisions: dict = field(default_factory=dict)
    futures: dict = field(default_factory=dict)
    rows: dict = field(default_factory=dict)


def collect_candidate_groups(config, env, policy, agent, scorer, candidates, *,
                             on_failure=None, on_progress=None):
    """Yield complete groups in candidate order under one frozen policy version.

    ``candidates`` lazily yields (example, candidate_id). At most one configured
    window is admitted, including completed groups waiting for ordered delivery.
    Closing this generator cancels queued scoring and waits for running requests
    before returning; the caller must close it when abandoning iteration.
    """
    execution = execution_settings(config)
    window_size = execution.get("sampling_group_concurrency", 1)
    batch_size = execution["reader_batch_size"]
    group_size = config["training"].get("group_size", 8)
    retries = config["training"].get("group_infrastructure_retries", 1)
    version = policy.policy_version
    candidates = iter(candidates)
    metrics = dict(reader_batches=0, reader_decisions=0, reader_seconds=0.0,
                   scoring_seconds_sum=0.0, scoring_wait_seconds=0.0,
                   peak_groups=0, peak_scoring_inflight=0, group_retries=0,
                   windows_completed=0, groups_completed=0)
    durations = SimpleQueue()
    futures, window, ready = set(), [], deque()
    pool = ThreadPoolExecutor(max_workers=execution["task_agent_workers"],
                              thread_name_prefix="agmem-scoring")

    def check_version():
        if policy.policy_version != version:
            raise ValueError("Candidate collection requires one unchanged policy version")

    def score(example, trajectory):
        started = time.monotonic()
        try:
            trajectory.prediction, trajectory.task_input_hash = agent.answer(
                example, trajectory.evidence, metadata=trajectory.task_completion)
            return (scorer.score(example, trajectory) if scorer else
                    score_trajectory(example, trajectory, config["reward"], max_ids=env.max_ids))
        finally:
            durations.put(time.monotonic() - started)

    def start(group):
        check_version()
        for index, seed in enumerate(group.seeds):
            episode = env.start_episode(group.example, seed, version)
            group.episodes[index] = episode
            group.decisions[index] = next(episode)

    def close_episodes(group):
        for episode in group.episodes.values():
            episode.close()
        group.episodes.clear()
        group.decisions.clear()

    def wait_for(pending, *, first=False):
        if pending:
            started = time.monotonic()
            wait(pending, return_when=FIRST_COMPLETED if first else ALL_COMPLETED)
            metrics["scoring_wait_seconds"] += time.monotonic() - started

    def retry_group(group, exc):
        # No old request from this group may survive into its replacement.
        for future in group.futures.values():
            future.cancel()
        wait_for(list(group.futures.values()))
        for future in group.futures.values():
            if not future.cancelled():
                error = future.exception()
                if error is not None and not isinstance(error, InfrastructureError):
                    raise error
        close_episodes(group)
        if on_failure is not None:
            on_failure(group.example, group.candidate_id, group.attempt, exc)
        if group.attempt >= retries:
            raise exc
        # Retain the original per-rollout seeds, discard every old score, and
        # repeat the complete group rather than filling a missing trajectory.
        group.attempt += 1
        metrics["group_retries"] += 1
        group.futures.clear()
        group.rows.clear()
        remaining = [(other, index) for other, index in ready if other is not group]
        ready.clear()
        ready.extend(remaining)
        start(group)
        ready.extend((group, index) for index in range(group_size))

    def poll():
        # Inspect errors before submitting further work. Iteration order is
        # stable, even when several HTTP workers finish at the same instant.
        for group in window:
            failure = None
            for index, future in group.futures.items():
                if index in group.rows or not future.done():
                    continue
                try:
                    group.rows[index] = future.result()
                except InfrastructureError as exc:
                    failure = exc
                    break
            if failure is not None:
                retry_group(group, failure)

    try:
        while True:
            check_version()
            admitted = list(islice(candidates, window_size))
            if not admitted:
                break
            window = [_Group(example, candidate_id, [
                seeded_int(config["seed"], example.sample_id, version, candidate_id, index) % (2 ** 31)
                for index in range(group_size)]) for example, candidate_id in admitted]
            metrics["peak_groups"] = max(metrics["peak_groups"], len(window))
            for group in window:
                start(group)
            # Interleave queries before rollout indices so even the first
            # batch can share Reader compute across different memory snapshots.
            ready.extend((group, index) for index in range(group_size) for group in window)
            next_yield = 0
            while next_yield < len(window):
                check_version()
                poll()
                while next_yield < len(window) and len(window[next_yield].rows) == group_size:
                    group = window[next_yield]
                    rows = [group.rows[index] for index in range(group_size)]
                    if any(row.policy_version != version for row in rows):
                        raise ValueError("Group contains a different policy version")
                    for row, advantage in zip(rows, advantages(
                            [row.reward for row in rows], config["training"].get("advantage_epsilon", 1e-6))):
                        row.advantage = advantage
                    check_version()
                    yield group.example, group.candidate_id, rows
                    check_version()
                    metrics["groups_completed"] += 1
                    next_yield += 1
                if ready:
                    batch = [ready.popleft() for _ in range(min(batch_size, len(ready)))]
                    messages = [group.decisions[index][0] for group, index in batch]
                    seeds = [group.decisions[index][1] for group, index in batch]
                    started = time.monotonic()
                    try:
                        if batch_size == 1:
                            completions = [policy.complete(messages[0], seed=seeds[0])]
                        else:
                            completions = policy.complete_batch(messages, seeds,
                                max_batch_tokens=execution["reader_batch_max_tokens"])
                    except CapacityError as exc:
                        ids = [group.example.sample_id for group, _ in batch]
                        raise CapacityError(f"Cross-query batched retrieval {ids}: {exc}") from exc
                    finally:
                        metrics["reader_seconds"] += time.monotonic() - started
                    check_version()
                    if len(completions) != len(batch):
                        raise ValueError("Batched Reader returned the wrong number of completions")
                    metrics["reader_batches"] += 1
                    metrics["reader_decisions"] += len(batch)
                    for (group, index), completion in zip(batch, completions):
                        try:
                            group.decisions[index] = group.episodes[index].send(completion)
                            ready.append((group, index))
                        except StopIteration as done:
                            del group.decisions[index]
                            future = pool.submit(score, group.example, done.value)
                            group.futures[index] = future
                            futures.add(future)
                            metrics["peak_scoring_inflight"] = max(metrics["peak_scoring_inflight"],
                                sum(not item.done() for item in futures))
                elif next_yield < len(window):
                    wait_for([future for group in window for future in group.futures.values()
                              if not future.done()], first=True)
            # All scores have been consumed and every group delivered. No new
            # candidates enter based on which HTTP response happened to win.
            for group in window:
                close_episodes(group)
            futures.clear()
            metrics["windows_completed"] += 1
            while True:
                try:
                    metrics["scoring_seconds_sum"] += durations.get_nowait()
                except Empty:
                    break
            if on_progress is not None:
                on_progress({"event": "sampling_window_completed", **metrics})
    finally:
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        for group in window:
            close_episodes(group)
