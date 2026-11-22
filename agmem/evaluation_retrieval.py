"""Bounded cross-query retrieval, advanced entirely by the caller's thread."""
from collections import deque
from dataclasses import dataclass

from .core import CapacityError


@dataclass
class _Episode:
    example: object
    generator: object
    decision: tuple | None = None


def iter_retrievals(env, policy, examples, seed, batch_size=4,
                    max_batch_tokens=16384, query_concurrency=4):
    """Yield ``(example, trajectory)`` as independent retrievals finish.

    At most ``query_concurrency`` queries are active, and each Reader call
    advances up to ``batch_size`` decisions in round-robin order. The policy
    handles padding, model capacity and further splitting by token budget.
    Every query retains the supplied seed; its environment increments decision
    seeds. The caller selects the policy temperature (zero for evaluation).

    Reader generation and environment/store operations stay on the consuming
    thread. A consumer can persist each yielded result and start HTTP scoring
    before requesting more retrievals. Close this iterator when abandoning it
    so unfinished environment generators are released promptly.
    """
    for name, value in (("batch_size", batch_size), ("max_batch_tokens", max_batch_tokens),
                        ("query_concurrency", query_concurrency)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    candidates = enumerate(examples)
    version = policy.policy_version
    active, ready = {}, deque()
    exhausted = False

    def check_version():
        if policy.policy_version != version:
            raise ValueError("Evaluation retrieval requires one unchanged policy version")

    try:
        while active or not exhausted:
            check_version()
            while len(active) < query_concurrency and not exhausted:
                try:
                    index, example = next(candidates)
                except StopIteration:
                    exhausted = True
                    break
                episode = _Episode(example, env.start_episode(example, seed, version))
                active[index] = episode
                try:
                    episode.decision = next(episode.generator)
                except StopIteration as exc:
                    raise ValueError(f"Retrieval {example.sample_id} ended before its first Reader decision") from exc
                except Exception as exc:
                    exc.add_note(f"Starting evaluation retrieval for {example.sample_id}")
                    raise
                ready.append(index)
            if not ready:
                break

            indices = [ready.popleft() for _ in range(min(batch_size, len(ready)))]
            batch = [active[index] for index in indices]
            sample_ids = [episode.example.sample_id for episode in batch]
            messages = [episode.decision[0] for episode in batch]
            seeds = [episode.decision[1] for episode in batch]
            try:
                completions = policy.complete_batch(messages, seeds, max_batch_tokens=max_batch_tokens)
            except CapacityError as exc:
                raise CapacityError(f"Evaluation batched retrieval {sample_ids}: {exc}") from exc
            except Exception as exc:
                exc.add_note(f"Generating evaluation Reader decisions for {sample_ids}")
                raise
            check_version()
            if len(completions) != len(batch):
                raise ValueError(f"Batched Reader returned the wrong number of completions for {sample_ids}: "
                                 f"expected {len(batch)}, got {len(completions)}")

            for index, episode, completion in zip(indices, batch, completions):
                check_version()
                try:
                    episode.decision = episode.generator.send(completion)
                except StopIteration as done:
                    del active[index]
                    yield episode.example, done.value
                except Exception as exc:
                    exc.add_note(f"Advancing evaluation retrieval for {episode.example.sample_id}")
                    raise
                else:
                    ready.append(index)
    finally:
        for episode in active.values():
            episode.generator.close()
