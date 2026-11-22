"""Bounded, resumable Task/Judge scoring across independent evaluation queries."""

from .core import trajectory_from_dict
from .execution import bounded_results


def score_pending(examples, retrieval, journal, agent, scorer, workers=4,
                  retrieval_rows=None):
    """Yield ``(index, example, row, is_new)`` with bounded query concurrency.

    Completed journal rows are yielded without another model request. Missing
    rows are scored in completion order; callers should place results at
    ``index`` to retain the original example order for metrics and categories.

    Journal/evidence reads and successful journal writes happen on the caller's
    thread. Each worker only answers one question, then judges that answer.
    At most ``workers`` full trajectories are scheduled at once. If loading or
    scoring fails, successful in-flight queries are persisted and yielded before
    the first error is raised. Failures never become zero-scored episodes.

    ``retrieval_rows`` optionally supplies a lazy ``(example, trajectory)``
    stream, already persisted by the retrieval stage. It is consumed on the
    caller thread while workers score earlier queries. Cached answers are
    skipped, but the stream is still consumed completely on success.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("Evaluation scoring workers must be a positive integer")
    examples = list(examples)
    sample_ids = [example.sample_id for example in examples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Evaluation scoring requires unique sample IDs")
    examples_by_id = {example.sample_id: (index, example)
                      for index, example in enumerate(examples)}

    pending = []
    for index, example in enumerate(examples):
        value = journal.read(example.sample_id)
        if value is None:
            pending.append((index, example))
            continue
        row = trajectory_from_dict(value)
        _validate_history(example, row)
        yield index, example, row, False

    def inputs():
        if retrieval_rows is not None:
            missing = {example.sample_id for _, example in pending}
            seen = set()
            for retrieved_example, row in retrieval_rows:
                sample_id = retrieved_example.sample_id
                if sample_id not in examples_by_id:
                    raise ValueError(f"Unexpected retrieval sample ID: {sample_id}")
                if sample_id in seen:
                    raise ValueError(f"Duplicate retrieval sample ID: {sample_id}")
                seen.add(sample_id)
                index, example = examples_by_id[sample_id]
                _validate_history(example, row)
                if sample_id not in missing:
                    continue
                missing.remove(sample_id)
                yield index, example, row
            if missing:
                raise ValueError(f"Missing retrieval for {min(missing)}; retrieval stream ended early")
            return
        for index, example in pending:
            value = retrieval.read(example.sample_id)
            if value is None:
                raise ValueError(f"Missing retrieval for {example.sample_id}; run the retrieve stage first")
            row = trajectory_from_dict(value)
            _validate_history(example, row)
            yield index, example, row

    def score(item):
        _, example, row = item
        row.prediction, row.task_input_hash = agent.answer(
            example, row.evidence, metadata=row.task_completion)
        scorer.score(example, row)
        return row

    for (index, example, _), row in bounded_results(score, inputs(), workers):
        journal.write(row)
        yield index, example, row, True


def _validate_history(example, row):
    if row.sample_id != example.sample_id or row.memory_id != example.memory_id:
        raise ValueError(f"Retrieved history does not match the test example: {example.sample_id}")
