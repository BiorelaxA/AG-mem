"""Bounded scheduling; execution settings do not change Writer cache identity."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


DEFAULTS = {"writer_workers": 1, "task_agent_workers": 1,
            "reader_batch_size": 1, "reader_batch_max_tokens": 16384}


def execution_settings(config):
    values = {**DEFAULTS, **config.get("execution", {})}
    unknown = values.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown execution settings: {sorted(unknown)}")
    for name, value in values.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"execution.{name} must be a positive integer")
    if values["reader_batch_size"] > 8:
        raise ValueError("execution.reader_batch_size cannot exceed the group size (8)")
    return values


def bounded_results(function, items, workers):
    """Yield (item, result) on completion, with at most `workers` items in flight.

    The caller prepares items and persists results on its own thread. On failure,
    stop submitting, drain successful in-flight results, then raise the first error.
    This keeps partial Writer work reusable without publishing an incomplete history.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if workers == 1:
        for item in items:
            yield item, function(item)
        return
    iterator = iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending, exhausted, failure = {}, False, None
        while pending or not exhausted:
            while len(pending) < workers and not exhausted and failure is None:
                try:
                    item = next(iterator)
                except StopIteration:
                    exhausted = True
                except Exception as exc:
                    failure, exhausted = exc, True
                else:
                    pending[pool.submit(function, item)] = item
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                item = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    if failure is None:
                        failure = exc
                    exhausted = True
                else:
                    yield item, result
        if failure is not None:
            raise failure


def make_job_config(source, destination, environ):
    """Overlay scheduling for a new job without modifying its frozen base config."""
    from pathlib import Path
    import yaml
    from .core import load_config
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise ValueError("Job configuration must be a new file, separate from the frozen configuration")
    config = load_config(source)
    config["execution"] = {**DEFAULTS, "writer_workers": 2, "task_agent_workers": 2,
                           "reader_batch_size": 2, **config.get("execution", {})}
    for key in DEFAULTS:
        value = environ.get("AGMEM_" + key.upper())
        if value is not None:
            config["execution"][key] = int(value)
    settings = execution_settings(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    temporary.replace(destination)
    return max(settings["writer_workers"], settings["task_agent_workers"])


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Create a per-job scheduling overlay; print required vLLM max-num-seqs")
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    print(make_job_config(args.source, args.destination, os.environ))
