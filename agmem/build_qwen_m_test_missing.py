"""Run the existing local evaluation Writer on the missing M test session addresses."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from . import build_qwen_m_eval_directories as evaluation
from .build_adjacent_directories_gpu import load_recipe
from .core import InfrastructureError, digest, file_digest
from .hierarchical.adapters import input_files
from .hierarchical.longmemeval_range import range_banks


class TestSupplementWriter(evaluation.LocalDirectoryWriter):
    """Handle an exhausted local Writer read timeout at the individual entry."""

    def _request_entry(self, payload, kind, stage):
        import httpx

        try:
            return super()._request_entry(payload, kind, stage)
        except InfrastructureError as exc:
            # SessionClient wraps the final HTTP exception after its retry loop.
            if not isinstance(exc.__context__, httpx.ReadTimeout):
                raise
            self.traces.append({"stage": stage, "valid": False,
                                "reason": "writer_read_timeout_after_retries",
                                "write_range": payload.get("write_range")})
            # The existing writer binds a placeholder to this exact source span.
            # For boundary reviews, it keeps the original boundary instead.
            return None


def missing_test_sessions(spec, sha, session_counts, output):
    baseline = Path(spec["exclude_committed_root"]).resolve()
    output = Path(output).resolve()
    if baseline == output:
        raise ValueError("Write the test supplement to its own journal")
    wanted = [{f"p{i:04d}" for i in range(1, n + 1)} for n in session_counts]
    for root in (baseline, output):
        journal = root / "build_journal.sqlite3"
        if root == output and not journal.exists():
            continue
        with closing(sqlite3.connect(journal.as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("BEGIN")
            identity = json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0])
            files = [f["sha256"] for d in identity["datasets"] if d["spec"]["name"] == spec["name"]
                     for f in d["files"] if f["split"] == "test"]
            if identity["protocol"] != evaluation.PROTOCOL or files != [sha]:
                raise ValueError(f"Dataset or evaluation protocol mismatch in {root}")
            for i, parents in enumerate(wanted):
                bid = "bank_" + digest([spec["name"], "test", sha, i])[:24]
                for (pid,) in db.execute("SELECT parent_id FROM sessions WHERE bank_id=? AND directory IS NOT NULL "
                                         "AND status IN ('complete','fallback','empty')", (bid,)):
                    parents.remove(pid)
    return {i: parents for i, parents in enumerate(wanted) if parents}


def main():
    args = evaluation.argument_parser().parse_args()
    config = load_recipe(args.config)
    spec, = config["datasets"]
    if spec["name"] != "split_longmemeval_m" or args.splits != ["test"]:
        raise ValueError("This job builds LongMemEval-M test only")
    (_, path), = input_files(spec, ["test"])
    sha = file_digest(path)
    counts = [len(s) for s in pq.read_table(path, columns=["haystack_session_ids"]).column(0)]
    wanted = missing_test_sessions(spec, sha, counts, args.output_root or config["output_root"])
    evaluation.engine.announce("qwen_test_missing", trajectories=len(wanted),
                               sessions=sum(map(len, wanted.values())), total_sessions=sum(counts))

    def selected_banks(dataset_spec, splits, *, count_tokens, tokenizer_id):
        return range_banks(path, sha, "test", 1, len(counts), count_tokens=count_tokens, wanted=wanted)

    original = evaluation.engine.iter_banks
    original_writer = evaluation.LocalDirectoryWriter
    evaluation.engine.iter_banks = selected_banks
    evaluation.LocalDirectoryWriter = TestSupplementWriter
    try:
        evaluation.main()
    finally:
        evaluation.LocalDirectoryWriter = original_writer
        evaluation.engine.iter_banks = original


if __name__ == "__main__":
    main()
