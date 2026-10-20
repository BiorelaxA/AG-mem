from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import json

from .core import canonical, load_config


def main():
    parser = argparse.ArgumentParser(description="AG-Mem session directory + full parameter GRPO")
    parser.add_argument("--config", default="configs/default.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare", help="Adapt fixed train/validation/test; clean annotations and report quarantines")
    command.add_argument("--limit-per-source", type=int)
    command.add_argument("--skip-file-verification", action="store_true", help="Disables fingerprint parsing acceleration too")
    command = commands.add_parser("build-directories", help="Offline session summaries using the frozen 27B service")
    command.add_argument("--limit-memories", type=int)
    command = commands.add_parser("train")
    command.add_argument("--max-groups", type=int)
    command.add_argument("--resume")
    command.add_argument("--output-dir", help="Override training.output_dir; use a fresh directory for a new run")
    command = commands.add_parser("evaluate")
    command.add_argument("--checkpoint", help="Checkpoint's model/ directory; omit for the initial reader baseline")
    command.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    command.add_argument("--limit", type=int)
    command.add_argument("--output", default="runs/evaluation")
    command = commands.add_parser("fingerprint-model")
    command.add_argument("path")
    command.add_argument("--full-weights", action="store_true")
    command = commands.add_parser("inspect-data", help="Validate every answer/choice without parsing large history bodies")
    command.add_argument("--output", default="runs/answer_schema_report.json")
    command = commands.add_parser("export-tensorboard", help="Convert a completed run's JSONL metrics into TensorBoard events")
    command.add_argument("run_dir")
    command = commands.add_parser("export-rollouts", help="Export Reader outputs and final Task Agent answers without retrieved text")
    command.add_argument("run_dir")
    args = parser.parse_args()
    if args.command == "fingerprint-model":
        from .policy import model_fingerprint
        print(json.dumps(model_fingerprint(args.path, args.full_weights), ensure_ascii=False, indent=2))
        return
    if args.command == "export-tensorboard":
        from .monitoring import export_tensorboard
        print(json.dumps(export_tensorboard(args.run_dir), ensure_ascii=False, indent=2))
        return
    if args.command == "export-rollouts":
        from .traces import export_rollouts
        print(json.dumps(export_rollouts(args.run_dir), ensure_ascii=False, indent=2))
        return
    config = load_config(args.config)
    from . import pipeline
    if args.command == "prepare":
        pipeline.prepare(config, args.limit_per_source, not args.skip_file_verification)
    elif args.command == "build-directories":
        pipeline.build_directories(config, args.limit_memories)
    elif args.command == "train":
        if args.output_dir:
            config["training"]["output_dir"] = args.output_dir
        pipeline.train(config, args.max_groups, args.resume)
    elif args.command == "evaluate":
        from .policy import HFPolicy, load_model, load_tokenizer
        path = args.checkpoint or config["models"]["reader_path"]
        policy = HFPolicy(load_model(path, config["reader"]["device"]), load_tokenizer(path), config["reader"], temperature=0.0)
        store = pipeline.open_prepared(config)
        try:
            print(json.dumps(pipeline.evaluate_with_policy(config, store, policy, args.split, args.output, args.limit), ensure_ascii=False, indent=2))
        finally:
            store.close()
    elif args.command == "inspect-data":
        from collections import Counter
        import pyarrow.parquet as pq
        from .core import atomic_json
        from .data import InvalidSample, adapt_row
        counts, issues = Counter(), []
        for source in config["data"]["sources"]:
            for split in config["data"]["splits"]:
                path = Path(config["data"]["input_root"]) / source / f"{split}.parquet"
                file = pq.ParquetFile(path)
                fields = [name for name in ("question", "answer", "question_id", "all_options", "correct_answer", "incorrect_answers") if name in file.schema_arrow.names]
                for index, row in enumerate(pq.read_table(path, columns=fields).to_pylist()):
                    try:
                        example = adapt_row(row, source, split, index)
                        counts[f"{source}/{split}/{example.answer_mode}"] += 1
                    except InvalidSample as exc:
                        counts[f"{source}/{split}/quarantined"] += 1
                        issues.append({"source": source, "split": split, "row": index, "question_id": row.get("question_id"), "reason": str(exc)})
        report = {"counts": dict(counts), "issues": issues}
        atomic_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
