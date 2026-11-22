"""Local Qwen Writer for the two held-out LongMemEval-M splits."""
from pathlib import Path

from . import build_adjacent_directories as engine
from .build_topic_event_directories import argument_parser
from .build_deepseek_no_repair import (
    BOUNDARY_PROMPT, BOUNDARY_PROMPT_VERSION, NoRepairDeepSeekDirectoryWriter,
)
from .build_qwen_continued_directories import QwenContinuationSegmenter, _fail_fast_results
from .core import file_digest
from .hierarchical.writer import DirectoryWriter

PROTOCOL = "adjacent_topic_event_local_qwen_eval_v1"
SPLITS = ("validation", "test")


class LocalDirectoryWriter(NoRepairDeepSeekDirectoryWriter):
    # The local vLLM receives the JSON schema via structured_outputs, outside chat messages.
    _available = DirectoryWriter._available


def main():
    parser = argument_parser()
    parser.set_defaults(config="configs/build_longmemeval_m_qwen_eval.yaml", splits=list(SPLITS))
    args = parser.parse_args()
    if set(args.splits) - set(SPLITS):
        parser.error("Only validation and test are supported by the M eval builder")
    # This dedicated worker runs in a separate interpreter. Keep original
    # journal fingerprints and entry points available for historical runs.
    original = engine.load_recipe

    def recipe(*values, **kwargs):
        config, identity = original(*values, **kwargs)
        if (len(config["datasets"]) != 1 or set(config["datasets"][0]["splits"]) != set(SPLITS)
                or config.get("retry_fallback") is not False or config["writer"]["context_blocks"] != 1
                or config["writer"]["semantic_retries"] != 0):
            raise ValueError("Expected the M validation/test one-block, no-repair recipe")
        identity["boundary_prompt_version"] = BOUNDARY_PROMPT_VERSION
        identity["prompt_hashes"]["boundary.txt"] = file_digest(BOUNDARY_PROMPT)
        identity["boundary_output_fields"] = ["decision", "evidence_blocks"]
        for name in (Path(__file__).name, "build_deepseek_no_repair.py", "build_deepseek_directories.py",
                     "build_qwen_continued_directories.py", "build_topic_event_directories.py"):
            identity["implementation_hashes"][name] = file_digest(Path(__file__).with_name(name))
        return config, identity

    saved = engine.PROTOCOL, engine.DATASETS, engine.load_recipe, engine.bounded_results
    engine.PROTOCOL, engine.DATASETS = PROTOCOL, {"split_longmemeval_m"}
    engine.load_recipe, engine.bounded_results = recipe, _fail_fast_results
    try:
        return engine.run(args, writer_factory=LocalDirectoryWriter, segmenter_factory=QwenContinuationSegmenter)
    finally:
        engine.PROTOCOL, engine.DATASETS, engine.load_recipe, engine.bounded_results = saved


if __name__ == "__main__":
    main()
