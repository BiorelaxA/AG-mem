"""Continue frozen DeepSeek M directories with one attempt per generated entry.

Committed source results, including fallbacks, are imported unchanged before
inference. Invalid generated entries become honest placeholders; remote service
failures still stop the build. Run this scoped adapter in its own process.
"""

from contextlib import contextmanager
from copy import copy, deepcopy
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock

from . import build_adjacent_directories as engine
from . import build_deepseek_directories as deepseek
from . import build_deepseek_directories_gpu as launcher
from . import import_deepseek_directories as importer
from .build_topic_event_directories import (
    BOUNDARY_PROMPT as SOURCE_BOUNDARY_PROMPT,
    BOUNDARY_PROMPT_VERSION as SOURCE_BOUNDARY_PROMPT_VERSION,
)
from .core import file_digest
from .hierarchical.adjacent_writer import _BudgetExceeded
from .hierarchical.writer import _strict_json, chat_token_count, token_count


DEFAULT_CONFIG = "configs/build_longmemeval_m_deepseek_no_repair.yaml"
PROTOCOL = "adjacent_topic_event_deepseek_no_repair_v3"
DEFAULT_SOURCE = importer.DEFAULT_SOURCE
DEFAULT_TARGET = "runs/parent_child_directories/deepseek_flash_lmem_adjacent_no_repair_v3"
BOUNDARY_PROMPT = (Path(__file__).with_name("hierarchical") / "resources" / "adjacent" /
                   "topic_event_no_reason_v1" / "boundary.txt")
BOUNDARY_PROMPT_VERSION = "topic_event.no_reason.v1"
_BASE_COMPATIBLE_IDENTITY = importer.compatible_identity
_LOCK = RLock()
NO_REPAIR_POLICY = {
    "version": "deepseek_entry_no_repair.v1",
    "logical_requests_per_entry": 1,
    "semantic_retries": 0,
    "retry_fallback": False,
    "invalid_json_schema_or_text_budget": "placeholder_without_retry",
    "output_finish_reason_length": "placeholder_without_retry",
    "other_invalid_finish_reason": "placeholder_without_retry",
    "output_budget_expansion": False,
    "output_failure_source_splitting": False,
    "unavailable_boundary_review": "keep_boundary",
    "input_overflow": "unchanged_complete_source_block_windows_and_reduction",
    "transport_failure": "configured_transport_retries_then_fail_session_and_stop_admission",
    "source_reuse": "committed_complete_empty_and_fallback_unchanged",
    "source_inference_caches_and_traces": "excluded",
}


class NoRepairDeepSeekDirectoryWriter(deepseek.DeepSeekDirectoryWriter):
    """Use one preceding source block and never retry an invalid generated entry."""

    def __init__(self, tokenizer, settings, complete):
        super().__init__(tokenizer, settings, complete)
        if self.settings["semantic_retries"] != 0:
            raise ValueError("No-repair Writer requires semantic_retries=0")
        if self.settings["context_blocks"] != 1:
            raise ValueError("This Writer requires context_blocks=1")
        self.adjacent_prompts["boundary"] = BOUNDARY_PROMPT.read_text()

    def _schema(self, kind):
        schema = super()._schema(kind)
        if kind == "boundary":
            del schema["properties"]["reason"]
            schema["required"].remove("reason")
        return schema

    def _validate_entry(self, output, kind, payload):
        if kind != "boundary":
            return super()._validate_entry(output, kind, payload)
        from jsonschema import Draft202012Validator

        Draft202012Validator(self._schema(kind)).validate(output)
        allowed = {block["index"] for side in ("left", "right") for block in payload[side]["blocks"]}
        if any(type(index) is not int or index not in allowed for index in output["evidence_blocks"]):
            raise ValueError("Boundary evidence must reference supplied source Blocks")

    def _request_entry(self, payload, kind, stage):
        from jsonschema import ValidationError

        schema = self._schema(kind)
        messages = self._entry_messages(payload, kind)
        budget = self.settings["output_tokens"]
        if min(self.settings["max_output_tokens"], self._available(messages)) < budget:
            # This exception is only possible BEFORE generation. The existing
            # engine may window the complete source to fit its input capacity.
            raise _BudgetExceeded("Original source plus output reservation exceeds context")
        response = self.complete(messages, budget, schema, stage)
        self.stats["calls"] += 1
        if "rollup" in stage:
            self.stats["rollup_calls"] += 1
        record = {"stage": stage, "max_tokens": budget, "repair_index": 0,
                  "input_tokens": chat_token_count(self.tokenizer, messages),
                  "finish_reason": response.get("finish_reason"),
                  "write_range": deepcopy(payload.get("write_range")),
                  "repair_policy": NO_REPAIR_POLICY["version"]}
        if response.get("finish_reason") == "length":
            record.update(valid=False, reason="truncated_output")
            self.traces.append(record)
            self._hook("reject", messages, budget, schema, stage, response, reason="truncated_output")
            return None
        try:
            if response.get("finish_reason") != "stop":
                raise ValueError("Entry generation did not finish normally")
            if not isinstance(response.get("text"), str) or not response["text"].strip():
                raise ValueError("Entry generation returned no text")
            result = _strict_json(response["text"])
            self._validate_entry(result, kind, payload)
        except (ValueError, TypeError, KeyError, ValidationError) as exc:
            errors = [str(getattr(exc, "message", exc))[:1600]]
            record.update(valid=False, reason="entry_validation", errors=errors)
            self.traces.append(record)
            self._hook("reject", messages, budget, schema, stage, response, reason=errors[0])
            return None
        record.update(valid=True, status="complete")
        self.traces.append(record)
        self._hook("accept", messages, budget, schema, stage, response, status="complete")
        return result


@contextmanager
def _no_repair_engine():
    with _LOCK, deepseek._LOCK:
        previous = engine.PROTOCOL, engine.DATASETS, engine.load_recipe, engine.bounded_results
        engine.PROTOCOL, engine.DATASETS = PROTOCOL, {"split_longmemeval_m"}
        engine.load_recipe, engine.bounded_results = load_recipe, deepseek._fail_fast_results
        try:
            yield
        finally:
            engine.PROTOCOL, engine.DATASETS, engine.load_recipe, engine.bounded_results = previous


def load_recipe(path, workers=None, *, embedding_model_path=None, output_root=None,
                allow_missing_embedding=False):
    # The frozen loader supplies the actual remote Writer identity and identical
    # prompts/tokenizer/budgets. Imported targets have a journal before their
    # first run manifest, so validate the FULL new identity below instead.
    with _LOCK, deepseek._LOCK:
        protocol, check_root = deepseek.PROTOCOL, deepseek._check_fresh_or_matching_root
        deepseek.PROTOCOL = PROTOCOL
        deepseek._check_fresh_or_matching_root = lambda root: None
        try:
            config, identity = deepseek.load_recipe(path, workers,
                embedding_model_path=embedding_model_path, output_root=output_root,
                allow_missing_embedding=allow_missing_embedding)
        finally:
            deepseek.PROTOCOL, deepseek._check_fresh_or_matching_root = protocol, check_root
    if config["writer"]["semantic_retries"] != 0 or config.get("retry_fallback") is not False:
        raise ValueError("No-repair continuation requires semantic_retries=0 and retry_fallback=false")
    if config["writer"]["context_blocks"] != 1:
        raise ValueError("This continuation requires context_blocks=1")
    if not config.get("source_root"):
        raise ValueError("Configure the frozen DeepSeek source_root")
    config["source_root"] = str(Path(config["source_root"]).resolve())
    identity = deepcopy(identity)
    identity["boundary_prompt_version"] = BOUNDARY_PROMPT_VERSION
    identity["prompt_hashes"]["boundary.txt"] = file_digest(BOUNDARY_PROMPT)
    identity["boundary_output_fields"] = ["decision", "evidence_blocks"]
    identity["no_repair_policy"] = deepcopy(NO_REPAIR_POLICY)
    identity["directory_reuse_policy"] = "preserve_deepseek_commits_then_deepseek_no_repair.v1"
    for resource in (Path(__file__), Path(importer.__file__)):
        identity["implementation_hashes"][resource.name] = file_digest(resource)
    importer._existing_target(Path(config["output_root"]), identity)
    return config, identity


def compatible_identity(source, target):
    """Admit one-block context, no repairs, and the versioned compact boundary output."""
    if source.get("protocol") != importer.SOURCE_PROTOCOL or target.get("protocol") != PROTOCOL:
        raise ValueError("No-repair import requires the frozen DeepSeek source and no-repair target protocols")
    source_retries = source.get("writer_settings", {}).get("semantic_retries")
    target_retries = target.get("writer_settings", {}).get("semantic_retries")
    if (type(source_retries) is not int or source_retries != 2
            or type(target_retries) is not int or target_retries != 0):
        raise ValueError("No-repair import permits only semantic_retries 2 -> 0")
    source_context = source.get("writer_settings", {}).get("context_blocks")
    target_context = target.get("writer_settings", {}).get("context_blocks")
    if (type(source_context) is not int or source_context != 2
            or type(target_context) is not int or target_context != 1):
        raise ValueError("This import permits only context_blocks 2 -> 1")
    if (source.get("boundary_prompt_version") != SOURCE_BOUNDARY_PROMPT_VERSION
            or source.get("prompt_hashes", {}).get("boundary.txt") != file_digest(SOURCE_BOUNDARY_PROMPT)
            or target.get("boundary_prompt_version") != BOUNDARY_PROMPT_VERSION
            or target.get("prompt_hashes", {}).get("boundary.txt") != file_digest(BOUNDARY_PROMPT)
            or target.get("boundary_output_fields") != ["decision", "evidence_blocks"]):
        raise ValueError("Only the verified topic/event boundary prompt -> no-reason prompt migration is allowed")
    comparison = deepcopy(target)
    comparison["protocol"] = importer.TARGET_PROTOCOL
    comparison["writer_settings"]["semantic_retries"] = 2
    comparison["writer_settings"]["context_blocks"] = 2
    comparison["boundary_prompt_version"] = source["boundary_prompt_version"]
    comparison["prompt_hashes"]["boundary.txt"] = source["prompt_hashes"]["boundary.txt"]
    _BASE_COMPATIBLE_IDENTITY(source, comparison)
    for key in ("writer", "request_options", "budget_tokenizer"):
        if source.get(key) != target.get(key):
            raise ValueError(f"Source and target {key} semantics differ")


def import_committed(source_root, target_root, target_identity, **kwargs):
    # Only the compatibility comparison sees normalized copies. Import manifests,
    # target journals and per-session reuse provenance retain the real identities.
    with _LOCK:
        previous = importer.compatible_identity
        importer.compatible_identity = compatible_identity
        try:
            return importer.import_committed(source_root, target_root, target_identity, **kwargs)
        finally:
            importer.compatible_identity = previous


def run(args, **factories):
    config, identity = load_recipe(args.config, args.workers,
        embedding_model_path=getattr(args, "embedding_model_path", None),
        output_root=getattr(args, "output_root", None), allow_missing_embedding=args.dry_run)
    source_root = getattr(args, "source_root", None) or config["source_root"]
    _, preview = engine.plan(config, identity, args.datasets, args.splits)
    limit = getattr(args, "limit_sessions", None)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit_sessions must be a positive integer")
    endpoints = getattr(args, "embedding_base_urls", None)
    if endpoints:
        endpoints = deepseek.embedding_endpoints(endpoints)
    if not args.dry_run and not getattr(args, "import_only", False) and not endpoints:
        raise ValueError("Provide --embedding-base-urls or use the two-embedding-GPU launcher")
    count_tokens = None
    if not args.dry_run:
        from .policy import load_tokenizer

        tokenizer = factories.get("tokenizer")
        if tokenizer is None:
            tokenizer = load_tokenizer(config["model_path"])
            factories["tokenizer"] = tokenizer
        count_tokens = lambda text: token_count(tokenizer, text)
    imported = import_committed(source_root, config["output_root"], identity,
                                dry_run=args.dry_run, count_tokens=count_tokens)
    engine.announce("deepseek_no_repair_import", **{key: imported[key] for key in (
        "version", "source_root", "output_root", "status", "source_states", "source_banks",
        "registered_sessions", "counters", "import_already_complete", "excluded_artifacts") if key in imported})
    if getattr(args, "import_only", False):
        return imported
    preview["no_repair_policy"] = deepcopy(NO_REPAIR_POLICY)
    preview["boundary_prompt_version"] = identity["boundary_prompt_version"]
    preview["boundary_output_fields"] = list(identity["boundary_output_fields"])
    engine.announce("deepseek_no_repair_plan", **preview)
    if args.dry_run:
        return preview
    planned = copy(args)
    planned.base_urls, planned.base_url = [config["service"]["base_url"]], None
    planned.embedding_base_url = endpoints[0]
    if factories.get("pool_factory") is None:
        factories["pool_factory"] = deepseek.DeepSeekWriterPool
    if factories.get("writer_factory") is None:
        factories["writer_factory"] = NoRepairDeepSeekDirectoryWriter
    if factories.get("segmenter_factory") is None:
        factories["segmenter_factory"] = deepseek.DeepSeekSegmenter
    if factories.get("embedding_factory") is None:
        def create_encoder(_base_url, tokenizer, **kwargs):
            return deepseek.MultiEndpointEmbeddingClient(endpoints, tokenizer, **kwargs)
        factories["embedding_factory"] = create_encoder
    with _no_repair_engine():
        return engine.run(planned, **factories)


def launch(args):
    """Complete the CPU import before the existing launcher probes API or GPU."""
    launcher._validate_options(args)
    cpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    with _LOCK:
        build_command = launcher.build_command

        def command(options, endpoints=None):
            result = build_command(options, endpoints)
            index = result.index("agmem.build_deepseek_directories")
            result[index:index + 1] = ["agmem.build_deepseek_no_repair", "worker"]
            if getattr(options, "source_root", None):
                result.extend(["--source-root", options.source_root])
            if getattr(options, "import_only", False):
                result.append("--import-only")
            return result

        launcher.build_command = command
        try:
            if args.dry_run or args.import_only:
                subprocess.run(command(args, args.embedding_base_urls), check=True, env=cpu_env)
                return None
            imported = copy(args)
            imported.import_only = True
            subprocess.run(command(imported), check=True, env=cpu_env)
            return launcher.launch(args)
        finally:
            launcher.build_command = build_command


def argument_parser(*, worker_mode=False):
    parser = deepseek.argument_parser() if worker_mode else launcher.argument_parser()
    parser.description = __doc__
    parser.set_defaults(config=DEFAULT_CONFIG, workers=256)
    parser.add_argument("--source-root")
    parser.add_argument("--import-only", action="store_true",
                        help="Validate and import frozen directories on CPU; do not start services")
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    worker_mode = bool(argv and argv[0] == "worker")
    if worker_mode:
        argv.pop(0)
    args = argument_parser(worker_mode=worker_mode).parse_args(argv)
    (run if worker_mode else launch)(args)


if __name__ == "__main__":
    main()
