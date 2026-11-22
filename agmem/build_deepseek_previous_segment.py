"""DeepSeek child summaries with the immediately preceding fixed source span."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
import json

from . import build_deepseek_directories as deepseek
from . import build_deepseek_no_repair as v3
from . import import_deepseek_directories as importer
from .build_portable_deepseek_directories import portable_worker
from .core import file_digest
from .hierarchical.adjacent_writer import validate_spans

PROTOCOL = "adjacent_topic_event_deepseek_previous_segment_v1"
CHILD_PROMPT = v3.BOUNDARY_PROMPT.parent.parent / "previous_segment_v1" / "child.txt"
OLD_CHILD = CHILD_PROMPT.parent.parent / "child.txt"
POLICY = {"unit": "previous_complete_segment", "count": 1, "source": "raw_blocks",
          "scope": "same_session", "overflow": "child_placeholder"}


class PreviousSegmentWriter(v3.NoRepairDeepSeekDirectoryWriter):
    def __init__(self, tokenizer, settings, complete):
        deepseek.DeepSeekDirectoryWriter.__init__(self, tokenizer, settings, complete)
        if self.settings["context_blocks"] != 0 or self.settings["semantic_retries"] != 0:
            raise ValueError("Previous-segment Writer requires zero block context and zero repairs")
        self.adjacent_prompts.update(child=CHILD_PROMPT.read_text(), boundary=v3.BOUNDARY_PROMPT.read_text())

    def build(self, session, spans):
        fixed = validate_spans(spans, len(session["blocks"]))
        self.previous_by_start = {span["start_block"]: fixed[i - 1] if i else None
                                  for i, span in enumerate(fixed)}
        return super().build(session, fixed)

    def _source_payload(self, start, end, kind, children):
        if kind != "child":
            return super()._source_payload(start, end, kind, children)
        data = self._raw_payload(start, end, kind, children)
        del data["context_blocks"]
        previous = self.previous_by_start[start]
        data["context_segments"] = [] if previous is None else [
            {**previous, "blocks": self.public["blocks"][previous["start_block"] - 1:previous["end_block"]]}]
        return data

    def _summarize_range(self, start, end, kind, children):
        # Keep the fixed segment and its context together even on overflow.
        if kind == "child" and not self._fits(self._entry_messages(self._source_payload(start, end, kind, children), kind)):
            self.traces.append({"stage": "child_context_overflow", "write_range": {"start": start, "end": end}})
            return self._placeholder(kind, start, end), True
        return super()._summarize_range(start, end, kind, children)


def load_recipe(path, workers=None, **options):
    original_protocol = deepseek.PROTOCOL
    try:
        deepseek.PROTOCOL = PROTOCOL
        config, identity = deepseek.load_recipe(path, workers, **options)
    finally:
        deepseek.PROTOCOL = original_protocol
    if (config.get("child_context_segments") != 1 or config["writer"]["context_blocks"] != 0
            or config["writer"]["semantic_retries"] != 0 or config.get("retry_fallback") is not False):
        raise ValueError("Expected one complete context segment, zero block context and no repairs")
    identity.update(boundary_prompt_version=v3.BOUNDARY_PROMPT_VERSION,
                    boundary_output_fields=["decision", "evidence_blocks"], child_context_policy=POLICY,
                    directory_reuse_policy="preserve_committed_then_previous_segment.v1",
                    no_repair_policy={**v3.NO_REPAIR_POLICY, "child_input_overflow": "placeholder"})
    identity["prompt_hashes"].update({"child.txt": file_digest(CHILD_PROMPT),
                                     "boundary.txt": file_digest(v3.BOUNDARY_PROMPT)})
    for path in (Path(__file__), Path(v3.__file__), Path(importer.__file__)):
        identity["implementation_hashes"][path.name] = file_digest(path)
    importer._existing_target(Path(config["output_root"]), identity)
    return config, identity


@contextmanager
def previous_segment_worker(portable=None):
    old_load, old_compare, old_files = v3.load_recipe, v3.compatible_identity, importer._verify_files

    def compare(source, target):
        if (target["protocol"] != PROTOCOL or target.get("child_context_policy") != POLICY
                or target["prompt_hashes"]["child.txt"] != file_digest(CHILD_PROMPT)):
            raise ValueError("Unknown previous-segment construction identity")
        adjusted = deepcopy(target)
        adjusted["protocol"] = v3.PROTOCOL
        adjusted["prompt_hashes"]["child.txt"] = file_digest(OLD_CHILD)
        adjusted["writer_settings"]["context_blocks"] = 1
        if source["protocol"] == importer.SOURCE_PROTOCOL:
            return old_compare(source, adjusted)
        if source["protocol"] != v3.PROTOCOL:
            raise ValueError("Expected a frozen original or v3 DeepSeek build")
        for key in ("writer", "request_options", "budget_tokenizer", "boundary_output_fields"):
            if source[key] != adjusted[key]:
                raise ValueError(f"v3 source differs in {key}")
        before = dict(source, protocol=importer.SOURCE_PROTOCOL)
        v3._BASE_COMPATIBLE_IDENTITY(before, dict(adjusted, protocol=importer.TARGET_PROTOCOL))

    def verify(source, target):
        if portable and source["protocol"] == v3.PROTOCOL:
            if source.get("portable_migration", {}).get("manifest_sha256") != portable.report["manifest_sha256"]:
                raise ValueError("v3 continuation comes from a different transfer manifest")
            return portable.verify_files(portable.source_identity, target)
        return old_files(source, target)

    v3.load_recipe, v3.compatible_identity, importer._verify_files = load_recipe, compare, verify
    try:
        yield
    finally:
        v3.load_recipe, v3.compatible_identity, importer._verify_files = old_load, old_compare, old_files


def main():
    parser = v3.argument_parser(worker_mode=True)
    parser.add_argument("--migration-manifest")
    args = parser.parse_args()
    context = (portable_worker(args.migration_manifest, args.config, args.embedding_model_path)
               if args.migration_manifest else nullcontext())
    with context as portable, previous_segment_worker(portable):
        if not args.source_root:
            config = v3.launcher.load_recipe(args.config)
            # Pin the first import's source for subsequent resumes.
            saved = Path(args.output_root or config["output_root"]) / "deepseek_import_manifest.json"
            if saved.exists():
                args.source_root = json.loads(saved.read_text())["source_root"]
            elif (Path(v3.DEFAULT_TARGET) / "deepseek_import_manifest.json").is_file():
                args.source_root = v3.DEFAULT_TARGET
            else:
                args.source_root = portable.source_root if portable else config["source_root"]
        v3.engine.announce("deepseek_child_context", **POLICY, source_root=str(args.source_root))
        v3.run(args, writer_factory=PreviousSegmentWriter)


if __name__ == "__main__":
    main()
