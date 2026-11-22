"""Opt-in child summaries with three complete preceding source segments.

The legacy writer and its resources remain immutable. Fixed child boundaries and
the public directory schema are unchanged; only the new workflow uses this class.
"""

from copy import deepcopy
from pathlib import Path

from ..core import CapacityError
from .adjacent_writer import AdjacentDirectoryWriter, _BudgetExceeded, validate_spans


RESOURCE_ROOT = Path(__file__).with_name("resources") / "adjacent"
CHILD_PROMPT = RESOURCE_ROOT / "segment_context_v1" / "child.txt"
BOUNDARY_PROMPT = RESOURCE_ROOT / "topic_event_v2" / "boundary.txt"
CHILD_PROMPT_VERSION = "segment_context.v1"
PREVIOUS_SEGMENTS = 3


def context_policy():
    """Construction identity for this input contract, independent of schemas."""
    return {"version": CHILD_PROMPT_VERSION,
            "unit": "complete_preceding_segments", "previous_segments": PREVIOUS_SEGMENTS,
            "input": "original_source", "scope": "same_session_only",
            "order": "oldest_to_newest", "following_segments": 0,
            "core_span": "complete_current_segment",
            "overflow": "fail_without_truncating_or_dropping_context"}


class SegmentContextDirectoryWriter(AdjacentDirectoryWriter):
    """Write complete child spans with their last three complete raw neighbors.

    A child request must fit with its entire current segment and required earlier
    segments. It is never silently shortened, windowed, or converted to summaries.
    Parent generation retains the legacy complete-source windowing behavior.
    """

    def __init__(self, tokenizer, settings, complete, *, previous_segments=PREVIOUS_SEGMENTS):
        if type(previous_segments) is not int or previous_segments != PREVIOUS_SEGMENTS:
            raise ValueError("segment_context.v1 requires previous_segments=3")
        super().__init__(tokenizer, settings, complete)
        if self.settings["context_blocks"] != 0:
            raise ValueError("Segment context requires writer.context_blocks=0; context is measured in segments")
        self.previous_segments = previous_segments
        self.adjacent_prompts["child"] = CHILD_PROMPT.read_text()
        self.adjacent_prompts["boundary"] = BOUNDARY_PROMPT.read_text()
        self._fixed_spans = None

    def _child_index(self, start, end):
        if self._fixed_spans is None:
            raise ValueError("Fixed segment boundaries must be bound before child summarization")
        for index, span in enumerate(self._fixed_spans):
            if (span["start_block"], span["end_block"]) == (start, end):
                return index
        raise ValueError("Child summarization requires one complete fixed segment")

    def _source_payload(self, start, end, kind, children):
        if kind != "child":
            return super()._source_payload(start, end, kind, children)
        index = self._child_index(start, end)
        payload = super()._raw_payload(start, end, kind, children, context_count=0)
        payload.pop("context_blocks")
        payload["context_segments"] = [
            {**span, "blocks": deepcopy(self.public["blocks"][
                span["start_block"] - 1:span["end_block"]])}
            for span in self._fixed_spans[max(0, index - self.previous_segments):index]
        ]
        payload["context_policy"] = context_policy()
        if not self._fits(self._entry_messages(payload, kind)):
            raise CapacityError(
                f"child_segment_context_exceeds_capacity: complete segment {start}-{end} plus "
                f"{len(payload['context_segments'])} preceding complete segments and output reservation "
                f"do not fit writer.context_tokens={self.settings['context_tokens']}; "
                "increase the context budget. Original source and required context were not truncated.")
        return payload

    def _summarize_range(self, start, end, kind, children):
        if kind != "child":
            return super()._summarize_range(start, end, kind, children)
        payload = self._source_payload(start, end, kind, children)
        self.stats["windows"] += 1
        self.traces.append({"stage": "child_source_window", "write_range": {"start": start, "end": end},
                            "source_hash": self._session_hash,
                            "block_indices": [block["index"] for block in payload["blocks"]],
                            "context_policy": context_policy(),
                            "context_segment_ranges": [
                                {key: segment[key] for key in ("start_block", "end_block")}
                                for segment in payload["context_segments"]]})
        try:
            result = self._request_entry(payload, kind, "adjacent_child")
        except _BudgetExceeded as exc:
            # Output expansion/repair exhaustion does not authorize shrinking
            # source input or changing a selected child boundary.
            self.traces.append({"stage": "child_generation_budget_exceeded",
                                "write_range": {"start": start, "end": end},
                                "reason": str(exc), "source_input_preserved": True})
            result = None
        if result is None:
            return self._placeholder(kind, start, end), True
        return result, False

    def build(self, session, spans):
        self.prepare(session)
        fixed = validate_spans(spans, len(session["blocks"]))
        if self._fixed_spans is not None and fixed != self._fixed_spans:
            raise ValueError("A SegmentContextDirectoryWriter cannot change its fixed segment boundaries")
        self._fixed_spans = deepcopy(fixed)
        # Check all complete child inputs before sending any child requests.
        # A later long context cannot leave a deceptively partial child build.
        for span in fixed:
            self._source_payload(span["start_block"], span["end_block"], "child", [])
        return super().build(session, fixed)
