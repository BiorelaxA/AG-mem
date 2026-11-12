"""Source-grounded child-first writing over program-fixed adjacent segments.

The public directory shape stays compatible with DirectoryReader. This module never
changes a selected child boundary. Large spans may be *read* in complete Block
windows, but those windows are not exposed as new children. Every parent is built
after its children and receives the original source, directly or through explicitly
traced complete-source windows when the session exceeds the model context.
"""

from copy import deepcopy
from pathlib import Path

from ..core import canonical, file_digest
from .records import validate_session
from .writer import (DirectoryWriter, _strict_json, _text_budget, chat_token_count,
                     public_session, validate_build_result, validate_schema)


RESOURCE_ROOT = Path(__file__).with_name("resources") / "adjacent"
PROMPT_VERSION = "directory-writer.adjacent.v1"


def adjacent_prompt_hashes():
    """Do not add these resources to the legacy Writer's immutable identity."""
    return {p.name: file_digest(p) for p in sorted(RESOURCE_ROOT.iterdir()) if p.is_file()}


def validate_spans(spans, block_count):
    """Accept segmenter metadata while verifying exact consecutive coverage."""
    if not isinstance(spans, (list, tuple)):
        raise ValueError("Fixed spans must be a sequence")
    expected, result = 1, []
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("A fixed span must be an object")
        start, end = span.get("start_block"), span.get("end_block")
        if (type(start) is not int or type(end) is not int or start != expected
                or end < start or end > block_count):
            raise ValueError("Fixed spans must cover original Blocks consecutively exactly once")
        result.append({"start_block": start, "end_block": end})
        expected = end + 1
    if expected != block_count + 1:
        raise ValueError("Fixed spans omit source Blocks")
    return result


class _BudgetExceeded(Exception):
    pass


class AdjacentDirectoryWriter(DirectoryWriter):
    """One instance per session; boundary review can precede ``build``.

    Requests within a session are sequential. The shared transport pool controls
    total concurrent requests across all session workers.
    """

    def __init__(self, tokenizer, settings, complete):
        super().__init__(tokenizer, settings, complete)
        self.adjacent_prompts = {kind: (RESOURCE_ROOT / f"{kind}.txt").read_text()
                                 for kind in ("child", "parent", "boundary")}
        self.stats = {name: 0 for name in (
            "calls", "semantic_repairs", "output_expansions", "windows", "rollup_calls",
            "rollup_levels", "fallback_windows", "child_entries", "parent_source_windows",
            "boundary_reviews", "boundary_review_failures")}
        self.traces = []
        self._session_hash = None
        self._windowed = False

    def prepare(self, session):
        validate_session(session)
        validate_schema(session, "normalized_session")
        if self._session_hash is not None and self._session_hash != session["source_hash"]:
            raise ValueError("An AdjacentDirectoryWriter instance belongs to one session")
        self._session_hash = session["source_hash"]
        self.public = public_session(session, self.settings["modality_policy"])

    def _schema(self, kind):
        if kind == "boundary":
            fields = {"decision": {"type": "string", "enum": ["merge", "keep_boundary"]},
                      "reason": {"type": "string", "minLength": 1},
                      "evidence_blocks": {"type": "array", "uniqueItems": True,
                                          "items": {"type": "integer", "minimum": 1}}}
        else:
            fields = {name: {"type": "string", "minLength": 1} for name in ("title", "content")}
            if kind == "child":
                fields["entities"] = {"type": "array", "uniqueItems": True,
                                      "maxItems": self.settings["max_entities"],
                                      "items": {"type": "string", "minLength": 1}}
        return {"type": "object", "additionalProperties": False,
                "properties": fields, "required": list(fields)}

    def _entry_messages(self, payload, kind, previous=None, errors=None):
        data = deepcopy(payload)
        if previous is not None:
            # Failed generated text may be shortened; original input never is.
            while True:
                data["repair"] = {"instruction": "Return a corrected JSON entry for the unchanged source.",
                                  "errors": errors, "previous_response": previous}
                messages = [{"role": "system", "content": self.adjacent_prompts[kind]},
                            {"role": "user", "content": canonical(data)}]
                if self._fits(messages):
                    return messages
                if not previous:
                    raise _BudgetExceeded("Repair wrapper cannot fit without removing original source")
                previous = previous[:len(previous) // 2]
        return [{"role": "system", "content": self.adjacent_prompts[kind]},
                {"role": "user", "content": canonical(data)}]

    def _validate_entry(self, output, kind, payload):
        from jsonschema import Draft202012Validator
        Draft202012Validator(self._schema(kind)).validate(output)
        if kind == "boundary":
            allowed = {b["index"] for side in ("left", "right") for b in payload[side]["blocks"]}
            if any(type(i) is not int or i not in allowed for i in output["evidence_blocks"]):
                raise ValueError("Boundary evidence must reference supplied source Blocks")
            _text_budget(output["reason"], "boundary.reason", 200, self.tokenizer)
            return
        _text_budget(output["title"], f"{kind}.title", self.settings["title_tokens"], self.tokenizer)
        _text_budget(output["content"], f"{kind}.content",
                     self.settings[f"{kind}_content_tokens"], self.tokenizer)
        if kind == "child" and any(not entity.strip() for entity in output["entities"]):
            raise ValueError("entities must not contain whitespace-only strings")

    def _request_entry(self, payload, kind, stage):
        from jsonschema import ValidationError
        previous, errors = None, None
        schema = self._schema(kind)
        for repair in range(self.settings["semantic_retries"] + 1):
            messages = self._entry_messages(payload, kind, previous, errors)
            available = min(self.settings["max_output_tokens"], self._available(messages))
            if available < self.settings["output_tokens"]:
                raise _BudgetExceeded("Original source plus output reservation exceeds context")
            budget = self.settings["output_tokens"]
            current_stage = stage + ("_repair" if repair else "")
            if repair:
                self.stats["semantic_repairs"] += 1
            while True:
                # Transport errors escape; they never become successful summaries.
                response = self.complete(messages, budget, schema, current_stage)
                self.stats["calls"] += 1
                if "rollup" in stage:
                    self.stats["rollup_calls"] += 1
                record = {"stage": current_stage, "max_tokens": budget, "repair_index": repair,
                          "input_tokens": chat_token_count(self.tokenizer, messages),
                          "finish_reason": response.get("finish_reason"),
                          "write_range": deepcopy(payload.get("write_range"))}
                if response.get("finish_reason") == "length":
                    record.update(valid=False, reason="truncated_output")
                    self.traces.append(record)
                    self._hook("reject", messages, budget, schema, current_stage, response,
                               reason="truncated_output")
                    expanded = min(available, budget * 2)
                    if expanded <= budget:
                        raise _BudgetExceeded("Truncated entry at the largest completion budget")
                    self.stats["output_expansions"] += 1
                    budget = expanded
                    continue
                try:
                    if response.get("finish_reason") != "stop":
                        raise ValueError("Entry generation did not finish normally")
                    if not isinstance(response.get("text"), str) or not response["text"].strip():
                        raise ValueError("Entry generation returned no text")
                    result = _strict_json(response["text"])
                    self._validate_entry(result, kind, payload)
                except (ValueError, TypeError, KeyError, ValidationError) as exc:
                    errors = [str(getattr(exc, "message", exc))[:1600]]
                else:
                    record.update(valid=True, status="complete")
                    self.traces.append(record)
                    self._hook("accept", messages, budget, schema, current_stage, response, status="complete")
                    return result
                record.update(valid=False, reason="entry_validation", errors=errors)
                self.traces.append(record)
                self._hook("reject", messages, budget, schema, current_stage, response, reason=errors[0])
                previous = response.get("text") if isinstance(response.get("text"), str) else ""
                break
        return None

    def _placeholder(self, kind, start, end):
        self.stats["fallback_windows"] += 1
        self.traces.append({"stage": f"{kind}_placeholder", "status": "fallback",
                            "write_range": {"start": start, "end": end},
                            "reason": "entry_generation_failed"})
        value = {"title": self._short_placeholder(f"Original records {start}-{end}", self.settings["title_tokens"]),
                 "content": self._short_placeholder(
                     "Automatic summary unavailable; read the original source span.",
                     self.settings[f"{kind}_content_tokens"])}
        if kind == "child":
            value["entities"] = []
        return value

    def _raw_payload(self, start, end, kind, children, context_count=0):
        result = {"input_kind": "original_source", "scope": kind,
                  "source_kind": self.public["source_kind"],
                  "session_metadata": deepcopy(self.public["session_metadata"]),
                  "headers": deepcopy(self.public["headers"]),
                  "output_language": self.settings["output_language"], "limits": self._limits(),
                  "write_range": {"start": start, "end": end},
                  "blocks": deepcopy(self.public["blocks"][start - 1:end]),
                  "context_blocks": deepcopy(self.public["blocks"][max(0, start - 1 - context_count):start - 1])}
        if kind == "parent":
            result["children"] = [deepcopy(c) for c in children
                                  if c["start_block"] <= end and c["end_block"] >= start]
        return result

    def _source_payload(self, start, end, kind, children):
        # Optional earlier context may be omitted, original core source may not.
        maximum = min(self.settings["context_blocks"], start - 1) if kind == "child" else 0
        for count in range(maximum, -1, -1):
            payload = self._raw_payload(start, end, kind, children, count)
            if self._fits(self._entry_messages(payload, kind)):
                return payload
        return None

    def _reduction_payload(self, entries, kind):
        return {"input_kind": "ordered_source_overviews", "scope": kind,
                "source_kind": self.public["source_kind"],
                "session_metadata": deepcopy(self.public["session_metadata"]),
                "output_language": self.settings["output_language"], "limits": self._limits(),
                "write_range": {"start": entries[0]["start_block"], "end": entries[-1]["end_block"]},
                "ordered_inputs": deepcopy(entries),
                "source_coverage": "All original Blocks were read in consecutive complete windows; no source was truncated."}

    def _reduce(self, entries, kind):
        """Bounded hierarchical reduction; all source windows retain provenance."""
        if len(entries) == 1:
            return {key: deepcopy(entries[0][key]) for key in ("title", "content", "entities")
                    if key in entries[0]}, False
        initial_start, initial_end = entries[0]["start_block"], entries[-1]["end_block"]
        failed = False
        while True:
            payload = self._reduction_payload(entries, kind)
            if self._fits(self._entry_messages(payload, kind)):
                try:
                    result = self._request_entry(payload, kind, f"adjacent_{kind}_rollup")
                except _BudgetExceeded:
                    result = None
                if result is not None:
                    return result, failed
                return self._placeholder(kind, initial_start, initial_end), True
            groups, group = [], []
            for entry in entries:
                proposed = group + [entry]
                if group and not self._fits(self._entry_messages(self._reduction_payload(proposed, kind), kind)):
                    groups.append(group)
                    group = []
                group.append(entry)
            if group:
                groups.append(group)
            if len(groups) >= len(entries):
                return self._placeholder(kind, initial_start, initial_end), True
            reduced = []
            for group in groups:
                if len(group) == 1:
                    reduced.append(group[0])
                    continue
                payload = self._reduction_payload(group, kind)
                try:
                    result = self._request_entry(payload, kind, f"adjacent_{kind}_rollup")
                except _BudgetExceeded:
                    result = None
                if result is None:
                    result = self._placeholder(kind, group[0]["start_block"], group[-1]["end_block"])
                    failed = True
                reduced.append({"start_block": group[0]["start_block"],
                                "end_block": group[-1]["end_block"], **result})
            entries = reduced
            self.stats["rollup_levels"] += 1

    def _summarize_range(self, start, end, kind, children):
        payload = self._source_payload(start, end, kind, children)
        if payload is not None:
            self.stats["windows"] += 1
            if kind == "parent":
                self.stats["parent_source_windows"] += 1
            self.traces.append({"stage": f"{kind}_source_window", "write_range": {"start": start, "end": end},
                                "source_hash": self._session_hash,
                                "block_indices": [b["index"] for b in payload["blocks"]]})
            try:
                result = self._request_entry(payload, kind, f"adjacent_{kind}")
            except _BudgetExceeded:
                result = None
                if start < end:
                    payload = None  # Retry with complete, smaller source windows.
            if payload is not None:
                if result is None:
                    return self._placeholder(kind, start, end), True
                return result, False
        if start == end:
            raise ValueError(f"oversized_block: Block {start} cannot fit the {kind} source request without truncation")
        self._windowed = True
        midpoint = (start + end) // 2
        left, left_failed = self._summarize_range(start, midpoint, kind, children)
        right, right_failed = self._summarize_range(midpoint + 1, end, kind, children)
        entries = [{"start_block": start, "end_block": midpoint, **left},
                   {"start_block": midpoint + 1, "end_block": end, **right}]
        merged, failed = self._reduce(entries, kind)
        return merged, left_failed or right_failed or failed

    def review_boundary(self, payload):
        """Review an uncertain merge without needing build() to run first."""
        session = payload["session"]
        self.prepare(session)
        left, right = payload["left"], payload["right"]
        start, middle, next_start, end = (left.get("start_block"), left.get("end_block"),
                                         right.get("start_block"), right.get("end_block"))
        if (any(type(i) is not int for i in (start, middle, next_start, end))
                or not 1 <= start <= middle < next_start <= end <= len(self.public["blocks"])
                or next_start != middle + 1):
            raise ValueError("Boundary review requires two adjacent original source spans")
        data = {"session_metadata": deepcopy(self.public["session_metadata"]),
                "source_kind": self.public["source_kind"],
                "left": {"start_block": start, "end_block": middle,
                         "blocks": deepcopy(self.public["blocks"][start - 1:middle])},
                "right": {"start_block": next_start, "end_block": end,
                          "blocks": deepcopy(self.public["blocks"][next_start - 1:end])}}
        self.stats["boundary_reviews"] += 1
        try:
            result = self._request_entry(data, "boundary", "adjacent_boundary")
        except _BudgetExceeded:
            result = None
        if result is None:
            self.stats["boundary_review_failures"] += 1
            self.traces.append({"stage": "boundary_review_unavailable", "left": [start, middle],
                                "right": [next_start, end], "decision": "keep_boundary"})
            return {"decision": "keep_boundary", "reason": "Boundary review unavailable; original boundary retained.",
                    "evidence_blocks": [], "available": False, "review_unavailable": True}
        return {**result, "available": True, "review_unavailable": False}

    def build(self, session, spans):
        self.prepare(session)
        spans = validate_spans(spans, len(session["blocks"]))
        if not spans:
            result = {"parent": {"title": self._short_placeholder("Empty session", self.settings["title_tokens"]),
                                 "content": self._short_placeholder("No body records.", self.settings["parent_content_tokens"])},
                      "children": [], "status": "empty", "mode": "single_session",
                      "stats": deepcopy(self.stats), "traces": deepcopy(self.traces)}
            return validate_build_result(result, session, self.tokenizer, self.settings)
        children, parent_inputs, fallback = [], [], False
        for i, span in enumerate(spans):
            start, end = span["start_block"], span["end_block"]
            entry, failed = self._summarize_range(start, end, "child", [])
            fallback |= failed
            children.append({"end_block": end, **entry,
                             "start_reason": "session_start" if i == 0 else "semantic"})
            parent_inputs.append({**span, **entry})
            self.stats["child_entries"] += 1
        parent, failed = self._summarize_range(1, len(session["blocks"]), "parent", parent_inputs)
        fallback |= failed
        result = {"parent": parent, "children": children,
                  "status": "fallback" if fallback else "complete",
                  "mode": "windowed" if self._windowed else "single_session",
                  "stats": deepcopy(self.stats), "traces": deepcopy(self.traces)}
        return validate_build_result(result, session, self.tokenizer, self.settings)
