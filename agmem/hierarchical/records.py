"""Immutable public records and deterministic, contiguous interaction blocks."""

from datetime import datetime
import re

from agmem.core import canonical, digest


RECORD_FIELDS = {"record_id", "seq", "kind", "speaker", "call_id", "text", "attachments"}
SESSION_FIELDS = {"schema_version", "bank_id", "parent_id", "source_kind", "order", "time",
                  "participants", "header_records", "records", "blocks", "source_hash"}
KINDS = {"message", "action", "observation", "note", "final", "task_instruction", "initial_observation"}
MODALITY_POLICIES = {"existing_captions", "references_only"}


def public_record(record, modality_policy="existing_captions"):
    """Project source text, never annotation IDs or evaluation metadata.

    Captions are visibly separate auxiliary attachment descriptions, not added
    to the original message text. ``references_only`` retains image references.
    """
    if modality_policy not in MODALITY_POLICIES:
        raise ValueError("Unsupported modality_policy")
    attachments = [{"kind": a["kind"], "ref": a["ref"],
                    "caption": a["caption"] if modality_policy == "existing_captions" else None}
                   for a in record["attachments"]]
    return {"kind": record["kind"], "speaker": record["speaker"],
            "text": record["text"], "attachments": attachments}


def render_records(records, modality_policy="existing_captions"):
    """The versioned body-token rendering: a stable JSON array of public records."""
    return canonical([public_record(record, modality_policy) for record in records])


def time_metadata(value):
    """Keep the original date; only parse unambiguous explicit formats."""
    if value is not None and not isinstance(value, str):
        raise ValueError("Session date must be a string or null")
    result = {"raw": value, "iso": None, "timezone": None}
    if not value:
        return result
    parsed = None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?", value):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            pass
    if parsed is None:
        for pattern in ("%I:%M %p on %d %B, %Y", "%d %B, %Y", "%Y/%m/%d (%a) %H:%M"):
            try:
                parsed = datetime.strptime(value, pattern)
                break
            except ValueError:
                continue
    if parsed is not None:
        date_only = bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{1,2} [A-Za-z]+, \d{4}", value))
        result["iso"] = parsed.date().isoformat() if date_only else parsed.isoformat()
        if parsed.utcoffset() is not None:
            result["timezone"] = "UTC" if parsed.utcoffset().total_seconds() == 0 else parsed.strftime("%z")
    return result


def _token_count(records, count_tokens):
    count = count_tokens(render_records(records))
    if type(count) is not int or count < 0:
        raise ValueError("The tokenizer must return a nonnegative integer token count")
    return count


def build_blocks(records, source_kind, count_tokens):
    """Messages are indivisible; agent calls close only after all returns arrive."""
    if source_kind not in {"dialogue", "agent"}:
        raise ValueError("Unknown source_kind")
    groups, current, pending, used_calls = [], [], set(), set()
    for record in records:
        if source_kind == "dialogue":
            if record["kind"] != "message":
                raise ValueError("Dialogue bodies must contain message records")
            groups.append([record])
            continue
        kind, call_id = record["kind"], record["call_id"]
        current.append(record)
        if kind == "action":
            if not call_id or call_id in used_calls:
                raise ValueError("An action needs a unique call_id")
            pending.add(call_id)
            used_calls.add(call_id)
        elif kind == "observation":
            if not call_id or call_id not in pending:
                raise ValueError("Observation has no matching pending action")
            pending.remove(call_id)
        if not pending:
            groups.append(current)
            current = []
    if pending or current:
        raise ValueError("Unclosed calls in completed session")
    return [{"index": i, "record_ids": [record["record_id"] for record in group],
             "token_count": _token_count(group, count_tokens)} for i, group in enumerate(groups, 1)]


def make_session(*, bank_id, parent_id, order, records, time, participants, count_tokens,
                 header_records=None, source_kind="dialogue"):
    session = {"schema_version": "session.v1", "bank_id": bank_id, "parent_id": parent_id,
               "source_kind": source_kind, "order": order, "time": time,
               "participants": list(dict.fromkeys(participants)), "header_records": header_records or [],
               "records": records, "blocks": build_blocks(records, source_kind, count_tokens)}
    session["source_hash"] = digest(session)
    # Counts were just computed above; avoid tokenizing every source message a
    # second time while adapting the very large M banks.
    validate_session(session)
    return session


def validate_session(session, count_tokens=None):
    if not isinstance(session, dict) or set(session) != SESSION_FIELDS:
        raise ValueError("NormalizedSessionV1 fields differ from the schema")
    if session["schema_version"] != "session.v1" or session["source_kind"] not in {"dialogue", "agent"}:
        raise ValueError("Unsupported normalized session schema/source_kind")
    if type(session["order"]) is not int or session["order"] < 1:
        raise ValueError("Session order must be a positive integer")
    for key in ("bank_id", "parent_id"):
        if not isinstance(session[key], str) or not session[key]:
            raise ValueError(f"Missing {key}")
    time = session["time"]
    if not isinstance(time, dict) or set(time) != {"raw", "iso", "timezone"} or any(
            value is not None and not isinstance(value, str) for value in time.values()):
        raise ValueError("Invalid time metadata")
    if (not isinstance(session["participants"], list)
            or any(not isinstance(x, str) or not x for x in session["participants"])
            or len(set(session["participants"])) != len(session["participants"])):
        raise ValueError("Invalid participants")
    if any(not isinstance(session[k], list) for k in ("header_records", "records", "blocks")):
        raise ValueError("Normalized records and blocks must be arrays")
    records = session["header_records"] + session["records"]
    ids = []
    for seq, record in enumerate(records, 1):
        if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
            raise ValueError("Record fields differ from the schema")
        if type(record["seq"]) is not int or record["seq"] != seq:
            raise ValueError("Records must preserve consecutive original order")
        if not isinstance(record["record_id"], str) or not record["record_id"]:
            raise ValueError("Missing neutral record ID")
        ids.append(record["record_id"])
        if record["kind"] not in KINDS or not isinstance(record["text"], str):
            raise ValueError("Invalid record kind/text")
        if any(record[k] is not None and not isinstance(record[k], str) for k in ("speaker", "call_id")):
            raise ValueError("Invalid speaker/call_id")
        if not isinstance(record["attachments"], list):
            raise ValueError("Attachments must be an array")
        for a in record["attachments"]:
            if (not isinstance(a, dict) or set(a) != {"kind", "ref", "caption"}
                    or a["kind"] not in {"image", "other"} or not isinstance(a["ref"], str) or not a["ref"]
                    or a["caption"] is not None and not isinstance(a["caption"], str)):
                raise ValueError("Invalid attachment")
    if len(set(ids)) != len(ids):
        raise ValueError("Record IDs are not unique within a session")
    flattened, by_id = [], {r["record_id"]: r for r in session["records"]}
    for index, block in enumerate(session["blocks"], 1):
        if (not isinstance(block, dict) or set(block) != {"index", "record_ids", "token_count"}
                or type(block["index"]) is not int or block["index"] != index
                or not isinstance(block["record_ids"], list) or not block["record_ids"]
                or type(block["token_count"]) is not int or block["token_count"] < 0):
            raise ValueError("Invalid Block schema or 1-based index")
        if any(rid not in by_id for rid in block["record_ids"]):
            raise ValueError("Block refers to unknown/header records")
        flattened.extend(block["record_ids"])
        if count_tokens is not None and block["token_count"] != _token_count(
                [by_id[rid] for rid in block["record_ids"]], count_tokens):
            raise ValueError("Block token count differs from the configured tokenizer")
    if flattened != [record["record_id"] for record in session["records"]]:
        raise ValueError("Blocks must cover every body record exactly once in original order")
    if session["source_hash"] != digest({k: v for k, v in session.items() if k != "source_hash"}):
        raise ValueError("Normalized source hash mismatch")
    return session
