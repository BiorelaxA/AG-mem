"""Bind model-selected Block boundaries to immutable, exact source addresses."""

from copy import deepcopy

from ..core import digest
from .records import render_records, validate_session


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Token counts must be nonnegative integers")
    return value


def _unique(values):
    return list(dict.fromkeys(values))


def _span(session, start, end, count_tokens):
    blocks = session["blocks"][start - 1:end]
    ids = [rid for block in blocks for rid in block["record_ids"]]
    by_id = {record["record_id"]: record for record in session["records"]}
    records = [by_id[rid] for rid in ids]
    result = {"start_block": start, "end_block": end,
              "start_record_id": ids[0], "end_record_id": ids[-1],
              "block_count": len(blocks), "record_count": len(ids)}
    if count_tokens is not None:
        result["body_token_count"] = _count(count_tokens(render_records(records)))
    return result


def directory_version(directory):
    return digest({key: value for key, value in directory.items() if key != "directory_version"})


def bind_directory(session, writer_result, writer_identity, tokenizer_id, *, count_tokens,
                   prompt_version="directory-writer.v1"):
    """Create one ParentDirectoryV1; model output never determines persistent IDs."""
    validate_session(session, count_tokens=count_tokens)
    if prompt_version not in {"directory-writer.v1", "directory-writer.adjacent.v1"}:
        raise ValueError("Unsupported directory Writer prompt version")
    _text(tokenizer_id, "tokenizer_id")
    model = _text(writer_identity.get("model"), "writer model")
    revision = _text(writer_identity.get("model_revision", writer_identity.get("revision")),
                     "writer revision")
    parent = writer_result.get("parent", {})
    if set(parent) != {"title", "content"}:
        raise ValueError("Writer parent must contain exactly title/content")
    title, content = (_text(parent[key], f"parent.{key}") for key in ("title", "content"))
    status, mode = writer_result.get("status"), writer_result.get("mode")
    if status not in {"complete", "fallback", "empty"} or mode not in {"single_session", "windowed"}:
        raise ValueError("Invalid Writer build status/mode")
    supplied = writer_result.get("children")
    if not isinstance(supplied, list):
        raise ValueError("Writer children must be an array")
    block_count = len(session["blocks"])
    if not block_count:
        if status != "empty" or supplied:
            raise ValueError("Empty sessions require status=empty and children=[]")
    elif status == "empty" or not supplied:
        raise ValueError("Nonempty sessions require complete child coverage")
    children, start = [], 1
    for order, child in enumerate(supplied, 1):
        required = {"end_block", "title", "content", "entities"}
        if not required.issubset(child) or set(child) - required - {"start_reason"}:
            raise ValueError("Unexpected Writer child fields")
        end = child["end_block"]
        if type(end) is not int or not start <= end <= block_count:
            raise ValueError("Invalid or non-increasing child end_block")
        entities = child["entities"]
        if (not isinstance(entities, list) or any(not isinstance(e, str) or not e.strip() for e in entities)
                or len(set(entities)) != len(entities)):
            raise ValueError("Child entities must be unique, nonempty strings")
        reason = child.get("start_reason", "session_start" if order == 1 else "semantic")
        if ((order == 1 and reason != "session_start")
                or (order > 1 and reason not in {"semantic", "window_limit"})
                or (reason == "window_limit" and mode != "windowed")):
            raise ValueError("Child start_reason is incompatible with its position/build mode")
        children.append({"id": f"{session['parent_id']}:c{order:04d}",
                         "parent_id": session["parent_id"], "order": order,
                         "title": _text(child["title"], "child.title"),
                         "content": _text(child["content"], "child.content"),
                         "entities": list(entities), "start_reason": reason,
                         "source": _span(session, start, end, count_tokens)})
        start = end + 1
    if start != block_count + 1:
        raise ValueError("Children omit the end of the session")
    body = session["records"]
    directory = {
        "schema_version": "directory.v1", "bank_id": session["bank_id"],
        "id": session["parent_id"], "source_kind": session["source_kind"], "order": session["order"],
        "time": deepcopy(session["time"]), "participants": list(session["participants"]),
        "title": title, "content": content,
        "entities": _unique(entity for child in children for entity in child["entities"]),
        "source": {"snapshot_id": "snapshot_" + session["source_hash"],
                   "header_record_ids": [r["record_id"] for r in session["header_records"]],
                   "start_record_id": body[0]["record_id"] if body else None,
                   "end_record_id": body[-1]["record_id"] if body else None,
                   "block_count": block_count, "record_count": len(body),
                   "body_token_count": _count(count_tokens(render_records(body))) if body else 0,
                   "tokenizer_id": tokenizer_id},
        "children": children,
        "build": {"status": status, "mode": mode, "writer_model": model, "writer_revision": revision,
                  "prompt_version": prompt_version, "block_policy": "block.v1",
                  "source_hash": session["source_hash"]},
    }
    directory["directory_version"] = directory_version(directory)
    validate_directory(directory, session, count_tokens=count_tokens)
    return directory


def validate_directory(directory, session, *, count_tokens=None):
    """Reject stale source snapshots, altered versions, and approximate source spans."""
    validate_session(session, count_tokens=count_tokens)
    fields = {"schema_version", "bank_id", "directory_version", "id", "source_kind", "order",
              "time", "participants", "title", "content", "entities", "source", "children", "build"}
    if not isinstance(directory, dict) or set(directory) != fields:
        raise ValueError("Invalid ParentDirectoryV1 fields")
    if directory["schema_version"] != "directory.v1" or directory_version(directory) != directory["directory_version"]:
        raise ValueError("Directory version does not match its immutable content")
    for key, expected in (("bank_id", session["bank_id"]), ("id", session["parent_id"]),
                          ("source_kind", session["source_kind"]), ("order", session["order"]),
                          ("time", session["time"]), ("participants", session["participants"])):
        if directory[key] != expected:
            raise ValueError(f"Directory {key} differs from source snapshot")
    for key in ("title", "content"):
        _text(directory[key], key)
    build = directory["build"]
    if (not isinstance(build, dict) or set(build) != {"status", "mode", "writer_model", "writer_revision",
            "prompt_version", "block_policy", "source_hash"} or build["source_hash"] != session["source_hash"]
            or build["status"] not in {"complete", "fallback", "empty"}
            or build["mode"] not in {"single_session", "windowed"}
            or build["prompt_version"] not in {"directory-writer.v1", "directory-writer.adjacent.v1"}
            or build["block_policy"] != "block.v1"):
        raise ValueError("Invalid directory build provenance")
    _text(build["writer_model"], "writer_model")
    _text(build["writer_revision"], "writer_revision")
    body, blocks = session["records"], session["blocks"]
    source = directory["source"]
    expected = {"snapshot_id": "snapshot_" + session["source_hash"],
                "header_record_ids": [r["record_id"] for r in session["header_records"]],
                "start_record_id": body[0]["record_id"] if body else None,
                "end_record_id": body[-1]["record_id"] if body else None,
                "block_count": len(blocks), "record_count": len(body)}
    if (not isinstance(source, dict) or set(source) != set(expected) | {"body_token_count", "tokenizer_id"}
            or any(source[k] != value for k, value in expected.items())):
        raise ValueError("Parent source does not cover the exact complete session")
    _text(source["tokenizer_id"], "tokenizer_id")
    _count(source["body_token_count"])
    if count_tokens is not None:
        total = count_tokens(render_records(body)) if body else 0
        if total != source["body_token_count"]:
            raise ValueError("Parent token count differs from source rendering")
    children = directory["children"]
    if not isinstance(children, list) or bool(children) != bool(blocks):
        raise ValueError("Children must cover every nonempty session")
    if (not blocks) != (build["status"] == "empty"):
        raise ValueError("Empty build status differs from actual session")
    start = 1
    for index, child in enumerate(children, 1):
        if not isinstance(child, dict) or set(child) != {"id", "parent_id", "order", "title", "content",
                                                        "entities", "start_reason", "source"}:
            raise ValueError("Invalid child fields")
        if (child["id"] != f"{session['parent_id']}:c{index:04d}" or child["parent_id"] != directory["id"]
                or child["order"] != index):
            raise ValueError("Child IDs/order do not follow the committed mapping")
        for key in ("title", "content"):
            _text(child[key], key)
        entities = child["entities"]
        if (not isinstance(entities, list) or any(not isinstance(e, str) or not e.strip() for e in entities)
                or len(entities) != len(set(entities))):
            raise ValueError("Invalid child entities")
        span = child["source"]
        end = span.get("end_block") if isinstance(span, dict) else None
        if type(end) is not int or not start <= end <= len(blocks):
            raise ValueError("Child end_block is out of order/range")
        expected_span = _span(session, start, end, count_tokens)
        if (set(span) != {"start_block", "end_block", "start_record_id", "end_record_id", "block_count",
                          "record_count", "body_token_count"}
                or any(span[k] != value for k, value in expected_span.items())):
            raise ValueError("Child source range differs from the exact Block/Record mapping")
        _count(span["body_token_count"])
        reason = child["start_reason"]
        if ((index == 1 and reason != "session_start")
                or (index > 1 and reason not in {"semantic", "window_limit"})
                or (reason == "window_limit" and build["mode"] != "windowed")):
            raise ValueError("Invalid child start_reason")
        start = end + 1
    if start != len(blocks) + 1:
        raise ValueError("Children omit trailing Blocks")
    if directory["entities"] != _unique(e for child in children for e in child["entities"]):
        raise ValueError("Parent entities are not the stable child union")

