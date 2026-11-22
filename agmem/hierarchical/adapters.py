"""Official dialogue inputs and an explicit compatibility path for saved parquet."""

import json
from pathlib import Path
import re

from agmem.core import digest, file_digest
from .records import make_session, time_metadata, public_record, MODALITY_POLICIES


SPLITS = {"train", "validation", "test"}


def _array(value, field):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise ValueError(f"Invalid JSON array: {field}") from exc
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _attachments(message):
    refs = message.get("img_url")
    caption = message.get("blip_caption")
    if refs is None:
        return []
    refs = [refs] if isinstance(refs, str) else refs
    if not isinstance(refs, list) or any(not isinstance(x, str) or not x for x in refs):
        raise ValueError("Image references must be nonempty strings")
    captions = caption if isinstance(caption, list) else [caption] * len(refs)
    if len(captions) != len(refs) or any(c is not None and not isinstance(c, str) for c in captions):
        raise ValueError("Image captions must align with image references")
    return [{"kind": "image", "ref": ref, "caption": cap} for ref, cap in zip(refs, captions)]


def _record(message, parent_id, index, *, official_locomo=False):
    if not isinstance(message, dict):
        raise ValueError("Every message must be an object")
    speaker = message.get("speaker") if official_locomo else message.get("role")
    text = message.get("text") if official_locomo else message.get("content")
    if speaker is not None and not isinstance(speaker, str):
        raise ValueError("Message speaker must be text or null")
    if not isinstance(text, str):
        raise ValueError("Message text must be a string; structured content needs an explicit adapter")
    return {"record_id": f"{parent_id}:r{index:04d}", "seq": index, "kind": "message",
            "speaker": speaker, "call_id": None, "text": text, "attachments": _attachments(message)}


def input_files(spec, selected_splits=None):
    """Resolve sources without loading records; suitable for a dry-run plan."""
    selected = list(selected_splits or spec.get("splits", ["train", "validation", "test"]))
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(SPLITS):
        raise ValueError("Select unique train/validation/test splits")
    if "input_files" in spec:
        files = [(split, Path(spec["input_files"][split]).resolve()) for split in selected if split in spec["input_files"]]
        if len(files) != len(selected):
            raise ValueError("Missing configured input file for a selected split")
    elif "path" in spec:
        split = spec.get("split", "test")
        files = [(split, Path(spec["path"]).resolve())] if split in selected else []
    else:
        root = Path(spec["input_root"])
        candidate = root / spec["name"]
        if candidate.is_dir():
            root = candidate
        files = [(split, (root / f"{split}.parquet").resolve()) for split in selected]
    for _, path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing dataset input: {path}")
    return files


def _rows(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            yield from batch.to_pylist()
    elif path.suffix == ".jsonl":
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError("Official dataset JSON must contain an array of items")
        yield from data


def _source(path, sha, split, index, row, source_format):
    return {"file": str(path), "file_sha256": sha, "split": split, "row_index": index,
            "original_question_id": row.get("question_id"), "source_format": source_format}


def _membership(name, source):
    return {"sample_id": f"{name}/{source['split']}/{source['row_index']:06d}",
            "split": source["split"], "row_index": source["row_index"],
            "original_question_id": source.get("original_question_id")}


def _error(bank_id, source, exc, parent_id=None):
    return {"status": "failed", "bank_id": bank_id, "parent_id": parent_id,
            "error_type": type(exc).__name__, "reason": str(exc),
            "source_file": source["file"], "row_index": source["row_index"]}


def _history_columns(row):
    if not isinstance(row, dict):
        raise ValueError("Dataset item must be an object")
    sessions = _array(row.get("haystack_sessions"), "haystack_sessions")
    ids = _array(row.get("haystack_session_ids"), "haystack_session_ids")
    dates = _array(row.get("haystack_dates"), "haystack_dates")
    if not (len(sessions) == len(ids) == len(dates)):
        raise ValueError("Session/ID/date array lengths disagree")
    return sessions, ids, dates


def _normalize_history(name, row, source, bank_id, count_tokens, *, locomo=False):
    sessions, source_ids, dates = _history_columns(row)
    bank = {"bank_id": bank_id, "dataset": name, "sessions": [], "source_map": [],
            "query_memberships": [_membership(name, source)], "issues": []}
    if locomo:
        bank["issues"].append({"status": "warning", "reason": "processed_locomo_compat",
            "details": "Saved role/content are retained verbatim. Official speaker_a/speaker_b metadata and image fields absent from this export cannot be reconstructed."})
    for pos, messages in enumerate(sessions):
        parent_id = f"p{pos+1:04d}"
        try:
            if not isinstance(messages, list):
                raise ValueError("Session must be an array of messages")
            records = [_record(message, parent_id, i) for i, message in enumerate(messages, 1)]
            participants = list(dict.fromkeys(r["speaker"] for r in records if r["speaker"]))
            normalized = make_session(bank_id=bank_id, parent_id=parent_id, order=pos+1,
                records=records, time=time_metadata(dates[pos]), participants=participants,
                count_tokens=count_tokens)
            bank["sessions"].append(normalized)
            if dates[pos] is None or dates[pos] == "":
                bank["issues"].append({"status": "warning", "parent_id": parent_id, "reason": "missing_session_date"})
            session_map = {**source, "bank_id": bank_id, "parent_id": parent_id,
                           "session_index": pos, "original_session_id": source_ids[pos],
                           "source_hash": normalized["source_hash"]}
            bank["source_map"].append(session_map)
            for message_index, (record, message) in enumerate(zip(records, messages)):
                bank["source_map"].append({**session_map, "record_id": record["record_id"],
                    "message_index": message_index, "original_message_id": message.get("dia_id", message.get("id"))})
        except (ValueError, TypeError, KeyError) as exc:
            bank["issues"].append(_error(bank_id, source, exc, parent_id))
    return bank


def _official_locomo(name, row, source, count_tokens):
    if not isinstance(row, dict) or not isinstance(row.get("conversation"), dict):
        raise ValueError("LoCoMo item needs a conversation object")
    conversation = row["conversation"]
    # The ID is neutral; the original sample ID lives only in the private map.
    original_sample = row.get("sample_id", source["row_index"])
    bank_id = "bank_" + digest([name, "locomo_official", source["file_sha256"], original_sample])[:24]
    bank = {"bank_id": bank_id, "dataset": name, "sessions": [], "source_map": [],
            "query_memberships": [], "issues": []}
    participants = []
    for key in ("speaker_a", "speaker_b"):
        speaker = conversation.get(key, row.get(key))
        if isinstance(speaker, str) and speaker:
            if speaker not in participants:
                participants.append(speaker)
        elif speaker is None or speaker == "":
            bank["issues"].append({"status": "warning", "reason": f"missing_{key}"})
        else:
            raise ValueError(f"Invalid {key}")
    session_keys = sorted((int(m.group(1)), k) for k in conversation if (m := re.fullmatch(r"session_([0-9]+)", k)))
    if not session_keys:
        raise ValueError("LoCoMo conversation contains no session arrays")
    if len({n for n, _ in session_keys}) != len(session_keys):
        raise ValueError("LoCoMo session numeric indices are not unique")
    for order, (source_number, key) in enumerate(session_keys, 1):
        parent_id = f"p{order:04d}"
        try:
            messages = conversation[key]
            if not isinstance(messages, list):
                raise ValueError("LoCoMo session must be a message array")
            records = [_record(message, parent_id, i, official_locomo=True) for i, message in enumerate(messages, 1)]
            date = conversation.get(f"{key}_date_time")
            if date is None or date == "":
                bank["issues"].append({"status": "warning", "parent_id": parent_id, "reason": "missing_session_date"})
            normalized = make_session(bank_id=bank_id, parent_id=parent_id, order=order,
                records=records, time=time_metadata(date), participants=participants, count_tokens=count_tokens)
            bank["sessions"].append(normalized)
            session_map = {**source, "bank_id": bank_id, "parent_id": parent_id,
                           "original_sample_id": original_sample, "original_session_id": key,
                           "session_index": source_number, "json_pointer": f"/{source['row_index']}/conversation/{key}",
                           "source_hash": normalized["source_hash"]}
            bank["source_map"].append(session_map)
            for i, message in enumerate(messages):
                bank["source_map"].append({**session_map, "record_id": records[i]["record_id"],
                    "message_index": i, "original_message_id": message.get("dia_id"),
                    "json_pointer": session_map["json_pointer"] + f"/{i}"})
        except (ValueError, TypeError, KeyError) as exc:
            bank["issues"].append(_error(bank_id, source, exc, parent_id))
    # QA is never copied into sessions/Writer input; membership is a private address only.
    qa = row.get("qa", [])
    if not isinstance(qa, list):
        bank["issues"].append({"status": "warning", "reason": "invalid_qa_membership_array"})
        qa = []
    bank["query_memberships"] = [{"sample_id": f"{name}/{source['split']}/{source['row_index']:06d}/qa/{i:06d}",
                                  "split": source["split"], "row_index": source["row_index"], "qa_index": i}
                                 for i in range(len(qa))]
    return bank


def _locomo_history_key(row):
    sessions, ids, dates = _history_columns(row)
    cleaned = []
    for pos, messages in enumerate(sessions):
        if not isinstance(messages, list):
            # Malformed source remains part of the private bank identity. Its
            # own failed item must not prevent adapting other valid sessions.
            cleaned.append({"invalid_session": messages})
            continue
        projected = []
        for i, message in enumerate(messages, 1):
            try:
                projected.append(public_record(_record(message, f"p{pos+1:04d}", i)))
            except (ValueError, TypeError, KeyError):
                projected.append({"invalid_message": message})
        cleaned.append(projected)
    return digest({"ids": ids, "dates": dates, "sessions": cleaned})


def iter_banks(dataset_spec, selected_splits=None, *, count_tokens, tokenizer_id=None):
    """Yield complete banks; session errors are explicit and do not hide other sessions.

    Saved LoCoMo query rows sharing the exact public history form one bank, even
    across splits. LongMemEval retains a separate bank for every question item.
    ``tokenizer_id`` is recorded by the orchestrator; schemas store actual counts.
    """
    if not callable(count_tokens):
        raise ValueError("An actual tokenizer count_tokens callback is required")
    name = dataset_spec["name"]
    if name not in {"locomo", "split_longmemeval_s", "split_longmemeval_m"}:
        raise ValueError("Only LoCoMo and classic LongMemEval-S/M are supported")
    modality = dataset_spec.get("modality_policy", "existing_captions")
    if modality not in MODALITY_POLICIES:
        raise ValueError("Unsupported modality_policy")
    # Snapshots retain attachment captions. The public renderer/Writer applies
    # references_only when requested; it must never edit the original text.
    grouped = {}
    for split, path in input_files(dataset_spec, selected_splits):
        sha = file_digest(path)
        for index, row in enumerate(_rows(path)):
            fmt = dataset_spec.get("format", ("locomo_json" if name == "locomo" and path.suffix != ".parquet"
                                              else "locomo_parquet" if name == "locomo" else "longmemeval"))
            if fmt not in {"locomo_json", "locomo_parquet", "longmemeval", "longmemeval_json", "longmemeval_parquet"}:
                raise ValueError("Unsupported dataset format")
            safe_row = row if isinstance(row, dict) else {}
            source = _source(path, sha, split, index, safe_row, fmt)
            bank_id = "bank_" + digest([name, split, sha, index])[:24]
            try:
                if fmt == "locomo_json":
                    bank = _official_locomo(name, row, source, count_tokens)
                elif name == "locomo":
                    key = _locomo_history_key(row)
                    if key in grouped:
                        grouped[key]["query_memberships"].append(_membership(name, source))
                        continue
                    bank_id = "bank_" + digest([name, "public_history", key])[:24]
                    bank = _normalize_history(name, row, source, bank_id, count_tokens, locomo=True)
                    grouped[key] = bank
                    continue
                else:
                    bank = _normalize_history(name, row, source, bank_id, count_tokens)
            except (ValueError, TypeError, KeyError) as exc:
                bank = {"bank_id": bank_id, "dataset": name, "sessions": [], "source_map": [source],
                        "query_memberships": [_membership(name, source)], "issues": [_error(bank_id, source, exc)]}
            yield bank
    yield from grouped.values()
