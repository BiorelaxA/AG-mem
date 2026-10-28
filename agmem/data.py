from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json
import random
import re

from .core import Example, canonical, digest, file_digest, seeded_int

ADAPTER_VERSION = "session_choice_v2_empty_sessions"
# Session/history adaptation is unchanged, so existing Writer directories remain
# valid. Public task context is versioned separately from history preparation.
TASK_CONTEXT_VERSION = "question_date_v1"
LONGMEMEVAL_SOURCES = {"split_longmemeval_s", "split_longmemeval_m"}


class InvalidSample(ValueError):
    pass


def decode_list(value, name: str) -> list:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise InvalidSample(f"{name}: invalid JSON") from exc
    if not isinstance(value, list):
        raise InvalidSample(f"{name}: expected list")
    return value


def option_label(value: str) -> str | None:
    match = re.fullmatch(r"\s*\(?([A-Za-z])\)?[.．、]?\s*", value or "")
    return match.group(1).upper() if match else None


def question_date_observation(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidSample("LongMemEval requires a nonempty question_date")
    return f"Current date (when the question was asked): {value.strip()}"


def labeled_options(options: list[str]) -> dict[str, str]:
    result = {}
    for text in options:
        match = re.match(r"^\s*\(?([A-Za-z])\)?[.．、:：)）]?\s+(.+)$", text, re.S)
        if not match or match.group(1).upper() in result:
            raise InvalidSample("Invalid or duplicate option label")
        result[match.group(1).upper()] = match.group(2).strip()
    if len(result) < 2:
        raise InvalidSample("Too few options")
    return result


def adapt_row(row: dict, source: str, split: str, index: int, memory_id: str = "") -> Example:
    sample_id = f"{source}/{split}/{index:06d}"
    query, answer = row.get("question"), row.get("answer")
    if not isinstance(query, str) or not query.strip() or not isinstance(answer, str) or not answer.strip():
        raise InvalidSample("Empty or non-string question/answer")
    example = Example(sample_id, source, split, memory_id, query, gold_answers=[answer])
    if source in LONGMEMEVAL_SOURCES:
        example.observation = question_date_observation(row.get("question_date"))
        example.judge_metadata = {key: row[key] for key in ("question_type", "question_id") if key in row}
    if source in {"personamem_32k", "personamem_128k"}:
        example.options = labeled_options(decode_list(row["all_options"], "all_options"))
        label = option_label(row.get("correct_answer"))
        if label not in example.options:
            raise InvalidSample("Unknown correct option")
        if example.options[label].strip() != answer.strip():
            raise InvalidSample("Correct option and answer text disagree")
        example.answer_mode, example.gold_answers = "choice", [label]
    elif source == "zh4o":
        matches = list(re.finditer(r"(?<![A-Za-z])([ABCD])(?:[.．、:：)）]|\s)+", query))
        if [match.group(1) for match in matches] != list("ABCD"):
            raise InvalidSample("Cannot parse all four ZH4O options unambiguously")
        example.options = {match.group(1): query[match.end():matches[j + 1].start() if j + 1 < len(matches) else None].strip()
                           for j, match in enumerate(matches)}
        label = option_label(answer)
        if label not in example.options or not all(example.options.values()):
            raise InvalidSample("Invalid ZH4O reference label")
        example.query = query[:matches[0].start()].strip()
        example.answer_mode, example.gold_answers = "choice", [label]
    elif source == "personamem_v2_128k":
        wrong = decode_list(row["incorrect_answers"], "incorrect_answers")
        wrong = [text for text in wrong if text is not None]  # Null padding is not an option.
        if not wrong or any(not isinstance(text, str) or not text.strip() for text in wrong):
            raise InvalidSample("Invalid PersonaMem-v2 distractors")
        texts = [answer] + wrong
        if len(set(text.strip() for text in texts)) != len(texts) or len(texts) > 26:
            raise InvalidSample("Duplicate or too many PersonaMem-v2 choices")
        # Correctness is never used for visible ordering. Shuffle all indexed options with a stable seed.
        order = list(range(len(texts)))
        random.Random(seeded_int("v2-options-v1", sample_id)).shuffle(order)
        example.options = {chr(65 + i): texts[j] for i, j in enumerate(order)}
        example.gold_answers = [chr(65 + order.index(0))]
        example.answer_mode = "choice"
    return example


def task_examples(config, store, split):
    """Read tasks with public dates, including legacy prepared examples.

    Restore only task context in memory. Do not rewrite live JSONL, memories,
    directory versions, or Writer caches. Verified scalar Parquet columns are
    read once per source/split per Store lifetime, without parsing histories.
    """
    examples = store.examples(split)
    sources = {e.source for e in examples} & LONGMEMEVAL_SOURCES
    if not sources:
        return examples
    import pyarrow.parquet as pq
    root = Path(config["data"]["input_root"])
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = {row["source"]: row for row in manifest["sources"]}
    identity = store.get_metadata("prepare_identity")
    needs_judge = config.get("reward", {}).get("formula") in {"judge_gated_v1", "judge_cost_v2"}
    cache = getattr(store, "_question_date_rows", None)
    if cache is None:
        cache = store._question_date_rows = {}
    for source in sorted(sources):
        key = (str(root.resolve()), source, split, needs_judge)
        if key in cache:
            continue
        prepared = identity["datasets"][source] if identity.get("layout") == "by_source" else identity
        if prepared["input_manifest_sha256"] != file_digest(manifest_path):
            raise ValueError("Question-date input manifest differs from prepared provenance")
        path = root / source / f"{split}.parquet"
        if file_digest(path) != entries[source]["output_files"][split]["sha256"]:
            raise ValueError(f"Question-date dataset content changed: {path}")
        file = pq.ParquetFile(path)
        if "question_date" not in file.schema_arrow.names:
            raise InvalidSample(f"LongMemEval question_date column missing: {path}")
        columns = ["question", "answer", "question_date"]
        if needs_judge:
            for field in ("question_type", "question_id"):
                if field not in file.schema_arrow.names:
                    raise InvalidSample(f"LongMemEval Judge requires {field}: {path}")
                columns.append(field)
        cache[key] = file.read(columns=columns).to_pylist()
    restored = []
    for example in examples:
        if example.source not in sources:
            restored.append(example)
            continue
        parts = example.sample_id.split("/")
        if len(parts) != 3 or parts[:2] != [example.source, split] or not parts[2].isdigit():
            raise ValueError(f"Cannot map prepared sample to question date: {example.sample_id}")
        index = int(parts[2])
        rows = cache[(str(root.resolve()), example.source, split, needs_judge)]
        if index >= len(rows):
            raise ValueError(f"Question-date row missing: {example.sample_id}")
        raw = rows[index]
        if raw["question"] != example.query or example.gold_answers != [raw["answer"]]:
            raise ValueError(f"Question-date row does not match prepared example: {example.sample_id}")
        observation = question_date_observation(raw["question_date"])
        if example.observation not in ("", observation):
            raise ValueError(f"Conflicting prepared question-date context: {example.sample_id}")
        metadata = example.judge_metadata
        if needs_judge:
            metadata = {key: raw[key] for key in ("question_type", "question_id")}
            if example.judge_metadata and example.judge_metadata != metadata:
                raise ValueError(f"Conflicting prepared Judge metadata: {example.sample_id}")
        restored.append(replace(example, observation=observation, judge_metadata=metadata))
    return restored


# Historical annotation and unexpected malformed columns cannot pass this allowlist.
MESSAGE_FIELDS = {"role", "content", "name", "timestamp", "time", "date", "dia_id", "tool_calls", "tool_call_id", "function_call"}


def clean_history(row: dict) -> list[dict]:
    sessions = decode_list(row.get("haystack_sessions"), "haystack_sessions")
    ids = row.get("haystack_session_ids")
    dates = row.get("haystack_dates")
    ids = decode_list(ids, "haystack_session_ids") if ids is not None else [None] * len(sessions)
    dates = decode_list(dates, "haystack_dates") if dates is not None else [None] * len(sessions)
    if len(ids) != len(sessions) or len(dates) != len(sessions):
        raise InvalidSample("Session/ID/date lengths disagree")
    records = []
    for position, (session, original_id, date) in enumerate(zip(sessions, ids, dates)):
        if not isinstance(session, list):
            raise InvalidSample("Malformed session")
        messages = []
        for message in session:
            if not isinstance(message, dict) or not isinstance(message.get("role"), str) or not isinstance(message.get("content"), (str, list)):
                raise InvalidSample("Malformed message role/content")
            messages.append({key: value for key, value in message.items() if key in MESSAGE_FIELDS})
        # Source conversion may add a gold-derived 'answer_' marker to a public session ID.
        sid = re.sub(r"^(?:answer_)+", "", str(original_id)) if original_id is not None else f"session_{position}"
        body = {"messages": messages}
        content_hash = digest(body)
        metadata = {"source_session_id": sid, "time": date, "order": position}
        record_id = f"r_{position:05d}_{digest([content_hash, metadata])[:12]}"
        records.append({"id": record_id, **metadata, "session_hash": content_hash, "messages": messages})
    if not records:
        raise InvalidSample("Empty history")
    return records


def parquet_rows(root: str | Path, sources: list[str], splits: list[str], limit: int | None = None, verify: bool = True):
    import pyarrow.parquet as pq
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    entries = {row["source"]: row for row in manifest["sources"]}
    for source in sources:
        if source not in entries:
            raise ValueError(f"Source missing from fixed split manifest: {source}")
        for split in splits:
            path = root / source / f"{split}.parquet"
            if split not in {"train", "validation", "test"}:
                raise ValueError("Unknown split")
            if verify and file_digest(path) != entries[source]["output_files"][split]["sha256"]:
                raise ValueError(f"Dataset content changed: {path}")
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
                for row in batch.to_pylist():
                    if limit is not None and index >= limit:
                        break
                    yield source, split, index, row
                    index += 1
                if limit is not None and index >= limit:
                    break
