"""Answer exported Task requests on a networked CPU machine.

This file is deliberately standalone and uses only the Python standard library.
Copy it together with a requests.json.gz bundle; no model weights, gold answers,
training data, or GPU are needed. Returned answers are judged separately by the
original local Judge after copying the response directory back.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from threading import Lock
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


REQUEST_PROTOCOL = "checkpoint_task_requests_v1"
TAGGED_REQUEST_PROTOCOL = "checkpoint_task_requests_v2"
RESPONSE_PROTOCOL = "checkpoint_task_responses_v1"
ADAPTIVE_RESPONSE_PROTOCOL = "checkpoint_task_responses_v2"
TAGGED_RESPONSE_PROTOCOL = "checkpoint_task_responses_v3"
COMPLETION_PROTOCOL = "score_nonempty_truncated_v1"
ANSWER_PARSER = "answer_tag_v1"
# Kept here as a stdlib-only contract so this file can still be copied to a
# separate network host. Repository tests compare it with prompts.py exactly.
HISTORY_ANSWER_TEMPLATE = """I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Provide your final answer wrapped in <answer> ... </answer>.

History Chats: {history_chats}

Question: {question}

Please provide your final answer wrapped in <answer> ... </answer>."""
WAIT_REPORT_SECONDS = 30
_SETTING_KEYS = {"model", "model_revision", "request_profile", "send_seed",
                 "temperature", "max_tokens_parameter", "max_output_tokens", "seed"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _is_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def extract_tagged_answer(text):
    """Read exactly one complete, nonempty answer; never guess from raw text."""
    if not isinstance(text, str):
        raise ValueError("Task response must contain one nonempty <answer>...</answer> block")
    # Count tag-like tokens as well as exact tags to reject nested, duplicate,
    # attributed or differently cased blocks instead of silently ignoring them.
    tags = re.findall(r"<\s*/?\s*answer\b[^>]*>", text, flags=re.IGNORECASE)
    if (tags != ["<answer>", "</answer>"]
            or len(re.findall(r"<\s*/?\s*answer\b", text, flags=re.IGNORECASE)) != 2):
        raise ValueError("Task response must contain exactly one <answer>...</answer> block")
    match = re.search(r"<answer>([\s\S]*?)</answer>", text)
    if match is None or not match.group(1).strip():
        raise ValueError("Task response has an empty or incomplete answer block")
    return match.group(1).strip()


def _tagged_contract():
    return {"task_context_version": "question_date_v1", "task_prompt_version": "history_answer_v1",
            "task_prompt_sha256": digest(HISTORY_ANSWER_TEMPLATE),
            "history_rendering": "session_time_raw_text_v1", "answer_parser": ANSWER_PARSER,
            "completion_policy": "require_stop_v1"}


def _validate_tagged_messages(messages):
    if (not isinstance(messages, list) or len(messages) != 1
            or not isinstance(messages[0], dict) or set(messages[0]) != {"role", "content"}
            or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str)):
        raise ValueError("Invalid exported history-answer Task messages")
    prefix, rest = HISTORY_ANSWER_TEMPLATE.split("{history_chats}")
    separator, suffix = rest.split("{question}")
    content = messages[0]["content"]
    if not content.startswith(prefix) or not content.endswith(suffix):
        raise ValueError("History-answer Task template differs from exported contract")
    body = content[len(prefix):-len(suffix)]
    if separator not in body or not body.rsplit(separator, 1)[1].strip():
        raise ValueError("History-answer Task input is missing its question boundary")


def validate_bundle(bundle):
    """Check portable request provenance without credentials or model imports."""
    if not isinstance(bundle, dict) or set(bundle) != {"manifest", "requests"}:
        raise ValueError("Invalid Task request bundle")
    manifest, requests = bundle["manifest"], bundle["requests"]
    if not isinstance(manifest, dict) or manifest.get("protocol") not in {REQUEST_PROTOCOL, TAGGED_REQUEST_PROTOCOL}:
        raise ValueError("Unsupported Task request protocol")
    tagged = manifest["protocol"] == TAGGED_REQUEST_PROTOCOL
    ids = manifest.get("sample_ids")
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(s, str) or not s for s in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Request sample IDs must be nonempty and unique")
    for key in ("retrieval_manifest_sha256", "requests_sha256"):
        if not _is_hash(manifest.get(key)):
            raise ValueError("Invalid request provenance hash: " + key)
    if tagged:
        if (manifest.get("source_kind") != "retrieval_reprompt"
                or "source_task_manifest_sha256" not in manifest
                or manifest["source_task_manifest_sha256"] is not None):
            raise ValueError("History-answer requests must identify their source as saved retrieval")
    elif not _is_hash(manifest.get("source_task_manifest_sha256")):
        raise ValueError("Invalid request provenance hash: source_task_manifest_sha256")
    for key in ("checkpoint", "dataset"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError("Missing request provenance field: " + key)
    contract = manifest.get("task_input_contract")
    if not isinstance(contract, dict):
        raise ValueError("Missing Task input contract")
    if tagged and contract != _tagged_contract():
        raise ValueError("Unsupported history-answer Task input contract")
    if not tagged and contract.get("task_prompt_version", "legacy_v1") != "legacy_v1":
        raise ValueError("Task prompt version requires the history-answer request protocol")
    settings = manifest.get("request_settings")
    if not isinstance(settings, dict) or set(settings) != _SETTING_KEYS:
        raise ValueError("Invalid portable Task request settings")
    for key in ("model", "model_revision"):
        if not isinstance(settings[key], str) or not settings[key].strip() or settings[key].startswith("REQUIRED"):
            raise ValueError("Missing remote model identity: " + key)
    if settings["request_profile"] != "openai_chat" or type(settings["send_seed"]) is not bool:
        raise ValueError("Invalid remote request profile or seed setting")
    temperature = settings["temperature"]
    if temperature is not None and (isinstance(temperature, bool)
            or not isinstance(temperature, (int, float)) or not math.isfinite(temperature)
            or not 0 <= temperature <= 2):
        raise ValueError("Invalid remote temperature")
    if settings["max_tokens_parameter"] not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("Invalid remote output limit parameter")
    if type(settings["max_output_tokens"]) is not int or settings["max_output_tokens"] < 1:
        raise ValueError("Invalid remote output limit")
    if type(settings["seed"]) is not int:
        raise ValueError("Invalid remote seed")
    if not isinstance(requests, list) or len(requests) != len(ids):
        raise ValueError("Request count differs from manifest")
    for expected_id, row in zip(ids, requests):
        if not isinstance(row, dict) or set(row) != {"sample_id", "task_input_hash", "messages"}:
            raise ValueError("Invalid request record")
        if row["sample_id"] != expected_id:
            raise ValueError("Request sample order differs from manifest")
        messages = row["messages"]
        if tagged:
            _validate_tagged_messages(messages)
            if row["task_input_hash"] != digest(messages):
                raise ValueError("Exported Task input hash mismatch")
            continue
        if (not isinstance(messages, list) or len(messages) != 2
                or any(not isinstance(m, dict) or set(m) != {"role", "content"}
                       or not isinstance(m["content"], str) for m in messages)
                or [m["role"] for m in messages] != ["system", "user"]
                or not messages[0]["content"].strip()):
            raise ValueError("Invalid exported Task messages")
        try:
            task = json.loads(messages[1]["content"])
        except (TypeError, ValueError):
            raise ValueError("Task input must contain the original public task and evidence") from None
        if (not isinstance(task, dict) or set(task) != {"query", "observation", "options", "retrieved_evidence"}
                or not isinstance(task["query"], str) or not isinstance(task["observation"], str)
                or not isinstance(task["options"], dict) or not isinstance(task["retrieved_evidence"], list)):
            raise ValueError("Task input must contain only the public task and evidence")
        if contract.get("task_prompt_sha256") is not None and digest(messages[0]["content"]) != contract["task_prompt_sha256"]:
            raise ValueError("Task system prompt differs from exported contract")
        if row["task_input_hash"] != digest(messages):
            raise ValueError("Exported Task input hash mismatch")
    if manifest["requests_sha256"] != digest(requests):
        raise ValueError("Request bundle content hash mismatch")
    return bundle


def load_bundle(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return validate_bundle(json.load(stream))


def _endpoint(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Set OPENAI_BASE_URL to the remote service endpoint")
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("Remote endpoint must be an HTTP(S) URL without credentials, query or fragment")
    return value


def task_provenance(settings, endpoint):
    return {"identity": {"model": settings["model"], "model_revision": settings["model_revision"],
                         "base_url": _endpoint(endpoint), "request_profile": "openai_chat"},
            "request_options": {key: settings[key] for key in
                                ("request_profile", "send_seed", "temperature", "max_tokens_parameter")}}


def _empty_answer_policy(settings, maximum):
    if maximum is None:
        return None
    initial = settings["max_output_tokens"]
    if type(maximum) is not int or maximum < initial:
        raise ValueError("--empty-answer-max-tokens must be an integer at least the exported token budget")
    if settings["max_tokens_parameter"] != "max_completion_tokens" or maximum == initial:
        return None
    return {"protocol": "empty_answer_budget_v1", "initial_max_tokens": initial,
            "max_completion_tokens": maximum, "multiplier": 4,
            "trigger": "empty_text_finish_length"}


def _tagged_answer_policy(settings, maximum):
    initial = settings["max_output_tokens"]
    if maximum is None:
        maximum = initial
    if type(maximum) is not int or maximum < initial:
        raise ValueError("--empty-answer-max-tokens must be an integer at least the exported token budget")
    if settings["max_tokens_parameter"] != "max_completion_tokens":
        maximum = initial
    return {"protocol": "answer_tag_complete_v1", "initial_max_tokens": initial,
            "max_completion_tokens": maximum, "multiplier": 4,
            "trigger": "finish_reason_length", "require_finish_reason": "stop"}


def _allowed_budgets(settings, policy):
    budget = settings["max_output_tokens"]
    budgets = {budget}
    if policy is not None:
        while budget < policy["max_completion_tokens"]:
            budget = min(budget * policy["multiplier"], policy["max_completion_tokens"])
            budgets.add(budget)
    return budgets


def response_manifest(bundle, endpoint, empty_answer_max_tokens=None):
    settings = bundle["manifest"]["request_settings"]
    manifest = {"protocol": RESPONSE_PROTOCOL, "request_manifest_sha256": digest(bundle["manifest"]),
                "sample_ids": bundle["manifest"]["sample_ids"],
                "task": task_provenance(settings, endpoint)}
    if bundle["manifest"]["protocol"] == TAGGED_REQUEST_PROTOCOL:
        manifest["protocol"] = TAGGED_RESPONSE_PROTOCOL
        manifest["task"]["request_options"].update(
            answer_parser=ANSWER_PARSER,
            answer_completion_policy=_tagged_answer_policy(settings, empty_answer_max_tokens))
        return manifest
    policy = _empty_answer_policy(settings, empty_answer_max_tokens)
    if policy is not None:
        # Preserve the exact v1 signature so existing successful base-budget
        # answers can remain byte-for-byte unchanged during an atomic upgrade.
        manifest["legacy_manifest_sha256"] = digest(manifest)
        manifest["protocol"] = ADAPTIVE_RESPONSE_PROTOCOL
        manifest["task"]["request_options"]["empty_answer_retry"] = policy
    return manifest


def completion_problem(completion, max_tokens):
    """Return a local reason code, never provider-controlled diagnostic text."""
    if not isinstance(completion, dict):
        return "invalid_completion_object"
    if completion.get("protocol") != COMPLETION_PROTOCOL:
        return "invalid_completion_protocol"
    text = completion.get("text")
    if text is None or isinstance(text, str) and not text.strip():
        return "empty_text"
    if not isinstance(text, str):
        return "non_text_content"
    if ("finish_reason" not in completion or completion["finish_reason"] is not None
            and not isinstance(completion["finish_reason"], str)):
        return "invalid_finish_reason"
    if (type(completion.get("truncated")) is not bool
            or completion["truncated"] != (completion["finish_reason"] == "length")):
        return "invalid_truncation_metadata"
    if ("usage" not in completion or completion["usage"] is not None
            and not isinstance(completion["usage"], dict)):
        return "invalid_usage"
    if type(completion.get("max_tokens")) is not int or completion["max_tokens"] != max_tokens:
        return "invalid_token_budget"
    try:
        canonical(completion)
    except (TypeError, ValueError):
        return "invalid_json_metadata"
    return None


def validate_completion(completion, max_tokens):
    reason = completion_problem(completion, max_tokens)
    if reason:
        raise ValueError(f"Invalid remote Task completion: {reason}; no task score assigned")
    return completion


def _tagged_completion_problem(completion, max_tokens):
    reason = completion_problem(completion, max_tokens)
    if reason is not None:
        return reason
    if completion["finish_reason"] != "stop":
        return "truncated_answer" if completion["finish_reason"] == "length" else "incomplete_answer"
    try:
        extract_tagged_answer(completion["text"])
    except ValueError:
        return "invalid_answer_tags"
    return None


def validate_tagged_completion(completion, max_tokens):
    reason = _tagged_completion_problem(completion, max_tokens)
    if reason is not None:
        raise ValueError(f"Invalid tagged Task completion: {reason}; no task score assigned")
    return completion


def _completion_diagnostics(completion):
    """Allowlist scalar metadata; omit prompts, answers, endpoints and headers."""
    if not isinstance(completion, dict):
        return {}
    finish = completion.get("finish_reason")
    if (not isinstance(finish, (str, type(None)))
            or finish not in {None, "stop", "length", "content_filter", "tool_calls", "function_call"}):
        finish = "other"
    content = completion.get("text")
    usage = completion.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("completion_tokens_details")
    details = details if isinstance(details, dict) else {}
    values = {"finish_reason": finish, "content_chars": len(content) if isinstance(content, str) else None}
    for key, value in (("completion_tokens", usage.get("completion_tokens")),
                       ("reasoning_tokens", details.get("reasoning_tokens"))):
        values[key] = value if type(value) is int and value >= 0 else None
    return values


def load_response_records(bundle, root, require_complete=True):
    """Read saved answers and validate their pairing; never makes an API call."""
    validate_bundle(bundle)
    root = Path(root)
    manifest = _read_json(root / "manifest.json")
    try:
        endpoint = manifest["task"]["identity"]["base_url"]
    except (TypeError, KeyError):
        raise ValueError("Missing response model provenance") from None
    maximum = None
    if manifest.get("protocol") in {ADAPTIVE_RESPONSE_PROTOCOL, TAGGED_RESPONSE_PROTOCOL}:
        policy_key = ("answer_completion_policy" if manifest["protocol"] == TAGGED_RESPONSE_PROTOCOL
                      else "empty_answer_retry")
        try:
            policy = manifest["task"]["request_options"][policy_key]
        except (KeyError, TypeError):
            raise ValueError("Missing adaptive completion budget policy") from None
        if not isinstance(policy, dict):
            raise ValueError("Missing adaptive completion budget policy")
        maximum = policy.get("max_completion_tokens")
        if maximum is None:
            raise ValueError("Missing adaptive completion budget cap")
    if manifest != response_manifest(bundle, endpoint, maximum):
        raise ValueError("Responses do not match the exported requests or remote model protocol")
    signature = digest(manifest)
    legacy_signature = manifest.get("legacy_manifest_sha256")
    valid_signatures = {signature}
    if legacy_signature is not None:
        valid_signatures.add(legacy_signature)
    settings = bundle["manifest"]["request_settings"]
    policy_key = ("answer_completion_policy" if manifest["protocol"] == TAGGED_RESPONSE_PROTOCOL
                  else "empty_answer_retry")
    policy = manifest["task"]["request_options"].get(policy_key)
    budgets = _allowed_budgets(settings, policy)
    expected_files = {f"{i:08d}.json" for i in range(len(bundle["requests"]))}
    if any(p.name not in expected_files for p in (root / "episodes").glob("*.json")):
        raise ValueError("Unexpected response episode file")
    rows = {}
    for index, request in enumerate(bundle["requests"]):
        path = root / "episodes" / f"{index:08d}.json"
        if not path.exists():
            if require_complete:
                raise ValueError("Remote answers are incomplete; resume the network runner first")
            continue
        row = _read_json(path)
        if (not isinstance(row, dict) or set(row) != {"manifest_sha256", "sample_id", "task_input_hash", "completion"}
                or not _is_hash(row["manifest_sha256"])
                or row["manifest_sha256"] not in valid_signatures or row["sample_id"] != request["sample_id"]
                or row["task_input_hash"] != request["task_input_hash"]):
            raise ValueError("Remote answer provenance or Task input hash mismatch")
        completion = row["completion"]
        budget = completion.get("max_tokens") if isinstance(completion, dict) else None
        if (type(budget) is not int or budget not in budgets
                or row["manifest_sha256"] == legacy_signature and budget != settings["max_output_tokens"]):
            raise ValueError("Remote completion budget does not match its recorded request policy")
        if bundle["manifest"]["protocol"] == TAGGED_REQUEST_PROTOCOL:
            validate_tagged_completion(completion, budget)
        else:
            validate_completion(completion, budget)
        rows[row["sample_id"]] = row
    complete = root / "complete.json"
    if complete.exists():
        marker = _read_json(complete)
        if (not isinstance(marker, dict) or set(marker) != {"manifest_sha256", "n"}
                or not _is_hash(marker["manifest_sha256"]) or marker["manifest_sha256"] not in valid_signatures
                or marker["n"] != len(bundle["requests"])
                or len(rows) != len(bundle["requests"])):
            raise ValueError("Invalid remote completion marker")
    elif require_complete:
        raise ValueError("Remote response completion marker is missing; resume the network runner first")
    return manifest, rows


@contextmanager
def _output_lock(root):
    # POSIX file locking is available on the Linux GPU/network hosts and macOS.
    # Windows support uses the equivalent byte-range lock from the stdlib.
    path = Path(root) / ".runner.lock"
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise RuntimeError("Another runner owns this response directory") from None
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise RuntimeError("Another runner owns this response directory") from None
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Credentials must stay on the explicitly configured provider endpoint.
        return None


def _complete(request, settings, endpoint, api_key, timeout, retries, report=None,
              empty_answer_max_tokens=None, *, answer_parser=None):
    if answer_parser not in {None, ANSWER_PARSER}:
        raise ValueError("Unsupported Task answer parser")
    tagged = answer_parser == ANSWER_PARSER
    policy = (_tagged_answer_policy(settings, empty_answer_max_tokens) if tagged
              else _empty_answer_policy(settings, empty_answer_max_tokens))
    budget = settings["max_output_tokens"]
    payload = {"model": settings["model"], "messages": request["messages"],
               settings["max_tokens_parameter"]: budget}
    if settings["temperature"] is not None:
        payload["temperature"] = settings["temperature"]
    if settings["send_seed"]:
        payload["seed"] = settings["seed"]
    opener = build_opener(_NoRedirect())
    error_type = "UnknownError"
    reason = "unknown"
    report = report or (lambda event, **fields: None)
    attempt, retries_used, budget_expansions = 0, 0, 0
    # Transport failures must not use up the opportunity to recover a
    # length-limited answer. Legacy requests expand only empty answers; tagged
    # requests require a complete answer even when a visible prefix is present.
    # Each budget has its own retry allowance and a separately bounded cap.
    while True:
        attempt += 1
        payload[settings["max_tokens_parameter"]] = budget
        http_request = Request(endpoint + "/chat/completions", data=canonical(payload).encode(),
                               headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
                               method="POST")
        started = time.monotonic()
        completion, status_code = None, None
        report("remote_task_attempt_started", sample_id=request["sample_id"],
               attempt=attempt, timeout_seconds=timeout, max_tokens=budget,
               retries_used_at_budget=retries_used, budget_expansions=budget_expansions)
        try:
            with opener.open(http_request, timeout=timeout) as response:
                status_code = getattr(response, "status", None)
                if type(status_code) is not int:
                    status_code = None
                result = json.load(response)
            choice = result["choices"][0]
            completion = {"protocol": COMPLETION_PROTOCOL, "text": choice["message"]["content"],
                          "finish_reason": choice.get("finish_reason"),
                          "truncated": choice.get("finish_reason") == "length", "usage": result.get("usage"),
                          "max_tokens": budget}
            if tagged:
                validate_tagged_completion(completion, budget)
            else:
                validate_completion(completion, budget)
        except Exception as exc:
            # Never print exception messages or provider response bodies: either
            # may echo request headers, credentials, or private task evidence.
            error_type = type(exc).__name__
            reason = ((_tagged_completion_problem(completion, budget) if tagged
                       else completion_problem(completion, budget))
                      if completion is not None else "response_or_transport_error")
            next_budget = budget
            if (policy is not None and completion is not None and (tagged or reason == "empty_text")
                    and completion.get("finish_reason") == "length"):
                next_budget = min(budget * policy["multiplier"], policy["max_completion_tokens"])
            increase_budget = next_budget > budget
            will_retry = increase_budget or retries_used < retries
            retry_kind = ("increase_token_budget" if increase_budget else
                          "retry_current_budget" if will_retry else None)
            transport = {}
            if isinstance(exc, URLError):
                transport["transport_error_type"] = type(exc.reason).__name__
                number = getattr(exc.reason, "errno", None)
                if type(number) is int:
                    transport["transport_errno"] = number
            if isinstance(exc, HTTPError):
                status_code = exc.code
                exc.close()
            report("remote_task_attempt_failed", sample_id=request["sample_id"], attempt=attempt,
                   elapsed_seconds=round(time.monotonic() - started, 3), status_code=status_code,
                   error_type=error_type, reason=reason, will_retry=will_retry,
                   max_tokens=budget, next_max_tokens=next_budget if will_retry else None,
                   retry_kind=retry_kind, retries_used_at_budget=retries_used,
                   budget_expansions=budget_expansions,
                   **transport, **_completion_diagnostics(completion))
            if not will_retry:
                break
            if increase_budget:
                budget = next_budget
                budget_expansions += 1
                retries_used = 0
            else:
                time.sleep(min(2 ** retries_used, 8))
                retries_used += 1
        else:
            report("remote_task_attempt_succeeded", sample_id=request["sample_id"], attempt=attempt,
                   elapsed_seconds=round(time.monotonic() - started, 3), status_code=status_code,
                   max_tokens=budget, retries_used_at_budget=retries_used,
                   budget_expansions=budget_expansions,
                   **_completion_diagnostics(completion))
            return completion
    hint = ""
    if reason == "empty_text" and completion is not None and completion.get("finish_reason") == "length":
        hint = f"; empty answer at token limit {budget}; inspect events.jsonl and the recorded completion budget"
    elif tagged and reason == "truncated_answer":
        hint = f"; tagged answer truncated at token limit {budget}; no partial answer saved"
    raise RuntimeError(f"Remote Task request failed ({error_type}, reason={reason}){hint}; no task score assigned") from None


def run(requests, output, workers=4, timeout=600, retries=3, empty_answer_max_tokens=None):
    """Run/resume the network-only phase, persisting each successful answer."""
    if type(workers) is not int or workers < 1:
        raise ValueError("--workers must be a positive integer")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("--timeout must be positive and finite")
    if type(retries) is not int or retries < 0:
        raise ValueError("--retries must be a nonnegative integer")
    bundle = load_bundle(requests)
    tagged = bundle["manifest"]["protocol"] == TAGGED_REQUEST_PROTOCOL
    if tagged:
        _tagged_answer_policy(bundle["manifest"]["request_settings"], empty_answer_max_tokens)
    else:
        _empty_answer_policy(bundle["manifest"]["request_settings"], empty_answer_max_tokens)
    endpoint = _endpoint(os.environ.get("OPENAI_BASE_URL", ""))
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY for the remote Task service")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with _output_lock(output):
        manifest_path = output / "manifest.json"
        previous = _read_json(manifest_path) if manifest_path.exists() else None
        # Library callers omitting the optional setting retain the existing
        # adaptive protocol on resume. A fresh run still defaults to v1.
        if (empty_answer_max_tokens is None and isinstance(previous, dict)
                and previous.get("protocol") in {ADAPTIVE_RESPONSE_PROTOCOL, TAGGED_RESPONSE_PROTOCOL}):
            policy_key = ("answer_completion_policy" if previous["protocol"] == TAGGED_RESPONSE_PROTOCOL
                          else "empty_answer_retry")
            try:
                empty_answer_max_tokens = previous["task"]["request_options"][policy_key]["max_completion_tokens"]
            except (TypeError, KeyError):
                raise ValueError("Missing adaptive completion budget policy") from None
        expected = response_manifest(bundle, endpoint, empty_answer_max_tokens)
        if manifest_path.exists():
            if previous != expected:
                legacy = response_manifest(bundle, endpoint)
                if expected["protocol"] != ADAPTIVE_RESPONSE_PROTOCOL or previous != legacy:
                    raise ValueError("Remote evaluation protocol changed; choose a new response directory")
                # Validate all old records before changing even the manifest.
                # Readers can then accept v1 records through the attested legacy
                # signature; no per-episode rewrite can leave a partial upgrade.
                load_response_records(bundle, output, require_complete=False)
                _atomic_json(manifest_path, expected)
        else:
            if any(p.name != ".runner.lock" for p in output.iterdir()):
                raise ValueError("Response artifacts exist without a manifest")
            _atomic_json(manifest_path, expected)
        manifest, cached = load_response_records(bundle, output, require_complete=False)
        signature = digest(manifest)
        settings = bundle["manifest"]["request_settings"]
        pending = iter((i, r) for i, r in enumerate(bundle["requests"]) if r["sample_id"] not in cached)
        completed = len(cached)
        first_error = None
        event_lock = Lock()

        def report(event, **fields):
            record = {"timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
            with event_lock:
                with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(canonical(record) + "\n")
                if event not in {"remote_task_attempt_started", "remote_task_attempt_succeeded"}:
                    print(canonical(record), flush=True)

        report("remote_task_resumed", completed=completed, total=len(bundle["requests"]),
               workers=workers, timeout_seconds=timeout, retries=retries,
               empty_answer_retry=manifest["task"]["request_options"].get("empty_answer_retry"))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            inflight = {}

            def submit():
                item = next(pending, None)
                if item is None:
                    return False
                index, request = item
                parser_options = {"answer_parser": ANSWER_PARSER} if tagged else {}
                future = pool.submit(_complete, request, settings, endpoint, api_key, timeout, retries,
                                     report, empty_answer_max_tokens, **parser_options)
                inflight[future] = (index, request, time.monotonic())
                return True

            for _ in range(workers):
                if not submit():
                    break
            while inflight:
                done, _ = wait(inflight, timeout=WAIT_REPORT_SECONDS, return_when=FIRST_COMPLETED)
                if not done:
                    report("remote_task_waiting", completed=completed, total=len(bundle["requests"]),
                           draining=first_error is not None,
                           pending=[{"sample_id": r["sample_id"],
                                     "elapsed_seconds": round(time.monotonic() - started, 1)}
                                    for _, r, started in inflight.values()])
                # Inspect all completed futures before scheduling anything else.
                for future in done:
                    index, request, _ = inflight.pop(future)
                    try:
                        completion = future.result()
                        if tagged:
                            policy = manifest["task"]["request_options"]["answer_completion_policy"]
                            budget = completion.get("max_tokens") if isinstance(completion, dict) else None
                            if type(budget) is not int or budget not in _allowed_budgets(settings, policy):
                                raise ValueError("Tagged completion budget differs from recorded policy")
                            validate_tagged_completion(completion, budget)
                        _atomic_json(output / "episodes" / f"{index:08d}.json", {
                            "manifest_sha256": signature, "sample_id": request["sample_id"],
                            "task_input_hash": request["task_input_hash"], "completion": completion})
                        completed += 1
                        report("remote_task_progress", sample_id=request["sample_id"],
                               completed=completed, total=len(bundle["requests"]))
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
                        report("remote_task_failed", sample_id=request["sample_id"],
                               error_type=type(exc).__name__, completed=completed,
                               total=len(bundle["requests"]), draining_requests=len(inflight))
                if first_error is None:
                    for _ in range(workers - len(inflight)):
                        if not submit():
                            break
            if first_error is not None:
                raise first_error
        _atomic_json(output / "complete.json", {"manifest_sha256": signature, "n": len(bundle["requests"])})
        load_response_records(bundle, output)
        report("remote_task_finished", completed=completed, responses=str(output.resolve()))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, help="Exported requests.json.gz")
    parser.add_argument("--output", required=True, help="Resumable response directory; copy it back for local judging")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--retries", type=int, default=3,
                        help="Ordinary error retries per token budget; empty-answer budget increases "
                             "have a separate allowance bounded by --empty-answer-max-tokens")
    parser.add_argument("--empty-answer-max-tokens", type=int, default=16384,
                        help="Maximum retry budget for empty length-ended answers, or any length-ended "
                             "history-answer response; the initial budget and successful saved answers stay unchanged")
    args = parser.parse_args()
    try:
        run(args.requests, args.output, args.workers, args.timeout, args.retries, args.empty_answer_max_tokens)
    except Exception as exc:
        # Validation errors are authored locally; API exceptions are sanitized
        # by _complete. Avoid an exception traceback that might include a URL.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
