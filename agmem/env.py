from __future__ import annotations

import json
import re

from .core import CapacityError, Decision, Trajectory, digest
from .prompts import reader_messages, task_messages


def parse_action(text: str) -> dict:
    # Accept one fenced JSON object; keep schema validation strict and preserve sampled tokens.
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", text.strip(), flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"ids", "stop"}:
        raise ValueError("Expected exactly ids and stop")
    if type(value["stop"]) is not bool or not isinstance(value["ids"], list) or any(not isinstance(item, str) for item in value["ids"]):
        raise ValueError("Invalid action types")
    if value["stop"] and value["ids"]:
        raise ValueError("stop=true requires ids=[]")
    return value


class MemoryEnvironment:
    def __init__(self, store, directory_version: str, max_ids=10, max_calls=8, capacity=None):
        self.store, self.version = store, directory_version
        self.max_ids, self.max_calls, self.capacity = max_ids, max_calls, capacity

    def run(self, example, policy, seed: int, policy_version=0):
        directory = self.store.directory(example.memory_id, self.version)
        entries = {row["id"]: row for row in directory}
        references = {row["id"]: row for row in self.store.memory(example.memory_id)}
        if len(entries) != len(directory) or entries.keys() != references.keys():
            raise ValueError("Directory does not align with the immutable memory snapshot")
        trajectory = Trajectory(example.sample_id, seed, example.memory_id, policy_version)
        evidence, error = {}, None
        for decision_index in range(self.max_calls + 1):
            messages = reader_messages(example, directory, list(evidence.values()), error)
            try:
                completion = policy.complete(messages, seed=seed + decision_index)
            except CapacityError as exc:
                raise CapacityError(f"{example.sample_id}, decision {decision_index}: {exc}") from exc
            decision = Decision(messages, completion, state_sha256=digest(messages))
            trajectory.decisions.append(decision)
            try:
                action = parse_action(completion.text)
                decision.action = action
            except (ValueError, TypeError):
                trajectory.termination_reason = "protocol_failure"
                break
            if action["stop"]:
                trajectory.termination_reason = "stop"
                break
            if trajectory.n_read_calls >= self.max_calls:
                trajectory.termination_reason = "max_read_calls"
                break
            trajectory.n_read_calls += 1
            ids = action["ids"]
            code = None
            if not 1 <= len(ids) <= self.max_ids:
                code = "invalid_batch_size"
            elif len(set(ids)) != len(ids):
                code = "duplicate_ids_in_batch"
            elif any(key not in references for key in ids):
                code = "unknown_or_unauthorized_id"
            records = []
            if code is None:
                for key in ids:
                    original = self.store.record(references[key])
                    records.append({key_: original[key_] for key_ in ("id", "source_session_id", "time", "order", "raw_text")} | {"content": entries[key]["content"]})
                candidate = {**evidence, **{row["id"]: row for row in records}}
                try:
                    if self.capacity:
                        self.capacity(reader_messages(example, directory, list(candidate.values())), task_messages(example, list(candidate.values())))
                except CapacityError:
                    code = "context_capacity_exceeded"
            if code is not None:
                error = {"code": code, "requested_ids": ids}
                decision.tool_result = {"ok": False, "requested_ids": ids, "records": [], "error": error}
            else:
                evidence.update({row["id"]: row for row in records})
                decision.tool_result = {"ok": True, "requested_ids": ids, "records": records}
                error = None
        else:
            trajectory.termination_reason = "max_decisions"
        trajectory.evidence = list(evidence.values())
        return trajectory
