"""Leaf-only ORM environment with every available child directory visible."""

from .core import CapacityError, Decision, InfrastructureError, Trajectory, digest
from .env import parse_action
from .full_directory_prompts import (READER_VERSION, TASK_VERSION, full_directory_interaction_history,
                                     full_directory_reader_messages, full_directory_reader_system)
from .hierarchical.reader import DirectoryReader
from .hierarchical_env import HierarchicalMemoryEnvironment


class FullDirectoryMemoryEnvironment(HierarchicalMemoryEnvironment):
    """Reuse immutable hierarchical banks and exact evidence, with no navigation.

    Every available parent is expanded before the first decision. This preparation
    is free; only model READ actions consume calls. Evidence and source-reader state
    are committed together after the complete next Reader and Task inputs fit.
    """

    def __init__(self, store, directory_version, max_ids=2, max_calls=10, capacity=None,
                 id_protocol="strict_v2", *, count_tokens=None,
                 prompt_version=READER_VERSION, task_prompt_version=TASK_VERSION,
                 modality_policy="existing_captions", visible_frontier=None):
        full_directory_reader_system(max_ids, max_calls)
        if id_protocol != "strict_v2" or prompt_version != READER_VERSION or task_prompt_version != TASK_VERSION:
            raise ValueError("Full-directory ORM requires strict_v2, hierarchical_full_v1 and hierarchical_evidence_v1")
        count_tokens = count_tokens or getattr(store, "hierarchical_count_tokens", None)
        if not callable(count_tokens):
            raise ValueError("Provide the actual directory tokenizer count_tokens callback")
        self.store, self.version = store, directory_version
        self.max_ids, self.max_calls, self.capacity = max_ids, max_calls, capacity
        self.count_tokens, self.visible_frontier = count_tokens, visible_frontier
        self.id_protocol, self.prompt_version, self.task_prompt_version = id_protocol, prompt_version, task_prompt_version
        self.modality_policy = modality_policy

    def _messages(self, example, reader, state, evidence, trajectory, error=None):
        return full_directory_reader_messages(
            example, reader.render(example.query, example.observation, state), self._ordered_evidence(evidence),
            max_ids=self.max_ids, max_calls=self.max_calls, read_calls_used=trajectory.n_read_calls,
            interaction_history=full_directory_interaction_history(trajectory.decisions), error=error)

    def _empty_result(self, trajectory):
        return {"directory_protocol": READER_VERSION, "ok": False, "action_kind": "invalid",
                "requested_ids": [], "selected_ids": [], "invalid_ids": [], "invalid_id_reasons": {},
                "already_read_ids": [], "duplicate_ids": [], "new_ids": [], "records": [],
                "retrieval_violations": [], "no_new_evidence": True, "read_call_consumed": False,
                "read_calls_used": trajectory.n_read_calls,
                "read_calls_remaining": self.max_calls - trajectory.n_read_calls}

    def _episode(self, example, seed, policy_version):
        directories, sessions = self.store.hierarchical_bank(example.memory_id, self.version)
        reader = DirectoryReader(directories, sessions, count_tokens=self.count_tokens,
                                 retention="cumulative", context_blocks=0,
                                 modality_policy=self.modality_policy, visible_frontier=self.visible_frontier)
        state = reader.initial_state()
        available_parents = [pid for pid in reader._order if reader._allowed(pid)]
        if available_parents:
            expanded = reader.read(available_parents, state, 2 ** 62)
            if not expanded["ok"]:
                raise InfrastructureError(f"Full-directory preparation failed: {expanded['error']['code']}")
            state = expanded["state"]
        trajectory = Trajectory(example.sample_id, seed, example.memory_id, policy_version)
        evidence, error = {}, None
        for index in range(self.max_calls + 1):
            messages = self._check_capacity(example, reader, state, evidence, trajectory, error)
            completion = yield messages, seed + index
            decision = Decision(messages, completion, state_sha256=digest(messages))
            decision.tool_result = self._empty_result(trajectory)
            trajectory.decisions.append(decision)
            try:
                decision.action = parse_action(completion.text)
            except (ValueError, TypeError):
                decision.format_violations.append("invalid_json_or_schema")
                self._feedback(decision.tool_result, decision.format_violations)
                trajectory.termination_reason = "protocol_failure"
                break
            if decision.action["stop"]:
                decision.tool_result.update(action_kind="stop", requested_ids=list(decision.action["ids"]))
                if decision.action["ids"]:
                    # STOP has one format violation regardless of ID values or count.
                    decision.format_violations.append("stop_with_ids")
                self._feedback(decision.tool_result, decision.format_violations)
                state = reader.stop(state)["state"]
                trajectory.termination_reason = "stop"
                break
            state, evidence, error = self._read(example, reader, state, evidence, trajectory, decision)
            if not decision.tool_result["read_call_consumed"]:
                trajectory.termination_reason = "max_read_calls"
                break
        else:
            trajectory.termination_reason = "max_decisions"
        trajectory.evidence = self._ordered_evidence(evidence)
        return trajectory

    @staticmethod
    def _feedback(result, violations, code=None):
        if code is None and violations:
            codes = {"invalid_json_or_schema": "invalid_json_or_schema", "stop_with_ids": "stop_with_ids",
                     "empty_ids": "invalid_batch_size", "too_many_ids": "too_many_ids",
                     "duplicate_ids": "duplicate_ids_in_batch", "unknown_id": "unknown_or_unauthorized_id",
                     "repeated_read": "repeated_read", "read_budget_exceeded": "max_read_calls"}
            code = codes[violations[0]]
        result["ok"] = code is None
        result.pop("error", None)
        if code is None:
            return None
        fields = ("requested_ids", "invalid_ids", "invalid_id_reasons", "already_read_ids", "duplicate_ids",
                  "new_ids", "no_new_evidence", "read_calls_used", "read_calls_remaining", "read_call_consumed")
        error = {"code": code, **{key: result[key] for key in fields}, "format_violations": list(violations)}
        messages = {
            "invalid_json_or_schema": "Return exactly one JSON object containing a string-list ids and boolean stop.",
            "stop_with_ids": "STOP requires an empty ids list. No listed ID was read.",
            "invalid_batch_size": "READ requires at least one leaf ID. No IDs were read.",
            "too_many_ids": "The request exceeds max_ids_per_call. No IDs were read; submit an allowed-size request.",
            "duplicate_ids_in_batch": "Each leaf ID may appear only once per request. Valid new IDs were retained if capacity allowed.",
            "unknown_or_unauthorized_id": "Copy an exact leaf ID from child_directories. Parent IDs cannot be read.",
            "repeated_read": "The already_read_ids remain in retrieved_evidence. Reuse their original text.",
            "max_read_calls": "The READ budget is exhausted. No IDs were read and no additional call was charged.",
            "context_capacity_exceeded": "The complete next input exceeds capacity. No new evidence was retained or shortened.",
        }
        error["message"] = messages[code]
        result["error"] = error
        return error

    def _read(self, example, reader, state, evidence, trajectory, decision):
        requested = decision.action["ids"]
        unique = list(dict.fromkeys(requested))
        available = reader._visible_children(state)
        invalid = [key for key in unique if key not in available]
        already_read = [key for key in unique if key in evidence]
        seen, duplicates = set(), []
        for key in requested:
            if key in seen and key not in duplicates:
                duplicates.append(key)
            seen.add(key)
        violations = decision.format_violations
        if not requested:
            violations.append("empty_ids")
        if len(requested) > self.max_ids:
            violations.append("too_many_ids")
        if duplicates:
            violations.append("duplicate_ids")
        if invalid:
            violations.append("unknown_id")
        if already_read:
            violations.append("repeated_read")
        result = self._empty_result(trajectory)
        result.update(requested_ids=list(requested), invalid_ids=invalid, already_read_ids=already_read,
                      duplicate_ids=duplicates,
                      invalid_id_reasons={key: "nonleaf" if key in reader._parents else
                                          "unavailable" if key in reader._children else "unknown" for key in invalid},
                      action_kind="segment" if any(key in available for key in requested) else "invalid",
                      retrieval_violations=[label for label in violations
                                            if label in {"duplicate_ids", "unknown_id", "repeated_read"}])
        decision.tool_result = result
        if trajectory.n_read_calls >= self.max_calls:
            violations.append("read_budget_exceeded")
            result["retrieval_violations"].append("read_budget_exceeded")
            return state, evidence, self._feedback(result, violations, "max_read_calls")
        trajectory.n_read_calls += 1
        result.update(read_call_consumed=True, read_calls_used=trajectory.n_read_calls,
                      read_calls_remaining=self.max_calls - trajectory.n_read_calls)
        selected = [key for key in unique if key in available and key not in evidence]
        if not 1 <= len(requested) <= self.max_ids:
            selected = []
        result["selected_ids"] = selected
        candidate_state, candidate_evidence = state, evidence
        if selected:
            read = reader.read(selected, state, 2 ** 62)
            if not read["ok"]:
                raise InfrastructureError(f"Full-directory source read failed: {read['error']['code']}")
            candidate_state = read["state"]
            records = [self._segment_record(reader, key) for key in selected]
            candidate_evidence = {**evidence, **{row["id"]: row for row in records}}
            result.update(records=records, new_ids=list(selected), no_new_evidence=False)
        error = self._feedback(result, violations)
        try:
            self._check_capacity(example, reader, candidate_state, candidate_evidence, trajectory, error)
        except CapacityError:
            result.update(records=[], new_ids=[], no_new_evidence=True)
            error = self._feedback(result, violations, "context_capacity_exceeded")
            # A history-only capacity failure propagates; no prompt is truncated.
            self._check_capacity(example, reader, state, evidence, trajectory, error)
            return state, evidence, error
        return candidate_state, candidate_evidence, error
