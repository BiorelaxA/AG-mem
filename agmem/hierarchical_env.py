"""Opt-in ORM environment over immutable parent/child directory banks."""

from copy import deepcopy

from .core import CapacityError, Decision, InfrastructureError, Trajectory, digest
from .env import MemoryEnvironment, parse_action
from .hierarchical.reader import DirectoryReader
from .hierarchical.records import render_records
from .hierarchical_prompts import (READER_VERSION, TASK_VERSION, compact_interaction_history,
                                   hierarchical_reader_messages, hierarchical_reader_system,
                                   hierarchical_task_messages)


class HierarchicalMemoryEnvironment(MemoryEnvironment):
    """Reuse rollout scheduling, but keep hierarchical execution opt-in.

    Store.hierarchical_bank(memory_id, version) returns (directories, sessions).
    count_tokens must use the directory Writer tokenizer for integrity checks;
    capacity separately checks complete Reader and Task inputs with their own
    tokenizers. Each rollout owns its state and its permanent evidence dictionary.
    """

    def __init__(self, store, directory_version, max_ids=1, max_calls=10, capacity=None,
                 id_protocol="strict_v2", *, count_tokens=None, max_open_directories=2,
                 prompt_version=READER_VERSION, task_prompt_version=TASK_VERSION,
                 modality_policy="existing_captions"):
        hierarchical_reader_system(max_ids, max_calls, max_open_directories)
        if id_protocol != "strict_v2" or prompt_version != READER_VERSION or task_prompt_version != TASK_VERSION:
            raise ValueError("Hierarchical ORM requires strict_v2, hierarchical_v1 and hierarchical_evidence_v1")
        count_tokens = count_tokens or getattr(store, "hierarchical_count_tokens", None)
        if not callable(count_tokens):
            raise ValueError("Provide the actual directory tokenizer count_tokens callback")
        self.store, self.version = store, directory_version
        self.max_ids, self.max_calls, self.capacity = max_ids, max_calls, capacity
        self.count_tokens, self.max_open_directories = count_tokens, max_open_directories
        self.id_protocol, self.prompt_version, self.task_prompt_version = id_protocol, prompt_version, task_prompt_version
        self.modality_policy = modality_policy

    @staticmethod
    def _ordered_evidence(evidence):
        return sorted(evidence.values(), key=lambda row: (row["order"], row["segment_order"], row["id"]))

    def _messages(self, example, reader, state, evidence, trajectory, error=None):
        view = reader.render(example.query, example.observation, state)
        view["open_parent_ids_lru"] = list(state["display_parent_ids"])
        return hierarchical_reader_messages(example, view, self._ordered_evidence(evidence),
            max_ids=self.max_ids, max_calls=self.max_calls, max_open_directories=self.max_open_directories,
            read_calls_used=trajectory.n_read_calls,
            interaction_history=compact_interaction_history(trajectory.decisions), error=error)

    def _check_capacity(self, example, reader, state, evidence, trajectory, error=None):
        messages = self._messages(example, reader, state, evidence, trajectory, error)
        if self.capacity:
            self.capacity(messages, hierarchical_task_messages(example, self._ordered_evidence(evidence)))
        return messages

    def _episode(self, example, seed, policy_version):
        directories, sessions = self.store.hierarchical_bank(example.memory_id, self.version)
        reader = DirectoryReader(directories, sessions, count_tokens=self.count_tokens,
                                 retention="lru", max_open_directories=self.max_open_directories,
                                 context_blocks=0, modality_policy=self.modality_policy)
        trajectory = Trajectory(example.sample_id, seed, example.memory_id, policy_version)
        state, evidence, error = reader.initial_state(), {}, None
        for index in range(self.max_calls + 1):
            messages = self._check_capacity(example, reader, state, evidence, trajectory, error)
            completion = yield messages, seed + index
            decision = Decision(messages, completion, state_sha256=digest(messages))
            # Identify even an immediate STOP or malformed action as belonging
            # to this protocol, so zero-READ episodes retain navigation metrics.
            decision.tool_result = {"directory_protocol": "hierarchical_v1", "action_kind": "invalid",
                                    "records": [], "new_ids": [], "read_call_consumed": False,
                                    "read_calls_used": trajectory.n_read_calls,
                                    "read_calls_remaining": self.max_calls - trajectory.n_read_calls}
            trajectory.decisions.append(decision)
            try:
                decision.action = parse_action(completion.text)
            except (ValueError, TypeError):
                decision.format_violations.append("invalid_json_or_schema")
                trajectory.termination_reason = "protocol_failure"
                break
            if decision.action["stop"]:
                decision.tool_result["action_kind"] = "stop"
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

    def _segment_record(self, reader, child_id):
        child = reader._children[child_id]
        session = reader._sessions[child["parent_id"]]
        source = child["source"]
        by_id = {row["record_id"]: row for row in session["records"]}
        records = [by_id[rid] for block in session["blocks"][source["start_block"] - 1:source["end_block"]]
                   for rid in block["record_ids"]]
        # No preceding Block, session header, parent summary or unread sibling
        # summary can enter a selected segment's Task evidence.
        return {"id": child_id, "parent_id": child["parent_id"], "source_session_id": child["parent_id"],
                "time": session["time"]["raw"] or session["time"]["iso"] or "Unknown",
                "order": session["order"] - 1, "segment_order": child["order"],
                "title": child["title"], "content": child["content"], "entities": deepcopy(child["entities"]),
                "raw_text": render_records(records, self.modality_policy),
                "source_record_ids": [row["record_id"] for row in records],
                "source_blocks": [source["start_block"], source["end_block"]]}

    @staticmethod
    def _feedback(result, violations, code=None):
        if code is None and violations:
            codes = {"empty_ids": "invalid_batch_size", "too_many_ids": "too_many_ids",
                     "duplicate_ids": "duplicate_ids_in_batch", "unknown_id": "unknown_or_unauthorized_id",
                     "repeated_read": "repeated_read", "visible_navigation": "visible_navigation"}
            code = codes[violations[0]]
        result["ok"] = code is None
        result.pop("error", None)
        if code is None:
            return None
        error = {"code": code, "requested_ids": list(result["requested_ids"]),
                 "invalid_ids": list(result["invalid_ids"]), "already_read_ids": list(result["already_read_ids"]),
                 "visible_navigation_ids": list(result["visible_navigation_ids"]),
                 "new_ids": list(result["new_ids"]), "read_calls_used": result["read_calls_used"],
                 "read_calls_remaining": result["read_calls_remaining"],
                 "read_call_consumed": result["read_call_consumed"], "format_violations": list(violations)}
        if result["already_read_ids"]:
            error["message"] = "These child IDs are already in retained evidence. Reuse their original text; do not read them again."
        elif result["invalid_ids"]:
            error["message"] = "Use an exact parent ID or a child ID from an open directory. Reopen the parent before reading hidden children."
        elif code == "visible_navigation":
            error["message"] = ("This child directory is already provided in the current prompt. Requesting "
                                "an already open directory incurs a format penalty each time. Use its child IDs, "
                                "open a directory that is not currently displayed, or STOP.")
        elif code == "context_capacity_exceeded":
            error["message"] = "The full next input would exceed capacity. No new directory or evidence was retained."
        result["error"] = error
        return error

    def _read(self, example, reader, state, evidence, trajectory, decision):
        requested = decision.action["ids"]
        unique = list(dict.fromkeys(requested))
        parents = {pid for pid in reader._parents if reader._allowed(pid)}
        visible_children = reader._visible_children(state)
        already_read = [key for key in unique if key in evidence]
        # Inspect the pre-action view, including unexecuted excess IDs. Parent
        # entries are always visible; only their currently open child lists count.
        visible_navigation = [key for key in unique if key in state["display_parent_ids"]]
        # Already-read children remain recognizable for repeat penalties even
        # when their parent list is evicted; they never execute another read.
        valid_ids = parents | visible_children | set(evidence)
        invalid = [key for key in unique if key not in valid_ids]
        violations = decision.format_violations
        if not requested:
            violations.append("empty_ids")
        if len(requested) > self.max_ids:
            violations.append("too_many_ids")
        if len(unique) != len(requested):
            violations.append("duplicate_ids")
        if invalid:
            violations.append("unknown_id")
        if already_read:
            violations.append("repeated_read")
        if visible_navigation:
            violations.append("visible_navigation")
        first = requested[0] if requested else None
        action_kind = "navigation" if first in reader._parents else "segment" if first in reader._children else "invalid"
        selected = [first] if first in valid_ids else []
        result = {"directory_protocol": "hierarchical_v1", "ok": False,
                  "action_kind": action_kind, "requested_ids": list(requested),
                  "selected_ids": selected, "records": [], "invalid_ids": invalid,
                  "already_read_ids": already_read, "new_ids": [], "no_new_evidence": True,
                  "read_call_consumed": False, "directory_navigation_success": False,
                  "repeated_navigation": action_kind == "navigation" and first in state["expanded_parent_ids"],
                  "visible_navigation_ids": visible_navigation,
                  "visible_navigation_noop": action_kind == "navigation" and first in state["display_parent_ids"],
                  "opened_parent_ids": [], "evicted_parent_ids": [],
                  "open_parent_ids_lru": list(state["display_parent_ids"])}
        decision.tool_result = result
        if len(requested) > self.max_ids:
            result["ids_truncated"] = True
        if trajectory.n_read_calls >= self.max_calls:
            violations.append("read_budget_exceeded")
            result.update(read_calls_used=trajectory.n_read_calls, read_calls_remaining=0)
            return state, evidence, self._feedback(result, violations, "max_read_calls")
        trajectory.n_read_calls += 1
        result.update(read_call_consumed=True, read_calls_used=trajectory.n_read_calls,
                      read_calls_remaining=self.max_calls - trajectory.n_read_calls)
        candidate_state, candidate_evidence = state, evidence
        if selected and first not in evidence:
            # This large per-tool ceiling does not truncate or authorize an
            # oversized model prompt: the complete next Reader/Task states are
            # checked below before either mutation is committed.
            read = reader.read(selected, state, 2 ** 62)
            if not read["ok"]:
                raise InfrastructureError(f"Hierarchical source read failed: {read['error']['code']}")
            candidate_state = read["state"]
            result["open_parent_ids_lru"] = list(candidate_state["display_parent_ids"])
            if action_kind == "navigation":
                result["directory_navigation_success"] = True
                result["opened_parent_ids"] = [first]
                result["evicted_parent_ids"] = [pid for pid in state["display_parent_ids"]
                                                 if pid not in candidate_state["display_parent_ids"]]
            else:
                record = self._segment_record(reader, first)
                candidate_evidence = {**evidence, first: record}
                result.update(records=[record], new_ids=[first], no_new_evidence=False)
        error = self._feedback(result, violations)
        try:
            self._check_capacity(example, reader, candidate_state, candidate_evidence, trajectory, error)
        except CapacityError:
            result.update(records=[], new_ids=[], no_new_evidence=True, directory_navigation_success=False,
                          opened_parent_ids=[], evicted_parent_ids=[],
                          open_parent_ids_lru=list(state["display_parent_ids"]))
            error = self._feedback(result, violations, "context_capacity_exceeded")
            # A failure-state history must also fit. If it cannot, propagate
            # CapacityError rather than scoring a truncated or hidden rollout.
            self._check_capacity(example, reader, state, evidence, trajectory, error)
            return state, evidence, error
        return candidate_state, candidate_evidence, error
