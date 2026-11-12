from __future__ import annotations

import json
import re
from copy import deepcopy

from .core import CapacityError, Decision, Trajectory, digest
from .prompts import (reader_messages, task_messages, validate_reader_prompt_version,
                      validate_task_prompt_version)


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
    # A stop action never executes a read. Preserve its IDs for the rollout
    # audit, but ignore their contents when deciding whether to stop.
    return value


class MemoryEnvironment:
    def __init__(self, store, directory_version: str, max_ids=4, max_calls=8, capacity=None,
                 id_protocol="strict_v1", *, prompt_version="legacy_v1",
                 task_prompt_version="legacy_v1"):
        if type(max_ids) is not int or max_ids < 1:
            raise ValueError("max_ids must be a positive integer")
        self.store, self.version = store, directory_version
        self.max_ids, self.max_calls, self.capacity = max_ids, max_calls, capacity
        if id_protocol not in {"strict_v1", "repair_v2", "strict_v2"}:
            raise ValueError("Unknown retrieval.id_protocol")
        self.id_protocol = id_protocol
        self.prompt_version = validate_reader_prompt_version(prompt_version)
        self.task_prompt_version = validate_task_prompt_version(task_prompt_version)
        if self.prompt_version in {"interaction_history_v4", "interaction_history_v5"} and self.id_protocol != "strict_v2":
            raise ValueError(f"{self.prompt_version} requires the strict_v2 retrieval protocol")

    def _reader_messages(self, example, directory, evidence, trajectory, error=None):
        settings = {}
        if self.prompt_version in {"interaction_history_v4", "interaction_history_v5"}:
            # At a generation boundary these are only completed prior steps.
            # At capacity preflight the caller first finalizes the current tool
            # result, making this exactly the proposed next Reader state.
            settings["interaction_history"] = [
                {"reader_output": step.completion.text, "action": deepcopy(step.action),
                 "tool_result": deepcopy(step.tool_result)} for step in trajectory.decisions]
        return reader_messages(example, directory, evidence, error, max_ids=self.max_ids,
                               id_protocol=self.id_protocol, prompt_version=self.prompt_version,
                               max_calls=self.max_calls, read_calls_used=trajectory.n_read_calls,
                               **settings)

    def run(self, example, policy, seed: int, policy_version=0):
        episode = self.start_episode(example, seed, policy_version)
        messages, decision_seed = next(episode)
        while True:
            try:
                completion = policy.complete(messages, seed=decision_seed)
            except CapacityError as exc:
                raise CapacityError(f"{example.sample_id}, decision {decision_seed - seed}: {exc}") from exc
            try:
                messages, decision_seed = episode.send(completion)
            except StopIteration as done:
                return done.value

    def run_batch(self, example, policy, seeds, policy_version=0, max_batch_tokens=16384):
        """Advance independent retrieval states together; tools remain ordered per rollout."""
        episodes = [self.start_episode(example, seed, policy_version) for seed in seeds]
        active = {index: next(episode) for index, episode in enumerate(episodes)}
        results = [None] * len(seeds)
        while active:
            indices = list(active)
            messages = [active[index][0] for index in indices]
            decision_seeds = [active[index][1] for index in indices]
            try:
                completions = policy.complete_batch(messages, decision_seeds, max_batch_tokens=max_batch_tokens)
            except CapacityError as exc:
                raise CapacityError(f"{example.sample_id}, batched retrieval: {exc}") from exc
            if len(completions) != len(indices):
                raise ValueError("Batched Reader returned the wrong number of completions")
            for index, completion in zip(indices, completions):
                try:
                    active[index] = episodes[index].send(completion)
                except StopIteration as done:
                    results[index] = done.value
                    del active[index]
        return results

    def start_episode(self, example, seed: int, policy_version=0):
        """Create an independent retrieval state, advanced only by its caller.

        The generator yields (messages, decision_seed) and accepts a Completion;
        its return value is the finished Trajectory. Store/cache access stays on
        the thread advancing the generator, including cross-query batching.
        """
        return self._episode(example, seed, policy_version)

    def _episode(self, example, seed, policy_version):
        directory = self.store.directory(example.memory_id, self.version)
        entries = {row["id"]: row for row in directory}
        references = {row["id"]: row for row in self.store.memory(example.memory_id)}
        if len(entries) != len(directory) or entries.keys() != references.keys():
            raise ValueError("Directory does not align with the immutable memory snapshot")
        trajectory = Trajectory(example.sample_id, seed, example.memory_id, policy_version)
        evidence, error = {}, None
        previous_invalid = None
        for decision_index in range(self.max_calls + 1):
            messages = self._reader_messages(example, directory, list(evidence.values()), trajectory, error)
            completion = yield messages, seed + decision_index
            decision = Decision(messages, completion, state_sha256=digest(messages))
            trajectory.decisions.append(decision)
            try:
                action = parse_action(completion.text)
                decision.action = action
            except (ValueError, TypeError):
                decision.format_violations.append("invalid_json_or_schema")
                trajectory.termination_reason = "protocol_failure"
                break
            if action["stop"]:
                trajectory.termination_reason = "stop"
                break
            if self.id_protocol == "strict_v2":
                error = self._strict_read(example, directory, entries, references,
                                          evidence, trajectory, decision)
                if not decision.tool_result["read_call_consumed"]:
                    trajectory.termination_reason = "max_read_calls"
                    break
                continue
            if self.id_protocol == "repair_v2":
                requested = action["ids"]
                if not requested:
                    decision.format_violations.append("empty_ids")
                if len(requested) > self.max_ids:
                    decision.format_violations.append("too_many_ids")
                if len(set(requested)) != len(requested):
                    decision.format_violations.append("duplicate_ids")
                ids, repairs, invalid = resolve_ids(requested[:self.max_ids], references)
                if repairs:
                    decision.format_violations.append("repaired_id")
                if invalid:
                    decision.format_violations.append("unknown_id")
                if len(ids) != len(set(ids)) and "duplicate_ids" not in decision.format_violations:
                    decision.format_violations.append("duplicate_ids")
                # Inspect even the final over-budget decision, but never execute it.
                if trajectory.n_read_calls >= self.max_calls:
                    trajectory.termination_reason = "max_read_calls"
                    break
                trajectory.n_read_calls += 1
                ids = list(dict.fromkeys(ids))
                records = []
                for key in ids:
                    original = self.store.record(references[key])
                    records.append({k: original[k] for k in ("id", "source_session_id", "time", "order", "raw_text")} | {"content": entries[key]["content"]})
                before = set(evidence)
                capacity_error = False
                if records:
                    candidate = {**evidence, **{row["id"]: row for row in records}}
                    try:
                        if self.capacity:
                            self.capacity(reader_messages(example, directory, list(candidate.values()),
                                                          max_ids=self.max_ids, id_protocol=self.id_protocol,
                                                          prompt_version=self.prompt_version, max_calls=self.max_calls,
                                                          read_calls_used=trajectory.n_read_calls),
                                          task_messages(example, list(candidate.values()), self.task_prompt_version))
                    except CapacityError:
                        capacity_error = True
                        records = []
                    else:
                        evidence = candidate
                error = None
                if invalid:
                    error = {"code": "unknown_or_unauthorized_id", "invalid_ids": invalid,
                             "message": "Copy valid IDs exactly from the current directory; keep all leading zeros and hashes."}
                elif not requested:
                    error = {"code": "invalid_batch_size"}
                elif capacity_error:
                    error = {"code": "context_capacity_exceeded"}
                if error is not None and capacity_error:
                    error["capacity_rejected"] = True
                decision.tool_result = {"ok": error is None, "requested_ids": requested,
                                        "selected_ids": ids, "records": records,
                                        "id_repairs": repairs, "invalid_ids": invalid}
                if error is not None:
                    decision.tool_result["error"] = error
                if len(requested) > self.max_ids:
                    decision.tool_result["ids_truncated"] = True
                signature = tuple(sorted(set(invalid))) if invalid else None
                if signature is not None and signature == previous_invalid and set(evidence) == before:
                    decision.format_violations.append("repeated_invalid_action")
                    trajectory.termination_reason = "repeated_invalid_action"
                    break
                previous_invalid = signature
                continue
            if trajectory.n_read_calls >= self.max_calls:
                trajectory.termination_reason = "max_read_calls"
                break
            trajectory.n_read_calls += 1
            # Preserve the sampled action for GRPO and format scoring. Execute
            # only its ordered prefix; ignored IDs are never fetched or checked.
            requested_ids = action["ids"]
            ids = requested_ids[:self.max_ids]
            code = None
            if not ids:
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
                        self.capacity(reader_messages(example, directory, list(candidate.values()),
                                                      max_ids=self.max_ids, id_protocol=self.id_protocol,
                                                      prompt_version=self.prompt_version, max_calls=self.max_calls,
                                                      read_calls_used=trajectory.n_read_calls),
                                      task_messages(example, list(candidate.values()), self.task_prompt_version))
                except CapacityError:
                    code = "context_capacity_exceeded"
            if code is not None:
                error = {"code": code, "requested_ids": ids}
                decision.tool_result = {"ok": False, "requested_ids": requested_ids, "records": [], "error": error}
            else:
                evidence.update({row["id"]: row for row in records})
                decision.tool_result = {"ok": True, "requested_ids": requested_ids, "records": records}
                error = None
            if len(requested_ids) > self.max_ids:
                decision.tool_result.update(ids_truncated=True, selected_ids=ids)
        else:
            trajectory.termination_reason = "max_decisions"
        trajectory.evidence = list(evidence.values())
        return trajectory

    def _strict_read(self, example, directory, entries, references, evidence, trajectory, decision):
        """Inspect the full action; fetch only unread, exact IDs in its allowed prefix.

        Evidence is also the successful-read history for this rollout. A failed
        capacity check never adds to that history, and prior records are never
        fetched again. Historical protocols above deliberately keep their old
        execution and reward semantics.
        """
        requested = decision.action["ids"]
        unique = list(dict.fromkeys(requested))
        invalid = [key for key in unique if key not in references]
        already_read = [key for key in unique if key in evidence]
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
        selected = list(dict.fromkeys(key for key in requested[:self.max_ids] if key in references))
        result = {"ok": False, "requested_ids": requested, "selected_ids": selected,
                  "records": [], "invalid_ids": invalid, "already_read_ids": already_read,
                  "new_ids": [], "no_new_evidence": True, "read_call_consumed": False}
        decision.tool_result = result
        if len(requested) > self.max_ids:
            result["ids_truncated"] = True

        code = None
        staged_evidence = None
        if trajectory.n_read_calls >= self.max_calls:
            violations.append("read_budget_exceeded")
            code = "max_read_calls"
        else:
            # Empty, invalid and entirely repeated requests still consume one call.
            trajectory.n_read_calls += 1
            result["read_call_consumed"] = True
            records = []
            for key in selected:
                if key in evidence:
                    continue
                original = self.store.record(references[key])
                records.append({k: original[k] for k in ("id", "source_session_id", "time", "order", "raw_text")}
                               | {"content": entries[key]["content"]})
            if records:
                candidate = {**evidence, **{row["id"]: row for row in records}}
                if self.prompt_version in {"interaction_history_v4", "interaction_history_v5"}:
                    # Complete the tool feedback before checking capacity: the
                    # next input includes this action/result as well as evidence.
                    staged_evidence = candidate
                    result["records"] = records
                    result["new_ids"] = [row["id"] for row in records]
                    result["no_new_evidence"] = False
                else:
                    try:
                        if self.capacity:
                            self.capacity(reader_messages(example, directory, list(candidate.values()),
                                                          max_ids=self.max_ids, id_protocol=self.id_protocol,
                                                          prompt_version=self.prompt_version, max_calls=self.max_calls,
                                                          read_calls_used=trajectory.n_read_calls),
                                          task_messages(example, list(candidate.values()), self.task_prompt_version))
                    except CapacityError:
                        code = "context_capacity_exceeded"
                    else:
                        evidence.update({row["id"]: row for row in records})
                        result["records"] = records
                        result["new_ids"] = [row["id"] for row in records]
                        result["no_new_evidence"] = False

        if code is None and violations:
            codes = {"empty_ids": "invalid_batch_size", "too_many_ids": "too_many_ids",
                     "duplicate_ids": "duplicate_ids_in_batch", "unknown_id": "unknown_or_unauthorized_id",
                     "repeated_read": "repeated_read"}
            code = codes[violations[0]]
        result["read_calls_used"] = trajectory.n_read_calls
        result["read_calls_remaining"] = max(0, self.max_calls - trajectory.n_read_calls)
        error = self._strict_feedback(result, violations, code)
        if self.prompt_version in {"interaction_history_v4", "interaction_history_v5"} and result["read_call_consumed"]:
            candidate = staged_evidence if staged_evidence is not None else evidence
            try:
                if self.capacity:
                    self.capacity(self._reader_messages(example, directory, list(candidate.values()),
                                                        trajectory, error),
                                  task_messages(example, list(candidate.values()), self.task_prompt_version))
            except CapacityError:
                # Roll back only the proposed evidence; the attempted READ and
                # its capacity failure remain visible in the interaction history.
                result.update(records=[], new_ids=[], no_new_evidence=True)
                error = self._strict_feedback(result, violations, "context_capacity_exceeded")
                self.capacity(self._reader_messages(example, directory, list(evidence.values()),
                                                    trajectory, error),
                              task_messages(example, list(evidence.values()), self.task_prompt_version))
                # If even the failure-state history cannot fit, propagate the
                # infrastructure error instead of scoring a truncated rollout.
            else:
                if staged_evidence is not None:
                    evidence.update(staged_evidence)
        return error

    @staticmethod
    def _strict_feedback(result, violations, code):
        result["ok"] = code is None
        result.pop("error", None)
        if code is None:
            return None
        # This feedback changes the next Reader state even when the attempted
        # read adds no records. Partial successes and all violations stay auditable.
        error = {"code": code, "requested_ids": result["requested_ids"], "invalid_ids": result["invalid_ids"],
                 "already_read_ids": result["already_read_ids"], "new_ids": result["new_ids"],
                 "no_new_evidence": result["no_new_evidence"],
                 "read_call_consumed": result["read_call_consumed"],
                 "read_calls_used": result["read_calls_used"],
                 "read_calls_remaining": result["read_calls_remaining"],
                 "format_violations": list(violations)}
        if result["already_read_ids"]:
            error["message"] = ("The already_read_ids are already in retrieved_evidence; reuse that evidence. "
                                "Request unread directory IDs only when needed, or stop.")
        elif result["invalid_ids"]:
            error["message"] = "Copy IDs exactly from the current directory, including leading zeros and hashes."
        result["error"] = error
        return error


def resolve_ids(ids, references):
    """Only normalize omitted leading zeros; never guess hashes or cross histories."""
    resolved, repairs, invalid = [], [], []
    for key in ids:
        if key in references:
            resolved.append(key)
            continue
        match = re.fullmatch(r"r_([0-9]{1,5})_([0-9a-f]{12})", key)
        candidates = []
        if match:
            number, suffix = match.groups()
            # Require a unique directory match, not just a constructed spelling.
            candidates = [known for known in references
                          if (m := re.fullmatch(r"r_([0-9]+)_([0-9a-f]{12})", known))
                          and int(m[1]) == int(number) and m[2] == suffix]
            canonical = f"r_{int(number):05d}_{suffix}"
            if candidates != [canonical] or len(number) >= 5:
                candidates = []
        if len(candidates) == 1:
            resolved.append(candidates[0])
            repairs.append({"requested_id": key, "resolved_id": candidates[0]})
        else:
            invalid.append(key)
    return resolved, repairs, invalid
