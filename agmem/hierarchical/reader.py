"""Parent expansion and exact child evidence reads, independent of Reader RL."""

from copy import deepcopy

from ..core import canonical
from .binding import validate_directory
from .records import MODALITY_POLICIES, public_record


class DirectoryReader:
    """One immutable bank, with an explicit policy for retaining expanded lists.

    ``visible_frontier`` is the largest committed parent order visible to this
    task. Agent replay at subtask j must pass j-1. States are internal data;
    only ``render`` and read ``result`` are intended for model-visible use.
    """

    def __init__(self, directories, sessions, *, count_tokens, retention,
                 visible_frontier=None, context_blocks=1, modality_policy="existing_captions",
                 max_open_directories=2):
        if retention not in {"cumulative", "latest", "lru"}:
            raise ValueError("Choose retention=cumulative, latest or lru explicitly")
        if type(max_open_directories) is not int or max_open_directories < 1:
            raise ValueError("max_open_directories must be a positive integer")
        self.max_open_directories = max_open_directories
        if context_blocks not in (0, 1) or type(context_blocks) is not int:
            raise ValueError("context_blocks must be 0 or 1")
        if visible_frontier is not None and (type(visible_frontier) is not int or visible_frontier < 0):
            raise ValueError("visible_frontier must be a nonnegative parent order")
        if modality_policy not in MODALITY_POLICIES:
            raise ValueError("Unsupported modality_policy")
        self.retention, self.context_blocks = retention, context_blocks
        self.modality_policy = modality_policy
        self.count_tokens, self.frontier = count_tokens, visible_frontier
        source_values = list(sessions.values()) if isinstance(sessions, dict) else list(sessions)
        self._sessions = {s["parent_id"]: deepcopy(s) for s in source_values}
        directory_values = list(directories.values()) if isinstance(directories, dict) else list(directories)
        self._parents = {d["id"]: deepcopy(d) for d in directory_values}
        if (not self._parents or len(self._parents) != len(directory_values)
                or len(self._sessions) != len(source_values) or set(self._parents) != set(self._sessions)):
            raise ValueError("Provide unique directories and matching source sessions for one bank")
        self.bank_id = directory_values[0]["bank_id"]
        if len({d["order"] for d in self._parents.values()}) != len(self._parents):
            raise ValueError("Parent orders must be unique")
        seen_records = set()
        self._children = {}
        for pid, directory in self._parents.items():
            session = self._sessions[pid]
            validate_directory(directory, session, count_tokens=count_tokens)
            if directory["bank_id"] != self.bank_id:
                raise ValueError("A Reader cannot combine distinct banks")
            if directory["source_kind"] == "agent" and visible_frontier is None:
                raise ValueError("Agent replay requires an explicit committed-subtask frontier")
            ids = {r["record_id"] for r in session["header_records"] + session["records"]}
            if seen_records & ids:
                raise ValueError("Record IDs must be unique within a bank")
            seen_records.update(ids)
            for child in directory["children"]:
                if child["id"] in self._children or child["id"] in self._parents:
                    raise ValueError("Directory IDs must be unique within a bank")
                self._children[child["id"]] = child
        self._order = sorted(self._parents, key=lambda pid: self._parents[pid]["order"])
        self._versions = {pid: self._parents[pid]["directory_version"] for pid in self._order}

    def _allowed(self, pid):
        return self.frontier is None or self._parents[pid]["order"] <= self.frontier

    def initial_state(self):
        result = {"schema_version": "hierarchical-reader.v1", "bank_id": self.bank_id,
                "directory_versions": dict(self._versions), "retention": self.retention,
                "modality_policy": self.modality_policy, "context_blocks": self.context_blocks,
                "visible_frontier": self.frontier,
                "expanded_parent_ids": [], "display_parent_ids": [], "retrieved_child_ids": [],
                "returned_record_ids": [], "stopped": False}
        if self.retention == "lru":
            result["max_open_directories"] = self.max_open_directories
        return result

    def _child_sort(self, cid):
        child = self._children[cid]
        return self._parents[child["parent_id"]]["order"], child["order"]

    def _visible_children(self, state):
        current = {child["id"] for pid in state["display_parent_ids"] for child in self._parents[pid]["children"]}
        return current if self.retention == "lru" else current | set(state["retrieved_child_ids"])

    def _validate_state(self, state):
        if not isinstance(state, dict) or set(state) != set(self.initial_state()):
            raise ValueError("invalid_state")
        if (state["schema_version"] != "hierarchical-reader.v1" or state["bank_id"] != self.bank_id
                or state["directory_versions"] != self._versions or state["retention"] != self.retention
                or state["modality_policy"] != self.modality_policy
                or state["context_blocks"] != self.context_blocks or state["visible_frontier"] != self.frontier
                or type(state["stopped"]) is not bool):
            raise ValueError("state_bank_or_version_mismatch")
        for key in ("expanded_parent_ids", "display_parent_ids", "retrieved_child_ids", "returned_record_ids"):
            values = state[key]
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values) or len(set(values)) != len(values):
                raise ValueError("invalid_state")
        expanded, displayed = set(state["expanded_parent_ids"]), set(state["display_parent_ids"])
        if self.retention == "lru" and (state["max_open_directories"] != self.max_open_directories
                                        or len(displayed) > self.max_open_directories):
            raise ValueError("state_visibility_mismatch")
        if (not expanded.issubset(self._parents) or not displayed.issubset(expanded)
                or any(not self._allowed(pid) for pid in expanded)
                or self.retention == "cumulative" and displayed != expanded):
            raise ValueError("state_visibility_mismatch")
        if any(cid not in self._children or self._children[cid]["parent_id"] not in expanded
               for cid in state["retrieved_child_ids"]):
            raise ValueError("state_visibility_mismatch")
        expected_records = [r["record_id"] for r in self._evidence(state["retrieved_child_ids"])]
        if state["returned_record_ids"] != expected_records:
            raise ValueError("state_evidence_mismatch")

    @staticmethod
    def _parent_view(parent):
        return {key: deepcopy(parent[key]) for key in ("id", "time", "participants", "title", "content")}

    @staticmethod
    def _child_view(child):
        return {key: deepcopy(child[key]) for key in ("id", "title", "content", "entities")}

    def _evidence(self, child_ids):
        by_address = {}
        for cid in sorted(set(child_ids), key=self._child_sort):
            child = self._children[cid]
            pid = child["parent_id"]
            session = self._sessions[pid]
            by_id = {r["record_id"]: r for r in session["records"]}
            start, end = child["source"]["start_block"], child["source"]["end_block"]
            assignments = [(r, "header") for r in session["header_records"]]
            if self.context_blocks and start > 1:
                assignments += [(by_id[rid], "context") for rid in session["blocks"][start - 2]["record_ids"]]
            assignments += [(by_id[rid], "core") for block in session["blocks"][start - 1:end]
                            for rid in block["record_ids"]]
            for record, role in assignments:
                address = (pid, record["record_id"])
                if address not in by_address:
                    by_address[address] = {"parent_id": pid, "record_id": record["record_id"],
                                           **public_record(record, self.modality_policy), "introduced_by": []}
                attribution = {"node_id": cid, "role": role}
                if attribution not in by_address[address]["introduced_by"]:
                    by_address[address]["introduced_by"].append(attribution)
        order = {(pid, r["record_id"]): (self._parents[pid]["order"], r["seq"])
                 for pid in {a[0] for a in by_address} for r in
                 self._sessions[pid]["header_records"] + self._sessions[pid]["records"]}
        return [by_address[address] for address in sorted(by_address, key=order.__getitem__)]

    def render(self, query, observation, state):
        self._validate_state(state)
        return {"query": query, "observation": observation,
                "parents": [self._parent_view(self._parents[pid]) for pid in self._order if self._allowed(pid)],
                "child_directories": [
                    {"parent_id": pid, "children": [self._child_view(child) for child in self._parents[pid]["children"]]}
                    for pid in self._order if pid in state["display_parent_ids"]],
                "retrieved_children": [
                    {"parent_id": self._children[cid]["parent_id"], **self._child_view(self._children[cid])}
                    for cid in sorted(state["retrieved_child_ids"], key=self._child_sort)],
                "evidence": self._evidence(state["retrieved_child_ids"])}

    @staticmethod
    def _error(state, code, **details):
        return {"ok": False, "state": deepcopy(state), "error": {"code": code, **details}}

    def read(self, node_ids, state, token_budget, *, bank_id=None, directory_versions=None):
        """An atomic READ: all IDs are checked against the pre-action view."""
        try:
            self._validate_state(state)
        except (ValueError, KeyError, TypeError) as exc:
            return self._error(state, "invalid_state", detail=str(exc))
        if state["stopped"]:
            return self._error(state, "reader_stopped")
        if bank_id is not None and bank_id != self.bank_id:
            return self._error(state, "bank_mismatch")
        if directory_versions is not None and directory_versions != self._versions:
            return self._error(state, "directory_version_mismatch")
        if type(token_budget) is not int or token_budget < 0:
            return self._error(state, "invalid_token_budget")
        if (not isinstance(node_ids, list) or not node_ids or any(not isinstance(n, str) for n in node_ids)
                or len(node_ids) != len(set(node_ids))):
            return self._error(state, "invalid_node_ids")
        parent_ids, child_ids = [], []
        visible = self._visible_children(state)
        for nid in node_ids:
            if nid in self._parents:
                if not self._allowed(nid):
                    return self._error(state, "beyond_visible_frontier")
                parent_ids.append(nid)
            elif nid in self._children:
                if not self._allowed(self._children[nid]["parent_id"]):
                    return self._error(state, "beyond_visible_frontier")
                if nid not in visible:
                    return self._error(state, "child_not_visible")
                child_ids.append(nid)
            else:
                return self._error(state, "unknown_id")
        try:
            for pid in set(parent_ids) | {self._children[cid]["parent_id"] for cid in child_ids}:
                validate_directory(self._parents[pid], self._sessions[pid], count_tokens=self.count_tokens)
        except (ValueError, KeyError, TypeError) as exc:
            return self._error(state, "source_or_directory_integrity_error", detail=str(exc))
        parent_ids.sort(key=self._order.index)
        child_ids.sort(key=self._child_sort)
        navigation = {
            "expanded_directories": [{"parent_id": pid, "children": [self._child_view(c) for c in self._parents[pid]["children"]]}
                                     for pid in parent_ids],
            "selected_children": [{"parent_id": self._children[cid]["parent_id"], **self._child_view(self._children[cid])}
                                  for cid in child_ids],
        }
        records = self._evidence(child_ids)
        payload = {"action": "READ", **navigation, "records": records}
        cost = {"returned_tokens": self.count_tokens(canonical(payload)),
                "directory_tokens": self.count_tokens(canonical(navigation)),
                "record_tokens": self.count_tokens(canonical(records)) if records else 0}
        if cost["returned_tokens"] > token_budget:
            return self._error(state, "token_budget_exceeded", required_tokens=cost["returned_tokens"],
                               token_budget=token_budget, **cost)
        updated = deepcopy(state)
        updated["expanded_parent_ids"] = [pid for pid in self._order if pid in
                                          set(updated["expanded_parent_ids"]) | set(parent_ids)]
        if self.retention == "cumulative":
            updated["display_parent_ids"] = list(updated["expanded_parent_ids"])
        elif self.retention == "lru":
            # Least-recently used first. Successful navigation and segment reads
            # both touch their parent; evidence never participates in eviction.
            touched = parent_ids + [self._children[cid]["parent_id"] for cid in child_ids]
            for pid in touched:
                updated["display_parent_ids"] = [p for p in updated["display_parent_ids"] if p != pid] + [pid]
            updated["display_parent_ids"] = updated["display_parent_ids"][-self.max_open_directories:]
        elif parent_ids:
            updated["display_parent_ids"] = list(parent_ids)
        updated["retrieved_child_ids"] = sorted(set(updated["retrieved_child_ids"]) | set(child_ids), key=self._child_sort)
        updated["returned_record_ids"] = [r["record_id"] for r in self._evidence(updated["retrieved_child_ids"])]
        new_ids = [rid for rid in updated["returned_record_ids"] if rid not in set(state["returned_record_ids"])]
        # Only result is the model-visible tool payload; accounting/state stay
        # outside it, so returned_tokens measures exactly what the model reads.
        return {"ok": True, "state": updated, "result": payload, "new_record_ids": new_ids, "cost": cost}

    def stop(self, state):
        try:
            self._validate_state(state)
        except (ValueError, KeyError, TypeError) as exc:
            return self._error(state, "invalid_state", detail=str(exc))
        if state["stopped"]:
            return self._error(state, "reader_stopped")
        updated = deepcopy(state)
        updated["stopped"] = True
        return {"ok": True, "state": updated, "result": {"action": "STOP"}}
