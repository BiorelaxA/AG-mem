from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import random


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path: str | Path) -> str:
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def seeded_int(*parts: Any) -> int:
    return int(digest(parts)[:15], 16)


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def append_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(canonical(value) + "\n")


@dataclass
class Example:
    sample_id: str
    source: str
    split: str
    memory_id: str
    query: str
    observation: str = ""
    options: dict[str, str] = field(default_factory=dict)
    gold_answers: list[str] = field(default_factory=list)
    answer_mode: str = "text"
    # Scoring-only metadata; public_task deliberately excludes these fields.
    judge_metadata: dict = field(default_factory=dict)

    def public_task(self) -> dict:
        # Explicit allowlist: no caller should serialize the whole Example into a prompt.
        return {"query": self.query, "observation": self.observation, "options": self.options}


@dataclass
class Completion:
    text: str
    prompt_ids: list[int] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    old_logprobs: list[float] = field(default_factory=list)
    sampling_seed: int | None = None
    sampling_batch_size: int = 1
    truncated: bool = False


@dataclass
class Decision:
    messages: list[dict]
    completion: Completion
    action: dict | None = None
    tool_result: dict | None = None
    state_sha256: str = ""
    format_violations: list[str] = field(default_factory=list)


@dataclass
class Trajectory:
    sample_id: str
    seed: int
    memory_id: str
    policy_version: int = 0
    decisions: list[Decision] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    n_read_calls: int = 0
    termination_reason: str = ""
    prediction: str = ""
    outcome: float = 0.0
    reward: float = 0.0
    advantage: float = 0.0
    task_input_hash: str = ""
    read_cost: float = 0.0
    format_failure: bool = False
    format_reward: float = 0.0
    outcome_reward: float = 0.0
    raw_read_cost: float = 0.0
    judge_score: float | None = None
    answer_f1: float | None = None
    likelihood: dict = field(default_factory=dict)
    auxiliary_reward: float = 0.0
    # Frozen Task Agent completion metadata, distinct from Reader truncation.
    task_completion: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def trajectory_from_dict(value: dict) -> Trajectory:
    value = dict(value)
    decisions = []
    for item in value.pop("decisions"):
        item = dict(item)
        item["completion"] = Completion(**item["completion"])
        decisions.append(Decision(**item))
    return Trajectory(**value, decisions=decisions)


class InfrastructureError(RuntimeError):
    """A failed service call invalidates the whole training group, not its task score."""


class CapacityError(RuntimeError):
    pass


def load_config(path: str | Path) -> dict:
    import yaml
    def merge(left, right):
        result = dict(left)
        for key, item in right.items():
            result[key] = merge(result[key], item) if isinstance(item, dict) and isinstance(result.get(key), dict) else item
        return result
    def read(current, seen):
        current = Path(current).resolve()
        if current in seen:
            raise ValueError("Cyclic base configuration")
        value = yaml.safe_load(current.read_text())
        if not isinstance(value, dict):
            raise ValueError("Configuration must be a mapping")
        base = value.pop("extends", None)
        return merge(read(current.parent / base, seen | {current}), value) if base else value
    value = read(path, set())
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a mapping")
    from .execution import execution_settings
    execution_settings(value)
    if value.get("data", {}).get("layout", "combined") not in {"combined", "by_source"}:
        raise ValueError("data.layout must be combined or by_source")
    algorithm = value["training"].get("algorithm", "grpo")
    if algorithm not in {"grpo", "dapo"}:
        raise ValueError("training.algorithm must be grpo or dapo")
    if algorithm == "grpo" and value["training"]["group_size"] != 8:
        raise ValueError("This experiment fixes GRPO group_size=8")
    if value["training"]["parameterization"] != "full":
        raise ValueError("Reader must use full parameter training")
    if value["memory"]["record_unit"] != "session" or value["writer"]["lookback_sessions"] < 3:
        raise ValueError("Use one entry per session and at least three preceding sessions")
    max_ids = value["retrieval"]["max_ids_per_call"]
    if value["retrieval"].get("id_protocol", "strict_v1") not in {"strict_v1", "strict_v2", "repair_v2"}:
        raise ValueError("Unknown retrieval.id_protocol")
    if type(max_ids) is not int or max_ids < 1:
        raise ValueError("retrieval.max_ids_per_call must be a positive integer")
    if value["reward"]["read_call_cost"] != 0.01:
        raise ValueError("Experiment fixes read_call_cost=0.01")
    penalty = value["reward"].get("format_penalty", 0.0)
    if value["reward"].get("formula", "outcome_cost_v1") not in {"outcome_cost_v1", "outcome_gated_v2", "judge_gated_v1", "judge_cost_v2"}:
        raise ValueError("Unknown reward.formula")
    if value["reward"].get("formula") == "judge_cost_v2" and value["retrieval"].get("id_protocol") != "strict_v2":
        raise ValueError("judge_cost_v2 requires retrieval.id_protocol=strict_v2 for repeat-read penalties")
    if isinstance(penalty, bool) or not isinstance(penalty, (int, float)) or not math.isfinite(penalty) or penalty < 0:
        raise ValueError("reward.format_penalty must be finite and nonnegative")
    if algorithm == "dapo":
        from .dapo_sampling import validate_dapo_config
        validate_dapo_config(value)
    return value
