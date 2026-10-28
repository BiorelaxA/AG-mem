"""Frozen binary answer judges, separate from task generation and Reader reward."""
from __future__ import annotations

import hashlib
import json
import re

from .core import InfrastructureError, digest
from .judge_prompts import LOCOMO_REFINED, LONGMEMEVAL_TEMPLATES, UPSTREAM
from .services import ChatService


PARSER_VERSION = "strict_binary_v1"
DEFAULT_PROTOCOLS = {
    "locomo": "locomo_refined",
    "split_longmemeval_s": "longmemeval_official",
    "split_longmemeval_m": "longmemeval_official",
}


def parse_refined(text: str) -> float:
    """Accept one JSON object (optionally fenced), never substring matching."""
    if not isinstance(text, str):
        raise ValueError("Judge output must be text")
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if fenced:
        text = fenced.group(1).strip()
    # Duplicate labels are ambiguous even when a JSON parser would keep the last.
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate Judge JSON key")
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=unique_keys)
    if not isinstance(value, dict) or set(value) != {"label"} or value["label"] not in ("CORRECT", "WRONG"):
        raise ValueError("Refined Judge must return exactly {label: CORRECT|WRONG}")
    return float(value["label"] == "CORRECT")


def parse_longmemeval(text: str) -> float:
    if not isinstance(text, str) or text.strip().lower() not in {"yes", "no"}:
        raise ValueError("LongMemEval Judge must return exactly yes or no")
    return float(text.strip().lower() == "yes")


def prompt_for(example, reference: str, prediction: str, protocol: str):
    """Use upstream question/gold/prediction inputs; no retrieved memory or actions."""
    if protocol == "locomo_refined":
        return LOCOMO_REFINED.format(question=example.query, gold_answer=reference,
                                     generated_answer=prediction), parse_refined
    if protocol == "longmemeval_official":
        metadata = getattr(example, "judge_metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("LongMemEval Judge metadata must be a mapping")
        question_id, question_type = metadata.get("question_id"), metadata.get("question_type")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("LongMemEval Judge requires original question_id (including _abs)")
        if question_type not in LONGMEMEVAL_TEMPLATES or question_type == "abstention":
            raise ValueError(f"Unsupported LongMemEval question_type: {question_type!r}")
        branch = "abstention" if "_abs" in question_id else question_type
        return LONGMEMEVAL_TEMPLATES[branch].format(example.query, reference, prediction), parse_longmemeval
    raise ValueError(f"Unknown Judge protocol: {protocol!r}")


class JudgeScorer:
    def __init__(self, service_config, store, judge_config=None):
        self.config = dict(judge_config or {})
        self.protocols = {**DEFAULT_PROTOCOLS, **self.config.get("protocols", {})}
        if any(value not in UPSTREAM for value in self.protocols.values()):
            raise ValueError("Judge protocols must be locomo_refined or longmemeval_official")
        self.max_tokens = self.config.get("max_output_tokens", 128)
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("Judge max_output_tokens must be a positive integer")
        self.seed = self.config.get("seed", 42)
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Judge seed must be a nonnegative integer")
        self.service = ChatService(service_config, store, "judge")
        self._provenance = {
            "model": dict(self.service.identity),
            "protocols": dict(self.protocols), "upstream": UPSTREAM,
            "template_sha256": {
                "locomo_refined": hashlib.sha256(LOCOMO_REFINED.encode()).hexdigest(),
                "longmemeval_official": {name: hashlib.sha256(text.encode()).hexdigest()
                                         for name, text in LONGMEMEVAL_TEMPLATES.items()},
            },
            "parser": PARSER_VERSION, "gold_aggregation": "any_complete_reference_v1",
            "input": "upstream_question_gold_prediction_v1",
            "temperature": 0.0, "enable_thinking": False, "seed": self.seed,
            "max_output_tokens": self.max_tokens,
        }
        self.cache_scope = "judge/" + digest(self._provenance)

    def provenance(self):
        return json.loads(json.dumps(self._provenance))

    def validate_example(self, example) -> None:
        """Validate routing, references and private metadata without service I/O."""
        references = example.gold_answers
        if (not isinstance(references, list) or not references
                or any(not isinstance(item, str) or not item.strip() for item in references)):
            raise ValueError("Judge requires nonempty complete reference answers")
        protocol = self.protocols.get(example.source)
        if protocol is None:
            raise ValueError(f"No Judge protocol configured for source: {example.source!r}")
        prompt_for(example, references[0], "", protocol)

    def score(self, example, prediction: str) -> float:
        self.validate_example(example)
        if not isinstance(prediction, str):
            raise ValueError("Judge prediction must be text")
        references = example.gold_answers
        protocol = self.protocols[example.source]
        # Alternatives are whole acceptable answers. Never split a list of facts
        # inside one reference into individually sufficient answers.
        for reference in dict.fromkeys(references):
            prompt, parse = prompt_for(example, reference, prediction, protocol)
            text = self.service.complete([{"role": "user", "content": prompt}], self.max_tokens,
                                         seed=self.seed, cache_scope=self.cache_scope, validator=parse)
            try:
                score = parse(text)
            except (ValueError, TypeError) as exc:
                # Defensive for externally injected/legacy caches; never turn a
                # malformed service response into an incorrect-answer reward.
                raise InfrastructureError("Invalid Judge label; no task score assigned") from exc
            if score == 1.0:
                return score
        return 0.0

    def close(self):
        self.service.client.close()
