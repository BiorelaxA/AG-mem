"""Contiguous agglomerative segmentation over immutable source Blocks.

Only adjacent spans may merge, but their score uses *all* cross-span pairs.
This is deliberately separate from sparse-graph linkage implementations whose
distance updates may retain only distances on the original connectivity edges.
"""

from copy import deepcopy
import math

import numpy as np

from .embedding import normalized_vector
from .records import MODALITY_POLICIES, public_record, validate_session


SEGMENTATION_VERSION = "adjacent-average.v1"
DEFAULTS = {
    # Engineering starting values, not thresholds calibrated on this corpus.
    "high_threshold": 0.78,
    "low_threshold": 0.55,
    "min_cohesion": 0.40,
    "max_segment_tokens": 4096,
    "context_weight": 0.15,
    "context_chars": 256,
    "review_enabled": True,
    "max_review_calls_per_session": 32,
}


def segmentation_settings(supplied=None):
    values = {**DEFAULTS, **(supplied or {})}
    unknown = values.keys() - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown segmentation settings: {sorted(unknown)}")
    for key in ("high_threshold", "low_threshold", "min_cohesion", "context_weight"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Segmentation {key} must be finite")
        if not -1 <= value <= 1:
            raise ValueError(f"Segmentation {key} must lie in [-1,1]")
    if values["low_threshold"] > values["high_threshold"]:
        raise ValueError("Segmentation low_threshold cannot exceed high_threshold")
    if not 0 <= values["context_weight"] <= 0.5:
        raise ValueError("Context is auxiliary: context_weight must lie in [0,0.5]")
    for key in ("max_segment_tokens", "context_chars", "max_review_calls_per_session"):
        minimum = 1 if key == "max_segment_tokens" else 0
        if type(values[key]) is not int or values[key] < minimum:
            raise ValueError(f"Segmentation {key} must be an integer >= {minimum}")
    if type(values["review_enabled"]) is not bool:
        raise ValueError("Segmentation review_enabled must be boolean")
    return values


def validate_spans(spans, block_count):
    if not isinstance(spans, list):
        raise ValueError("Segmentation spans must be a list")
    start = 1
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("Segmentation span must be an object")
        left, right = span.get("start_block"), span.get("end_block")
        if (type(left) is not int or type(right) is not int or left != start
                or not left <= right <= block_count):
            raise ValueError("Segmentation must cover consecutive Blocks exactly once")
        start = right + 1
    if start != block_count + 1:
        raise ValueError("Segmentation omitted trailing Blocks")
    return spans


def block_texts(session, modality_policy="existing_captions"):
    """Public source only: no query/gold labels, source IDs, or private metadata."""
    records = {r["record_id"]: r for r in session["records"]}
    texts = []
    for block in session["blocks"]:
        parts = []
        for rid in block["record_ids"]:
            record = public_record(records[rid], modality_policy)
            parts.append(f"Speaker: {record['speaker'] or 'unknown'}\n"
                         f"Kind: {record['kind']}\nText: {record['text']}")
            for attachment in record["attachments"]:
                if attachment.get("caption"):
                    parts.append("Auxiliary image caption: " + attachment["caption"])
        texts.append("\n\n".join(parts))
    return texts


class AdjacentSegmenter:
    def __init__(self, embedding_client, settings=None, review_boundary=None, *,
                 modality_policy="existing_captions"):
        if modality_policy not in MODALITY_POLICIES:
            raise ValueError("Unsupported segmentation modality policy")
        self.embedding_client = embedding_client
        self.settings = segmentation_settings(settings)
        self.review_boundary = review_boundary
        self.modality_policy = modality_policy

    def _similarities(self, texts):
        vectors = self.embedding_client.encode(texts)
        if len(vectors) != len(texts):
            raise ValueError("Embedding client omitted segmentation inputs")
        dimensions = len(vectors[0])
        raw = np.asarray([normalized_vector(v, dimensions) for v in vectors], dtype=np.float64)
        similarity = raw @ raw.T
        count = len(texts)
        contextual = raw.copy()
        chars, weight = self.settings["context_chars"], self.settings["context_weight"]
        selected, queries = [], []
        if chars and weight:
            for i in range(1, count):
                if len(texts[i]) <= chars:
                    selected.append(i)
                    queries.append("Earlier context (reference only): " + texts[i - 1][-chars:]
                                   + "\nCurrent passage to represent:\n" + texts[i])
        if queries:
            values = self.embedding_client.encode(queries)
            if len(values) != len(queries):
                raise ValueError("Embedding client omitted contextual segmentation inputs")
            for i, vector in zip(selected, values):
                contextual[i] = normalized_vector(vector, dimensions)
            similarity = (1 - weight) * similarity + weight * (contextual @ contextual.T)
        return np.clip(similarity, -1.0, 1.0), count + len(queries)

    @staticmethod
    def _score(matrix, left, right):
        a, b = left["start_block"] - 1, left["end_block"]
        c, d = right["start_block"] - 1, right["end_block"]
        return float(matrix[a:b, c:d].mean())

    @staticmethod
    def _cohesion(matrix, start, end):
        submatrix = matrix[start - 1:end, start - 1:end]
        n = end - start + 1
        if n < 2:
            return 1.0
        # The least-supported message identifies a topic drift masked by a large
        # majority. Low cohesion requests review; it is not an absolute veto on
        # a semantically dependent question, correction, or tool observation.
        return float(((submatrix.sum(axis=1) - np.diag(submatrix)) / (n - 1)).min())

    def segment(self, session):
        validate_session(session)
        settings, blocks = self.settings, session["blocks"]
        stats = {"initial_blocks": len(blocks), "embedding_texts": 0, "merges": 0,
                 "automatic_merges": 0, "review_merges": 0, "review_calls": 0,
                 "review_failures": 0, "review_budget_exhausted": 0, "review_unavailable": 0,
                 "review_disabled": 0, "needs_review": 0, "oversized_blocks": 0}
        traces = []
        spans = []
        for block in blocks:
            oversized = block["token_count"] > settings["max_segment_tokens"]
            stats["oversized_blocks"] += int(oversized)
            spans.append({"start_block": block["index"], "end_block": block["index"],
                          "token_count": block["token_count"], "oversized": oversized})
            if oversized:
                traces.append({"event": "oversized_atomic_block", "block": block["index"],
                               "token_count": block["token_count"], "action": "preserve_complete_block"})
        if len(blocks) > 1:
            matrix, stats["embedding_texts"] = self._similarities(block_texts(session, self.modality_policy))
            # Rejected span pairs remain rejected while their exact membership
            # is unchanged. A neighboring merge creates a new candidate key.
            rejected = set()
            while len(spans) > 1:
                candidates = []
                for i, (left, right) in enumerate(zip(spans, spans[1:])):
                    key = (left["start_block"], left["end_block"], right["start_block"], right["end_block"])
                    if key in rejected:
                        continue
                    score = self._score(matrix, left, right)
                    candidates.append((-score, left["start_block"], i, key))
                candidates.sort()  # Highest score first; left position breaks ties.
                merged = False
                for negative_score, _, i, key in candidates:
                    left, right = spans[i:i + 2]
                    score = -negative_score
                    total = left["token_count"] + right["token_count"]
                    event = {"left": list(key[:2]), "right": list(key[2:]), "score": score,
                             "token_count": total}
                    if total > settings["max_segment_tokens"]:
                        rejected.add(key)
                        traces.append({"event": "adjacent_boundary_kept", "reason": "token_budget", **event})
                        continue
                    if score < settings["low_threshold"]:
                        rejected.add(key)
                        traces.append({"event": "adjacent_boundary_kept", "reason": "low_similarity", **event})
                        continue
                    cohesion = self._cohesion(matrix, key[0], key[3])
                    event["cohesion"] = cohesion
                    automatic = score >= settings["high_threshold"] and cohesion >= settings["min_cohesion"]
                    accepted, review = automatic, None
                    if not automatic:
                        if not settings["review_enabled"]:
                            reason = "review_disabled"
                            stats["review_disabled"] += 1
                        elif self.review_boundary is None:
                            reason = "review_unavailable"
                            stats["review_unavailable"] += 1
                            stats["needs_review"] += 1
                        elif stats["review_calls"] >= settings["max_review_calls_per_session"]:
                            reason = "review_budget_exhausted"
                            stats["review_budget_exhausted"] += 1
                            stats["needs_review"] += 1
                        else:
                            stats["review_calls"] += 1
                            try:
                                review = self.review_boundary({"session": session, "left": deepcopy(left),
                                                               "right": deepcopy(right), "score": score,
                                                               "cohesion": cohesion})
                                if not isinstance(review, dict) or review.get("decision") not in {"merge", "keep_boundary"}:
                                    raise ValueError("Invalid boundary review decision")
                                if review.get("review_unavailable", False) or review.get("available", True) is False:
                                    accepted = False
                                    reason = "review_unavailable"
                                    stats["review_unavailable"] += 1
                                    stats["needs_review"] += 1
                                else:
                                    accepted = review["decision"] == "merge"
                                    reason = "review_merge" if accepted else "review_keep_boundary"
                                event["review"] = review
                            except Exception as exc:
                                stats["review_failures"] += 1
                                stats["needs_review"] += 1
                                event["error_type"] = type(exc).__name__
                                reason = "review_failed"
                        if not accepted:
                            rejected.add(key)
                            traces.append({"event": "adjacent_boundary_kept", "reason": reason, **event})
                            continue
                    replacement = {"start_block": left["start_block"], "end_block": right["end_block"],
                                   "token_count": total, "oversized": False}
                    spans[i:i + 2] = [replacement]
                    stats["merges"] += 1
                    stats["automatic_merges" if automatic else "review_merges"] += 1
                    traces.append({"event": "adjacent_merge", "reason": "automatic" if automatic else "review_merge", **event})
                    merged = True
                    break  # Recompute all current neighbors after changing spans.
                if not merged:
                    break
        validate_spans(spans, len(blocks))
        stats["segments"] = len(spans)
        return {"spans": spans, "stats": stats, "traces": traces}
