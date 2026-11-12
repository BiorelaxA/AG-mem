"""Explicit cross-span linkage and source integrity for adjacent segmentation."""

from copy import deepcopy
import unittest

import numpy as np

from agmem.core import digest
from agmem.hierarchical.records import make_session
from agmem.hierarchical.segmentation import AdjacentSegmenter, block_texts, segmentation_settings, validate_spans


def session(texts, tokens=None):
    records = [{"record_id": f"p1:r{i}", "seq": i, "kind": "message",
                "speaker": "user" if i % 2 else "assistant", "call_id": None,
                "text": text, "attachments": []} for i, text in enumerate(texts, 1)]
    value = make_session(bank_id="private_bank", parent_id="p1", order=1, records=records,
                         time={"raw": None, "iso": None, "timezone": None},
                         participants=["user", "assistant"], count_tokens=lambda text: 10)
    if tokens is not None:
        for block, count in zip(value["blocks"], tokens):
            block["token_count"] = count
        value["source_hash"] = digest({k: v for k, v in value.items() if k != "source_hash"})
    return value


class FixedEmbeddings:
    def __init__(self, vectors):
        self.vectors, self.calls = vectors, []

    def encode(self, texts):
        self.calls.append(list(texts))
        return self.vectors[:len(texts)]


class MatrixSegmenter(AdjacentSegmenter):
    def __init__(self, matrix, settings=None, review_boundary=None):
        super().__init__(None, {"context_weight": 0, **(settings or {})}, review_boundary)
        self.matrix = np.asarray(matrix, dtype=float)

    def _similarities(self, texts):
        return self.matrix, len(texts)


def ranges(result):
    return [(s["start_block"], s["end_block"]) for s in result["spans"]]


class AdjacentSegmentationTests(unittest.TestCase):
    def test_similar_nonadjacent_messages_are_not_reordered_or_combined(self):
        value = session(["computer purchase", "holiday plan", "computer return"])
        original = deepcopy(value)
        embeddings = FixedEmbeddings([[1, 0], [0, 1], [1, 0]])
        result = AdjacentSegmenter(embeddings, {"context_weight": 0}).segment(value)
        self.assertEqual(ranges(result), [(1, 1), (2, 2), (3, 3)])
        self.assertEqual(value, original)
        public = str(embeddings.calls)
        for private in ("private_bank", "p1:r", "source_hash", "record_ids"):
            self.assertNotIn(private, public)

    def test_same_topic_followups_merge_but_next_topic_stays_separate(self):
        value = session(["light color", "color answer", "light brightness", "brightness answer",
                         "speaker news", "speaker answer"])
        embeddings = FixedEmbeddings([[1, 0]] * 4 + [[0, 1]] * 2)
        result = AdjacentSegmenter(embeddings, {"context_weight": 0}).segment(value)
        self.assertEqual(ranges(result), [(1, 4), (5, 6)])
        self.assertEqual(result["stats"]["merges"], 4)
        self.assertEqual(result["stats"]["review_calls"], 0)

    def test_average_linkage_uses_all_cross_pairs_not_original_chain_edges(self):
        # Every original neighboring edge exceeds the high threshold, but the
        # first and third units are unrelated. Single-link/chain-edge updates
        # would incorrectly merge all three.
        matrix = [[1, .95, .1], [.95, 1, .9], [.1, .9, 1]]
        result = MatrixSegmenter(matrix, {"high_threshold": .8, "low_threshold": .6,
                                          "review_enabled": False}).segment(session(["A", "A/B", "B"]))
        self.assertEqual(ranges(result), [(1, 2), (3, 3)])
        boundary = [t for t in result["traces"] if t["event"] == "adjacent_boundary_kept"][-1]
        self.assertAlmostEqual(boundary["score"], .5)

    def test_low_cohesion_requires_review_even_when_cross_span_score_is_high(self):
        matrix = [[1, .3, .9], [.3, 1, .9], [.9, .9, 1]]
        reviews = []
        def review(payload):
            reviews.append(payload)
            return {"decision": "keep_boundary"}
        result = MatrixSegmenter(matrix, {"low_threshold": .5, "high_threshold": .55,
                                          "min_cohesion": .7}, review).segment(session(["A", "B", "bridge"]))
        self.assertEqual(ranges(result), [(1, 1), (2, 3)])
        self.assertEqual(len(reviews), 1)
        self.assertAlmostEqual(reviews[0]["score"], .6)
        self.assertLess(reviews[0]["cohesion"], .7)

    def test_rejected_best_candidate_does_not_prevent_another_valid_merge(self):
        matrix = [[1, .99, 0], [.99, 1, .9], [0, .9, 1]]
        value = session(["long", "short1", "short2"], [90, 20, 20])
        result = MatrixSegmenter(matrix, {"max_segment_tokens": 100}).segment(value)
        self.assertEqual(ranges(result), [(1, 1), (2, 3)])

    def test_ambiguous_boundary_can_merge_only_after_explicit_review(self):
        seen = []
        def review(payload):
            seen.append(payload)
            return {"decision": "merge", "reason": "same need clarified", "evidence_blocks": [1, 2]}
        value = session(["request", "clarification"])
        result = MatrixSegmenter([[1, .65], [.65, 1]], review_boundary=review).segment(value)
        self.assertEqual(ranges(result), [(1, 2)])
        self.assertEqual(result["stats"]["review_merges"], 1)
        self.assertEqual(seen[0]["left"]["start_block"], 1)
        self.assertEqual(seen[0]["right"]["end_block"], 2)
        self.assertIs(seen[0]["session"], value)

    def test_review_failure_and_budget_exhaustion_keep_source_and_report(self):
        def broken(_):
            raise RuntimeError("service unavailable")
        result = MatrixSegmenter([[1, .65, .65], [.65, 1, .65], [.65, .65, 1]],
                                 {"max_review_calls_per_session": 1}, broken).segment(session(["a", "b", "c"]))
        self.assertEqual(ranges(result), [(1, 1), (2, 2), (3, 3)])
        self.assertEqual(result["stats"]["review_calls"], 1)
        self.assertEqual(result["stats"]["review_failures"], 1)
        self.assertEqual(result["stats"]["review_budget_exhausted"], 1)
        self.assertEqual(result["stats"]["needs_review"], 2)
        self.assertTrue(any(t.get("reason") == "review_failed" for t in result["traces"]))

    def test_unavailable_review_is_distinct_from_a_valid_keep_decision(self):
        value = session(["a", "b"])
        result = MatrixSegmenter([[1, .65], [.65, 1]], review_boundary=lambda _: {
            "decision": "keep_boundary", "review_unavailable": True,
            "reason": "context budget exceeded"}).segment(value)
        self.assertEqual(ranges(result), [(1, 1), (2, 2)])
        self.assertEqual(result["stats"]["review_unavailable"], 1)
        self.assertEqual(result["stats"]["needs_review"], 1)

    def test_explicitly_disabled_review_is_not_a_service_failure(self):
        result = MatrixSegmenter([[1, .65], [.65, 1]], {"review_enabled": False}).segment(session(["a", "b"]))
        self.assertEqual(result["stats"]["review_disabled"], 1)
        self.assertEqual(result["stats"]["review_unavailable"], 0)
        self.assertEqual(result["stats"]["review_failures"], 0)
        self.assertEqual(result["stats"]["needs_review"], 0)

    def test_modality_policy_is_respected_before_embedding(self):
        value = session(["image record", "second record"])
        value["records"][0]["attachments"] = [{"kind": "image", "ref": "private-image-ref",
                                                  "caption": "CAPTION_VISIBLE_ONLY_WHEN_ALLOWED"}]
        value["source_hash"] = digest({k: v for k, v in value.items() if k != "source_hash"})
        self.assertIn("CAPTION_VISIBLE_ONLY_WHEN_ALLOWED", str(block_texts(value)))
        embeddings = FixedEmbeddings([[1, 0], [0, 1]])
        AdjacentSegmenter(embeddings, {"context_weight": 0}, modality_policy="references_only").segment(value)
        self.assertNotIn("CAPTION_VISIBLE_ONLY_WHEN_ALLOWED", str(embeddings.calls))
        self.assertNotIn("private-image-ref", str(embeddings.calls))

    def test_oversized_atomic_message_is_retained_intact_without_embedding_when_alone(self):
        value = session(["all original text must remain"], [500])
        original = deepcopy(value)
        embeddings = FixedEmbeddings([])
        result = AdjacentSegmenter(embeddings, {"max_segment_tokens": 100}).segment(value)
        self.assertEqual(ranges(result), [(1, 1)])
        self.assertTrue(result["spans"][0]["oversized"])
        self.assertEqual(result["stats"]["oversized_blocks"], 1)
        self.assertEqual(value, original)
        self.assertFalse(embeddings.calls)

    def test_equal_scores_have_deterministic_leftmost_tie_breaking(self):
        matrix = [[1, .9, .1], [.9, 1, .9], [.1, .9, 1]]
        for _ in range(3):
            result = MatrixSegmenter(matrix).segment(session(["a", "b", "c"]))
            self.assertEqual(ranges(result), [(1, 2), (3, 3)])
            self.assertEqual(result["traces"][0]["left"], [1, 1])

    def test_empty_sessions_and_invalid_spans_or_settings(self):
        self.assertEqual(ranges(AdjacentSegmenter(None).segment(session([]))), [])
        for spans in ([{"start_block": 2, "end_block": 3}],
                      [{"start_block": 1, "end_block": 2}],
                      [{"start_block": 1, "end_block": 2}, {"start_block": 2, "end_block": 3}]):
            with self.assertRaises(ValueError):
                validate_spans(spans, 3)
        for settings in ({"high_threshold": .1, "low_threshold": .2}, {"context_weight": .8},
                         {"max_segment_tokens": 0}, {"review_enabled": 1}, {"low_threshold": float("nan")}):
            with self.assertRaises(ValueError):
                segmentation_settings(settings)


if __name__ == "__main__":
    unittest.main()
