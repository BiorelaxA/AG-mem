"""Source isolation and lossless Block adaptation, using deterministic token counts."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from agmem.core import digest
from agmem.hierarchical.adapters import iter_banks
from agmem.hierarchical.records import (build_blocks, make_session, public_record,
                                         render_records, time_metadata, validate_session)


def count_tokens(text):
    # Deliberately nonzero and deterministic: prove the adapter invokes its tokenizer.
    return len(text.encode("utf-8"))


class HierarchicalAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, rows):
        path = self.root / name
        path.write_text(json.dumps(rows, ensure_ascii=False))
        return str(path)

    def banks(self, name, path, **extra):
        return list(iter_banks({"name": name, "path": path, **extra}, ["test"], count_tokens=count_tokens))

    def locomo(self):
        return {"sample_id": "private_sample", "qa": [{"question": "PRIVATE QUESTION", "answer": "PRIVATE GOLD"}],
                "observation": "PRIVATE OBSERVATION", "event_summary": "PRIVATE EVENT",
                "conversation": {"speaker_a": "Alice", "speaker_b": "Bob",
                    "session_10": [{"speaker": "Bob", "dia_id": "D10:1", "text": "Later correction: no.",
                                    "has_answer": True}], "session_10_date_time": "2024-01-10",
                    "session_2": [{"speaker": "Alice", "dia_id": "D2:1", "text": "Original words, unchanged!",
                                   "img_url": "photo.jpg", "blip_caption": "A red bicycle",
                                   "query": "PRIVATE IMAGE QUERY", "re-download": True}],
                    "session_2_date_time": "2024-01-02"}}

    def lme(self):
        return {"question_id": "PRIVATE QUESTION ID", "question": "PRIVATE QUESTION", "answer": "PRIVATE GOLD",
                "question_type": "PRIVATE TYPE", "answer_session_ids": ["answer_private"],
                "question_date": "2099-01-01", "haystack_dates": ["2024-01-02 12:30"],
                "haystack_session_ids": ["answer_private"],
                "haystack_sessions": [[{"role": "user", "content": "The words has_answer are actual speech.", "has_answer": True},
                                       {"role": "user", "content": "Another same-role message."},
                                       {"role": "assistant", "content": "Correction: not accepted."}]]}

    def test_official_locomo_numeric_order_and_exact_messages_image_caption(self):
        bank = self.banks("locomo", self.write("locomo.json", [self.locomo()]))[0]
        self.assertEqual(len(bank["sessions"]), 2)
        first, last = bank["sessions"]
        self.assertEqual(first["records"][0]["text"], "Original words, unchanged!")
        self.assertEqual(last["records"][0]["text"], "Later correction: no.")
        self.assertEqual(first["participants"], ["Alice", "Bob"])
        self.assertEqual(first["time"], {"raw": "2024-01-02", "iso": "2024-01-02", "timezone": None})
        self.assertEqual(first["blocks"][0]["token_count"], count_tokens(render_records(first["records"])))
        self.assertEqual(first["records"][0]["attachments"],
                         [{"kind": "image", "ref": "photo.jpg", "caption": "A red bicycle"}])
        visible = json.dumps([public_record(r) for s in bank["sessions"] for r in s["records"]])
        for private in ("PRIVATE", "D2:1", "D10:1", "has_answer", "re-download"):
            self.assertNotIn(private, visible)
        self.assertIn("D2:1", json.dumps(bank["source_map"]))
        self.assertEqual(len(bank["query_memberships"]), 1)

    def test_longmemeval_keeps_each_message_and_does_not_scrub_valid_speech(self):
        row = self.lme()
        row["haystack_sessions"] = json.dumps(row["haystack_sessions"])
        bank = self.banks("split_longmemeval_s", self.write("lme.json", [row]))[0]
        session = bank["sessions"][0]
        self.assertEqual([r["speaker"] for r in session["records"]], ["user", "user", "assistant"])
        self.assertEqual([b["record_ids"] for b in session["blocks"]],
                         [["p0001:r0001"], ["p0001:r0002"], ["p0001:r0003"]])
        visible = render_records(session["records"])
        self.assertIn("has_answer are actual speech", visible)
        self.assertNotIn("PRIVATE", visible)
        self.assertNotIn("answer_private", json.dumps(session))
        self.assertNotIn("2099", json.dumps(session))
        self.assertIn("answer_private", json.dumps(bank["source_map"]))

    def test_longmemeval_questions_have_separate_banks_even_for_identical_history(self):
        first, second = self.lme(), self.lme()
        second["question_id"] = "another question"
        banks = self.banks("split_longmemeval_m", self.write("lme.json", [first, second]))
        self.assertEqual(len(banks), 2)
        self.assertNotEqual(banks[0]["bank_id"], banks[1]["bank_id"])
        self.assertEqual(render_records(banks[0]["sessions"][0]["records"]),
                         render_records(banks[1]["sessions"][0]["records"]))

    def test_array_length_mismatch_is_failed_without_silent_zip(self):
        row = self.lme();row["haystack_dates"] = []
        bank = self.banks("split_longmemeval_s", self.write("lme.json", [row]))[0]
        self.assertEqual(bank["sessions"], [])
        self.assertEqual(bank["issues"][0]["status"], "failed")
        self.assertIn("lengths disagree", bank["issues"][0]["reason"])

    def test_bad_session_does_not_hide_other_sessions_or_empty_sessions(self):
        row = self.lme()
        row.update(haystack_sessions=[[{"role": "user", "content": {"invalid": "object"}}], [],
                                      [{"role": "assistant", "content": "Still valid."}]],
                   haystack_session_ids=["a", "b", "c"], haystack_dates=[None, None, None])
        bank = self.banks("split_longmemeval_s", self.write("lme.json", [row]))[0]
        self.assertEqual([s["parent_id"] for s in bank["sessions"]], ["p0002", "p0003"])
        self.assertEqual(bank["sessions"][0]["blocks"], [])
        self.assertEqual([i["parent_id"] for i in bank["issues"] if i["status"] == "failed"], ["p0001"])

    def test_processed_locomo_reuses_across_splits_and_keeps_exported_prefix(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        row = self.lme();row["haystack_sessions"][0][0].update(role="Alice", content="Alice: Existing prefix.", dia_id="D1:1")
        other = deepcopy(row);other["question_id"] = "second question"
        other["haystack_sessions"][0][0]["has_answer"] = False
        files = {}
        for split, value in [("train", row), ("test", other)]:
            path = self.root / f"{split}.parquet"
            value["haystack_sessions"] = json.dumps(value["haystack_sessions"])
            pq.write_table(pa.Table.from_pylist([value]), path);files[split] = str(path)
        spec = {"name": "locomo", "format": "locomo_parquet", "input_files": files}
        banks = list(iter_banks(spec, ["train", "test"], count_tokens=count_tokens))
        self.assertEqual(len(banks), 1)
        self.assertEqual(len(banks[0]["query_memberships"]), 2)
        self.assertEqual(banks[0]["sessions"][0]["records"][0]["text"], "Alice: Existing prefix.")
        self.assertEqual(banks[0]["issues"][0]["reason"], "processed_locomo_compat")

    def test_processed_locomo_bad_session_does_not_suppress_other_sessions(self):
        row=self.lme()
        row.update(haystack_sessions=[{"not":"a session"},[{"role":"Alice","content":"Valid later session"}]],
                   haystack_dates=[None,None],haystack_session_ids=["s1","s2"])
        bank=self.banks("locomo",self.write("compat.json",[row]),format="locomo_parquet")[0]
        self.assertEqual([s["parent_id"] for s in bank["sessions"]],["p0002"])
        self.assertEqual([i["parent_id"] for i in bank["issues"] if i["status"]=="failed"],["p0001"])

    def test_missing_metadata_is_reported_without_guessing_speakers_or_timezone(self):
        sample = self.locomo();sample["conversation"].pop("speaker_b")
        sample["conversation"].pop("session_2_date_time")
        bank = self.banks("locomo", self.write("locomo.json", [sample]))[0]
        self.assertEqual(bank["sessions"][0]["participants"], ["Alice"])
        self.assertEqual(bank["sessions"][0]["time"], {"raw": None, "iso": None, "timezone": None})
        self.assertIn("missing_speaker_b", [i["reason"] for i in bank["issues"]])
        self.assertIsNone(time_metadata("7:31 pm on 21 January, 2022")["timezone"])
        self.assertEqual(time_metadata("2024-01-01T00:00:00Z")["timezone"], "UTC")
        self.assertIsNone(time_metadata("last month")["iso"])

    def test_source_hash_and_exact_block_cover_are_checked(self):
        s = self.banks("split_longmemeval_s", self.write("lme.json", [self.lme()]))[0]["sessions"][0]
        validate_session(s, count_tokens)
        corrupt = deepcopy(s);corrupt["records"][0]["text"] = "changed"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_session(corrupt)
        corrupt = deepcopy(s);corrupt["blocks"][0]["record_ids"] = ["p0001:r0002"]
        corrupt["source_hash"] = digest({k:v for k,v in corrupt.items() if k != "source_hash"})
        with self.assertRaisesRegex(ValueError, "exactly once"):
            validate_session(corrupt)

    def test_parallel_calls_remain_one_contiguous_block_and_unclosed_calls_fail(self):
        def rec(i, kind, call_id=None):
            return {"record_id":f"p0001:r{i:04d}","seq":i,"kind":kind,"speaker":"agent",
                    "call_id":call_id,"text":f"Exact {kind} {i}","attachments":[]}
        records = [rec(1,"action","c1"),rec(2,"action","c2"),rec(3,"note"),
                   rec(4,"observation","c2"),rec(5,"observation","c1"),rec(6,"final")]
        blocks = build_blocks(records,"agent",count_tokens)
        self.assertEqual([len(b["record_ids"]) for b in blocks],[5,1])
        with self.assertRaisesRegex(ValueError,"Unclosed"):
            build_blocks(records[:4],"agent",count_tokens)
        with self.assertRaisesRegex(ValueError,"no matching"):
            build_blocks([rec(1,"observation","unknown")],"agent",count_tokens)

    def test_caption_policy_never_merges_auxiliary_text_into_speech(self):
        record = self.banks("locomo",self.write("locomo.json",[self.locomo()]))[0]["sessions"][0]["records"][0]
        self.assertEqual(public_record(record,"references_only")["attachments"][0]["caption"],None)
        self.assertEqual(record["attachments"][0]["caption"],"A red bicycle")
        self.assertNotIn("A red bicycle",record["text"])


if __name__ == "__main__":
    unittest.main()
