import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from agmem.core import Example, atomic_json, file_digest
from agmem.data import InvalidSample, adapt_row, task_examples
from agmem.prompts import reader_messages, task_messages
from agmem.services import ChatService, TaskAgent
from agmem.store import Store


class QuestionDateTests(unittest.TestCase):
    def test_date_reaches_both_models_without_labels_or_wall_clock(self):
        raw = {"question": "How many days ago did we meet?", "answer": "SECRET_ANSWER",
               "question_date": "2023/04/20 (Thu) 10:12", "answer_session_ids": ["SECRET_ID"],
               "question_type": "SECRET_TYPE"}
        for source in ["split_longmemeval_s", "split_longmemeval_m"]:
            example = adapt_row(raw, source, "train", 0)
            for messages in [reader_messages(example, [], []), task_messages(example, [])]:
                public = json.loads(messages[-1]["content"])
                self.assertIn(raw["question_date"], public["observation"])
                self.assertEqual(public["query"], raw["question"])
                self.assertNotIn("SECRET", json.dumps(messages))
            for missing in [None, "", 123]:
                with self.assertRaises(InvalidSample):
                    adapt_row({**raw, "question_date": missing}, source, "train", 0)
        self.assertEqual(adapt_row(raw, "locomo", "train", 0).observation, "")

    def test_legacy_examples_restore_each_split_without_mutating_store(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = "split_longmemeval_s"
            directory = root / source
            directory.mkdir()
            files, prepared = {}, {}
            for split, date in [("train", "2023/04/20 (Thu) 10:12"),
                                ("validation", "2023/05/20 (Sat) 10:12"),
                                ("test", "2023/06/20 (Tue) 10:12")]:
                path = directory / f"{split}.parquet"
                # Index 0 may be quarantined; matching must use source row IDs,
                # not the position in the surviving prepared examples list.
                pq.write_table(pa.Table.from_pylist([
                    {"question": "", "answer": "bad", "question_date": date},
                    {"question": "q", "answer": "gold", "question_date": date}]), path)
                files[split] = {"sha256": file_digest(path)}
                prepared[split] = [Example(f"{source}/{split}/000001", source, split, "same-memory", "q", gold_answers=["gold"])]
            atomic_json(root / "manifest.json", {"sources": [{"source": source, "output_files": files}]})
            identity = {"input_manifest_sha256": file_digest(root / "manifest.json")}
            store = SimpleNamespace(examples=lambda split: copy.deepcopy(prepared[split]),
                                    get_metadata=lambda key: {"layout": "by_source", "datasets": {source: identity}})
            config = {"data": {"input_root": str(root)}}
            for split in prepared:
                example = task_examples(config, store, split)[0]
                self.assertTrue(example.observation)
                self.assertEqual(example.memory_id, "same-memory")
                self.assertEqual(prepared[split][0].observation, "")
                with patch("pyarrow.parquet.ParquetFile", side_effect=AssertionError("must reuse in-memory scalar rows")):
                    self.assertEqual(task_examples(config, store, split), [example])
            prepared["train"][0].query = "different query"
            with self.assertRaisesRegex(ValueError, "does not match"):
                task_examples(config, store, "train")
            prepared["train"][0].query = "q"
            prepared["train"][0].observation = "Conflicting date"
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                task_examples(config, store, "train")
            prepared["train"][0].observation = ""
            store._question_date_rows.clear()
            with (directory / "train.parquet").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "dataset content changed"):
                task_examples(config, store, "train")

    def test_task_prompt_and_date_changes_do_not_hit_old_answer_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(temporary)
            config = {"model": "fixture", "model_revision": "fixture", "base_url": "http://fixture/v1", "retries": 0}
            service = ChatService(config, store, "task_agent")
            service.client.close()
            requests = []
            def respond(request):
                requests.append(json.loads(request.content))
                return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": str(len(requests))}}]})
            service.client = httpx.Client(transport=httpx.MockTransport(respond))
            try:
                example = Example("fixture", "fixture", "train", "memory", "When?")
                with patch("agmem.prompts.TASK_SYSTEM", "Old concise answer instructions"):
                    self.assertEqual(TaskAgent(service).answer(example, [])[0], "1")
                self.assertEqual(TaskAgent(service).answer(example, [])[0], "2")
                self.assertEqual(TaskAgent(service).answer(example, [])[0], "2")
                example.observation = "Current date: 2023/04/20"
                self.assertEqual(TaskAgent(service).answer(example, [])[0], "3")
                self.assertEqual(len(requests), 3)
            finally:
                service.client.close()
                store.close()


if __name__ == "__main__":
    unittest.main()
