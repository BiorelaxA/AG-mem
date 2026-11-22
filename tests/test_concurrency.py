from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident
from types import SimpleNamespace
import json
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
import yaml

from agmem.core import Completion, Example, InfrastructureError, Trajectory, load_config, seeded_int
from agmem.data import clean_history
from agmem.env import MemoryEnvironment
from agmem.execution import bounded_results, execution_settings, make_job_config
from agmem.pipeline import collect_group
from agmem.services import ChatService, DirectoryBuilder
from agmem.store import Store


class WordTokenizer:
    def encode(self, text, **kwargs):
        return text.split()
    def decode(self, ids, **kwargs):
        return ' '.join(ids)


class ConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / 'store')
        self.identity = {'model': 'fixture', 'model_revision': 'v1', 'base_url': 'http://fixture/v1', 'retries': 0}
        self.records = clean_history({'haystack_sessions': [
            [{'role': 'user', 'content': f'fact {i}'}] for i in range(7)]})
        self.memory_id = self.store.add_memory(self.records)
        self.writer_config = {'lookback_sessions': 3, 'content_max_tokens': 300}

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def builder(self, complete):
        return DirectoryBuilder(self.store, SimpleNamespace(identity=self.identity, complete=complete),
                                self.writer_config, WordTokenizer())

    def service(self, handler):
        service = ChatService(self.identity, self.store, 'task_agent')
        service.client.close()
        service.client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(service.client.close)
        return service

    def test_writer_concurrent_order_lookback_single_writer_and_reuse(self):
        barrier, lock = Barrier(2), Lock()
        calls, thread_ids = [], set()
        active = peak = 0
        def complete(messages, *args, **kwargs):
            nonlocal active, peak
            value = json.loads(messages[1]['content'])
            i = value['TARGET_SESSION']['order']
            self.assertEqual([r['order'] for r in value['PRECEDING_CONTEXT']], list(range(max(0, i - 3), i)))
            with lock:
                calls.append(i)
                thread_ids.add(get_ident())
                active += 1
                peak = max(peak, active)
            if i < 2:
                barrier.wait(timeout=5)
            time.sleep(0.005 * (2 - i % 2))
            with lock:
                active -= 1
            return f'summary {i}'
        builder = self.builder(complete)
        owner = get_ident()
        original = self.store.put_directory_entry
        def save(*args):
            self.assertEqual(get_ident(), owner)
            return original(*args)
        with patch.object(self.store, 'put_directory_entry', side_effect=save):
            entries = builder.build(self.memory_id, workers=2)
        self.assertEqual(peak, 2)
        self.assertNotIn(owner, thread_ids)
        self.assertEqual([r['content'] for r in entries], [f'summary {i}' for i in range(7)])
        self.assertEqual(self.store.directory(self.memory_id, builder.version), entries)
        self.assertEqual(builder.build(self.memory_id, workers=4), entries)
        self.assertEqual(sorted(calls), list(range(7)))
        serial_builder = self.builder(lambda *a, **k: self.fail('Completed cache should be reused'))
        self.assertEqual(serial_builder.version, builder.version)
        self.assertEqual(serial_builder.build(self.memory_id), entries)

    def test_writer_failure_drains_successes_and_resume_reuses_them(self):
        barrier = Barrier(2)
        def complete(messages, *args, **kwargs):
            index = json.loads(messages[1]['content'])['TARGET_SESSION']['order']
            if index < 2:
                barrier.wait(timeout=5)
            if index == 1:
                raise InfrastructureError('fixture failure')
            return f'summary {index}'
        builder = self.builder(complete)
        with self.assertRaises(InfrastructureError):
            builder.build(self.memory_id, workers=2)
        successful = {i for i, ref in enumerate(self.store.memory(self.memory_id))
                      if self.store.directory_entry(self.memory_id, builder.version, ref['id']) is not None}
        self.assertIn(0, successful)
        self.assertNotIn(1, successful)
        with self.assertRaisesRegex(ValueError, 'Directory missing'):
            self.store.directory(self.memory_id, builder.version)
        self.store.close()
        self.store = Store(self.store.root)
        retried = []
        def retry(messages, *args, **kwargs):
            index = json.loads(messages[1]['content'])['TARGET_SESSION']['order']
            retried.append(index)
            return f'summary {index}'
        entries = self.builder(retry).build(self.memory_id, workers=2)
        self.assertEqual(set(retried), set(range(7)) - successful)
        self.assertEqual(len(entries), 7)

    def test_http_requests_overlap_and_completion_cache_survives_reopen(self):
        barrier, lock = Barrier(2), Lock()
        calls = []
        def response(request):
            text = json.loads(request.content)['messages'][0]['content']
            with lock:
                calls.append(text)
            barrier.wait(timeout=5)
            return httpx.Response(200, json={'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]})
        service = self.service(response)
        def ask(i):
            return service.complete([{'role': 'user', 'content': str(i)}], 10)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(ask, range(12)))
        self.assertEqual(results, list(map(str, range(12))))
        self.assertEqual(len(self.store.completions.keys()), 12)
        self.store.close()
        self.store = Store(self.store.root)
        service.store = self.store
        self.assertEqual([ask(i) for i in range(12)], results)
        self.assertEqual(len(calls), 12)

    def test_identical_concurrent_task_requests_share_one_response(self):
        started, release, barrier = Event(), Event(), Barrier(8)
        calls = []
        def response(request):
            calls.append(request)
            started.set()
            self.assertTrue(release.wait(timeout=5))
            return httpx.Response(200, json={'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}]})
        service = self.service(response)
        def ask(_):
            barrier.wait(timeout=5)
            return service.complete([{'role': 'user', 'content': 'same'}], 10)
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(ask, i) for i in range(8)]
            self.assertTrue(started.wait(timeout=5))
            release.set()
            self.assertEqual([f.result() for f in futures], ['answer'] * 8)
        self.assertEqual(len(calls), 1)
        self.assertFalse(service._inflight)

    def test_batched_environment_matches_serial_with_divergent_actions(self):
        builder = self.builder(lambda *a, **k: 'summary')
        directory = builder.build(self.memory_id)
        keys = [row['id'] for row in directory]
        example = Example('fixture/train/0', 'fixture', 'train', self.memory_id, 'question')
        actions = {
            100: {'ids': keys[:5], 'stop': False}, 101: {'ids': ['ignored'], 'stop': True},
            200: 'not json',
            300: {'ids': ['unknown'], 'stop': False}, 301: {'ids': [keys[1]], 'stop': False},
            302: {'ids': [], 'stop': True},
        }
        batch_sizes = []
        def complete(messages, seed):
            action = actions[seed]
            return Completion(action if isinstance(action, str) else json.dumps(action))
        def complete_batch(messages, seeds, **kwargs):
            batch_sizes.append(len(seeds))
            return [complete(m, s) for m, s in zip(messages, seeds)]
        policy = SimpleNamespace(complete=complete, complete_batch=complete_batch)
        env = MemoryEnvironment(self.store, builder.version)
        serial = [env.run(example, policy, seed, 7).to_dict() for seed in (100, 200, 300)]
        batched = [r.to_dict() for r in env.run_batch(example, policy, [100, 200, 300], 7)]
        self.assertEqual(serial, batched)
        self.assertEqual(batch_sizes, [3, 2, 1])
        self.assertEqual([r['n_read_calls'] for r in batched], [1, 0, 2])
        self.assertEqual([len(r['evidence']) for r in batched], [4, 0, 1])
        self.assertEqual(batched[0]['decisions'][0]['action']['ids'], keys[:5])
        self.assertEqual(batched[0]['decisions'][0]['tool_result']['selected_ids'], keys[:4])

    def test_group_scoring_concurrent_but_policy_is_fixed_and_order_preserved(self):
        example = Example('fixture/train/0', 'fixture', 'train', self.memory_id, 'q', gold_answers=['yes'])
        policy = SimpleNamespace(policy_version=7)
        seeds = [seeded_int(42, example.sample_id, 7, 0, i) % (2 ** 31) for i in range(8)]
        batches, finished, barrier, lock = [], [], Barrier(2), Lock()
        owner = get_ident()
        def run_batch(ex, policy_, seeds_, version, **kwargs):
            self.assertEqual(get_ident(), owner)
            self.assertEqual(policy_.policy_version, version)
            batches.append(seeds_)
            return [Trajectory(ex.sample_id, seed, ex.memory_id, version, evidence=[{'i': seeds.index(seed)}],
                               termination_reason='stop') for seed in seeds_]
        def answer(ex, evidence, *, metadata=None):
            index = evidence[0]['i']
            self.assertNotEqual(get_ident(), owner)
            self.assertEqual(policy.policy_version, 7)
            if index < 2:
                barrier.wait(timeout=5)
            with lock:
                finished.append(index)
            return ('yes' if index % 2 else 'no'), str(index)
        config = {'seed': 42, 'execution': {'reader_batch_size': 2, 'task_agent_workers': 2},
                  'training': {'advantage_epsilon': 1e-6}, 'reward': {'read_call_cost': .01}}
        env, agent = SimpleNamespace(run_batch=run_batch, max_ids=2), SimpleNamespace(answer=answer)
        group = collect_group(config, env, policy, agent, example, 0)
        self.assertEqual([row.seed for row in group], seeds)
        self.assertEqual([row.task_input_hash for row in group], list(map(str, range(8))))
        self.assertEqual([len(batch) for batch in batches], [2] * 4)
        self.assertEqual(sorted(finished), list(range(8)))
        self.assertEqual([row.outcome for row in group], [0, 1] * 4)
        self.assertEqual(policy.policy_version, 7)
        self.assertAlmostEqual(sum(row.advantage for row in group), 0)
        agent.answer = lambda *args, **kwargs: (_ for _ in ()).throw(InfrastructureError('offline'))
        with self.assertRaises(InfrastructureError):
            collect_group(config, env, policy, agent, example, 0)

    def test_scheduler_does_not_eagerly_consume_all_items(self):
        pulled, consumed = [], []
        barrier = Barrier(2)
        def items():
            for i in range(10):
                self.assertLessEqual(len(pulled) - len(consumed), 1)
                pulled.append(i)
                yield i
        def work(i):
            if i < 2:
                barrier.wait(timeout=5)
            return i * i
        for item, result in bounded_results(work, items(), 2):
            consumed.append(item)
            self.assertEqual(result, item * item)
        self.assertEqual(sorted(consumed), list(range(10)))

    def test_job_overlay_preserves_frozen_identity_and_supports_serial_fallback(self):
        config = load_config('configs/default.yaml')
        config.pop('execution')  # Historical frozen configurations lack scheduling settings.
        source = Path(self.temp.name) / 'frozen.yaml'
        destination = Path(self.temp.name) / 'job.yaml'
        source.write_text(yaml.safe_dump(config))
        original = source.read_bytes()
        self.assertEqual(make_job_config(source, destination, {}), 2)
        runtime = load_config(destination)
        self.assertEqual(execution_settings(runtime)['reader_batch_size'], 2)
        self.assertEqual({k: v for k, v in runtime.items() if k != 'execution'}, config)
        self.assertEqual(source.read_bytes(), original)
        serial = Path(self.temp.name) / 'serial.yaml'
        env = {f'AGMEM_{key.upper()}': '1' for key in ('writer_workers', 'task_agent_workers', 'reader_batch_size')}
        self.assertEqual(make_job_config(source, serial, env), 1)
        self.assertEqual(execution_settings(load_config(serial))['reader_batch_size'], 1)
        for bad in (0, -1, True, 1.5, '2'):
            with self.assertRaises(ValueError):
                execution_settings({'execution': {'writer_workers': bad}})
        with self.assertRaises(ValueError):
            execution_settings({'execution': {'reader_batch_size': 9}})


if __name__ == '__main__':
    unittest.main()
