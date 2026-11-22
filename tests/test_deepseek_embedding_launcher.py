"""Remote Writer/local-embedding deployment and pooled client, without live services."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Lock
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

from agmem import build_deepseek_directories_gpu as launcher
from agmem.core import InfrastructureError
from agmem.hierarchical.embedding import EmbeddingClient
from agmem.hierarchical.embedding_pool import MultiEndpointEmbeddingClient


ROOT = Path(__file__).resolve().parents[1]


class Process:
    counter = 3000

    def __init__(self, kind, code=0):
        Process.counter += 1
        self.pid, self.kind = Process.counter, kind
        self.returncode = None if kind == "embedding" else code

    def poll(self):
        return self.returncode


class DeepSeekLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def args(self, **changes):
        args = launcher.argument_parser().parse_args([])
        args.job_dir = str(self.root / "job")
        for name, value in changes.items():
            setattr(args, name, value)
        return args

    def run_launch(self, args=None, *, wait_error=None, busy_error=None, probe_error=None,
                   worker_code=0, cleanup_error=False, environment=None):
        calls, stopped = [], []

        def start(command, **kwargs):
            kind = "embedding" if "pooling" in command else "worker"
            process = Process(kind, worker_code)
            calls.append((kind, command, kwargs, process))
            return process

        def stop(process):
            stopped.append(process)
            if cleanup_error and process.kind == "worker":
                raise RuntimeError("cleanup failed")

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {
                "DEEPSEEK_API_KEY": "test-secret", "CUDA_VISIBLE_DEVICES": "4,5",
                **(environment or {}),
            }, clear=True))
            stack.enter_context(patch.object(launcher, "load_recipe", return_value={
                "service": {"model": "deepseek-flash"},
                "embedding": {"workers": 8, "batch_size": 32, "expected_dimensions": 2560},
            }))
            stack.enter_context(patch.object(launcher, "local_embedding_model", return_value="/embedding"))
            dry = stack.enter_context(patch.object(launcher.subprocess, "run"))
            idle = stack.enter_context(patch.object(launcher, "ensure_gpus_idle", side_effect=busy_error))
            probe = stack.enter_context(patch.object(launcher, "probe_deepseek", side_effect=probe_error,
                                                   return_value=("https://api.deepseek.com", "deepseek-flash")))
            stack.enter_context(patch.object(launcher.subprocess, "Popen", side_effect=start))
            stack.enter_context(patch.object(launcher, "free_port", side_effect=[10100, 10101]))
            waits = stack.enter_context(patch.object(launcher, "wait_embedding_ready", side_effect=wait_error))
            stack.enter_context(patch.object(launcher, "stop_group", side_effect=stop))
            stack.enter_context(patch.object(launcher.signal, "signal"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                result = launcher.launch(args or self.args())
            except BaseException as exc:
                result = exc
        return SimpleNamespace(calls=calls, stopped=stopped, result=result, dry=dry,
                               idle=idle, probe=probe, waits=waits)

    def test_both_selected_gpus_embed_and_remote_writer_has_no_local_gpu(self):
        run = self.run_launch()
        self.assertIsInstance(run.result, Path)
        self.assertEqual([(kind, options["env"]["CUDA_VISIBLE_DEVICES"])
                          for kind, _, options, _ in run.calls],
                         [("embedding", "4"), ("embedding", "5"), ("worker", "")])
        for _, command, options, _ in run.calls[:2]:
            for flag, value in {"--tensor-parallel-size": "1", "--max-num-seqs": "64",
                                "--max-model-len": "8192", "--max-num-batched-tokens": "32768",
                                "--gpu-memory-utilization": "0.35", "--runner": "pooling",
                                "--convert": "embed"}.items():
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertNotIn("agmem-27b", command)
            self.assertNotIn("DEEPSEEK_API_KEY", options["env"])
            self.assertTrue(options["start_new_session"])
        worker = run.calls[-1][1]
        self.assertEqual(run.calls[-1][2]["env"]["DEEPSEEK_API_KEY"], "test-secret")
        self.assertIn("agmem.build_deepseek_directories", worker)
        self.assertEqual(worker[worker.index("--workers") + 1], "64")
        self.assertEqual(worker[-3:], ["--embedding-base-urls", "http://127.0.0.1:10100/v1",
                                      "http://127.0.0.1:10101/v1"])
        self.assertEqual(run.stopped, [row[3] for row in reversed(run.calls)])
        self.assertEqual(run.waits.call_count, 2)
        self.assertTrue(all(call.kwargs["expected_dimensions"] == 2560 for call in run.waits.call_args_list))
        deployment_text = (run.result / "deployment.json").read_text()
        self.assertNotIn("test-secret", deployment_text)
        deployment = json.loads(deployment_text)
        self.assertEqual(deployment["embedding"]["total_client_concurrency"], 8)
        self.assertEqual([row["gpu"] for row in deployment["embedding_replicas"]], ["4", "5"])
        self.assertIn("--dry-run", run.dry.call_args.args[0])
        self.assertEqual(run.dry.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "")

    def test_embedding_gpu_override_and_default_allocation(self):
        run = self.run_launch(environment={"AGMEM_EMBEDDING_GPUS": "6,7"})
        self.assertEqual([row[2]["env"]["CUDA_VISIBLE_DEVICES"] for row in run.calls[:2]], ["6", "7"])
        for visible, expected in (("4", ["4"]), (" 5 ", ["5"]), ("4,5", ["4", "5"])):
            with self.subTest(visible=visible):
                self.assertEqual(launcher.allocated_devices(visible), expected)
        for visible in ("", "4,4", "4,,5", "-1", "NoDevFiles", "-1,5", "NoDevFiles,5", "0,1,2"):
            with self.subTest(visible=visible), self.assertRaises(ValueError):
                launcher.allocated_devices(visible)

    def test_single_gpu_five_embeds_with_256_remote_writer_workers(self):
        run = self.run_launch(self.args(workers=256), environment={
            "AGMEM_EMBEDDING_GPUS": "5", "CUDA_VISIBLE_DEVICES": "5",
        })
        self.assertIsInstance(run.result, Path)
        self.assertEqual([(kind, options["env"]["CUDA_VISIBLE_DEVICES"])
                          for kind, _, options, _ in run.calls],
                         [("embedding", "5"), ("worker", "")])
        embedding = run.calls[0][1]
        for flag, value in {"--tensor-parallel-size": "1", "--max-num-seqs": "64",
                            "--max-model-len": "8192", "--max-num-batched-tokens": "32768",
                            "--gpu-memory-utilization": "0.35"}.items():
            self.assertEqual(embedding[embedding.index(flag) + 1], value)
        self.assertNotIn("DEEPSEEK_API_KEY", run.calls[0][2]["env"])
        worker = run.calls[1][1]
        self.assertEqual(worker[worker.index("--workers") + 1], "256")
        self.assertEqual(worker.count("--embedding-base-urls"), 1)
        self.assertEqual(worker[-2:], ["--embedding-base-urls", "http://127.0.0.1:10100/v1"])
        run.idle.assert_called_once_with(["5"], .35)
        run.waits.assert_called_once_with(run.calls[0][3], 10100,
                                         run.result / "embedding_1.log", expected_dimensions=2560)
        self.assertEqual(run.stopped, [run.calls[1][3], run.calls[0][3]])
        deployment = json.loads((run.result / "deployment.json").read_text())
        self.assertEqual(deployment["writer"]["workers"], 256)
        self.assertEqual(deployment["embedding_replicas"], [
            {"gpu": "5", "base_url": "http://127.0.0.1:10100/v1"},
        ])
        self.assertEqual(deployment["embedding"]["total_client_concurrency"], 8)

    def test_dry_run_has_no_key_cuda_model_or_network_requirement(self):
        with patch.dict(os.environ, {}, clear=True), ExitStack() as stack:
            mocks = {name: stack.enter_context(patch.object(launcher, name)) for name in
                     ("local_embedding_model", "load_recipe", "ensure_gpus_idle", "probe_deepseek")}
            start = stack.enter_context(patch.object(launcher.subprocess, "Popen"))
            run = stack.enter_context(patch.object(launcher.subprocess, "run"))
            self.assertIsNone(launcher.launch(self.args(dry_run=True, limit_sessions=3,
                                                        output_root="/tmp/new build")))
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertIn("--dry-run", command)
        self.assertEqual(command[command.index("--limit-sessions") + 1], "3")
        self.assertEqual(command[command.index("--output-root") + 1], "/tmp/new build")
        start.assert_not_called()
        for mock in mocks.values():
            mock.assert_not_called()

    def test_missing_key_fails_before_preflight_or_gpu(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(launcher.subprocess, "run") as run, \
                patch.object(launcher.subprocess, "Popen") as start, \
                self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
            launcher.launch(self.args())
        run.assert_not_called()
        start.assert_not_called()

    def test_busy_gpu_or_bad_api_never_starts_or_stops_a_process(self):
        for error in ({"busy_error": RuntimeError("occupied")},
                      {"probe_error": RuntimeError("bad API")}):
            with self.subTest(error=error):
                run = self.run_launch(**error)
                self.assertIsInstance(run.result, RuntimeError)
                self.assertEqual(run.calls, [])
                self.assertEqual(run.stopped, [])

    def test_startup_failure_only_stops_the_owned_embedding_processes(self):
        run = self.run_launch(wait_error=[None, RuntimeError("second replica failed")])
        self.assertIsInstance(run.result, RuntimeError)
        self.assertEqual([row[0] for row in run.calls], ["embedding", "embedding"])
        self.assertEqual(run.stopped, [row[3] for row in reversed(run.calls)])

    def test_single_gpu_startup_failure_only_stops_its_owned_embedding_process(self):
        run = self.run_launch(environment={"AGMEM_EMBEDDING_GPUS": "5"},
                              wait_error=RuntimeError("embedding startup failed"))
        self.assertIsInstance(run.result, RuntimeError)
        self.assertEqual([(kind, options["env"]["CUDA_VISIBLE_DEVICES"])
                          for kind, _, options, _ in run.calls], [("embedding", "5")])
        run.waits.assert_called_once()
        self.assertEqual(run.stopped, [run.calls[0][3]])

    def test_cleanup_failure_does_not_skip_other_owned_processes(self):
        with patch("sys.stderr", new=io.StringIO()):
            run = self.run_launch(cleanup_error=True)
        self.assertIsInstance(run.result, RuntimeError)
        self.assertIn("GPU release is unconfirmed", str(run.result))
        self.assertEqual(run.stopped, [row[3] for row in reversed(run.calls)])

    def test_existing_embedding_endpoints_need_no_local_deployment(self):
        run = self.run_launch(self.args(embedding_base_urls=["http://one/v1", "http://two/v1"]))
        self.assertIsNone(run.result)
        run.idle.assert_not_called()
        run.probe.assert_called_once()
        self.assertEqual(run.calls, [])
        self.assertEqual(run.dry.call_count, 2)

    def test_legacy_result_reuse_is_not_an_option(self):
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            launcher.argument_parser().parse_args(["--reuse-root", "old-run"])

    def test_idle_check_rejects_compute_processes_without_stopping_them(self):
        outputs = [SimpleNamespace(stdout="GPU-a, 80000, 79000\nGPU-b, 80000, 79000\n"),
                   SimpleNamespace(stdout="GPU-other, 11\nGPU-b, 12\n")]
        with patch.object(launcher.subprocess, "run", side_effect=outputs) as run, \
                patch.object(launcher, "stop_group") as stop, \
                self.assertRaisesRegex(RuntimeError, "busy.*left running"):
            launcher.ensure_gpus_idle(["4", "5"], .35)
        self.assertEqual(run.call_count, 2)
        self.assertIn("--id=4,5", run.call_args_list[0].args[0])
        stop.assert_not_called()

    def test_idle_check_ignores_other_gpus_and_fails_closed_without_status(self):
        outputs = [SimpleNamespace(stdout="GPU-a, 80000, 79000\nGPU-b, 80000, 79000\n"),
                   SimpleNamespace(stdout="GPU-other, 11\n")]
        with patch.object(launcher.subprocess, "run", side_effect=outputs):
            launcher.ensure_gpus_idle(["4", "5"], .35)
        with patch.object(launcher.subprocess, "run", side_effect=FileNotFoundError), \
                self.assertRaisesRegex(RuntimeError, "no services started"):
            launcher.ensure_gpus_idle(["4", "5"], .35)

    def test_single_gpu_idle_check_ignores_other_gpus_and_requires_capacity(self):
        outputs = [SimpleNamespace(stdout="GPU-five, 80000, 79000\n"),
                   SimpleNamespace(stdout="GPU-other, 11\n")]
        with patch.object(launcher.subprocess, "run", side_effect=outputs) as run, \
                patch.object(launcher, "stop_group") as stop:
            launcher.ensure_gpus_idle(["5"], .35)
        self.assertEqual(run.call_count, 2)
        self.assertIn("--id=5", run.call_args_list[0].args[0])
        stop.assert_not_called()
        with patch.object(launcher.subprocess, "run", return_value=SimpleNamespace(
                stdout="GPU-five, 80000, 20000\n")) as run, \
                patch.object(launcher, "stop_group") as stop, \
                self.assertRaisesRegex(RuntimeError, "insufficient free memory.*left running"):
            launcher.ensure_gpus_idle(["5"], .35)
        run.assert_called_once()
        stop.assert_not_called()

    def test_free_model_probe_uses_get_and_does_not_disclose_response(self):
        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        client.get.return_value = httpx.Response(200, json={"data": [{"id": "deepseek-flash"}]})
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "do-not-print",
                                     "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1/"}), \
                patch("httpx.Client", return_value=client):
            endpoint, model = launcher.probe_deepseek({})
            self.assertEqual((endpoint, model), ("https://api.deepseek.com/v1", "deepseek-flash"))
            self.assertEqual(client.get.call_args.args[0], "https://api.deepseek.com/v1/models")
            client.post.assert_not_called()
            client.get.return_value = httpx.Response(401, text="do-not-print")
            with self.assertRaisesRegex(RuntimeError, "HTTP 401") as raised:
                launcher.probe_deepseek({})
            self.assertNotIn("do-not-print", str(raised.exception))


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


class EmbeddingPoolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = Path(self.temp.name) / "cache.sqlite3"

    def client(self, transport=None, **changes):
        settings = {"instruction": "Group.", "expected_dimensions": 2,
                    "max_input_tokens": 128, "max_batch_tokens": 256,
                    "batch_size": 1, "network_retries": 0, **changes}
        client = MultiEndpointEmbeddingClient(["http://one/v1", "http://two/v1"], Tokenizer(),
            cache_path=self.cache, identity={"weights": "same", "tokenizer": "chars"},
            settings=settings, transport=transport)
        self.addCleanup(client.close)
        return client

    def test_every_request_rotates_but_shared_cache_skips_requests(self):
        transport = Mock()
        transport.post.return_value = httpx.Response(200, json={"data": [{"index": 0, "embedding": [3, 4]}]})
        client = self.client(transport)
        result = client.encode(["a", "b", "c"])
        self.assertEqual([call.args[0] for call in transport.post.call_args_list],
                         ["http://one/v1/embeddings", "http://two/v1/embeddings", "http://one/v1/embeddings"])
        self.assertEqual(result, [[.6, .8]] * 3)
        self.assertEqual(client.encode(["a", "b", "c"]), result)
        self.assertEqual(transport.post.call_count, 3)
        plain = EmbeddingClient("http://elsewhere/v1", Tokenizer(), cache_path=self.cache,
            identity={"weights": "same", "tokenizer": "chars"}, settings=client.settings, transport=transport)
        try:
            self.assertEqual(plain.namespace, client.namespace)
            self.assertEqual(plain.encode(["a"]), [[.6, .8]])
            self.assertEqual(transport.post.call_count, 3)
        finally:
            plain.close()

    def test_total_concurrency_and_same_key_coalescing_remain_shared(self):
        lock = Lock()
        active, maximum = 0, 0

        def post(url, **kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(active, maximum)
            time.sleep(.03)
            with lock:
                active -= 1
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})

        transport = Mock()
        transport.post.side_effect = post
        client = self.client(transport, workers=2)
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: client.encode([str(i)]), range(8)))
            list(pool.map(lambda _: client.encode(["identical"]), range(8)))
        self.assertLessEqual(maximum, 2)
        self.assertGreater(maximum, 1)
        self.assertEqual(transport.post.call_count, 9)
        self.assertGreater(client.stats["coalesced"], 0)

    def test_owns_and_closes_default_http_client_exactly_once(self):
        transport = Mock()
        with patch("httpx.Client", return_value=transport):
            client = self.client()
        self.assertFalse(client._owns_client)
        client.close()
        client.close()
        transport.close.assert_called_once()

    def test_failed_initialization_closes_owned_http_client(self):
        transport = Mock()
        with patch("httpx.Client", return_value=transport), self.assertRaises(ValueError):
            MultiEndpointEmbeddingClient(["http://one/v1", "http://two/v1"], Tokenizer(),
                                         cache_path=self.cache, identity={})
        transport.close.assert_called_once()

    def test_inflight_close_leaves_owned_transport_available_until_drained(self):
        transport = Mock()
        with patch("httpx.Client", return_value=transport):
            client = self.client()
        client.pending["in-flight"] = object()
        with self.assertRaisesRegex(RuntimeError, "in flight"):
            client.close()
        transport.close.assert_not_called()
        client.pending.clear()
        client.close()
        transport.close.assert_called_once()

    def test_retry_rotates_to_the_other_replica(self):
        transport = Mock()
        transport.post.side_effect = [httpx.Response(503), httpx.Response(200, json={
            "data": [{"index": 0, "embedding": [1, 0]}]})]
        client = self.client(transport, network_retries=1)
        with patch("agmem.hierarchical.embedding.time.sleep"):
            self.assertEqual(client.encode(["retry"]), [[1, 0]])
        self.assertEqual([call.args[0] for call in transport.post.call_args_list],
                         ["http://one/v1/embeddings", "http://two/v1/embeddings"])

    def test_external_transport_is_not_owned_and_invalid_vectors_are_not_cached(self):
        transport = Mock()
        transport.post.return_value = httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 2, 3]}]})
        client = self.client(transport)
        with self.assertRaises(InfrastructureError):
            client.encode(["bad vector"])
        self.assertEqual(client.db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0], 0)
        client.close()
        transport.close.assert_not_called()

    def test_every_endpoint_is_validated_before_any_cache_write(self):
        for endpoints in ([], "http://one/v1", ["http://one/v1", "http://secret:password@two/v1"],
                          ["http://one/v1", "http://two/v1?key=secret"], ["http://one/v1"] * 2):
            with self.subTest(endpoints=endpoints), self.assertRaises(ValueError):
                MultiEndpointEmbeddingClient(endpoints, Tokenizer(), cache_path=self.cache,
                                              identity={"model": "same"})
        self.assertFalse(self.cache.exists())


class DeepSeekScriptTests(unittest.TestCase):
    def test_defaults_and_argument_forwarding_without_running_python_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "python"
            executable.write_text(f"#!{sys.executable}\n" + "import json, os, sys\n"
                "print(json.dumps({'args': sys.argv[1:], 'gpus': os.environ.get('CUDA_VISIBLE_DEVICES')}))\n")
            executable.chmod(0o755)
            env = {**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"],
                   "AGMEM_PROJECT_DIR": str(ROOT)}
            for key in ("CUDA_VISIBLE_DEVICES", "AGMEM_EMBEDDING_GPUS", "AGMEM_WRITER_WORKERS",
                        "AGMEM_EMBEDDING_MAX_NUM_SEQS"):
                env.pop(key, None)
            for override in (False, True):
                if override:
                    env.update(AGMEM_EMBEDDING_GPUS="6,7", AGMEM_WRITER_WORKERS="96",
                               AGMEM_EMBEDDING_MAX_NUM_SEQS="80")
                result = subprocess.run(["bash", str(ROOT / "scripts/build_longmemeval_m_deepseek.sh"),
                    "--dry-run", "--limit-sessions", "2", "--output-root", "/tmp/new output"],
                    check=True, env=env, capture_output=True, text=True)
                row = json.loads(result.stdout)
                args = row["args"]
                self.assertEqual(args[2], "agmem.build_deepseek_directories_gpu")
                self.assertEqual(row["gpus"], "6,7" if override else "4,5")
                self.assertEqual(args[args.index("--workers") + 1], "96" if override else "64")
                self.assertEqual(args[args.index("--embedding-max-num-seqs") + 1], "80" if override else "64")
                self.assertEqual(args[-5:], ["--dry-run", "--limit-sessions", "2", "--output-root", "/tmp/new output"])
                self.assertIn("split_longmemeval_m", args)
                self.assertNotIn("--reuse-root", args)


if __name__ == "__main__":
    unittest.main()
