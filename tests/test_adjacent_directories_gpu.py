"""Deployment routing and process ownership without model loads or GPU allocation."""

from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from agmem import build_adjacent_directories_gpu as launcher


class Process:
    counter = 2500

    def __init__(self, kind, code=0):
        Process.counter += 1
        self.pid, self.kind = Process.counter, kind
        self.returncode = None if kind in {"writer", "embedding"} else code

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode


class AdjacentDirectoryLauncherTests(unittest.TestCase):
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

    def run_launch(self, args, *, preflight_code=0, worker_code=0,
                   embedding_error=None, writer_error=None, expected_dimensions=2560):
        calls, stopped = [], []

        def start(command, **kwargs):
            kind = ("embedding" if "pooling" in command else
                    "writer" if "vllm.entrypoints.openai.api_server" in command else
                    "worker" if any(module in command for module in (
                        "agmem.build_adjacent_directories", "agmem.repair_adjacent_fallbacks",
                        "agmem.build_topic_event_directories")) else "preflight")
            process = Process(kind, preflight_code if kind == "preflight" else worker_code)
            calls.append((kind, command, kwargs, process))
            return process

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,4,6"}, clear=True))
            stack.enter_context(patch.object(launcher, "load_recipe", return_value={
                "model_path": "/writer", "writer": {"context_tokens": 65536},
                "embedding": {"workers": 4, "batch_size": 32,
                              "expected_dimensions": expected_dimensions}}))
            stack.enter_context(patch.object(launcher, "local_models", return_value=("/writer", "/embedding")))
            stack.enter_context(patch.object(launcher.subprocess, "run"))
            stack.enter_context(patch.object(launcher.subprocess, "Popen", side_effect=start))
            stack.enter_context(patch.object(launcher, "free_port", side_effect=[10100, 10101, 10102]))
            embedding_wait = stack.enter_context(patch.object(
                launcher, "wait_embedding_ready", side_effect=embedding_error))
            stack.enter_context(patch.object(launcher, "wait_ready", side_effect=writer_error))
            stack.enter_context(patch.object(launcher, "stop_group", side_effect=stopped.append))
            stack.enter_context(patch.object(launcher.signal, "signal"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            try:
                result = launcher.launch(args)
            except BaseException as exc:
                result = exc
            if embedding_wait.called:
                self.assertEqual(embedding_wait.call_args.kwargs["expected_dimensions"], expected_dimensions)
        return calls, stopped, result

    def test_gpu_roles_and_bounded_concurrency_are_forwarded(self):
        calls, stopped, result = self.run_launch(self.args())
        self.assertIsInstance(result, Path)
        model_calls = [row for row in calls if row[0] in {"writer", "embedding"}]
        self.assertEqual([(row[0], row[2]["env"]["CUDA_VISIBLE_DEVICES"]) for row in model_calls],
                         [("embedding", "2"), ("writer", "4"), ("writer", "6")])
        embed_cmd = model_calls[0][1]
        self.assertEqual(embed_cmd[embed_cmd.index("--runner") + 1], "pooling")
        self.assertEqual(embed_cmd[embed_cmd.index("--convert") + 1], "embed")
        self.assertNotIn("--enable-chunked-prefill", embed_cmd)
        for _, command, options, _ in model_calls[1:]:
            self.assertEqual(command[command.index("--max-num-seqs") + 1], "64")
            self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "1")
            self.assertEqual(options["env"]["HF_HUB_OFFLINE"], "1")
            self.assertTrue(options["start_new_session"])
        worker = next(row for row in calls if row[0] == "worker")
        self.assertEqual(worker[2]["env"]["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(worker[1][-5:], ["--base-urls", "http://127.0.0.1:10101/v1",
                                        "http://127.0.0.1:10102/v1", "--embedding-base-url",
                                        "http://127.0.0.1:10100/v1"])
        self.assertEqual(stopped, [row[3] for row in reversed(calls)])
        deployment = json.loads((result / "deployment.json").read_text())
        self.assertEqual(deployment["writer_workers"], 128)
        self.assertEqual(deployment["embedding"]["client_concurrency"], 4)
        self.assertEqual(deployment["embedding"]["client_batch_size"], 32)
        self.assertEqual(deployment["embedding"]["expected_dimensions"], 2560)

    def test_explicit_legacy_dimensions_are_forwarded_to_probe(self):
        _, _, result = self.run_launch(self.args(), expected_dimensions=1024)
        self.assertIsInstance(result, Path)
        deployment = json.loads((result / "deployment.json").read_text())
        self.assertEqual(deployment["embedding"]["expected_dimensions"], 1024)

    def test_topic_event_routes_preflight_and_worker_to_new_version(self):
        calls, _, result = self.run_launch(self.args(topic_event=True))
        self.assertIsInstance(result, Path)
        worker = next(row for row in calls if row[0] == "worker")
        self.assertIn("agmem.build_topic_event_directories", worker[1])
        dry_command = launcher.build_command(self.args(topic_event=True, dry_run=True))
        self.assertIn("agmem.build_topic_event_directories", dry_command)
        self.assertIn("--dry-run", dry_command)

    def test_topic_event_rejects_legacy_repair_or_import_before_starting(self):
        for options in ({"repair_fallbacks": True}, {"reuse_root": "old", "reuse_mapping": "map"}):
            with self.subTest(options=options), patch.object(launcher.subprocess, "run") as run, \
                    patch.object(launcher.subprocess, "Popen") as start, \
                    self.assertRaisesRegex(ValueError, "separate build"):
                launcher.launch(self.args(topic_event=True, dry_run=True, **options))
            run.assert_not_called()
            start.assert_not_called()

    def test_incremental_topic_event_import_then_build_uses_distinct_modules(self):
        args = self.args(topic_event=True, preserve_existing=True, reuse_root="old",
                         reuse_mapping="mapping.jsonl", dry_run=True)
        with patch.object(launcher.subprocess, "run") as run, \
                patch.object(launcher.subprocess, "Popen") as start, redirect_stdout(io.StringIO()):
            launcher.launch(args)
        self.assertEqual(run.call_count, 2)
        self.assertIn("agmem.reuse_topic_event_directories", run.call_args_list[0].args[0])
        self.assertIn("agmem.build_topic_event_directories", run.call_args_list[1].args[0])
        for call in run.call_args_list:
            self.assertIn("--dry-run", call.args[0])
        start.assert_not_called()

    def test_incremental_policy_requires_explicit_topic_and_source(self):
        for options in ({}, {"topic_event": True},
                        {"reuse_root": "old", "reuse_mapping": "mapping.jsonl"}):
            with self.subTest(options=options), patch.object(launcher.subprocess, "run") as run, \
                    patch.object(launcher.subprocess, "Popen") as start, \
                    self.assertRaisesRegex(ValueError, "preserve-existing requires"):
                launcher.launch(self.args(preserve_existing=True, dry_run=True, **options))
            run.assert_not_called()
            start.assert_not_called()

    def test_invalid_dimensions_fail_before_deployment(self):
        for dimensions in (0, -1, True, "2560"):
            with self.subTest(dimensions=dimensions):
                calls, stopped, result = self.run_launch(self.args(), expected_dimensions=dimensions)
                self.assertIsInstance(result, ValueError)
                self.assertIn("expected_dimensions", str(result))
                self.assertEqual(calls, [])
                self.assertEqual(stopped, [])

    def test_preflight_failure_never_starts_models(self):
        calls, stopped, result = self.run_launch(self.args(), preflight_code=1)
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("preflight", str(result))
        self.assertEqual([row[0] for row in calls], ["preflight"])
        self.assertEqual(stopped, [calls[0][3]])

    def test_embedding_failure_cleans_up_only_started_groups(self):
        calls, stopped, result = self.run_launch(self.args(), embedding_error=RuntimeError("embedding startup"))
        self.assertIsInstance(result, RuntimeError)
        self.assertEqual([row[0] for row in calls], ["preflight", "embedding"])
        self.assertEqual(stopped, [row[3] for row in reversed(calls)])

    def test_second_writer_failure_also_stops_embedding(self):
        calls, stopped, result = self.run_launch(self.args(), writer_error=[None, RuntimeError("writer startup")])
        self.assertIsInstance(result, RuntimeError)
        self.assertEqual([row[0] for row in calls], ["preflight", "embedding", "writer", "writer"])
        self.assertEqual(stopped, [row[3] for row in reversed(calls)])

    def test_builder_failure_preserves_logs_and_releases_models(self):
        calls, stopped, result = self.run_launch(self.args(), worker_code=1)
        self.assertIsInstance(result, RuntimeError)
        self.assertIn("Rerun to resume", str(result))
        self.assertEqual(stopped, [row[3] for row in reversed(calls)])
        self.assertTrue((self.root / "job" / "deployment.json").exists())

    def test_dry_run_and_external_services_never_load_models_or_touch_cuda(self):
        for args in (self.args(dry_run=True), self.args(base_urls=["http://localhost:8000/v1"],
                                                       embedding_base_url="http://localhost:8001/v1")):
            with self.subTest(args=args), patch.dict(os.environ, {}, clear=True), \
                    patch.object(launcher.subprocess, "run") as run, \
                    patch.object(launcher.subprocess, "Popen") as start, \
                    patch.object(launcher, "local_models") as models:
                launcher.launch(args)
                run.assert_called_once()
                start.assert_not_called()
                models.assert_not_called()

    def test_partial_external_service_configuration_is_rejected(self):
        for args in (self.args(base_urls=["http://localhost:8000/v1"]),
                     self.args(embedding_base_url="http://localhost:8001/v1")):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "both"):
                launcher.launch(args)

    def test_overcommit_requires_explicit_server_limit_change(self):
        with self.assertRaisesRegex(ValueError, "combined max_num_seqs"):
            launcher.launch(self.args(workers=129))
        launcher._validate_options(self.args(workers=128, max_num_seqs=64))
        for changes in ({"embedding_max_model_len": 32769},
                        {"embedding_max_num_batched_tokens": 4096},
                        {"embedding_gpu_memory_utilization": 0}, {"limit_sessions": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                launcher._validate_options(self.args(**changes))

    def test_builder_overrides_and_dataset_scope(self):
        args = self.args(dry_run=True, output_root="/tmp/new build", limit_sessions=4,
                         embedding_model_path="/tmp/encoder path", datasets=["locomo"], splits=["test"])
        cmd = launcher.build_command(args)
        self.assertIn("agmem.build_adjacent_directories", cmd)
        self.assertEqual(cmd[cmd.index("--output-root") + 1], "/tmp/new build")
        self.assertEqual(cmd[cmd.index("--embedding-model-path") + 1], "/tmp/encoder path")
        self.assertEqual(cmd[cmd.index("--limit-sessions") + 1], "4")
        self.assertEqual(cmd[-1], "--dry-run")
        with redirect_stdout(io.StringIO()), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            launcher.argument_parser().parse_args(["--datasets", "split_longmemeval_m"])

    def test_repair_worker_routing_preserves_scope_and_service_arguments(self):
        args = self.args(repair_fallbacks=True,
                         config="configs/build_adjacent_qwen_locomo_lmes_parent400.yaml",
                         datasets=["split_longmemeval_s"], splits=["train", "test"],
                         output_root="/tmp/repair directory", limit_sessions=3, dry_run=True)
        command = launcher.build_command(args, ["http://writer/v1"], "http://encoder/v1")
        self.assertIn("agmem.repair_adjacent_fallbacks", command)
        self.assertNotIn("agmem.build_adjacent_directories", command)
        self.assertNotIn("--repair-fallbacks", command)
        self.assertEqual(command[command.index("--splits") + 1:command.index("--splits") + 3],
                         ["train", "test"])
        self.assertEqual(command[command.index("--output-root") + 1], "/tmp/repair directory")
        self.assertEqual(command[command.index("--limit-sessions") + 1], "3")
        self.assertEqual(command[-5:], ["--base-urls", "http://writer/v1",
                                      "--embedding-base-url", "http://encoder/v1", "--dry-run"])

    def test_repair_launch_still_owns_three_services_and_cleans_up(self):
        calls, stopped, result = self.run_launch(self.args(repair_fallbacks=True))
        self.assertIsInstance(result, Path)
        worker = next(row for row in calls if row[0] == "worker")
        self.assertIn("agmem.repair_adjacent_fallbacks", worker[1])
        self.assertEqual([row[0] for row in calls], ["preflight", "embedding", "writer", "writer", "worker"])
        self.assertEqual(stopped, [row[3] for row in reversed(calls)])

    def test_reuse_dry_run_checks_mapping_before_builder_without_starting_services(self):
        args = self.args(dry_run=True, reuse_root="/old journal", reuse_mapping="/new mapping.jsonl",
                         datasets=["split_longmemeval_s"], splits=["train", "validation", "test"],
                         output_root="/new journal", embedding_model_path="/encoder")
        with patch.object(launcher.subprocess, "run") as run, \
                patch.object(launcher.subprocess, "Popen") as start, redirect_stdout(io.StringIO()):
            launcher.launch(args)
        self.assertEqual(run.call_count, 2)
        importer, builder = [call.args[0] for call in run.call_args_list]
        self.assertIn("agmem.reuse_adjacent_directories", importer)
        self.assertIn("agmem.build_adjacent_directories", builder)
        self.assertEqual(importer[importer.index("--reuse-root") + 1], "/old journal")
        self.assertEqual(importer[importer.index("--mapping") + 1], "/new mapping.jsonl")
        self.assertEqual(importer[importer.index("--output-root") + 1], "/new journal")
        self.assertEqual(importer[-1], "--dry-run")
        self.assertNotIn("--reuse-root", builder)
        start.assert_not_called()

    def test_reuse_requires_mapping_and_cannot_mix_with_in_place_repair(self):
        for changes in ({"reuse_root": "/old"}, {"reuse_mapping": "/map"},
                        {"reuse_root": "/old", "reuse_mapping": "/map", "repair_fallbacks": True}):
            with self.subTest(changes=changes), patch.object(launcher.subprocess, "run") as run, \
                    self.assertRaises(ValueError):
                launcher.launch(self.args(dry_run=True, **changes))
            run.assert_not_called()

    def test_failed_reuse_prevents_build_and_model_start(self):
        args = self.args(dry_run=True, reuse_root="/old", reuse_mapping="/map")
        with patch.object(launcher.subprocess, "run", side_effect=RuntimeError("changed source")) as run, \
                patch.object(launcher.subprocess, "Popen") as start, redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, "changed source"):
            launcher.launch(args)
        run.assert_called_once()
        start.assert_not_called()

    def test_missing_encoder_has_actionable_message_before_cuda(self):
        writer = self.root / "writer"
        writer.mkdir()
        for filename in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"):
            (writer / filename).write_text("{}")
        args = self.args(embedding_model_path=str(self.root / "missing"))
        with self.assertRaisesRegex(ValueError, "download_directory_embedding.sh"):
            launcher.local_models(args, {"model_path": str(writer)})

    def test_server_input_budget_must_cover_client_budget_before_deployment(self):
        calls, stopped, result = self.run_launch(self.args(embedding_max_model_len=2048))
        self.assertIsInstance(result, ValueError)
        self.assertIn("undersized embedding service", str(result))
        self.assertEqual(calls, [])
        self.assertEqual(stopped, [])

    def test_download_reuses_pinned_revision_and_does_not_touch_identical_metadata(self):
        script = Path("scripts/download_directory_embedding.sh").read_text()
        program = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        target = self.root / "encoder"
        target.mkdir()
        metadata_path = target / "agmem_download.json"
        metadata_path.write_text(json.dumps({"repo_id": "Qwen/Qwen3-Embedding-4B", "revision": "pinned-commit"}))
        os.utime(metadata_path, ns=(10**15, 10**15))
        prior = metadata_path.stat().st_mtime_ns
        calls = []
        hub = ModuleType("huggingface_hub")

        class Api:
            def model_info(self, repo, revision):
                calls.append((repo, revision))
                return SimpleNamespace(sha=revision)

        hub.HfApi = Api
        hub.snapshot_download = lambda **kwargs: None
        with patch.dict("sys.modules", {"huggingface_hub": hub}), \
                patch.dict(os.environ, {}, clear=True), patch("sys.argv", ["download", str(target)]), \
                redirect_stdout(io.StringIO()):
            exec(compile(program, "download_directory_embedding.sh", "exec"), {})
        self.assertEqual(calls, [("Qwen/Qwen3-Embedding-4B", "pinned-commit")])
        self.assertEqual(metadata_path.stat().st_mtime_ns, prior)
        with patch.dict("sys.modules", {"huggingface_hub": hub}), \
                patch.dict(os.environ, {"AGMEM_EMBEDDING_REVISION": "new-commit"}, clear=True), \
                patch("sys.argv", ["download", str(target)]), redirect_stdout(io.StringIO()):
            exec(compile(program, "download_directory_embedding.sh", "exec"), {})
        self.assertEqual(calls[-1], ("Qwen/Qwen3-Embedding-4B", "new-commit"))
        self.assertEqual(json.loads(metadata_path.read_text())["revision"], "new-commit")

    def test_embedding_probe_uses_embeddings_endpoint_and_checks_dimensions(self):
        class Response(io.StringIO):
            status = 200

        class Client:
            def __init__(self, vector):
                self.vector, self.requests = vector, []

            def open(self, request, timeout):
                self.requests.append(request)
                if isinstance(request, str):
                    return Response("")
                return Response(json.dumps({"data": [{"index": 0, "embedding": self.vector}]}))

        for valid, dimensions, vector in ((True, 2560, [1.0] * 2560), (False, 2560, [1.0] * 1024),
                                         (False, 2560, [1.0] * 12), (False, 2560, [0.0] * 2560),
                                         (False, 2560, [float("nan")] * 2560),
                                         (True, 1024, [1.0] * 1024)):
            client = Client(vector)
            with self.subTest(valid=valid, dimensions=dimensions, length=len(vector)), \
                    patch.object(launcher.urllib.request, "build_opener", return_value=client):
                if valid:
                    launcher.wait_embedding_ready(Process("embedding"), 8001, Path("embedding.log"),
                                                  expected_dimensions=dimensions)
                else:
                    with self.assertRaisesRegex(RuntimeError, f"invalid {dimensions}-dimensional"):
                        launcher.wait_embedding_ready(Process("embedding"), 8001, Path("embedding.log"),
                                                      expected_dimensions=dimensions)
                request = client.requests[-1]
                self.assertEqual(request.full_url, "http://127.0.0.1:8001/v1/embeddings")
                self.assertEqual(json.loads(request.data)["model"], "agmem-embedding")


if __name__ == "__main__":
    unittest.main()
