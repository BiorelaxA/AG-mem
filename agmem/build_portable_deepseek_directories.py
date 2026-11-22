"""Continue transferred M directories with DeepSeek and a reference tokenizer.

Only the tokenizer assets are needed from Qwen3.5-27B. The separately installed
embedding checkpoint must match the transfer's full content proof. All identity
adapters are scoped to this process; saved source identities remain unchanged.
"""

from contextlib import contextmanager
from copy import copy, deepcopy
import os
from pathlib import Path
import subprocess
import sys
from threading import RLock

from . import build_adjacent_directories as engine
from . import build_deepseek_no_repair as worker
from . import build_deepseek_directories as deepseek
from . import build_deepseek_directories_gpu as launcher
from . import import_deepseek_directories as importer
from .core import digest, file_digest
from .portable_directory_identity import PortableIdentity


DEFAULT_MANIFEST = "migration/longmemeval_m_continuation.json"
_LOCK = RLock()


@contextmanager
def portable_worker(manifest, config_path, embedding_model_path=None):
    """Verify content before exposing narrowly scoped local identity aliases."""
    import yaml

    config = yaml.safe_load(Path(config_path).read_text())
    with _LOCK, worker._LOCK, deepseek._LOCK:
        model_identity, verify_files = engine._model_identity, importer._verify_files
        base_load, compatible = deepseek._BASE_LOAD_RECIPE, worker.compatible_identity
        portable = PortableIdentity(manifest)
        writer_path = Path(config["model_path"]).expanduser().resolve()
        reference = portable.manifest.get("reference_tokenizer_path")
        if reference is not None and writer_path != portable._project_path(reference):
            raise ValueError("Recipe model_path differs from the manifest reference_tokenizer_path")
        content = [{key: entry[key] for key in ("path", "bytes", "sha256")}
                   for entry in sorted(portable.manifest["models"]["writer_tokenizer"]["files"],
                                       key=lambda entry: entry["path"])]

        def reference_identity(path, *, allow_missing=False):
            # PortableIdentity checks the exact tokenizer file set and every
            # SHA256 before invoking this callback. No Writer weights are read.
            if Path(path).expanduser().resolve() == writer_path:
                return ({"ready": True, "model_revision": "reference-tokenizer@" + digest(content)},
                        portable.source_identity["tokenizer_id"])
            return model_identity(path, allow_missing=allow_missing)

        portable.verify_models(writer_path, embedding_model_path or config["embedding"]["model_path"],
                               reference_identity)
        proof = {"version": "content_verified_directory_relocation.v1",
                 "manifest_sha256": portable.report["manifest_sha256"],
                 "source_identity_sha256": digest(portable.source_identity),
                 "models": deepcopy(portable.report["models"])}

        def load(*args, **kwargs):
            loaded, identity = base_load(*args, **kwargs)
            loaded["source_root"] = str(portable.source_root)
            identity = deepcopy(identity)
            identity["portable_migration"] = deepcopy(proof)
            for path in (Path(__file__), Path(__file__).with_name("portable_directory_identity.py")):
                identity["implementation_hashes"][path.name] = file_digest(path)
            return loaded, identity

        def compare(source, target):
            if source != portable.source_identity:
                raise ValueError("Source journal identity differs from transfer manifest")
            source_budget, target_budget = source.get("budget_tokenizer"), target.get("budget_tokenizer")
            if not isinstance(source_budget, dict) or not isinstance(target_budget, dict):
                raise ValueError("Missing source/target budget_tokenizer identity")
            if target_budget.get("path") != str(writer_path):
                raise ValueError("Target budget_tokenizer path differs from verified reference tokenizer")
            comparison = deepcopy(target)
            comparison["budget_tokenizer"]["path"] = source_budget.get("path")
            if comparison["budget_tokenizer"] != source_budget:
                raise ValueError("Source and target budget_tokenizer semantics differ")
            return compatible(source, comparison)

        engine._model_identity = portable.model_identity
        importer._verify_files = portable.verify_files
        # This runs before NoRepair's _existing_target check, including when the
        # inference engine reloads the recipe after the CPU import creates it.
        deepseek._BASE_LOAD_RECIPE = load
        worker.compatible_identity = compare
        try:
            engine.announce("portable_model_content_verified", **portable.report["models"])
            yield portable
        finally:
            engine._model_identity, importer._verify_files = model_identity, verify_files
            deepseek._BASE_LOAD_RECIPE, worker.compatible_identity = base_load, compatible


def run(args, **factories):
    manifest = str(Path(args.migration_manifest).expanduser().resolve(strict=True))
    with portable_worker(manifest, args.config, getattr(args, "embedding_model_path", None)) as portable:
        if args.source_root is not None and Path(args.source_root).expanduser().resolve() != portable.source_root:
            raise ValueError("--source-root differs from the transferred manifest")
        selected = copy(args)
        selected.source_root = str(portable.source_root)
        return worker.run(selected, **factories)


def launch(args):
    """Import on CPU first, then route every GPU-launcher check to this worker."""
    manifest = str(Path(args.migration_manifest).expanduser().resolve(strict=True))
    launcher._validate_options(args)
    cpu_env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    with _LOCK, worker._LOCK:
        build_command = launcher.build_command

        def command(options, endpoints=None):
            result = build_command(options, endpoints)
            index = result.index("agmem.build_deepseek_directories")
            result[index:index + 1] = ["agmem.build_portable_deepseek_directories", "worker"]
            result.extend(["--migration-manifest", manifest])
            if getattr(options, "source_root", None):
                result.extend(["--source-root", str(options.source_root)])
            if getattr(options, "import_only", False):
                result.append("--import-only")
            return result

        launcher.build_command = command
        try:
            if args.dry_run or args.import_only:
                subprocess.run(command(args, args.embedding_base_urls), check=True, env=cpu_env)
                return None
            imported = copy(args)
            imported.import_only = True
            subprocess.run(command(imported), check=True, env=cpu_env)
            # Calling the GPU launcher directly avoids nesting NoRepair's own
            # module rewrite around this already portable command builder.
            return launcher.launch(args)
        finally:
            launcher.build_command = build_command


def argument_parser(*, worker_mode=False):
    parser = worker.argument_parser(worker_mode=worker_mode)
    parser.description = __doc__
    parser.add_argument("--migration-manifest", default=DEFAULT_MANIFEST)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    worker_mode = bool(argv and argv[0] == "worker")
    if worker_mode:
        argv.pop(0)
    args = argument_parser(worker_mode=worker_mode).parse_args(argv)
    (run if worker_mode else launch)(args)


if __name__ == "__main__":
    main()
