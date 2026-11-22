"""Relocate a frozen M build using verified content, without editing source or models."""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
from threading import RLock

from . import build_adjacent_directories as engine
from . import build_qwen_continued_directories as worker
from . import build_qwen_continued_directories_gpu as launcher
from . import import_deepseek_directories as importer
from .core import digest, file_digest
from .portable_directory_identity import PortableIdentity


DEFAULT_MANIFEST = "migration/longmemeval_m_continuation.json"
_LOCK = RLock()


@contextmanager
def portable_worker(manifest, config_path, embedding_model_path=None):
    """Scope verified identity aliases and dataset relocation to this process."""
    import yaml

    config = yaml.safe_load(Path(config_path).read_text())
    with _LOCK:
        model_identity, verify_files, base_load = engine._model_identity, importer._verify_files, worker._BASE_LOAD_RECIPE
        portable = PortableIdentity(manifest)
        portable.verify_models(config["model_path"], embedding_model_path or config["embedding"]["model_path"],
                               model_identity)
        proof = {"version": "content_verified_directory_relocation.v1",
                 "manifest_sha256": file_digest(manifest),
                 "source_identity_sha256": digest(portable.manifest["source_identity"]),
                 "models": deepcopy(portable.report["models"])}

        def load(*args, **kwargs):
            config, identity = base_load(*args, **kwargs)
            identity["portable_migration"] = proof
            for path in (Path(__file__), Path(__file__).with_name("portable_directory_identity.py")):
                identity["implementation_hashes"][path.name] = file_digest(path)
            return config, identity

        engine._model_identity = portable.model_identity
        importer._verify_files = portable.verify_files
        worker._BASE_LOAD_RECIPE = load
        try:
            engine.announce("portable_model_content_verified", **portable.report["models"])
            yield portable
        finally:
            engine._model_identity, importer._verify_files, worker._BASE_LOAD_RECIPE = model_identity, verify_files, base_load


def run(args, **factories):
    manifest = str(Path(args.migration_manifest).resolve(strict=True))
    with portable_worker(manifest, args.config, getattr(args, "embedding_model_path", None)) as portable:
        expected_source = (portable.project_root / portable.manifest["source_root_relative"]).resolve()
        if args.source_root is not None and Path(args.source_root).resolve() != expected_source:
            raise ValueError("--source-root differs from the transferred manifest")
        args.source_root = str(expected_source)
        return worker.run(args, **factories)


def launch(args):
    manifest = str(Path(args.migration_manifest).resolve(strict=True))
    with _LOCK:
        build_command = launcher.build_command

        def command(*values, **kwargs):
            result = build_command(*values, **kwargs)
            index = result.index("agmem.build_qwen_continued_directories")
            result[index:index + 1] = ["agmem.build_portable_qwen_directories", "worker"]
            return [*result, "--migration-manifest", manifest]

        launcher.build_command = command
        try:
            return launcher.launch(args)
        finally:
            launcher.build_command = build_command


def argument_parser(*, worker_mode=False):
    parser = worker.argument_parser() if worker_mode else launcher.argument_parser()
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
