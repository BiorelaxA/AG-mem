"""Package and verify LoCoMo / LongMemEval-S/M project data (Linux, GNU zstd/split)."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import io
import json
from pathlib import Path
import stat
import subprocess
import tarfile

NAMES = {"locomo", "split_longmemeval_s", "split_longmemeval_m"}
CODE = {".py", ".sh", ".sbatch", ".ipynb"}
VERSION = "agmem-three-datasets.v1"


def select(root, supplement=False):
    selected = {}
    locks = set()
    def tree(path):
        for p in sorted(path.rglob("*")):
            if p.is_symlink() or any(x.startswith("checkpoint-") or x in {"__pycache__", ".cache"} for x in p.parts):
                continue
            if p.is_file() and not p.name.endswith((".tmp", ".pyc")) and p.suffix not in CODE | {".pt", ".safetensors"}:
                selected[p] = None

    if supplement:
        tree(root/"runs/joint/locomo_query_split_seed42_v1")
        tree(root/"runs/analysis")
        for p in (root/"reports").iterdir():
            if p.name.startswith("migration_"):
                continue
            if p.is_dir():
                tree(p)
            elif p.is_file() and p.suffix not in CODE:
                selected[p] = None
        return selected, locks
    for ds in NAMES:
        tree(root/"data"/ds)
    for manifest in sorted((root/"data/processed").glob("*/manifest.json")):
        folder = manifest.parent
        datasets = {x["source"] for x in json.loads(manifest.read_text())["sources"]}
        if not datasets & NAMES:
            continue
        for ds in datasets & NAMES:
            tree(folder/ds)
        for p in folder.iterdir():
            if not p.is_file():
                continue
            if datasets <= NAMES or p.name == "manifest.json":
                selected[p] = None  # Original parent manifest hash is used by later splits.
            elif p.suffix == ".jsonl":
                with p.open("rb") as f:
                    selected[p] = b"".join(line for line in f if line.strip() and json.loads(line)["source"] in NAMES)
    for folder in sorted((root/"runs/parent_child_directories").iterdir()):
        if not folder.is_dir():
            continue
        v = json.loads((folder/"run_manifest.json").read_text())
        datasets = {d["spec"]["name"] for d in v.get("datasets", [])}
        if not datasets:  # Repair overlays are bound to a known source directory.
            origin = Path(v["source_root"]).name
            source = json.loads((folder.parent/origin/"run_manifest.json").read_text())
            datasets = {d["spec"]["name"] for d in source["datasets"]}
        if datasets <= NAMES and datasets:
            tree(folder)
            if (folder/".build.lock").is_file():
                locks.add(folder/".build.lock")
        elif datasets & NAMES:
            raise ValueError(f"Mixed directory bank needs explicit filtering: {folder}")
    runs = root/"runs"
    for folder in sorted(runs.glob("locomo*")):
        tree(folder)  # Dataset-specific prepared stores, rollouts, validation and logs.
    for folder in ("format05_best_final", "longmemeval_811_orm", "longmemeval_gpt5mini"):
        tree(runs/"evaluation"/folder)
    for p in selected:
        if p.name in {".job.lock", ".prepare_mixed.lock"}:
            locks.add(p)
    return selected, locks


def stamp(p):
    s = p.stat()
    return s.st_size, s.st_mtime_ns, s.st_ino


class HashReader:
    def __init__(self, stream):
        self.stream, self.hash = stream, hashlib.sha256()

    def read(self, n=-1):
        block = self.stream.read(n)
        self.hash.update(block)
        return block


def sha(path):
    with path.open("rb") as f:
        reader = HashReader(f)
        while reader.read(8 << 20):
            pass
    return reader.hash.hexdigest()


def pack(root, out, plan, supplement=False):
    files, locks = select(root, supplement)
    sizes = {p: stamp(p) for p in files}
    print(f"{len(files)} files, {sum(s[0] for s in sizes.values()) / 2**30:.2f} GiB before compression", flush=True)
    if plan:
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty package directory")
    out.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for lock in sorted(locks):
            f = stack.enter_context(lock.open("rb"))
            try:
                fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Active writer holds {lock}. Retry after stopping its job.") from None
        compressor = subprocess.Popen(["zstd", "-q", "-1", "-T6", "-c"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        splitter = subprocess.Popen(["split", "-b", "4G", "-d", "-a", "4", "-", str(out/"data.part-")], stdin=compressor.stdout)
        compressor.stdout.close()
        records = []
        try:
            with compressor.stdin, tarfile.open(fileobj=compressor.stdin, mode="w|", bufsize=1<<20) as tar:
                for p, filtered in sorted(files.items()):
                    rel = p.relative_to(root).as_posix()
                    size, ns, _ = sizes[p]
                    info = tarfile.TarInfo(rel)
                    info.mode = stat.S_IMODE(p.stat().st_mode)
                    info.size = size if filtered is None else len(filtered)
                    info.mtime = ns // 10**9
                    info.pax_headers = {"mtime": f"{ns//10**9}.{ns%10**9:09d}"}
                    with (p.open("rb") if filtered is None else io.BytesIO(filtered)) as f:
                        reader = HashReader(f)
                        tar.addfile(info, reader)
                    records.append({"path": rel, "bytes": info.size, "mtime_ns": ns, "sha256": reader.hash.hexdigest()})
            if compressor.wait() or splitter.wait():
                raise RuntimeError("Compression/splitting failed")
        finally:
            for proc in (compressor, splitter):
                if proc.poll() is None:
                    proc.terminate()
                proc.wait()
        after, _ = select(root, supplement)
        if after.keys() != files.keys() or any(stamp(p) != sizes[p] for p in files):
            raise RuntimeError("Source files changed. Keep writers stopped and retry into a fresh output directory.")
        (out/"FILES.json").write_text(json.dumps(records, ensure_ascii=False))
        parts = sorted(out.glob("data.part-*"))
        complete = {"version": VERSION, "parts": [p.name for p in parts], "files": len(records),
                    "input_bytes": sum(x["bytes"] for x in records)}
        (out/"COMPLETE.json").write_text(json.dumps(complete, indent=2))
        with (out/"SHA256SUMS.tmp").open("w") as f:
            for p in [*parts, out/"FILES.json", out/"COMPLETE.json"]:
                f.write(f"{sha(p)}  {p.name}\n")
        (out/"SHA256SUMS.tmp").rename(out/"SHA256SUMS")
        print("Ready:", out, flush=True)


def verify(root, out):
    manifest = json.loads((out/"COMPLETE.json").read_text())
    records = json.loads((out/"FILES.json").read_text())
    if manifest["version"] != VERSION or manifest["files"] != len(records):
        raise ValueError("Wrong or incomplete manifest")
    for row in records:
        p = root / row["path"]
        if not p.resolve().is_relative_to(root) or not p.is_file() or p.stat().st_size != row["bytes"] or sha(p) != row["sha256"]:
            raise ValueError(f"Content mismatch: {p}")
        if p.stat().st_mtime_ns != row["mtime_ns"]:
            raise ValueError(f"Timestamp mismatch: {p}")
    print(f"PASS: {len(records)} restored files, SHA256, lengths and mtimes")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=["plan", "pack", "verify"])
    cli.add_argument("--project", type=Path, default=Path.cwd())
    cli.add_argument("--package", type=Path, required=True)
    cli.add_argument("--supplement", action="store_true", help="Select dataset-derived analysis and the joint LoCoMo split")
    args = cli.parse_args()
    if args.action == "verify":
        verify(args.project.resolve(), args.package.resolve())
    else:
        pack(args.project.resolve(), args.package.resolve(), args.action == "plan", args.supplement)
