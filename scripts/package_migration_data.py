#!/usr/bin/env python3
"""Archive all data/runs/reports and root logs, excluding all checkpoint trees.

Stop all source writers before running. This does not stop jobs, upload files,
modify source data, include code/models/environments, or overwrite old packages.
Parts form ONE zstd-compressed pax tar stream and must be concatenated in order.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys


CHECKPOINT = re.compile(r"checkpoint-\d+(?:\.building)?\Z")
PREFIX = "archive.tar.zst.part-"


def snapshot(project):
    """Return deterministic metadata; never follow links out of the project."""
    records, excluded = [], []

    def visit(path):
        info = path.lstat()
        relative = path.relative_to(project).as_posix()
        if stat.S_ISDIR(info.st_mode) and CHECKPOINT.fullmatch(path.name):
            excluded.append(relative)
            return
        if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ValueError(f"Symlink/special file requires explicit review: {relative}")
        if path.suffix == ".safetensors" or path.name == "trainer.pt":
            raise ValueError(f"Weight/training state outside a recognized checkpoint: {relative}")
        directory = stat.S_ISDIR(info.st_mode)
        records.append({"path": relative, "type": "directory" if directory else "file",
                        "bytes": 0 if directory else info.st_size,
                        "mtime_ns": info.st_mtime_ns, "mode": stat.S_IMODE(info.st_mode)})
        if directory:
            for child in sorted(path.iterdir()):
                visit(child)

    for name in ("data", "runs", "reports"):
        path = project / name
        if path.exists() or path.is_symlink():
            visit(path)
    for path in sorted(project.iterdir()):
        if path.suffix in {".out", ".err", ".log"}:
            visit(path)
    if not any(row["type"] == "file" for row in records):
        raise ValueError("No source data files found")
    return records, excluded


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pack_stream(project, output, records, part_bytes, threads):
    names = output / ".tar-files.nul"
    names.write_bytes(b"".join(os.fsencode(row["path"]) + b"\0" for row in records))
    processes, parts = [], []
    try:
        with (output / "tar.log").open("wb") as tar_log, (output / "zstd.log").open("wb") as zstd_log:
            archive = subprocess.Popen([
                "tar", "--format=pax", "--no-recursion", "--null", "--verbatim-files-from",
                "-C", str(project), "-cf", "-", "--files-from", str(names)],
                stdout=subprocess.PIPE, stderr=tar_log)
            processes.append(archive)
            compressed = subprocess.Popen(["zstd", f"-T{threads}", "-3", "-q", "-c"],
                                          stdin=archive.stdout, stdout=subprocess.PIPE, stderr=zstd_log)
            processes.append(compressed)
            archive.stdout.close()
            # Hash parts as they are written; no second full-size tar is stored.
            with compressed.stdout:
                part = None
                try:
                    size = 0
                    for block in iter(lambda: compressed.stdout.read(1024 * 1024), b""):
                        offset = 0
                        while offset < len(block):
                            if part is None:
                                name = f"{PREFIX}{len(parts):05d}"
                                part = (output / name).open("xb")
                                digest, size = hashlib.sha256(), 0
                            piece = block[offset:offset + min(part_bytes - size, len(block) - offset)]
                            part.write(piece)
                            digest.update(piece)
                            offset += len(piece)
                            size += len(piece)
                            if size == part_bytes:
                                part.close()
                                parts.append({"name": name, "bytes": size, "sha256": digest.hexdigest()})
                                part = None
                    if part is not None:
                        part.close()
                        parts.append({"name": name, "bytes": size, "sha256": digest.hexdigest()})
                finally:
                    if part is not None:
                        part.close()
            codes = [process.wait() for process in processes]
            if any(codes):
                raise RuntimeError(f"Archive pipeline failed (tar/zstd={codes}); inspect logs in {output}")
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        names.unlink(missing_ok=True)
    if not parts:
        raise RuntimeError("Archive pipeline produced no parts")
    return parts


def package(project, output, *, dry_run=False, part_bytes=4 * 1024**3, threads=4):
    project = Path(project).resolve(strict=True)
    output = Path(output)
    if not output.is_absolute():
        raise ValueError("--output must be an absolute path")
    output = output.resolve()
    if output == project or project in output.parents:
        raise ValueError("Output must be outside the source project")
    if part_bytes < 1 or threads < 1:
        raise ValueError("Part size and compression threads must be positive")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output already exists and is not empty; use a fresh directory")
    records, excluded = snapshot(project)
    summary = {"source_project": str(project), "output": str(output),
               "file_count": sum(row["type"] == "file" for row in records),
               "source_bytes": sum(row["bytes"] for row in records),
               "excluded_checkpoint_directories": excluded, "part_bytes": part_bytes}
    print(json.dumps({"event": "migration_package_plan", **summary}, ensure_ascii=False), flush=True)
    if dry_run:
        return summary
    for program in ("tar", "zstd"):
        if shutil.which(program) is None:
            raise ValueError(f"Required program not found: {program}")
    output.mkdir(parents=True, exist_ok=True)
    source_list = output / "SOURCE_FILES.jsonl"
    with source_list.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    parts = pack_stream(project, output, records, part_bytes, threads)
    if snapshot(project) != (records, excluded):
        raise RuntimeError("Source changed during packaging; stop writers and use a fresh output directory")
    revision = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    status = subprocess.run(["git", "-C", str(project), "status", "--porcelain"],
                            capture_output=True, text=True)
    manifest = {"protocol": "agmem_migration_data_v1", **summary,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
                "git_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
                "archive": "one zstd-compressed pax tar stream; concatenate parts in listed order",
                "source_manifest": source_list.name, "parts": parts}
    write_json(output / "ARCHIVE_MANIFEST.json", manifest)
    checksums = [(part["sha256"], part["name"]) for part in parts]
    checksums += [(sha256_file(output / name), name)
                  for name in (source_list.name, "ARCHIVE_MANIFEST.json")]
    (output / "SHA256SUMS").write_text("".join(f"{sha}  {name}\n" for sha, name in checksums))
    write_json(output / "COMPLETE.json", {"protocol": manifest["protocol"],
               "sha256sums_sha256": sha256_file(output / "SHA256SUMS"), "part_count": len(parts)})
    print(json.dumps({"event": "migration_package_complete", "output": str(output),
                      "parts": len(parts), "compressed_bytes": sum(p["bytes"] for p in parts)}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", required=True, help="New/empty absolute directory outside the project")
    parser.add_argument("--dry-run", action="store_true", help="Inspect selection without writing or compressing")
    parser.add_argument("--part-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    package(args.project, args.output, dry_run=args.dry_run, part_bytes=args.part_bytes, threads=args.threads)


if __name__ == "__main__":
    main()
