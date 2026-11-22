"""Export this project's Codex CLI rollout files privately, preserving the sessions layout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile


def export(project, home, output, aliases):
    roots = {project.resolve(), *[p.resolve() for p in aliases]}
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Choose a new, empty private output directory")
    records = []
    with tempfile.TemporaryDirectory(dir=output) as tmp:
        stage = Path(tmp)
        for collection in ("sessions", "archived_sessions"):
            for path in sorted((home/collection).rglob("rollout-*.jsonl")):
                with path.open("rb") as src:
                    size = os.fstat(src.fileno()).st_size
                    try:
                        meta = json.loads(src.readline())["payload"]
                    except (KeyError, ValueError):
                        continue
                    if not meta.get("cwd") or not any(Path(meta["cwd"]).resolve().is_relative_to(root) for root in roots):
                        continue
                    rel = path.relative_to(home)
                    dest = stage/rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    src.seek(0)
                    length = 0
                    checksum = hashlib.sha256()
                    with dest.open("wb") as target:
                        for line in src:
                            if length + len(line) > size or not line.endswith(b"\n"):
                                break
                            target.write(line)
                            checksum.update(line)
                            length += len(line)
                    if not length:
                        raise ValueError(f"No complete session records: {rel}")
                    records.append((rel.as_posix(), checksum.hexdigest(), meta["id"]))
        if not records:
            raise ValueError("No matching project sessions")
        with (stage/"SESSION_SHA256SUMS").open("w") as f:
            for name, sha, _ in records:
                f.write(f"{sha}  {name}\n")
        with tarfile.open(output/"project-sessions.tar.gz", "x:gz", compresslevel=1) as archive:
            for name, _, _ in records:
                archive.add(stage/name, arcname=name)
            archive.add(stage/"SESSION_SHA256SUMS", arcname="SESSION_SHA256SUMS")
    with (output/"project-sessions.tar.gz").open("rb") as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    (output/"SHA256SUMS").write_text(f"{h.hexdigest()}  project-sessions.tar.gz\n")
    print(f"Exported {len(records)} project rollouts to {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path.cwd())
    p.add_argument("--codex-dir", type=Path, default=Path(os.environ.get("CODEX_HOME", Path.home()/".codex")))
    p.add_argument("--also-cwd", action="append", default=[], type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    export(a.project, a.codex_dir, a.output, a.also_cwd)
