"""Atomic session commits and validated Writer caches, with JSONL exports."""

from collections import Counter, defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from threading import RLock

from agmem.core import atomic_json, canonical


class BuildStore:
    """One locked connection; a transaction commits each complete session.

    SQLite is an internal recovery journal. The public artifacts remain the
    records/blocks/directories/source_map/manifest/traces JSONL files in the spec.
    DELETE journaling avoids relying on WAL shared-memory support on home shares.
    """

    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.db = sqlite3.connect(self.root / "build_journal.sqlite3", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS banks (bank_id TEXT PRIMARY KEY, dataset TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions (
                bank_id TEXT NOT NULL, parent_id TEXT NOT NULL, source_hash TEXT NOT NULL,
                snapshot TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                directory TEXT, error TEXT, result_stats TEXT,
                PRIMARY KEY(bank_id,parent_id));
            CREATE TABLE IF NOT EXISTS writer_cache (key TEXT PRIMARY KEY, status TEXT NOT NULL, result TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS writer_traces (seq INTEGER PRIMARY KEY AUTOINCREMENT, payload TEXT NOT NULL);
        """)
        prior = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
        if prior is not None and json.loads(prior[0]) != identity:
            self.db.close()
            raise ValueError("Directory build identity changed; use a fresh output_root")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('identity',?)", (canonical(identity),))
        atomic_json(self.root / "run_manifest.json", identity)

    def add_bank(self, bank):
        payload = {key: value for key, value in bank.items() if key != "sessions"}
        with self.lock, self.db:
            old = self.db.execute("SELECT dataset,payload FROM banks WHERE bank_id=?", (bank["bank_id"],)).fetchone()
            if old:
                if old[0] != bank["dataset"]:
                    raise ValueError("Bank belongs to another dataset")
                previous = json.loads(old[1])
                for key in ("query_memberships", "source_map", "issues"):
                    values = {canonical(v): v for v in previous.get(key, []) + payload.get(key, [])}
                    payload[key] = [values[k] for k in sorted(values)]
            self.db.execute("INSERT OR REPLACE INTO banks VALUES (?,?,?)",
                            (bank["bank_id"], bank["dataset"], canonical(payload)))

    def add_session(self, session):
        """Persist the immutable source snapshot before any Writer request."""
        values = (session["bank_id"], session["parent_id"])
        snapshot = canonical(session)
        with self.lock, self.db:
            old = self.db.execute("SELECT snapshot FROM sessions WHERE bank_id=? AND parent_id=?", values).fetchone()
            if old and old[0] != snapshot:
                raise ValueError("An existing session ID has different immutable source content")
            self.db.execute("INSERT OR IGNORE INTO sessions(bank_id,parent_id,source_hash,snapshot) VALUES (?,?,?,?)",
                            (*values, session["source_hash"], snapshot))

    def committed(self, bank_id, parent_id):
        with self.lock:
            row = self.db.execute("SELECT status,directory FROM sessions WHERE bank_id=? AND parent_id=?",
                                  (bank_id, parent_id)).fetchone()
        return (row[0], json.loads(row[1]) if row[1] else None) if row else (None, None)

    def commit(self, session, directory, stats):
        with self.lock, self.db:
            self.db.execute("UPDATE sessions SET status=?, directory=?, error=NULL, result_stats=? "
                            "WHERE bank_id=? AND parent_id=? AND source_hash=?",
                            (directory["build"]["status"], canonical(directory), canonical(stats),
                             session["bank_id"], session["parent_id"], session["source_hash"]))

    def failed(self, session, error_type, reason):
        with self.lock, self.db:
            self.db.execute("UPDATE sessions SET status='failed',directory=NULL,error=? WHERE bank_id=? AND parent_id=?",
                            (canonical({"error_type": error_type, "reason": reason}),
                             session["bank_id"], session["parent_id"]))

    def cached(self, key):
        with self.lock:
            row = self.db.execute("SELECT result FROM writer_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache(self, key, result):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO writer_cache VALUES (?,?,?)",
                            (key, result["status"], canonical(result)))

    def trace(self, row):
        with self.lock, self.db:
            self.db.execute("INSERT INTO writer_traces(payload) VALUES (?)", (canonical(row),))

    def report(self):
        counts, datasets = Counter(), defaultdict(Counter)
        with self.lock:
            for source, status, n in self.db.execute(
                    "SELECT b.dataset,s.status,COUNT(*) FROM sessions s JOIN banks b USING(bank_id) "
                    "GROUP BY b.dataset,s.status"):
                counts[status] += n
                datasets[source][status] += n
            issues = []
            for (payload,) in self.db.execute("SELECT payload FROM banks"):
                bank = json.loads(payload)
                issues.extend({"bank_id": bank["bank_id"], **issue} for issue in bank.get("issues", []))
        for key in ("complete", "fallback", "empty", "failed", "pending"):
            counts.setdefault(key, 0)
        return {"sessions": sum(counts.values()), "states": dict(counts),
                "by_dataset": {k: dict(v) for k, v in datasets.items()}, "source_issues": issues,
                "source_failures": sum(issue.get("status") == "failed" for issue in issues)}

    @contextmanager
    def _jsonl(self, name):
        path = self.root / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            yield lambda value: stream.write(canonical(value) + "\n")
        temporary.replace(path)

    def export(self, execution=None):
        """Rebuild exports from committed state; never append duplicates on resume."""
        from contextlib import ExitStack
        with self.lock, ExitStack() as stack:
            write = {name: stack.enter_context(self._jsonl(name + ".jsonl")) for name in
                     ("records", "blocks", "directories", "source_map", "bank_manifest", "writer_traces")}
            failures = []
            for bank_id, _, payload in self.db.execute("SELECT bank_id,dataset,payload FROM banks ORDER BY bank_id"):
                bank = json.loads(payload)
                for source in bank.pop("source_map", []):
                    write["source_map"]({"bank_id": bank_id, **source})
                parents = []
                query = self.db.execute("SELECT parent_id,snapshot,status,directory,error FROM sessions "
                                        "WHERE bank_id=? ORDER BY parent_id", (bank_id,))
                for parent_id, snapshot, status, directory, error in query:
                    session = json.loads(snapshot)
                    address = {"bank_id": bank_id, "parent_id": parent_id,
                               "snapshot_id": "snapshot_" + session["source_hash"]}
                    for field, section in (("header_records", "header"), ("records", "body")):
                        for record in session[field]:
                            write["records"]({**address, "section": section, **record})
                    for block in session["blocks"]:
                        write["blocks"]({**address, **block})
                    item = {"parent_id": parent_id, "order": session["order"], "status": status,
                            "source_hash": session["source_hash"]}
                    if directory is not None:
                        directory = json.loads(directory)
                        write["directories"](directory)
                        item["directory_version"] = directory["directory_version"]
                    if error:
                        item["error"] = json.loads(error)
                    if status in {"failed", "pending"}:
                        failures.append({"bank_id": bank_id, **item})
                    parents.append(item)
                bank["parents"] = sorted(parents, key=lambda p: p["order"])
                write["bank_manifest"](bank)
            for (payload,) in self.db.execute("SELECT payload FROM writer_traces ORDER BY seq"):
                write["writer_traces"](json.loads(payload))
        report = self.report()
        report["execution"] = execution or {}
        atomic_json(self.root / "build_report.json", report)
        atomic_json(self.root / "retry_failed.json", failures + [i for i in report["source_issues"] if i.get("status") == "failed"])
        return report

    def close(self):
        with self.lock:
            self.db.close()
