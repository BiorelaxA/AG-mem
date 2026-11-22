"""Inspect transferred dataset splits and build journals; compare source and destination."""
import argparse
import json
import sqlite3
from pathlib import Path

import pyarrow.parquet as pq


def inventory(root):
    splits = {str(p.relative_to(root)): pq.read_metadata(p).num_rows
              for p in sorted((root/"data").rglob("*.parquet"))
              if p.parent.name in {"locomo", "split_longmemeval_s", "split_longmemeval_m"}}
    builds = {}
    for dbfile in sorted((root/"runs/parent_child_directories").glob("*/build_journal.sqlite3")):
        db = sqlite3.connect(dbfile.resolve().as_uri() + "?mode=ro", uri=True, timeout=4)
        try:
            db.execute("BEGIN")
            counts = {}
            query = "SELECT b.dataset,s.status,count(*) FROM sessions s JOIN banks b USING(bank_id) GROUP BY b.dataset,s.status"
            for dataset, status, n in db.execute(query):
                counts.setdefault(dataset, {})[status] = n
            builds[dbfile.parent.name] = {"banks": db.execute("SELECT count(*) FROM banks").fetchone()[0],
                                          "states_by_dataset": counts}
        finally:
            db.close()
    return {"parquet_rows": splits, "directory_journals": builds}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--save", type=Path)
    choice.add_argument("--compare", type=Path)
    a = parser.parse_args()
    actual = inventory(a.project.resolve())
    if a.save:
        a.save.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
    elif actual != json.loads(a.compare.read_text()):
        raise SystemExit("Dataset rows or SQLite session states differ")
    print(f"Checked {len(actual['parquet_rows'])} split files and {len(actual['directory_journals'])} directory journals")
