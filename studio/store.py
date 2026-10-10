import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from .domain import normalize_project

class Conflict(ValueError):
    pass

class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "studio.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, revision INTEGER, updated REAL, data TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, project TEXT, data TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS images (id TEXT PRIMARY KEY, project TEXT, data TEXT)")
    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()
    def projects(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT data FROM projects ORDER BY updated DESC")]
    def project(self, pid):
        with self.connect() as db:
            row = db.execute("SELECT data FROM projects WHERE id=?", (pid,)).fetchone()
        if not row:
            raise KeyError("Project not found")
        return json.loads(row[0])
    def save_project(self, value, pid=None):
        item = normalize_project(value)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if pid:
                row = db.execute("SELECT revision FROM projects WHERE id=?", (pid,)).fetchone()
                if not row:
                    raise KeyError("Project not found")
                if value.get("revision") != row[0]:
                    raise Conflict("This project changed in another window. Reload before saving; your local draft is retained.")
                revision = row[0] + 1
            else:
                pid, revision = uuid.uuid4().hex, 1
            item.update(id=pid, revision=revision, updated=time.time())
            db.execute("INSERT OR REPLACE INTO projects VALUES (?,?,?,?)", (pid, revision, item["updated"], json.dumps(item)))
        return item
    def settings(self):
        with self.connect() as db:
            row = db.execute("SELECT data FROM settings WHERE id=1").fetchone()
        return json.loads(row[0]) if row else {}
    def save_settings(self, data):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES (1,?)", (json.dumps(data),))
    def items(self, table, pid=None):
        assert table in {"images", "jobs"}
        with self.connect() as db:
            rows = db.execute(f"SELECT data FROM {table}" + (" WHERE project=?" if pid else ""), (pid,) if pid else ())
            return [json.loads(row[0]) for row in rows]
    def item(self, table, item_id):
        assert table in {"images", "jobs"}
        with self.connect() as db:
            row = db.execute(f"SELECT data FROM {table} WHERE id=?", (item_id,)).fetchone()
        if not row:
            raise KeyError("Item not found")
        return json.loads(row[0])
    def put_item(self, table, item):
        assert table in {"images", "jobs"}
        with self.connect() as db:
            db.execute(f"INSERT OR REPLACE INTO {table} VALUES (?,?,?)", (item["id"], item["project"], json.dumps(item)))
