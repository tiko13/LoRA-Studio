import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from .domain import normalize_project
from .concept_preparation import upgrade_project

class Conflict(ValueError):
    pass

class PreparationLocked(Conflict):
    pass

class EmptyProjectExists(Conflict):
    def __init__(self, project):
        super().__init__('An empty LoRA project already exists.')
        self.project = project

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
            db.execute('CREATE TABLE IF NOT EXISTS training_presets (id TEXT PRIMARY KEY, model TEXT, data TEXT)')
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
            return [upgrade_project(json.loads(r[0])) for r in db.execute("SELECT data FROM projects ORDER BY updated DESC")]
    def project(self, pid):
        with self.connect() as db:
            row = db.execute("SELECT data FROM projects WHERE id=?", (pid,)).fetchone()
        if not row:
            raise KeyError("Project not found")
        return upgrade_project(json.loads(row[0]))
    def save_project(self, value, pid=None, *, guard_empty=False, selected_project_id=None):
        if not isinstance(value, dict):
            raise ValueError('Project must be an object')
        if pid is None and 'training' not in value:
            preset = next((p for p in self.presets() if p['model'] == value.get('model', 'krea2') and p.get('default')), None)
            if preset:
                value = {**value, 'training': preset['training'], 'training_preset': preset['id'],
                         'preparation': value.get('preparation', preset['preparation'])}
        if pid is None and 'trigger' not in value:
            value = {**value, 'trigger': 'lora_' + uuid.uuid4().hex[:10]}
        item = normalize_project(value)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if pid is None and guard_empty:
                # Serialize the check and insert across tabs and concurrent requests.
                row = db.execute('''SELECT p.data FROM projects p
                    WHERE NOT EXISTS (SELECT 1 FROM images i WHERE i.project=p.id)
                    ORDER BY (p.id=?) DESC, p.updated DESC, p.id LIMIT 1''',
                    (selected_project_id or '',)).fetchone()
                if row:
                    raise EmptyProjectExists(upgrade_project(json.loads(row[0])))
            if pid:
                row = db.execute("SELECT revision,data FROM projects WHERE id=?", (pid,)).fetchone()
                if not row:
                    raise KeyError("Project not found")
                self._assert_editable(json.loads(row[1]))
                if value.get("revision") != row[0]:
                    raise Conflict("This project changed in another window. Reload before saving; your local draft is retained.")
                previous = json.loads(row[1])
                item['sample_prompt_history'] = previous.get('sample_prompt_history', [])
                if previous.get('sample_prompts') and previous['sample_prompts'] != item['sample_prompts']:
                    item['sample_prompt_history'].append({'text': previous['sample_prompts'], 'at': time.time()})
                revision = row[0] + 1
            else:
                pid, revision = uuid.uuid4().hex, 1
            item.update(id=pid, revision=revision, updated=time.time())
            db.execute("INSERT OR REPLACE INTO projects VALUES (?,?,?,?)", (pid, revision, item["updated"], json.dumps(item)))
        return item
    @staticmethod
    def _assert_editable(project):
        if project.get('preparation_lock'):
            raise PreparationLocked('Preparation is locked for this training run. Use Copy to new run to make an editable project.')

    def assert_editable(self, pid):
        self._assert_editable(self.project(pid))

    def lock_preparation(self, pid, job_id, *, started=False, expected=None, images=None):
        """Reserve frozen inputs; the first optimization step makes the lock permanent."""
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT data FROM projects WHERE id=?', (pid,)).fetchone()
            if not row:
                raise KeyError('Project not found')
            project = json.loads(row[0])
            previous = project.get('preparation_lock')
            if previous and (previous.get('started') or previous['job'] == job_id and not started):
                return
            if expected is not None:
                self._assert_editable(project)
                current = [json.loads(r[0]) for r in db.execute('SELECT data FROM images WHERE project=?', (pid,))]
                if project['revision'] != expected['revision'] or {i['id']: i['revision'] for i in current} != {i['id']: i['revision'] for i in images}:
                    raise Conflict('Preparation changed before training launched. Review your changes and start again.')
            project['preparation_lock'] = {'job': job_id, 'started': started, 'at': time.time()}
            project['revision'] += 1
            db.execute('UPDATE projects SET revision=?,data=? WHERE id=?', (project['revision'], json.dumps(project), pid))

    def release_preparation(self, pid, job_id):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT data FROM projects WHERE id=?', (pid,)).fetchone()
            if not row:
                return
            project = json.loads(row[0])
            lock = project.get('preparation_lock', {})
            if lock.get('job') != job_id or lock.get('started'):
                return
            del project['preparation_lock']
            project['revision'] += 1
            db.execute('UPDATE projects SET revision=?,data=? WHERE id=?', (project['revision'], json.dumps(project), pid))
    def presets(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute('SELECT data FROM training_presets ORDER BY rowid')]
    def save_preset(self, preset):
        if not preset['name']:
            raise ValueError('Give the preset a name')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            existing = [json.loads(r[0]) for r in db.execute('SELECT data FROM training_presets WHERE model=?', (preset['model'],))]
            same = next((p for p in existing if p['source']==preset['source'] and p['name']==preset['name']), None)
            preset = {**preset, 'id': same['id'] if same else uuid.uuid4().hex, 'created': same['created'] if same else time.time()}
            if preset.get('default'):
                for old in existing:
                    old['default'] = False
                    db.execute('UPDATE training_presets SET data=? WHERE id=?', (json.dumps(old), old['id']))
            db.execute('INSERT OR REPLACE INTO training_presets VALUES (?,?,?)', (preset['id'], preset['model'], json.dumps(preset)))
        return preset
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
            if table == 'images':
                db.execute('BEGIN IMMEDIATE')
                row = db.execute('SELECT data FROM projects WHERE id=?', (item['project'],)).fetchone()
                if not row:
                    raise KeyError('Project not found')
                self._assert_editable(json.loads(row[0]))
            db.execute(f"INSERT OR REPLACE INTO {table} VALUES (?,?,?)", (item["id"], item["project"], json.dumps(item)))
