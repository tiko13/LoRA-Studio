"""Training preparation lifecycle and independent project copies."""
import copy
import json
import shutil
import time
import uuid
import re
from contextlib import nullcontext

from . import sample_prompts
from .monitoring import parse_log, read_json, write_json
from .processes import running
from .store import Conflict


def _owned_path(root, *parts):
    path = root.joinpath(*parts)
    if path.resolve() != path.absolute() or not path.resolve().is_relative_to(root) or path == root:
        raise ValueError('Cannot delete session files through linked or external folders')
    return path


def _check_tree(path):
    # Reject links and Windows junctions before moving or recursively removing anything.
    if path.is_symlink() or (path.exists() and getattr(path.lstat(), 'st_file_attributes', 0) & 0x400):
        raise ValueError('Cannot delete linked session files')
    if path.is_dir():
        for child in path.iterdir():
            _check_tree(child)


def delete_project(service, pid, value):
    """Remove a session, optionally removing only its Studio-owned artifacts."""
    if type(value.get('delete_files')) is not bool or type(value.get('revision')) is not int:
        raise ValueError('Choose whether to delete files and supply the session revision')
    if service.owner_url:
        raise Conflict('Open the main Studio window to delete sessions.')
    if not service.job_lock.acquire(blocking=False):
        raise Conflict('Wait for active work to finish before deleting a session.')
    store = service.store
    comparisons = getattr(service, 'comparisons', None)
    moved, trash = [], None
    try:
        with service.lock, comparisons.lock if comparisons else nullcontext():
            try:
                with store.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    row = db.execute('SELECT revision FROM projects WHERE id=?', (pid,)).fetchone()
                    if not row:
                        raise KeyError('Project not found')
                    if row[0] != value['revision']:
                        raise Conflict('This session changed. Close the popup and review deletion again.')
                    jobs = [json.loads(r[0]) for r in db.execute('SELECT data FROM jobs WHERE project=?', (pid,))]
                    image_ids = [r[0] for r in db.execute('SELECT id FROM images WHERE project=?', (pid,))]
                    for job in jobs:
                        if (job['status'] in {'queued', 'running', 'stopping'} or
                                job['status'] == 'interrupted' and job['kind'] == 'train' or
                                job['id'] in service.cancel or job['id'] in service.reconnected or
                                job.get('pid') and running(job['pid'])):
                            raise Conflict('Stop active work and resolve interrupted runs before deleting this session.')
                    related = [c for job in jobs for c in (comparisons.list(job['id']) if comparisons else [])]
                    if any(c['id'] in comparisons.active or c['status'] in {'preparing', 'running', 'stopping', 'interrupted'} for c in related):
                        raise Conflict('Finish or resolve this session’s comparisons before deleting it.')
                    if value['delete_files']:
                        ids = [pid, *(j['id'] for j in jobs), *(c['id'] for c in related)]
                        if any(not re.fullmatch(r'[A-Za-z0-9_-]+', identity) for identity in ids):
                            raise ValueError('Invalid session artifact identity')
                        targets = [_owned_path(store.root, 'images', pid),
                                   _owned_path(store.root, 'sample-prompts', pid + '.json'),
                                   _owned_path(store.root, 'sample-prompts', pid + '.json.tmp')]
                        targets += [_owned_path(store.root, folder, j['id']) for j in jobs
                                    for folder in ('runs', 'analysis', 'caption-jobs')]
                        targets += [_owned_path(store.root, 'comparisons', c['id']) for c in related]
                        for path in targets:
                            _check_tree(path)
                        trash = _owned_path(store.root, 'session-trash', uuid.uuid4().hex)
                        trash.mkdir(parents=True)
                        for index, path in enumerate(targets):
                            if path.exists():
                                destination = trash / str(index)
                                path.replace(destination)
                                moved.append((path, destination))
                    db.execute('DELETE FROM images WHERE project=?', (pid,))
                    db.execute('DELETE FROM jobs WHERE project=?', (pid,))
                    db.execute('DELETE FROM projects WHERE id=?', (pid,))
            except Exception:
                for original, temporary in reversed(moved):
                    temporary.replace(original)
                if trash is not None and trash.exists():
                    trash.rmdir()
                raise
            service.metadata_cache = {key: data for key, data in service.metadata_cache.items() if key[0] not in image_ids}
            cleanup_pending = False
            if trash is not None:
                try:
                    _check_tree(trash)
                    shutil.rmtree(trash)
                except (OSError, ValueError):
                    cleanup_pending = True
            return {'deleted': True, 'delete_files': value['delete_files'], 'image_ids': image_ids,
                    'cleanup_pending': cleanup_pending}
    finally:
        service.job_lock.release()


def sync_training_lock(store, job, parsed=None):
    if job['kind'] != 'train':
        return
    if store.project(job['project']).get('preparation_lock', {}).get('started'):
        return
    root = store.root / 'runs' / job['id']
    if parsed is None:
        config = read_json(root / 'config.json')
        initial = config.get('config', {}).get('process', [{}])[0].get('train', {}).get('steps', 0)
        total = job.get('continuations', [{}])[-1].get('total', initial) if job.get('continuations') else initial
        path = root / 'training.log'
        text = ''
        if path.exists():
            with path.open('rb') as stream:
                offset = max(0, path.stat().st_size - 4 * 1024 * 1024)
                stream.seek(offset)
                text = stream.read().decode('utf-8', errors='replace')
                if offset:
                    initial = total
        parsed = parse_log(text, total, initial_total=initial)
    if parsed.get('training_started'):
        store.lock_preparation(job['project'], job['id'], started=True)
    elif job['status'] in {'failed', 'cancelled', 'complete'}:
        store.release_preparation(job['project'], job['id'])


def copy_project(service, pid):
    """Copy all preparation, including excluded originals, without sharing mutable files."""
    store = service.store
    created_files = []
    image_dir = None
    try:
        with service.lock, store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT data FROM projects WHERE id=?', (pid,)).fetchone()
            if not row:
                raise KeyError('Project not found')
            source = json.loads(row[0])
            images = [json.loads(r[0]) for r in db.execute('SELECT data FROM images WHERE project=? ORDER BY rowid', (pid,))]
            project = copy.deepcopy(source)
            project.pop('preparation_lock', None)
            project.update(id=uuid.uuid4().hex, revision=1, updated=time.time(),
                           name=source['name'][:113] + ' (copy)')
            copied = []
            for original in images:
                item = copy.deepcopy(original)
                item.update(id=uuid.uuid4().hex, project=project['id'], revision=1)
                path = service.image_path(item)
                image_dir = path.parent
                image_dir.mkdir(parents=True, exist_ok=True)
                created_files.append(path)
                shutil.copyfile(service.image_path(original), path)
                copied.append(item)
            cache = read_json(store.root / 'sample-prompts' / (pid + '.json'))
            if cache:
                # IDs change, but the caption/intent evidence is identical. Preserve a
                # valid suite as valid and leave already-stale suites stale.
                if cache.get('signature') == sample_prompts.signature(source, images):
                    cache['signature'] = sample_prompts.signature(project, copied)
                path = store.root / 'sample-prompts' / (project['id'] + '.json')
                created_files.extend([path, path.with_suffix('.json.tmp')])
                write_json(path, cache)
            db.execute('INSERT INTO projects VALUES (?,?,?,?)',
                       (project['id'], project['revision'], project['updated'], json.dumps(project)))
            for item in copied:
                db.execute('INSERT INTO images VALUES (?,?,?)', (item['id'], project['id'], json.dumps(item)))
        return project
    except Exception:
        for path in created_files:
            path.unlink(missing_ok=True)
        if image_dir is not None and image_dir.exists():
            image_dir.rmdir()
        raise
