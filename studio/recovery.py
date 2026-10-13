"""Reconnect to existing trainers without launching or replacing training work."""
import sqlite3
import subprocess
import time
from contextlib import closing
from pathlib import Path

from . import processes
from .monitoring import read_json, run_config, write_json
from .results import checkpoint_info


def native_status(root, job_id):
    path = root / 'toolkit-control.sqlite3'
    if not path.is_file():
        return {}
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=1)) as db:
        db.row_factory = sqlite3.Row
        row = db.execute('SELECT status,step FROM Job WHERE id=?', (job_id,)).fetchone()
        return dict(row) if row else {}


def verified_identity(root, job):
    actual = processes.identity(job['pid'])
    if actual is None:
        return None
    expected = job.get('process_identity')
    if expected:
        return actual if processes.same_process(actual, expected) else None
    # Older runs predate creation markers. Require their exact frozen attempt
    # argument, and recheck identity around command inspection to catch PID reuse.
    args = processes.command_args(job['pid'])
    config = root / 'attempts' / f'{job.get("attempt", 1):03d}.json'
    if not config.is_file():
        config = root / 'config.json'
    if (len(args) != 3 or Path(args[1]).name != 'run.py' or
            not Path(args[2]).is_absolute() or Path(args[2]).resolve() != config.resolve() or
            not processes.same_process(processes.identity(job['pid']), actual)):
        raise ValueError('Previous trainer identity could not be confirmed. No process was stopped or started.')
    job['process_identity'] = actual
    return actual


def finish(service, job, root):
    """No exit code is available for an adopted process. Require backend evidence."""
    native = native_status(root, job['id'])
    control = read_json(root / 'monitor-control.json')
    config = run_config(root, job).get('config', {})
    process = config.get('process', [{}])[0]
    total = process.get('train', {}).get('steps')
    output = Path(process.get('training_folder', root / 'output')) / config.get('name', '')
    final = output / (config.get('name', '') + '.safetensors')
    step, ready = checkpoint_info(final) if final.resolve().is_relative_to(root.resolve()) else (None, False)
    if native.get('status') == 'completed' and ready and step == total and total:
        job.update(status='complete', phase='Training complete')
        job.pop('error', None)
    elif control.get('action') == 'pause' and control.get('checkpoint_confirmed') is not None:
        job.update(status='paused', phase='Paused at saved checkpoint', paused_step=control['checkpoint_confirmed'])
        job.pop('error', None)
    elif control.get('action') == 'cancel' or native.get('status') == 'stopped':
        job.update(status='cancelled', phase='Training stopped')
        job.pop('error', None)
    else:
        job.update(status='failed', phase='Trainer exited',
                   error='The trainer exited while Studio was disconnected. Successful completion could not be verified. Saved results are preserved; inspect the training log or resume from a complete checkpoint.')
    job.update(finished=time.time(), reconnected=False)
    service.reconnected.discard(job['id'])
    service.store.put_item('jobs', job)
    service.sync_training_lock(job)


def reconnect(service, job):
    root = service.store.root / 'runs' / job['id']
    if not job.get('pid'):
        job.update(status='interrupted', error='Studio restarted before a trainer process was recorded. Prepared files are preserved; check the run before retrying.')
        service.store.put_item('jobs', job)
        return
    try:
        actual = verified_identity(root, job)
        if actual is None:
            finish(service, job, root)
            return
        job.update(status='running', phase='Training', reconnected=True)
        job.pop('error', None)
        job.pop('finished', None)
        service.store.put_item('jobs', job)
        service.reconnected.add(job['id'])
    except (OSError, ValueError, sqlite3.Error, RuntimeError, subprocess.SubprocessError) as exc:
        job.update(status='interrupted', reconnected=False, error=str(exc))
        service.store.put_item('jobs', job)


def refresh(service):
    for job_id in list(service.reconnected):
        job = service.store.item('jobs', job_id)
        root = service.store.root / 'runs' / job_id
        if verified_identity(root, job) is None:
            finish(service, job, root)


def stop(service, job_id):
    job = service.store.item('jobs', job_id)
    root = service.store.root / 'runs' / job_id
    if verified_identity(root, job) is None:
        finish(service, job, root)
        raise ValueError('The trainer has already exited')
    # Backend IPC avoids killing a PID or unrelated child processes. It also works
    # after multiple Studio restarts and while the trainer is initializing.
    control = read_json(root / 'monitor-control.json')
    if control.get('action') != 'pause':
        write_json(root / 'monitor-control.json', {'action': 'cancel', 'requested': time.time()})
    path = root / 'toolkit-control.sqlite3'
    if not path.is_file():
        raise ValueError('This older backend has no reconnectable stop control. Wait for the trainer to finish.')
    with closing(sqlite3.connect(path)) as db, db:
        if db.execute('UPDATE Job SET stop=1 WHERE id=?', (job_id,)).rowcount != 1:
            raise ValueError('The backend control record is missing; the trainer was left running.')
    return {'requested': True}
