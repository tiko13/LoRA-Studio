"""Delete an exact checkpoint and its owned result artifacts."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from .processes import running
from .results import artifact_path, results, run_root
from .store import Conflict


@contextmanager
def checkpoint_access(service, job_id):
    # The same reservation protects training launches and comparison snapshots.
    if service.owner_url:
        raise Conflict('Open the main Studio window to manage checkpoints.')
    if not service.job_lock.acquire(blocking=False):
        raise Conflict('Wait for active work to finish or pause training before managing checkpoints.')
    try:
        with service.lock:
            job, root = run_root(service.store, job_id)
            if (service.store.root / 'runs' / job_id).absolute() != root:
                raise ValueError('Cannot delete checkpoints through linked run folders')
            if job['status'] in {'queued', 'running', 'stopping', 'interrupted'} or (job.get('pid') and running(job['pid'])):
                raise Conflict('Pause or stop this run and resolve interrupted processes before managing checkpoints.')
            yield job, root
    finally:
        service.job_lock.release()


def selected_checkpoint(store, job_id, name):
    _, root = run_root(store, job_id)
    path = artifact_path(root, name)
    if (root / name).absolute() != path:
        raise ValueError('Cannot delete linked checkpoint artifacts')
    if path.suffix.lower() != '.safetensors':
        raise ValueError('Choose a checkpoint')
    data = results(store, job_id)
    row = next((c for c in data['checkpoints'] if c['path'] == name), None)
    if row is None:
        raise ValueError('Checkpoint is no longer listed. Refresh Results.')
    return path, row, data


def latest_checkpoint(root, checkpoints):
    # A partial newer weight write cannot own the earlier completed optimizer.
    # Among complete saves, use AI Toolkit's filesystem creation-time ordering.
    complete = [c for c in checkpoints if c['ready']]
    return max((artifact_path(root, c['path']) for c in complete or checkpoints), key=lambda p: p.stat().st_ctime_ns)


def deletion_plan(service, job_id, name):
    """Enumerate exact owned artifacts; preserve shared inputs and independent sessions."""
    path, checkpoint, data = selected_checkpoint(service.store, job_id, name)
    _, root = run_root(service.store, job_id)
    targets = {path}

    def add(candidate, boundary):
        resolved = candidate.resolve()
        if not resolved.is_relative_to(boundary.resolve()) or not resolved.is_relative_to(service.store.root):
            raise ValueError('A related artifact points outside its storage folder')
        if candidate.is_symlink() or candidate.absolute() != resolved:
            raise ValueError('Cannot delete linked checkpoint artifacts')
        if candidate.is_file():
            targets.add(resolved)

    for suffix in ('.yaml', '.yml', '.json', '.txt'):
        add(path.with_suffix(suffix), path.parent)
        add(Path(str(path) + suffix), path.parent)
    # Two saves at the same step may share previews. Keep them for the surviving save.
    shared = any(c['path'] != name and c['step'] == checkpoint['step'] and
                 Path(c['path']).parent == Path(name).parent for c in data['checkpoints'])
    if not shared:
        for preview in checkpoint['previews']:
            image = artifact_path(root, preview['path'])
            add(image, root / 'output')
            add(image.with_suffix('.txt'), root / 'output')
            thumbs = image.parent / '.thumbs'
            if thumbs.exists():
                for thumb in thumbs.iterdir():
                    if thumb.name.startswith(image.stem + '.'):
                        add(thumb, root / 'output')
    siblings = [c for c in data['checkpoints'] if Path(c['path']).parent == Path(name).parent]
    # Toolkit overwrites one optimizer/SNR state at each save. It belongs to the
    # latest file in that output generation, never to an arbitrary older step.
    latest = latest_checkpoint(root, siblings)
    optimizer_removed = False
    if latest == path:
        for filename in ('optimizer.pt', 'learnable_snr.json'):
            add(path.parent / filename, path.parent)
        optimizer_removed = (path.parent / 'optimizer.pt').is_file()
    comparisons = getattr(service, 'comparisons', None)
    related = []
    directories = []
    if comparisons:
        for comparison in comparisons.list(job_id):
            if not any(c.get('path') == name for c in comparison['columns']):
                continue
            if comparison['id'] in comparisons.active or comparison['status'] in {'preparing', 'running', 'stopping', 'interrupted'}:
                raise Conflict('Finish or resolve comparisons using this checkpoint before deleting it.')
            directory = comparisons.directory(comparison['id'])
            if (comparisons.root / comparison['id']).absolute() != directory:
                raise ValueError('Cannot delete linked comparison folders')
            for file in directory.rglob('*'):
                add(file, directory)
                if file.is_dir():
                    directories.append(file.resolve())
            directories.append(directory)
            related.append(comparison['id'])
    targets = sorted(targets)
    fingerprint = [(str(p.relative_to(service.store.root)), p.stat().st_size, p.stat().st_mtime_ns) for p in targets]
    token = hashlib.sha256(json.dumps(fingerprint).encode()).hexdigest()
    return {'checkpoint': checkpoint['name'], 'files': len(targets), 'bytes': sum(p.stat().st_size for p in targets),
            'comparisons': len(related), 'optimizer': optimizer_removed,
            'shared_previews': shared, 'token': token}, targets, directories


def delete_checkpoint(service, job_id, name, token=None):
    with checkpoint_access(service, job_id):
        # Comparisons use this lock before attempting the job reservation.
        comparisons = getattr(service, 'comparisons', None)
        if comparisons:
            comparisons.lock.acquire()
        try:
            summary, targets, directories = deletion_plan(service, job_id, name)
            if token is None:
                return summary
            if token != summary['token']:
                raise Conflict('Checkpoint artifacts changed. Review deletion again.')
            trash = (service.store.root / 'checkpoint-trash' / uuid.uuid4().hex).resolve()
            if not trash.is_relative_to(service.store.root / 'checkpoint-trash'):
                raise ValueError('Invalid checkpoint cleanup location')
            trash.mkdir(parents=True)
            moved = []
            try:
                for index, path in enumerate(targets):
                    destination = trash / str(index)
                    path.replace(destination)
                    moved.append((path, destination))
            except Exception:
                for original, temporary in reversed(moved):
                    temporary.replace(original)
                raise
            finally:
                # Only this operation's verified, app-owned directory is removed.
                if not any(p.exists() for _, p in moved):
                    trash.rmdir()
            if trash.exists():
                shutil.rmtree(trash)
            for directory in sorted(set(directories), key=lambda p: len(p.parts), reverse=True):
                if directory.exists() and not any(directory.iterdir()):
                    directory.rmdir()
            return {**summary, 'deleted': True}
        finally:
            if comparisons:
                comparisons.lock.release()
