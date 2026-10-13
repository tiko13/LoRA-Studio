"""Read the AI Toolkit artifacts of a frozen run without loading model tensors."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys


MEDIA = {'.png', '.jpg', '.jpeg', '.webp'}


def run_root(store, job_id):
    job = store.item('jobs', job_id)
    if job['kind'] != 'train':
        raise ValueError('Results are available for training runs only')
    runs = (store.root / 'runs').resolve()
    root = (runs / job_id).resolve()
    if not root.is_relative_to(runs):
        raise ValueError('Invalid run location')
    return job, root


def artifact_path(root, name):
    if not isinstance(name, str) or '\\' in name:
        raise ValueError('Invalid result file')
    relative = Path(name)
    path = (root / relative).resolve()
    if (relative.is_absolute() or '..' in relative.parts or not relative.parts or
            relative.parts[0] != 'output' or not path.is_relative_to(root / 'output') or
            path.suffix.lower() not in MEDIA | {'.safetensors'}):
        raise ValueError('Invalid result file')
    if not path.is_file():
        raise KeyError('Result file is no longer available')
    return path


def checkpoint_info(path):
    """Read bounded metadata and detect incomplete writes; never deserialize tensors."""
    try:
        with path.open('rb') as stream:
            size = int.from_bytes(stream.read(8), 'little')
            if not 2 <= size <= 16 * 1024 * 1024:
                return None, False
            header = json.loads(stream.read(size))
        tensors = [v for k, v in header.items() if k != '__metadata__']
        end = max(v['data_offsets'][1] for v in tensors)
        complete = 8 + size + end == path.stat().st_size
        info = header.get('__metadata__', {}).get('training_info', '{}')
        info = json.loads(info) if isinstance(info, str) else info
        step = info.get('step') if isinstance(info, dict) else None
        return step if type(step) is int and step >= 0 else None, complete
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
        return None, False


def results(store, job_id):
    job, root = run_root(store, job_id)
    from .monitoring import run_config
    try:
        config = run_config(root, job)['config']
        process = config['process'][0]
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        config, process = {}, {}
    total = process.get('train', {}).get('steps')
    samples_config = process.get('sample', {})
    checkpoints, previews = [], []
    for candidate in sorted((root / 'output').rglob('*')):
        # Toolkit .tmp files are still being written; .thumbs are cropped duplicates.
        if (any(part.startswith('.') for part in candidate.relative_to(root).parts) or
                candidate.suffix.lower() not in MEDIA | {'.safetensors'}):
            continue
        name = candidate.relative_to(root).as_posix()
        try:
            path = artifact_path(root, name)
            stat = path.stat()
        except (OSError, ValueError, KeyError):
            continue
        row = {'path': name, 'name': candidate.name, 'size': stat.st_size, 'modified': stat.st_mtime}
        if path.suffix.lower() == '.safetensors':
            step, ready = checkpoint_info(path)
            final = candidate.stem == config.get('name')
            match = re.search(r'_(\d+)$', candidate.stem)
            if step is None:
                step = int(match[1]) if match and not final else total if final and job['status'] == 'complete' else None
            checkpoints.append({**row, 'step': step, 'final': final, 'ready': ready})
        else:
            match = re.search(r'_(\d+)_(\d+)\.(?:png|jpe?g|webp)$', candidate.name, re.I)
            step, index = (int(match[1]), int(match[2])) if match else (None, None)
            specs = samples_config.get('samples') or samples_config.get('prompts') or []
            spec = specs[index] if index is not None and index < len(specs) else {}
            prompt = spec.get('prompt', '') if isinstance(spec, dict) else str(spec)
            previews.append({**row, 'step': step, 'index': index, 'prompt': prompt})
    matched = set()
    for checkpoint in checkpoints:
        # Keep baseline and unknown-step images separate. Never substitute a nearby step.
        checkpoint['previews'] = [p for p in previews if checkpoint['step'] is not None and
                                  p['step'] == checkpoint['step'] and p['step'] != 0 and
                                  Path(p['path']).parent.parent == Path(checkpoint['path']).parent]
        matched.update(p['path'] for p in checkpoint['previews'])
    checkpoints.sort(key=lambda c: (c['final'], c['step'] if c['step'] is not None else -1, c['name']), reverse=True)
    return {'job': job, 'total_steps': total, 'output_available': (root / 'output').is_dir() and
            (root / 'output').resolve().is_relative_to(root), 'checkpoints': checkpoints,
            'other_previews': [p for p in previews if p['path'] not in matched],
            'sampling_disabled': process.get('train', {}).get('disable_sampling', False)}


def open_location(store, job_id, name=None):
    _, root = run_root(store, job_id)
    folder = artifact_path(root, name).parent if name else (root / 'output').resolve()
    if not folder.is_relative_to(root) or not folder.is_dir():
        raise ValueError('The output folder is not available for this run')
    try:
        if sys.platform == 'win32':
            os.startfile(str(folder))
        else:
            subprocess.Popen(['open' if sys.platform == 'darwin' else 'xdg-open', str(folder)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as exc:
        raise ValueError('Could not open the output folder on this computer') from exc
    return {'opened': True}
