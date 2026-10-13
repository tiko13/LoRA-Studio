"""Read-only training telemetry and checkpoint-aware lifecycle coordination.

AI Toolkit owns checkpoints and optimizer state. All derived data stays in a run.
"""
import csv
from contextlib import closing
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import threading
import time
import zipfile

ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
NUMBER = r'(?:[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf)'
PROGRESS = re.compile(r'(\d+)/(\d+)\s*\[([^\]]+)\]')
METRIC = re.compile(r'([\w/]+):\s*(' + NUMBER + r')', re.I)


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {} if default is None else default


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def seconds(value):
    try:
        parts = [float(v) for v in value.split(':')]
        return sum(v * 60 ** n for n, v in enumerate(reversed(parts)))
    except ValueError:
        return None


def parse_log(text, total, *, initial_total=None):
    """Keep last repaint per step, ignore sampling/cache bars and preserve missing data."""
    clean = ANSI.sub('', text).replace('\r', '\n')
    points, events, invalid = {}, [], 0
    training_started = False
    checkpoint, saved_step, phase, saving = None, None, 'Initializing', False
    attempt_total = initial_total if initial_total is not None else total
    for line in re.split(r'[\r\n]+', clean):
        if line.startswith('STUDIO_SEED '):
            seed = json.loads(line[len('STUDIO_SEED '):])
            saved_step, checkpoint, saving = seed['step'], seed['path'], False
        if line.startswith('STUDIO_RESUME '):
            events.append({'kind': 'resume', 'message': line})
            target = re.search(r'\btotal=(\d+)', line)
            if target:
                attempt_total = int(target[1])
        if 'Saving at step ' in line:
            saving, phase = True, 'Saving checkpoint'
        if 'Sampling at step ' in line:
            phase = 'Generating previews'
        for prefix, name in [('Loading ', 'Loading model'), ('Quantizing', 'Quantizing model'),
                             ('Caching latents', 'Caching images'), ('Caching text', 'Caching captions'),
                             ('Generating baseline', 'Baseline previews'), ('Generating images', 'Generating previews')]:
            if line.startswith(prefix):
                phase = name
        if 'Saved checkpoint to ' in line:
            checkpoint = line.split('Saved checkpoint to ', 1)[1].strip()
            saving = True
        if 'Saved optimizer to ' in line and checkpoint:
            match = re.search(r'_(\d+)\.safetensors$', checkpoint)
            saved_step = int(match[1]) if match else attempt_total
            saving = False
            events.append({'kind': 'checkpoint', 'step': saved_step, 'message': f'Checkpoint and optimizer saved at step {saved_step}'})
        if 'Error running job:' in line or 'Could not save optimizer' in line or 'Failed to load optimizer' in line:
            events.append({'kind': 'error', 'message': line[:1500]})
        match = PROGRESS.search(line)
        if not match or int(match[2]) != attempt_total or 'loss:' not in line:
            continue
        # Toolkit emits the first loss before advancing its zero-based progress bar.
        training_started = True
        step = int(match[1])
        metrics = {}
        for key, value in METRIC.findall(line):
            number = float(value)
            if math.isfinite(number):
                metrics[key] = number
            else:
                invalid += 1
        if not metrics:
            continue
        timing = match[3]
        elapsed = seconds(timing.split('<')[0])
        rate = re.search(r'([\d.]+)\s*(s/it|it/s)', timing)
        duration = None
        if rate and float(rate[1]) > 0:
            duration = float(rate[1]) if rate[2] == 's/it' else 1 / float(rate[1])
        points[step] = {'step': step, 'elapsed': elapsed, 'seconds_per_step': duration, **metrics}
        phase = 'Optimizing'
    return {'training_started': training_started or bool(saved_step and saved_step > 0),
            'points': [points[k] for k in sorted(points)], 'events': events[-200:],
            'saved_step': saved_step, 'phase': phase, 'invalid_values': invalid,
            'save_incomplete': saving, 'checkpoint_path': checkpoint, 'log_tail': clean[-48000:]}


def resume_checkpoint(root, parsed, total, *, validate=False, allow_complete=False, output=None):
    """Require the latest backend-selected weights and completed optimizer save to agree."""
    step = parsed['saved_step']
    if step is None or step > total or (step == total and not allow_complete) or parsed['save_incomplete']:
        raise ValueError('No complete resumable checkpoint: the last save is missing, incomplete, or already at the final step.')
    weights = list((output or root / 'output').glob('**/*.safetensors'))
    if not weights:
        raise ValueError('Saved checkpoint is missing')
    latest = max(weights, key=lambda p: p.stat().st_ctime_ns)
    expected = re.split(r'[/\\]', parsed['checkpoint_path'] or '')[-1]
    optimizer = latest.parent / 'optimizer.pt'
    if latest.name != expected or not optimizer.is_file() or optimizer.stat().st_size == 0:
        raise ValueError('Latest weights and completed optimizer save do not match; inspect the run files.')
    if validate:
        # Validate structure without executing pickle or importing a GPU runtime.
        try:
            with latest.open('rb') as f:
                size = int.from_bytes(f.read(8), 'little')
                if not 2 <= size <= 16 * 1024 * 1024:
                    raise ValueError('Invalid checkpoint header')
                header = json.loads(f.read(size))
                end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
                if end + 8 + size != latest.stat().st_size:
                    raise ValueError('Truncated checkpoint')
            with zipfile.ZipFile(optimizer) as archive:
                if not any(n.endswith('/data.pkl') for n in archive.namelist()) or archive.testzip():
                    raise ValueError('Invalid optimizer archive')
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            raise ValueError('Checkpoint or optimizer state is damaged; resume has been blocked.') from exc
    return latest, optimizer


def run_config(root, job):
    """Apply explicit continuation overrides without rewriting the original recipe."""
    config = read_json(root / 'config.json')
    if job.get('continuations') and config:
        continuation = job['continuations'][-1]
        process = config['config']['process'][0]
        process['train']['steps'] = continuation['total']
        process['training_folder'] = str(root / 'output' / f"continuation-{len(job['continuations'])}")
        # Public metadata describes the effective planned total for this attempt.
        meta = config.get('meta', {}).get('lora_studio')
        if meta:
            public = json.loads(meta)
            public['training']['train']['steps'] = continuation['total']
            config['meta']['lora_studio'] = json.dumps(public, ensure_ascii=False).replace('[name]', '\\u005bname]')
    return config


def gpu_metrics():
    fields = ['index', 'name', 'uuid', 'memory.used', 'memory.total', 'utilization.gpu',
              'utilization.memory', 'temperature.gpu', 'power.draw', 'power.limit', 'fan.speed', 'clocks.sm']
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=4,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if result.returncode:
            return {'available': False, 'gpus': []}
        rows = []
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) != len(fields):
                continue
            values = dict(zip(fields, [v.strip() for v in row]))
            for key in fields:
                if key not in {'name', 'uuid'}:
                    try:
                        values[key] = float(values[key])
                    except ValueError:
                        values[key] = None
            rows.append(values)
        return {'available': bool(rows), 'gpus': rows}
    except (OSError, subprocess.SubprocessError):
        return {'available': False, 'gpus': []}


def control_database(root, job_id):
    """The installed DiffusionTrainer's documented-in-code Job control contract."""
    path = root / 'toolkit-control.sqlite3'
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE IF NOT EXISTS Job (id TEXT PRIMARY KEY, stop INTEGER DEFAULT 0, '
                   'return_to_queue INTEGER DEFAULT 0, save_now INTEGER DEFAULT 0, sample_now INTEGER DEFAULT 0, '
                   'status TEXT, info TEXT, step INTEGER DEFAULT 0, speed_string TEXT)')
        db.execute('INSERT OR REPLACE INTO Job (id,status,info) VALUES (?, ?, ?)', (job_id, 'queued', 'Starting'))
    return path


def native_flag(root, job_id, key):
    if key not in {'save_now', 'sample_now'}:
        raise ValueError('Invalid training control')
    path = root / 'toolkit-control.sqlite3'
    if not path.is_file():
        raise ValueError('This older run supports saving at its scheduled checkpoints only.')
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(f'UPDATE Job SET {key}=1 WHERE id=?', (job_id,))


class TrainingMonitor:
    def __init__(self, service, owner_url=None):
        self.service, self.store, self.owner_url = service, service.store, owner_url
        self.lock = threading.RLock()
        self.cache = {}
        self.hardware = {'available': False, 'gpus': []}
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()

    def close(self):
        self.closed.set()
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=5)

    def root(self, job_id):
        job = self.store.item('jobs', job_id)
        if job['kind'] != 'train':
            raise ValueError('Training telemetry is available for training runs only')
        return self.store.root / 'runs' / job_id

    def parsed(self, root, total):
        path = root / 'training.log'
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size, total)
            cached = self.cache.get(str(path))
            if cached and cached[0] == stamp:
                return cached[1]
            # A 20k-step log normally stays well below this bound. Never read an unbounded file.
            with path.open('rb') as f:
                f.seek(max(0, stat.st_size - 64 * 1024 * 1024))
                text = f.read().decode('utf-8', errors='replace')
            original = read_json(root / 'config.json').get('config', {}).get('process', [{}])[0].get('train', {}).get('steps', total)
            result = parse_log(text, total, initial_total=original)
            result['log_updated'] = stat.st_mtime
            result['history_truncated'] = stat.st_size > 64 * 1024 * 1024
            self.cache[str(path)] = (stamp, result)
            return result
        except OSError:
            return {**parse_log('', total), 'log_updated': None, 'history_truncated': False}

    def snapshot(self, job_id):
        with self.lock:
            root = self.root(job_id)
            job = self.store.item('jobs', job_id)
            config = run_config(root, job)
            process = config.get('config', {}).get('process', [{}])[0]
            total = process.get('train', {}).get('steps', job.get('snapshot', {}).get('steps', 0))
            parsed = self.parsed(root, total)
            self.service.sync_training_lock(job, parsed)
            native = {}
            db_path = root / 'toolkit-control.sqlite3'
            if db_path.exists():
                try:
                    with closing(sqlite3.connect(db_path, timeout=.1)) as db:
                        db.row_factory = sqlite3.Row
                        row = db.execute('SELECT status,info,step,speed_string,save_now,sample_now FROM Job WHERE id=?', (job_id,)).fetchone()
                        native = dict(row) if row else {}
                except sqlite3.Error:
                    pass
            points = parsed['points']
            last = points[-1] if points else {}
            control = read_json(root / 'monitor-control.json')
            if job['status'] in {'running', 'queued'} and control.get('action') == 'pause':
                display_status = 'pausing'
            else:
                display_status = job['status']
            files = self.service.run_files(job_id)
            checkpoints = [f for f in files if f['path'].endswith('.safetensors') and f['path'].startswith('output/')]
            samples = [f for f in files if f['path'].startswith('output/') and f['path'].lower().endswith(('.jpg', '.png'))]
            history = []
            hp = root / 'hardware.jsonl'
            if hp.exists():
                with hp.open('rb') as f:
                    f.seek(max(0, hp.stat().st_size - 4 * 1024 * 1024))
                    for line in f.read().splitlines():
                        try:
                            history.append(json.loads(line))
                        except ValueError:
                            pass
            manifest = read_json(root / 'manifest.json')
            images = manifest.get('images', [])
            dataset = {'count': len(images), 'manual': sum(bool(i.get('manual')) for i in images),
                       'reviewed': sum(bool(i.get('reviewed')) for i in images),
                       'caption_warnings': sum(bool(i.get('warnings')) for i in images),
                       'visual_concerns': sum(i.get('visual_review', {}).get('status') == 'concerns' for i in images)}
            active = job['status'] in {'running', 'queued'}
            training = active and bool(job.get('pid')) and job.get('phase') == 'Training'
            controllable = not job.get('reconnected') or (root / 'toolkit-control.sqlite3').exists()
            speed = last.get('seconds_per_step')
            step = max(last.get('step', 0), parsed['saved_step'] or 0)
            try:
                resume_checkpoint(root, parsed, total, allow_complete=job['status'] == 'complete',
                                  output=Path(process['training_folder']) / config['config']['name'] if config else None)
                resumable, resume_error = True, None
            except ValueError as exc:
                resumable, resume_error = False, str(exc)
            return {'job': job, 'status': display_status, 'total': total, 'step': step,
                    'progress': min(100, 100 * step / total) if total else 0,
                    'eta_seconds': max(0, total - step) * speed if active and speed else None,
                    'wall_seconds': max(0, job.get('finished', time.time()) - job['created']),
                    'latest': last, **parsed, 'phase': (native.get('info') or parsed['phase']) if config else job.get('phase', 'Preparing dataset'), 'native_status': native,
                    'hardware': self.hardware, 'hardware_history': history[-5000:],
                    'config': config, 'dataset': dataset, 'gpu_index': manifest.get('project', {}).get('training', {}).get('gpu', 0), 'backend_commit': manifest.get('backend_commit'),
                    'checkpoints': checkpoints, 'samples': samples, 'files': files,
                    'control': control, 'capabilities': {'pause': training and controllable, 'cancel': (active or display_status == 'pausing') and controllable,
                        'resume': job['status'] in {'paused', 'cancelled', 'failed', 'interrupted'} and resumable,
                        'extend': job['status'] == 'complete' and resumable and parsed['saved_step'] == total,
                        'save': training and (root / 'toolkit-control.sqlite3').exists(),
                        'sample': training and (root / 'toolkit-control.sqlite3').exists(),
                        'native': (root / 'toolkit-control.sqlite3').exists()},
                    'resume_error': resume_error,
                    'resume_note': 'Restores saved weights, optimizer and step. Data-loader order and random state are not guaranteed bit-identical; the backend reapplies configured learning rate.',
                    'telemetry_note': 'Loss and LR are trainer log measurements (rounded by the trainer), not validation quality. GPU usage includes other processes. Hardware history begins when monitoring attaches.'}

    def _stop_owned(self, job_id):
        if job_id in self.service.cancel or job_id in self.service.reconnected:
            return self.service.stop(job_id)
        if self.owner_url:
            from .integrations import request
            remote = request(self.owner_url + '/api/jobs/' + job_id)['job']
            local = self.store.item('jobs', job_id)
            if remote.get('pid') != local.get('pid') or remote.get('project') != local.get('project'):
                raise ValueError('Trainer ownership changed; refusing to stop a different process')
            return request(self.owner_url + '/api/jobs/' + job_id + '/stop', {})
        raise ValueError('This server does not own the trainer. Reconnect to its original Studio server.')

    def action(self, job_id, action, value=None):
        with self.lock:
            root = self.root(job_id)
            snap = self.snapshot(job_id)
            if action == 'extend':
                count = (value or {}).get('checkpoints', 1)
                if type(count) is not int or not 1 <= count <= 1000:
                    raise ValueError('Additional checkpoints must be a whole number from 1 to 1000')
                return self.service.resume_training(job_id, additional_checkpoints=count)
            if action == 'resume':
                if not snap['capabilities']['resume']:
                    raise ValueError('No complete weights and optimizer checkpoint is available to resume')
                return self.service.resume_training(job_id)
            if action == 'continue':
                if snap['status'] != 'pausing' or snap['control'].get('stop_sent'):
                    raise ValueError('No withdrawable pause is pending; wait until the run has paused, then resume.')
                write_json(root / 'monitor-control.json', {})
                return {'requested': True}
            if action not in {'pause', 'cancel', 'save', 'sample'} or not snap['capabilities'].get(action):
                raise ValueError('This action is unavailable in the current run state')
            if action == 'pause':
                if snap['status'] == 'pausing':
                    return {'requested': True}
                write_json(root / 'monitor-control.json', {'action': 'pause', 'requested': time.time(),
                           'after_saved_step': snap['saved_step'], 'log_size': (root / 'training.log').stat().st_size if (root / 'training.log').exists() else 0})
                if snap['capabilities']['native']:
                    native_flag(root, job_id, 'save_now')
            elif action == 'cancel':
                root.mkdir(parents=True, exist_ok=True)
                write_json(root / 'monitor-control.json', {'action': 'cancel', 'requested': time.time()})
                self._stop_owned(job_id)
            else:
                native_flag(root, job_id, 'save_now' if action == 'save' else 'sample_now')
            return {'requested': True}

    def _watch(self):
        while not self.closed.wait(3):
            try:
                self.tick()
            except Exception as exc:
                # Keep monitoring alive; surface a bounded diagnostic, never silently claim success.
                self.hardware = {**self.hardware, 'monitor_error': str(exc)[:1000]}

    def tick(self):
        from .recovery import refresh
        refresh(self.service)
        jobs = [j for j in self.store.items('jobs') if j['kind'] == 'train' and j['status'] in {'running', 'queued', 'cancelled'}]
        if any(j['status'] in {'running', 'queued'} for j in jobs):
            self.hardware = {**gpu_metrics(), 'at': time.time()}
        for job in jobs:
            root = self.store.root / 'runs' / job['id']
            if not root.is_dir():
                continue
            self.service.sync_training_lock(job)
            with self.lock:
                control = read_json(root / 'monitor-control.json')
                if job['status'] in {'running', 'queued'}:
                    with (root / 'hardware.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(self.hardware) + '\n')
                if control.get('action') != 'pause':
                    continue
                if job['status'] == 'cancelled' and control.get('checkpoint_confirmed') is not None:
                    job.update(status='paused', phase='Paused at saved checkpoint', paused_step=control['checkpoint_confirmed'])
                    self.store.put_item('jobs', job)
                    control['action'] = 'paused'
                    write_json(root / 'monitor-control.json', control)
                elif job['status'] == 'running' and not control.get('stop_sent'):
                    # A complete optimizer marker occurs only after torch.save has returned.
                    path = root / 'training.log'
                    with path.open('rb') as f:
                        f.seek(control['log_size'])
                        fresh = f.read().decode('utf-8', errors='replace')
                    if 'Saved optimizer to ' not in fresh:
                        continue
                    # Never terminate during a subsequent checkpoint write.
                    completed = fresh.rfind('Saved optimizer to ')
                    if max(fresh.rfind('Saving at step '), fresh.rfind('Saved checkpoint to ')) > completed:
                        continue
                    snap = self.snapshot(job['id'])
                    if snap['saved_step'] is None or not snap['checkpoints']:
                        continue
                    control.update(checkpoint_confirmed=snap['saved_step'], stop_sent=True)
                    write_json(root / 'monitor-control.json', control)
                    try:
                        self._stop_owned(job['id'])
                    except Exception:
                        control.pop('stop_sent', None)
                        write_json(root / 'monitor-control.json', control)
                        raise
