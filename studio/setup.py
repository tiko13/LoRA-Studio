"""First-run checks and an isolated, upstream-managed AI Toolkit installation."""
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import uuid

from PIL import Image

from .integrations import PromptStudio, standalone_caption, request, PS
from .monitoring import read_json, write_json
from .processes import running, terminate_tree
from .store import Conflict
from .trainer import backend_status

TOOLKIT_REPOSITORY = 'https://github.com/ostris/ai-toolkit.git'
ACTIVE = {'queued', 'running'}
PROBE = r'''
import importlib.util, json, sys
result = {'python': '.'.join(map(str, sys.version_info[:3])), 'missing': []}
for name in ('torch', 'torchvision', 'diffusers', 'transformers', 'accelerate', 'safetensors', 'yaml', 'PIL', 'peft', 'bitsandbytes'):
    if importlib.util.find_spec(name) is None: result['missing'].append(name)
try:
    import torch
    result.update(torch=torch.__version__, cuda=torch.cuda.is_available(), devices=[])
    if result['cuda']:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            result['devices'].append({'name': p.name, 'vram_gb': round(p.total_memory / 2**30, 1)})
except Exception:
    result['error'] = 'PyTorch could not initialize. Repair this environment using its installer.'
print('STUDIO_SETUP ' + json.dumps(result))
'''


def python_for(root):
    for name in ('.venv/Scripts/python.exe', 'venv/Scripts/python.exe', '.venv/bin/python',
                 'venv/bin/python', '.venv/bin/python3', 'venv/bin/python3'):
        if (root / name).is_file():
            return str((root / name).resolve())
    return ''


def runtime_check(executable):
    if not executable or not Path(executable).is_file():
        return {'available': False, 'reason': 'Select the Python executable from your AI Toolkit environment.'}
    try:
        result = subprocess.run([executable, '-c', PROBE], capture_output=True, text=True,
                                timeout=90, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        lines = [s[13:] for s in result.stdout.splitlines() if s.startswith('STUDIO_SETUP ')]
        data = json.loads(lines[-1]) if lines else {}
        if result.returncode or not data:
            raise ValueError('probe failed')
        data['available'] = not data.get('missing') and not data.get('error')
        data['reason'] = data.get('error') or ('Missing packages: ' + ', '.join(data['missing']) if data['missing']
                         else 'Core packages found and PyTorch imported. Training has not been run.')
        return data
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {'available': False, 'reason': 'Python could not complete the runtime check within 90 seconds. Check the executable and repair its environment.'}


def hardware_check():
    executable = shutil.which('nvidia-smi')
    if executable:
        try:
            result = subprocess.run([executable, '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=10,
                                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            devices = [{'name': name.strip(), 'vram_gb': round(float(memory) / 1024, 1), 'driver': driver.strip()}
                       for name, memory, driver in csv.reader(result.stdout.splitlines())]
            if result.returncode == 0 and devices:
                return {'available': True, 'devices': devices}
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    return {'available': False, 'reason': 'NVIDIA hardware/driver could not be detected. Install or check the GPU driver before configuring CUDA training.'}


class Setup:
    def __init__(self, service):
        self.service = service
        self.root = service.store.root / 'setup'
        self.root.mkdir(exist_ok=True)
        self.path = self.root / 'state.json'
        self.lock = threading.RLock()
        self.process = None
        self.cancel = threading.Event()
        self.check_lock = threading.Lock()
        self.report = None
        self.state = read_json(self.path, {'version': 1, 'disposition': 'new', 'install': {}})
        install = self.state.get('install', {})
        if not service.owner_url and install.get('status') in ACTIVE:
            install.update(status='interrupted', message='Studio restarted during installation. Check the recorded installer before retrying; downloaded files are preserved.')
            self._save()
        manager = self.state.get('promptstudio_install', {})
        if not service.owner_url and manager.get('status') == 'running':
            manager.update(status='interrupted', message='Studio restarted while Manager was installing. Check ComfyUI Manager before retrying; do not submit a duplicate installation.')
            self._save()

    def _save(self):
        write_json(self.path, self.state)

    def status(self):
        with self.lock:
            return json.loads(json.dumps({**self.state, 'managed_parent': str(self.root.resolve()), 'report': self.report}))

    def dismiss(self, disposition):
        if disposition not in {'later', 'complete'}:
            raise ValueError('Unknown setup action')
        with self.lock:
            self.state['disposition'] = disposition
            self._save()
            return self.status()

    def check(self):
        if not self.check_lock.acquire(blocking=False):
            raise Conflict('A prerequisite check is already running. Try again shortly.')
        try:
            settings = self.service.settings()
            backend = backend_status(settings)
            runtime = runtime_check(settings['toolkit_python'])
            connection = self.service.connection(settings)
            free = round(shutil.disk_usage(self.root).free / 2**30, 1)
            report = {'checked': time.time(), 'settings_revision': settings['settings_revision'],
                      'backend': backend, 'runtime': runtime, 'connection': connection,
                      'git': bool(shutil.which('git')), 'disk_free_gb': free, 'hardware': hardware_check(),
                      'training_ready': bool(backend['available'] and backend['architectures'] and runtime.get('available') and runtime.get('cuda'))}
            with self.lock:
                self.report = report
            return report
        finally:
            self.check_lock.release()

    def vision(self):
        settings = self.service.settings()
        image = Image.new('RGB', (64, 64), (255, 0, 0))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        prompt = 'What is the dominant color of this image? Reply with just the English color name.'
        bridge = None
        if settings['integration'] != 'standalone':
            try:
                candidate = PromptStudio(settings['prompt_studio_url'])
                connection = candidate.connection()
                bridge = candidate
            except (ValueError, RuntimeError, KeyError):
                if settings['integration'] == 'promptstudio':
                    raise
        if bridge:
            response = bridge.caption(prompt, buffer.getvalue(), 'setup-check.png', connection, max_tokens=40)
        else:
            response = standalone_caption(settings, prompt, buffer.getvalue(), 'image/png', max_tokens=40)
        passed = str(response).strip().lower().strip('.! \"\'') == 'red'
        return {'passed': passed, 'reason': 'Synthetic image check passed. Review generated captions for accuracy.' if passed
                else 'The model did not identify the synthetic image. Select a vision-capable model and try again.'}

    def manager_status(self):
        url = self.service.settings()['prompt_studio_url']
        result = {'comfyui': False, 'manager': False, 'promptstudio': False, 'url': url}
        try:
            stats = request(url + '/system_stats', timeout=5)
            result['comfyui'] = isinstance(stats, dict) and 'system' in stats
            if not result['comfyui']:
                return result
        except (ValueError, RuntimeError):
            return result
        try:
            queue = request(url + '/manager/queue/status', timeout=5)
            result['manager'] = isinstance(queue, dict) and 'is_processing' in queue
            result['manager_busy'] = bool(queue.get('is_processing') or queue.get('in_progress_count') or queue.get('total_count', 0) > queue.get('done_count', 0))
        except (ValueError, RuntimeError):
            pass
        try:
            data = request(url + PS + '/chats?summaries=1&limit=1', timeout=5)
            result['promptstudio'] = isinstance(data, dict) and 'chats' in data
        except (ValueError, RuntimeError):
            pass
        return result

    def install_promptstudio(self, acknowledged=False, expected_url=None):
        if acknowledged is not True:
            raise ValueError('Confirm the described ComfyUI Manager installation first.')
        if expected_url != self.service.settings()['prompt_studio_url']:
            raise Conflict('The ComfyUI address changed. Check ComfyUI & Manager again before installing.')
        with self.lock:
            previous = self.state.get('promptstudio_install', {})
            if previous.get('status') == 'running':
                return self.status()
            status = self.manager_status()
            if status['promptstudio']:
                raise ValueError('Prompt Studio is already installed. Open it in ComfyUI and configure a vision model.')
            if not status['comfyui'] or not status['manager']:
                raise ValueError('Start ComfyUI with ComfyUI Manager, then check again, or use the manual installation guide.')
            if status.get('manager_busy'):
                raise ValueError('ComfyUI Manager has pending work. Wait until it finishes.')
            queue = request(status['url'] + '/queue', timeout=5)
            if not isinstance(queue, dict) or 'queue_running' not in queue or 'queue_pending' not in queue:
                raise ValueError('Could not verify the ComfyUI queue. Install through Manager directly.')
            if queue['queue_running'] or queue['queue_pending']:
                raise ValueError('ComfyUI has active or queued work. Wait before installing Prompt Studio.')
            self.state['promptstudio_install'] = {'status': 'running', 'message': 'ComfyUI Manager is installing Prompt Studio. Follow its console for dependency progress.', 'url': status['url']}
            self._save()
            threading.Thread(target=self._install_promptstudio, args=(status['url'],), daemon=True).start()
            return self.status()

    def _install_promptstudio(self, url):
        try:
            request(url + '/customnode/install/git_url', {'url': 'https://github.com/tiko13/ComfyUI_PromptStudio'}, timeout=900, accept_text=True)
            result = {'status': 'restart_required', 'message': 'Manager accepted the installation. When ComfyUI is idle, restart it yourself, refresh its browser, then open Prompt Studio and select a vision LLM. Recheck here afterward.'}
        except (ValueError, RuntimeError):
            result = {'status': 'needs_attention', 'message': 'Manager could not confirm installation. It may have rejected Git URL installation or still be working after a timeout. Check its console before retrying, or follow the manual guide. LoRA Studio does not change Manager security settings.'}
        with self.lock:
            self.state['promptstudio_install'].update(result)
            self._save()

    def _update(self, **values):
        with self.lock:
            self.state['install'].update(values)
            self._save()

    def start(self):
        with self.lock:
            if self.service.owner_url:
                raise ValueError('Run setup in the main Studio server.')
            previous = self.state.get('install', {})
            if previous.get('status') in ACTIVE:
                return self.status()
            if previous.get('pid') and running(previous['pid']):
                raise ValueError('The previous installer may still be running. Wait for it to exit before retrying.')
            if any(j['status'] in {'queued', 'running', 'stopping', 'paused'} for j in self.service.store.items('jobs')):
                raise ValueError('Wait for Studio jobs to finish before installing the training backend.')
            if not shutil.which('git'):
                raise ValueError('Install Git, restart LoRA Studio, then retry. An existing AI Toolkit installation can also be selected.')
            if shutil.disk_usage(self.root).free < 10 * 2**30:
                raise ValueError('Free at least 10 GB for backend installation. Training models and outputs need additional space.')
            target = (self.root / ('ai-toolkit-' + uuid.uuid4().hex[:12])).resolve()
            old = Path(previous.get('target') or self.root).resolve()
            # Retry only our own completed clone; never modify a user-selected checkout.
            if old.parent == self.root.resolve() and old.name.startswith('ai-toolkit-') and (old / '.git').is_dir() and (old / 'manager/__main__.py').is_file():
                target = old
            baseline = self.service.settings()
            self.cancel.clear()
            self.state['install'] = {'status': 'queued', 'phase': 'Preparing', 'target': str(target), 'started': time.time(), 'pid': None}
            self._save()
            threading.Thread(target=self._install, args=(target, baseline), daemon=True).start()
            return self.status()

    def _command(self, args, cwd, log):
        if self.cancel.is_set():
            raise InterruptedError()
        with self.lock:
            process = subprocess.Popen(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'PYTHONUNBUFFERED': '1',
                                       'UV_CACHE_DIR': str(self.root / 'cache'), 'PIP_CACHE_DIR': str(self.root / 'cache' / 'pip')},
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), start_new_session=os.name != 'nt')
            self.process = process
            self._update(pid=process.pid)
        deadline = time.monotonic() + 7200
        try:
            while process.poll() is None:
                if self.cancel.wait(.25):
                    terminate_tree(process)
                    raise InterruptedError()
                if time.monotonic() > deadline:
                    terminate_tree(process)
                    raise RuntimeError('The installer timed out after two hours. Check the private installer log and retry.')
            if process.returncode:
                raise RuntimeError('The upstream installer failed. Check setup/install.log in your private data folder, resolve the error, then retry.')
        finally:
            with self.lock:
                # Retain ownership evidence if termination failed; retry must not
                # start a second installer while the first may still be alive.
                exited = process.poll() is not None
                self.process = None if exited else process
                self._update(pid=None if exited else process.pid)

    def _install(self, target, baseline):
        try:
            with (self.root / 'install.log').open('ab') as log:
                self._update(status='running', phase='Downloading AI Toolkit')
                if not (target / '.git').is_dir():
                    self._command([shutil.which('git'), 'clone', '--depth', '1', '--recurse-submodules', TOOLKIT_REPOSITORY, str(target)], self.root, log)
                self._command([shutil.which('git'), '-C', str(target), 'submodule', 'update', '--init', '--recursive'], target, log)
                if not (target / 'manager/__main__.py').is_file():
                    raise RuntimeError('This AI Toolkit revision does not include its installer manager. Use an existing installation or the upstream installation guide.')
                self._update(phase='Installing Python, PyTorch and backend dependencies')
                self._command([sys.executable, '-m', 'manager', 'install'], target, log)
            if self.cancel.is_set():
                raise InterruptedError()
            self._update(phase='Checking the installed runtime')
            python = python_for(target)
            runtime = runtime_check(python)
            if self.cancel.is_set():
                raise InterruptedError()
            if not runtime.get('available'):
                raise RuntimeError(runtime['reason'])
            with self.service.lock:
                current = self.service.settings()
                if any(current[k] != baseline[k] for k in ('toolkit_path', 'toolkit_python')):
                    self._update(status='complete', phase='Installed', message='Installed successfully. Backend settings changed during installation; select this installation manually to use it.', python=python)
                else:
                    self.service.save_settings({'toolkit_path': str(target), 'toolkit_python': python})
                    self._update(status='complete', phase='Installed', message='AI Toolkit installed and selected. Run the prerequisite check to review GPU and model support.', python=python)
        except InterruptedError:
            self._update(status='cancelled', phase='Cancelled', message='Installation stopped. Downloaded files are preserved; retry to continue.')
        except Exception as exc:
            # Raw subprocess output stays in the private log, not in HTTP diagnostics.
            message = str(exc) if isinstance(exc, RuntimeError) else 'Installation could not finish. Check the private setup/install.log and retry.'
            self._update(status='failed', phase='Installation failed', message=message)

    def stop(self):
        self.cancel.set()
        return self.status()

    def close(self):
        self.cancel.set()
