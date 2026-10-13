"""Durable checkpoint comparisons, owned by LoRA Studio; ComfyUI executes cells."""
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re
import secrets
import shutil
import threading
import time
import urllib.parse
import urllib.request
import uuid

from PIL import Image, ImageDraw, ImageOps

from .integrations import PS, NoRedirect, IntegrationHTTPError, local_url, request
from .monitoring import read_json, write_json
from .results import results, run_root, artifact_path, checkpoint_info
from .store import Conflict

MAX_CELLS = 128
LORA = 'KCPP_PromptStudioLoraLoader'
MODEL = 'KCPP_PromptStudioModelLoader'
SAMPLERS = {'KCPP_PromptStudioSampler', 'KCPP_QwenImage21TurboSampler', 'KCPP_QwenImage21OfficialTurboSampler'}
FAMILIES = {'krea2': {'krea2'}, 'qwen21': {'qwenimage21', 'qwen21'},
            'flux': {'flux', 'flux1', 'flux1dev'}, 'sdxl': {'sdxl'}}


def family(value):
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def connected(graph, target, source, seen=None):
    if str(target) == str(source):
        return True
    seen = set() if seen is None else seen
    if target in seen:
        return False
    seen.add(target)
    return any(connected(graph, str(v[0]), source, seen) for v in graph.get(str(target), {}).get('inputs', {}).values()
               if isinstance(v, list) and len(v) == 2 and type(v[1]) is int)


def supported_profile(profile, model):
    """Require explicit family descriptors and connected, controllable graph targets."""
    if profile.get('kind') != 'create' or profile.get('stale') or profile.get('error'):
        return None
    graph = profile.get('snapshot', {}).get('output', {})
    prompt = str(profile.get('promptNodeId', ''))
    if graph.get(prompt, {}).get('class_type') not in {'KCPP_PromptSlot', 'KCPP_PromptAmplify'}:
        return None
    outputs = [str(n) for n in profile.get('resultNodeIds', []) if str(n) in graph]
    if not outputs:
        return None
    active = lambda n: any(connected(graph, out, n) for out in outputs)
    loras = [n for n in profile.get('loraNodes', []) if
             family(n.get('loraType')) in FAMILIES.get(model, set()) and active(str(n['id']))
             and graph.get(str(n['id']), {}).get('class_type') == LORA]
    samplers = [str(n['id']) for n in profile.get('samplingNodes', []) if active(str(n['id']))
                and graph.get(str(n['id']), {}).get('class_type') in SAMPLERS
                and type(graph[str(n['id'])].get('inputs', {}).get('seed')) is int]
    if not loras or not samplers or not active(prompt):
        return None
    for node_id, node in graph.items():
        if not active(node_id):
            continue
        if node['class_type'] in {'KCPP_ChatImageInput', 'LoadImage'}:
            return None  # No hidden image dependencies in Create-only comparisons.
        if node['class_type'] == 'KCPP_PromptAmplify' and str(node_id) != prompt:
            return None  # Additional native LLM stages cannot run inside the GPU reservation.
        if node['class_type'] == 'PS_ReferenceCollection' and node.get('inputs', {}).get('minimum', 0):
            return None
    selected = copy.deepcopy(profile)
    selected['comparison_loras'] = loras
    selected['comparison_samplers'] = samplers
    selected['comparison_resolution'] = all(any(v == [prompt, port]
        for node_id, node in graph.items() if active(node_id)
        for v in node.get('inputs', {}).values()) for port in (2, 3))
    return selected


def cell_graph(manifest, cell, checkpoint_id=None):
    profile, config = manifest['profile'], manifest['settings']
    graph = copy.deepcopy(profile['snapshot']['output'])
    prompt = graph[str(profile['promptNodeId'])]
    old = prompt.get('inputs', {})
    prompt['class_type'] = 'KCPP_PromptSlot'
    prompt['inputs'] = {k: v for k, v in old.items() if k in {'slot_name', 'secondary_instructions',
        'aspect_ratio', 'megapixels', 'multiple', 'resolution_width', 'resolution_height'}}
    prompt['inputs'].update(prompt=config['prompt'], slot_name=old.get('slot_name', 'Comparison'))
    if config.get('width'):
        prompt['inputs'].update(resolution_width=config['width'], resolution_height=config['height'])
    for node_id, name in config['models'].items():
        graph[node_id]['inputs']['unet_name'] = name
    for node_id in profile['comparison_samplers']:
        node = graph[node_id]
        node['inputs']['seed'] = cell['seed']
        if node['class_type'] == 'KCPP_PromptStudioSampler':
            for field in ('steps', 'cfg'):
                if config.get(field) is not None:
                    node['inputs'][field] = config[field]
    if checkpoint_id:
        target = config['loader']
        identity = 'lora_studio_' + manifest['id']
        # Keep the workflow's original LoRA stack fixed, and append the test LoRA.
        for node in graph.values():
            for key, value in node.get('inputs', {}).items():
                if isinstance(value, list) and value == [target, 0]:
                    node['inputs'][key] = [identity, 0]
        graph[identity] = {'class_type': 'PS_LoraStudioCheckpoint', 'inputs': {
            'model': [target, 0], 'checkpoint_id': checkpoint_id, 'strength': config['strength']}}
    # Force a single sample per cell for built-in latent sources.
    for node in graph.values():
        if node['class_type'].startswith('Empty') and 'batch_size' in node.get('inputs', {}):
            node['inputs']['batch_size'] = 1
    return graph


class ComparisonReservation:
    """Keep the server lease alive even while a cell blocks on generation or IO."""
    def __init__(self, endpoint, identity, stopping):
        self.url = endpoint + '/promptstudio/lora-studio/generation/'
        self.stopping = stopping
        self.done = threading.Event()
        self.error = None
        self.token = ''
        self.thread = None
        self.identity = identity

    def begin(self):
        result = request(self.url + 'begin', {'comparison': self.identity})
        self.token = result['token']
        self.thread = threading.Thread(target=self.heartbeat, daemon=True)
        self.thread.start()
        while not result.get('active'):
            self.check()
            if self.stopping():
                return False
            self.done.wait(.5)
            result = request(self.url + 'pulse', {'token': self.token})
        return not self.stopping()

    def heartbeat(self):
        while not self.done.wait(5):
            try:
                request(self.url + 'pulse', {'token': self.token}, timeout=10)
            except Exception:
                self.error = RuntimeError('Prompt Studio connection interrupted. Resume the comparison to reserve generation again.')
                return

    def check(self):
        if self.error:
            raise self.error

    def close(self):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=11)
        if self.token:
            try:
                request(self.url + 'end', {'token': self.token}, timeout=10)
            except Exception:
                pass  # The server lease expires if the connection is lost.


class Comparisons:
    def __init__(self, service, origin):
        self.service, self.origin = service, local_url(origin)
        self.root = service.store.root / 'comparisons'
        self.root.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.capabilities = {}
        self.active = set()
        # Do not silently resubmit work after a process restart.
        for path in self.root.glob('*/comparison.json'):
            data = read_json(path)
            if not service.owner_url and data.get('status') in {'preparing', 'running', 'stopping'}:
                data.update(status='interrupted', error='Studio restarted. Resume checks already queued cells before submitting more.')
                write_json(path, data)

    def directory(self, identity):
        if not isinstance(identity, str) or not re.fullmatch(r'[a-f0-9]{32}', identity):
            raise ValueError('Invalid comparison')
        path = (self.root / identity).resolve()
        if path.parent != self.root.resolve():
            raise ValueError('Invalid comparison location')
        return path

    def load(self, identity):
        data = read_json(self.directory(identity) / 'comparison.json')
        if not data:
            raise KeyError('Comparison not found')
        return data

    def save(self, data):
        data['updated'] = time.time()
        write_json(self.directory(data['id']) / 'comparison.json', data)

    @staticmethod
    def public(data):
        return {k: copy.deepcopy(v) for k, v in data.items() if k not in {'profile', 'endpoint', 'provider'}}

    def list(self, job):
        with self.lock:
            return sorted([self.public(d) for p in self.root.glob('*/comparison.json')
                           if (d := read_json(p)) and d.get('job') == job], key=lambda d: d['created'], reverse=True)

    def frozen_project(self, job):
        _, root = run_root(self.service.store, job)
        project = read_json(root / 'manifest.json').get('project', {})
        if not project.get('model'):
            raise ValueError('This run has no frozen model information. Its checkpoints can still be downloaded.')
        return project

    def catalog(self, job):
        project = self.frozen_project(job)
        endpoint = self.service.settings()['prompt_studio_url']
        try:
            bridge = request(endpoint + '/promptstudio/lora-studio/capabilities')
            if bridge.get('version') != 1 or bridge.get('generation_reservation') != 1:
                raise ValueError('Unsupported comparison bridge')
            cache = request(endpoint + PS + '/workflows')
        except Exception as exc:
            raise ValueError('Requires Prompt Studio with the LoRA Studio comparison bridge. Start or update Prompt Studio and restart ComfyUI, then try again.') from exc
        profiles = []
        for entry in cache.get('templates', []):
            profile = supported_profile(entry, project['model'])
            if not profile:
                continue
            for node in profile.get('modelNodes', []):
                node['choices'] = request(endpoint + PS + '/models?type=' + urllib.parse.quote(node['modelType'])).get('models', [])
            profiles.append(profile)
        _, root = run_root(self.service.store, job)
        process = read_json(root / 'config.json').get('config', {}).get('process', [{}])[0]
        prompts = process.get('sample', {}).get('samples') or process.get('sample', {}).get('prompts') or []
        first = prompts[0] if prompts else ''
        return {'workflows': profiles, 'prompt': first.get('prompt', '') if isinstance(first, dict) else first,
                'trigger': project.get('trigger', ''), 'max_cells': MAX_CELLS}

    def reserve(self):
        if self.service.owner_url:
            raise Conflict('Open the main LoRA Studio window to generate comparisons.')
        if any(j['status'] in {'running', 'queued', 'stopping', 'interrupted'} for j in self.service.store.items('jobs')):
            raise Conflict('Wait for active work to finish or pause training before generating a comparison.')
        if not self.service.job_lock.acquire(blocking=False):
            raise Conflict('Another job or comparison is active. Wait for it to finish.')

    def start(self, job, value):
        self.reserve()
        try:
            catalog = self.catalog(job)
            profile = next((p for p in catalog['workflows'] if p['id'] == value.get('workflow')), None)
            if not profile:
                raise ValueError('Choose a compatible Create workflow. Refresh workflows in Prompt Studio if needed.')
            available = {c['path']: c for c in results(self.service.store, job)['checkpoints'] if c['ready']}
            selection = value.get('checkpoints', [])
            if not isinstance(selection, list) or not selection or any(not isinstance(c, str) or c not in available for c in selection):
                raise ValueError('Select at least one complete checkpoint')
            selection = list(dict.fromkeys(selection))
            seeds = value.get('seeds')
            if not isinstance(seeds, list) or not seeds or any(type(s) is not int or not 0 <= s <= 9007199254740991 for s in seeds):
                raise ValueError('Enter nonnegative whole-number seeds (up to 9007199254740991)')
            seeds = list(dict.fromkeys(seeds))
            if type(value.get('baseline', False)) is not bool:
                raise ValueError('Invalid baseline option')
            if (len(selection) + bool(value.get('baseline'))) * len(seeds) > MAX_CELLS:
                raise ValueError(f'Comparisons support at most {MAX_CELLS} images')
            prompt = value.get('prompt', '')
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16000:
                raise ValueError('Enter a prompt (up to 16000 characters)')
            strength = value.get('strength', 1)
            if type(strength) not in (int, float) or not math.isfinite(strength) or not -100 <= strength <= 100:
                raise ValueError('LoRA strength must be between -100 and 100')
            loader = str(value.get('loader', profile['comparison_loras'][0]['id']))
            if loader not in {str(n['id']) for n in profile['comparison_loras']}:
                raise ValueError('Choose a compatible LoRA loader')
            models = value.get('models', {})
            if not isinstance(models, dict):
                raise ValueError('Invalid model selection')
            selected_models = {}
            for node in profile.get('modelNodes', []):
                selected = models.get(str(node['id']), node.get('modelName'))
                if selected not in {m['name'] for m in node['choices']}:
                    raise ValueError('Choose an available model for every model selector')
                selected_models[str(node['id'])] = selected
            config = {'prompt': prompt, 'strength': strength, 'loader': loader, 'models': selected_models,
                      'seeds': seeds, 'workflow': profile['id'], 'baseline': bool(value.get('baseline'))}
            for field, low, high in [('width', 64, 4096), ('height', 64, 4096), ('steps', 1, 200), ('cfg', 0, 100)]:
                val = value.get(field)
                if val is not None:
                    if type(val) not in (int, float) or not math.isfinite(val) or not low <= val <= high or (field != 'cfg' and type(val) is not int):
                        raise ValueError(f'Invalid {field}')
                    if field in {'width', 'height'} and val % 32:
                        raise ValueError('Dimensions must be multiples of 32')
                    config[field] = val
            if bool(config.get('width')) != bool(config.get('height')):
                raise ValueError('Set both width and height, or leave both at workflow defaults')
            if config.get('width') and not profile['comparison_resolution']:
                raise ValueError('This workflow does not expose resolution through its Prompt Studio prompt node')
            if any(config.get(k) is not None for k in ('steps', 'cfg')) and not any(
                    profile['snapshot']['output'][n]['class_type'] == 'KCPP_PromptStudioSampler' for n in profile['comparison_samplers']):
                raise ValueError('This workflow uses fixed sampling settings')
            selection.sort(key=lambda p: (available[p]['step'] is None, available[p]['step'] or 0, available[p]['final']))
            columns = [{**available[p], 'label': ('Final · ' if available[p]['final'] else '') +
                        (f"Step {available[p]['step']}" if available[p]['step'] is not None else available[p]['name'])} for p in selection]
            if config['baseline']:
                columns.insert(0, {'label': 'Without trained LoRA', 'baseline': True})
            identity = uuid.uuid4().hex
            root = self.directory(identity)
            root.mkdir()
            (root / 'weights').mkdir()
            (root / 'images').mkdir()
            data = {'id': identity, 'job': job, 'created': time.time(), 'status': 'preparing', 'error': '',
                    'workflow_name': profile['name'], 'profile': profile, 'settings': config,
                    'endpoint': self.service.settings()['prompt_studio_url'], 'columns': columns,
                    'cells': [{'id': uuid.uuid4().hex, 'column': col, 'seed': seed, 'status': 'pending', 'images': []}
                              for col in range(len(columns)) for seed in seeds]}
            with self.lock:
                self.save(data)
            self.launch(identity, prepare=True)
            return self.public(data)
        except Exception:
            self.service.job_lock.release()
            raise

    def launch(self, identity, prepare=False):
        self.active.add(identity)
        threading.Thread(target=self.worker, args=(identity, prepare), daemon=True).start()

    def action(self, identity, action):
        if self.service.owner_url:
            raise Conflict('Use the main LoRA Studio window to control comparisons.')
        with self.lock:
            data = self.load(identity)
            if action == 'stop':
                if identity in self.active:
                    data['status'] = 'stopping'
                    self.save(data)
                return self.public(data)
            if action == 'rerun':
                self.reserve()
                try:
                    source = identity
                    identity = uuid.uuid4().hex
                    root = self.directory(identity)
                    root.mkdir()
                    (root / 'weights').mkdir()
                    (root / 'images').mkdir()
                    data.update(id=identity, created=time.time(), status='preparing', error='', source_comparison=source)
                    data['cells'] = [{'id': uuid.uuid4().hex, 'column': c['column'], 'seed': c['seed'],
                                      'status': 'pending', 'images': []} for c in data['cells']]
                    self.save(data)
                    self.launch(identity, prepare=True)
                    return self.public(data)
                except Exception:
                    self.service.job_lock.release()
                    raise
            if action != 'retry':
                raise ValueError('Unknown comparison action')
            if identity in self.active:
                raise Conflict('This comparison is already active')
            # A lost HTTP response can mean the cell was submitted; never silently duplicate it.
            if any(c['status'] == 'submitting' and not c.get('prompt_id') for c in data['cells']):
                raise Conflict('A submission response was lost. Check ComfyUI history before starting a new comparison; this run cannot safely resubmit that cell.')
            self.reserve()
            try:
                for cell in data['cells']:
                    if cell['status'] == 'failed':
                        cell.update(status='pending', error='')
                        cell.pop('prompt_id', None)
                data.update(status='running', error='')
                self.save(data)
                self.launch(identity, prepare=True)
            except Exception:
                self.service.job_lock.release()
                raise
            return self.public(data)

    def freeze(self, data):
        _, run = run_root(self.service.store, data['job'])
        root = self.directory(data['id'])
        for column in data['columns']:
            if column.get('baseline'):
                continue
            if column.get('digest'):
                destination = root / 'weights' / (column['digest'] + '.safetensors')
                if not destination.exists() and data.get('source_comparison'):
                    source = self.directory(data['source_comparison']) / 'weights' / destination.name
                    temporary = destination.with_suffix('.tmp')
                    shutil.copyfile(source, temporary)
                    temporary.replace(destination)
                if not checkpoint_info(destination)[1]:
                    raise ValueError('A retained checkpoint snapshot is missing or incomplete. Create a new comparison from Results.')
                continue
            source = artifact_path(run, column['path'])
            if not checkpoint_info(source)[1]:
                raise ValueError('A selected checkpoint is incomplete. Refresh Results and try again.')
            before = source.stat()
            temporary = root / 'weights' / 'snapshot.tmp'
            digest = hashlib.sha256()
            with source.open('rb') as stream, temporary.open('wb') as output:
                while block := stream.read(1024 * 1024):
                    if self.stopped.is_set():
                        raise RuntimeError('Studio is shutting down; resume the comparison after restart.')
                    digest.update(block)
                    output.write(block)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or not checkpoint_info(temporary)[1]:
                temporary.unlink(missing_ok=True)
                raise ValueError('A checkpoint changed during preparation. Try again after training is paused.')
            column['digest'] = digest.hexdigest()
            temporary.replace(root / 'weights' / (column['digest'] + '.safetensors'))
            with self.lock:
                current = self.load(data['id'])
                current['columns'] = data['columns']
                self.save(current)

    def grant(self, identity, digest):
        token = secrets.token_hex(32)
        with self.lock:
            self.capabilities = {t: v for t, v in self.capabilities.items() if v[0] > time.monotonic()}
            self.capabilities[token] = (time.monotonic() + 60, identity, digest)
        return token

    def capability(self, token):
        with self.lock:
            grant = self.capabilities.pop(token, None) if isinstance(token, str) else None
        if not grant or grant[0] < time.monotonic():
            raise KeyError('Checkpoint capability expired')
        _, identity, digest = grant
        root = self.directory(identity)
        return {'app': 'lora-studio', 'version': 1, 'root': str(root),
                'path': str(root / 'weights' / (digest + '.safetensors')), 'sha256': digest}

    def update_cell(self, identity, cell):
        with self.lock:
            data = self.load(identity)
            data['cells'] = [cell if c['id'] == cell['id'] else c for c in data['cells']]
            self.save(data)

    def collect(self, data, cell, history):
        entry = history.get(cell['prompt_id'])
        if not entry:
            return False
        status = entry.get('status', {})
        if status.get('status_str') == 'error':
            raise ValueError('ComfyUI could not generate this cell. Check its console, then retry failed cells.')
        if not status.get('completed'):
            return False
        images = []
        for node in data['profile']['resultNodeIds']:
            images.extend(entry.get('outputs', {}).get(str(node), {}).get('images', []))
        if not images:
            raise ValueError('Workflow completed without an image from its configured result nodes')
        saved = []
        for index, image in enumerate(images[:8]):
            query = urllib.parse.urlencode({k: image[k] for k in ('filename', 'subfolder', 'type') if k in image})
            url = data['endpoint'] + '/view?' + query
            with urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open(url, timeout=60) as response:
                raw = response.read(64 * 1024 * 1024 + 1)
            if len(raw) > 64 * 1024 * 1024:
                raise ValueError('Generated image exceeds the local comparison size limit')
            name = cell['id'] + f'-{index}.png'
            with Image.open(io.BytesIO(raw)) as img:
                img.convert('RGB').save(self.directory(data['id']) / 'images' / name)
            saved.append(name)
        cell.update(status='complete', images=saved, error='')
        return True

    def wait_cell(self, data, cell):
        missing = 0
        while not self.stopped.is_set():
            history = request(data['endpoint'] + '/history/' + urllib.parse.quote(cell['prompt_id']))
            if self.collect(data, cell, history):
                return
            queue = request(data['endpoint'] + '/queue')
            present = any(row[1] == cell['prompt_id'] for key in ('queue_running', 'queue_pending') for row in queue.get(key, []))
            missing = 0 if present else missing + 1
            if missing >= 3:
                raise ValueError('This cell is no longer queued and has no saved ComfyUI result. Retry to generate it again.')
            self.stopped.wait(2)
        raise RuntimeError('Studio stopped. Resume to retrieve the queued result.')

    def worker(self, identity, prepare):
        reservation = None
        try:
            data = self.load(identity)
            if prepare:
                self.freeze(data)
            with self.lock:
                data = self.load(identity)
                if data['status'] != 'stopping':
                    data['status'] = 'running'
                    self.save(data)
            registrations = {}
            from . import llm
            settings = self.service.settings()
            owned_llm = llm._BACKENDS.get(settings['_llm_root'])
            if owned_llm is not None:
                owned_llm.release_used(settings)
            reservation = ComparisonReservation(data['endpoint'], identity,
                lambda: self.stopped.is_set() or self.load(identity)['status'] == 'stopping')
            ready = reservation.begin()
            for cell in data['cells']:
                if not ready or self.stopped.is_set() or self.load(identity)['status'] == 'stopping':
                    break
                reservation.check()
                if cell['status'] == 'complete':
                    continue
                try:
                    if not cell.get('prompt_id'):
                        column = data['columns'][cell['column']]
                        registered = None
                        if not column.get('baseline'):
                            digest = column['digest']
                            if digest not in registrations:
                                registrations[digest] = request(data['endpoint'] + '/promptstudio/lora-studio/register',
                                    {'origin': self.origin, 'token': self.grant(identity, digest)}, timeout=120)['checkpoint_id']
                            registered = registrations[digest]
                        graph = cell_graph(data, cell, registered)
                        reservation.check()
                        cell['status'] = 'submitting'
                        self.update_cell(identity, cell)
                        queued = request(data['endpoint'] + '/prompt', {'prompt': graph, 'client_id': 'lora-studio-' + identity,
                            'extra_data': {'lora_studio': {'comparison': identity, 'cell': cell['id']}}}, timeout=120,
                            headers={'X-LoRA-Studio-Reservation': reservation.token})
                        if not queued.get('prompt_id'):
                            raise RuntimeError('ComfyUI returned no prompt identifier')
                        cell.update(status='queued', prompt_id=queued['prompt_id'])
                        self.update_cell(identity, cell)
                    self.wait_cell(data, cell)
                except Exception as exc:
                    if cell['status'] == 'submitting' and isinstance(exc, IntegrationHTTPError) and exc.status == 409:
                        cell.update(status='pending', error='Generation reservation expired. Resume to continue.')
                        self.update_cell(identity, cell)
                        raise
                    if cell['status'] == 'submitting' and isinstance(exc, IntegrationHTTPError) and exc.status == 400:
                        cell.update(status='failed', error='ComfyUI rejected the workflow. Check its missing nodes or model selections, then retry.')
                        self.update_cell(identity, cell)
                        continue
                    # Retain known prompt IDs on transport failure to reconcile on resume.
                    if cell.get('prompt_id') and not isinstance(exc, ValueError):
                        cell.update(status='queued', error='Connection interrupted. Resume to check the queued result.')
                        self.update_cell(identity, cell)
                        raise
                    if cell['status'] == 'submitting':
                        self.update_cell(identity, cell)
                        raise RuntimeError('Submission could not be confirmed. Check ComfyUI history before creating another comparison.') from exc
                    cell.update(status='failed', error=str(exc))
                self.update_cell(identity, cell)
            with self.lock:
                data = self.load(identity)
                data['status'] = ('complete' if all(c['status'] == 'complete' for c in data['cells']) else
                                  'stopped' if data['status'] == 'stopping' else 'failed')
                self.save(data)
        except Exception as exc:
            with self.lock:
                data = self.load(identity)
                data.update(status='interrupted', error=str(exc))
                self.save(data)
        finally:
            if reservation:
                reservation.close()
            self.active.discard(identity)
            self.service.job_lock.release()

    def image(self, identity, name):
        data = self.load(identity)
        if name not in {n for cell in data['cells'] for n in cell['images']}:
            raise KeyError('Comparison image not found')
        path = (self.directory(identity) / 'images' / name).resolve()
        if path.parent != self.directory(identity) / 'images':
            raise ValueError('Invalid comparison image')
        return path

    def composite(self, identity):
        with self.lock:
            data = self.load(identity)
        width, height = 256, 300
        seeds, columns = data['settings']['seeds'], data['columns']
        # Bounded preview tiles keep the export predictable even for large comparisons.
        canvas = Image.new('RGB', (100 + width * len(columns), 60 + height * len(seeds)), '#19171f')
        draw = ImageDraw.Draw(canvas)
        for col, column in enumerate(columns):
            draw.text((110 + col * width, 20), column['label'], fill='white')
        for row, seed in enumerate(seeds):
            draw.text((8, 75 + row * height), f'Seed\n{seed}', fill='white')
            for col in range(len(columns)):
                cell = next(c for c in data['cells'] if c['column'] == col and c['seed'] == seed)
                x, y = 100 + col * width, 60 + row * height
                if cell['images']:
                    with Image.open(self.image(identity, cell['images'][0])) as img:
                        tile = ImageOps.contain(img.convert('RGB'), (width - 12, height - 30))
                    canvas.paste(tile, (x + (width - tile.width) // 2, y + (height - 30 - tile.height) // 2))
                else:
                    draw.text((x + 12, y + 30), cell['status'], fill='white')
        path = self.directory(identity) / 'grid.png'
        with self.lock:
            canvas.save(path)
        return path

    def close(self):
        self.stopped.set()
