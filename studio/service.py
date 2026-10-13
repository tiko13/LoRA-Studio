import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
import io
import copy
import shutil
from contextlib import nullcontext

from .domain import caption_instruction, intent, intent_digest, parse_caption, bounded_text
from .integrations import PromptStudio, local_url, standalone_caption, request
from .store import Store, Conflict
from .trainer import backend_status, export_run, preflight
from .processes import running, terminate_tree
from .preparation import metadata, plan, decorate, normalize_options, normalize_coverage, png_bytes
from .captioners import CAPTION_SOURCES, raw_caption_instruction, captioner_status, joy_files, joy_image, joy_runtime
from .caption_processing import format_observations, caption_text_current
from .verification import normalize_checks, instruction as verification_instruction, parse_verification, evidence_digest, verification_state
from . import sample_prompts
from . import object_preparation as objects
from . import llm

DEFAULT_SETTINGS = {"integration": "promptstudio", "prompt_studio_url": "http://127.0.0.1:8188",
                    "unload_comfy": True, "llm_url": "http://127.0.0.1:8080/v1", "llm_model": "",
                    "toolkit_path": "", "toolkit_python": "", "detector_python": "",
                    "caption_source": "connected", "caption_url": "http://127.0.0.1:8091/v1", "caption_model": "",
                    "caption_runtime": "", "joycaption_model": "", "joycaption_projector": "", "caption_gpu": 0,
                    "settings_revision": 0}
DEFAULT_SETTINGS.update(llm.DEFAULTS)

class Service:
    def __init__(self, root, *, owner_url=None):
        self.store = Store(root)
        self.owner_url = local_url(owner_url) if owner_url else None
        self._monitor = None
        self._setup = None
        self.lock = threading.RLock()
        self.job_lock = threading.Lock()
        self.cancel = {}
        self.processes = {}
        self.reconnected = set()
        self.metadata_cache = {}
        for job in self.store.items("jobs"):
            if not owner_url and job["status"] in {"queued", "running", "stopping", "interrupted"}:
                if job['kind'] == 'train':
                    from .recovery import reconnect
                    reconnect(self, job)
                else:
                    job.update(status="interrupted", error="Studio restarted. Completed captions and exported files are preserved.")
                    self.store.put_item("jobs", job)
            self.sync_training_lock(job)
        if self.reconnected:
            self.monitor
    def sync_training_lock(self, job, parsed=None):
        from .project_lifecycle import sync_training_lock
        sync_training_lock(self.store, job, parsed)
    def copy_project(self, pid):
        from .project_lifecycle import copy_project
        return copy_project(self, pid)
    def delete_project(self, pid, value):
        from .project_lifecycle import delete_project
        return delete_project(self, pid, value)
    def settings(self):
        saved = self.store.settings()
        data = {**copy.deepcopy(DEFAULT_SETTINGS), **saved, '_llm_root': str((self.store.root / 'llm').resolve())}
        if 'llm_provider' not in saved and saved.get('llm_model'):
            data['llm_provider'] = 'openai'
        return data
    @property
    def setup(self):
        with self.lock:
            if self._setup is None:
                from .setup import Setup
                self._setup = Setup(self)
            return self._setup
    def save_training_preset(self, job_id, value):
        from .presets import from_run
        job = self.store.item('jobs', job_id)
        if job['kind'] != 'train':
            raise ValueError('Choose a training run to save a preset')
        default = value.get('default', False)
        if not isinstance(default, bool):
            raise ValueError('Default must be true or false')
        root = self.store.root / 'runs' / job_id
        config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        return self.store.save_preset(from_run(value.get('name', ''), config, manifest, default=default))
    def apply_training_preset(self, pid, value):
        with self.lock:
            project = self.store.project(pid)
            preset = next((p for p in self.store.presets() if p['id'] == value.get('preset')), None)
            if not preset or preset['model'] != project['model']:
                raise ValueError('Choose a preset for this model')
            project.update(revision=value.get('revision'), training={**project['training'], **preset['training']},
                           training_preset=preset['id'], preparation=preset['preparation'])
            return self.store.save_project(project, pid)
    def preview_suite(self, project, images):
        from .monitoring import read_json
        if sample_prompts.mode(project) == 'manual':
            return sample_prompts.effective(project, images)
        path = self.store.root / 'sample-prompts' / (project['id'] + '.json')
        cached = read_json(path)
        if cached.get('signature') == sample_prompts.signature(project, images):
            return {k: v for k, v in cached.items() if k != 'history'}
        return {**sample_prompts.effective(project, images), 'needs_generation': True,
                'note': 'Starter preview. Training automatically tailors these prompts to the prepared dataset using the local LLM.'}
    def resolve_preview_suite(self, project, images, settings, *, force=False):
        from .monitoring import read_json, write_json
        current = self.preview_suite(project, images)
        if sample_prompts.mode(project) == 'manual' or (not force and not current.get('needs_generation')):
            return current
        result = sample_prompts.generate(project, images, settings)
        with self.lock:
            fresh = self.store.project(project['id'])
            self.store.assert_editable(project['id'])
            if fresh['revision'] != project['revision'] or sample_prompts.signature(fresh, self.store.items('images', project['id'])) != result['signature']:
                raise Conflict('The target or dataset changed while preview prompts were generated. Generate again; manual prompts are preserved.')
            path = self.store.root / 'sample-prompts' / (project['id'] + '.json')
            path.parent.mkdir(exist_ok=True)
            previous = read_json(path)
            history = previous.pop('history', [])
            if previous:
                history.append(previous)
            write_json(path, {**result, 'history': history[-10:]})
        return result
    def generate_preview_suite(self, pid):
        project, images, settings = self.store.project(pid), self.dataset(pid), self.settings()
        if sample_prompts.mode(project) != 'automatic':
            raise ValueError('Switch to Automatic to generate a suite; manual prompts are kept separately')
        if not project['trigger']:
            raise ValueError('Set the trigger in Base info first')
        def work(job, stop):
            job['phase'] = 'Generating preview prompts'
            self.store.put_item('jobs', job)
            suite = self.resolve_preview_suite(project, images, settings, force=True)
            job['messages'].append(suite.get('note', 'Preview prompts are ready in Training settings.'))
        return self.new_job(pid, 'preview_prompts', work)
    def save_settings(self, value):
        with self.lock:
            return self._save_settings(value)
    def llm_profile(self, value):
        with self.lock:
            settings = self.settings()
            result = llm.backend(settings).profile(value)
            if result.get('deleted') == settings['llamacpp_config_profile']:
                self._save_settings({'llamacpp_config_profile': '', 'llamacpp_autostart': False})
            return result
    def llm_process(self, action):
        settings = self.settings()
        if settings['integration'] != 'standalone':
            raise ValueError('Select the standalone backend to manage its server.')
        if action == 'status':
            return llm.backend(settings).process(action, settings)
        if not self.job_lock.acquire(blocking=False):
            raise Conflict('Wait for the current Studio job before changing the LLM server.')
        try:
            if any(j['kind'] == 'train' and j['status'] == 'interrupted' for j in self.store.items('jobs')):
                raise Conflict('Check the interrupted training process before starting or changing an LLM server.')
            return llm.backend(settings).process(action, settings)
        finally:
            self.job_lock.release()
    def _save_settings(self, value):
        revision = self.settings()['settings_revision']
        if 'settings_revision' in value and value['settings_revision'] != revision:
            raise Conflict('Settings changed in another window. Reopen setup or Settings to load the latest values; your current form is preserved.')
        data = {**self.settings(), **{k: v for k, v in value.items() if k in DEFAULT_SETTINGS}}
        # Legacy clients and dedicated captioners retain their compatible endpoint.
        if 'llm_model' in value and 'llm_provider' not in value and 'llm_provider' not in self.store.settings():
            data['llm_provider'] = 'openai'
        llm.validate(data)
        if data["integration"] not in {"auto", "promptstudio", "standalone"}:
            raise ValueError("Unknown integration mode")
        if not isinstance(data["caption_source"], str) or data["caption_source"] not in CAPTION_SOURCES:
            raise ValueError("Unknown caption source")
        if isinstance(data["caption_gpu"], bool) or not isinstance(data["caption_gpu"], int) or not 0 <= data["caption_gpu"] <= 15:
            raise ValueError("Caption GPU must be an integer from 0 to 15")
        for key in ("prompt_studio_url", "llm_url", "caption_url"):
            data[key] = local_url(bounded_text(data[key], 500, key))
        for key in ("llm_model", "toolkit_path", "toolkit_python", "detector_python", "caption_model", "caption_runtime", "joycaption_model", "joycaption_projector"):
            data[key] = bounded_text(data[key], 2000, key)
        if not isinstance(data["unload_comfy"], bool):
            raise ValueError("Unload setting must be true or false")
        data['settings_revision'] = revision + 1
        self.store.save_settings({k: v for k, v in data.items() if k != '_llm_root'})
        return data
    def connection(self, settings=None):
        settings = settings or self.settings()
        status = {"backend": backend_status(settings), "promptstudio": {"available": False}, "standalone": {"available": False},
                  "captioner": captioner_status(settings)}
        if settings["integration"] != "standalone":
            try:
                status["promptstudio"] = PromptStudio(settings["prompt_studio_url"]).inspect()
            except (ValueError, RuntimeError, KeyError) as exc:
                status["promptstudio"]["reason"] = str(exc)
        if not status["promptstudio"]["available"] and settings["integration"] != "promptstudio":
            try:
                if settings['llm_provider'] == 'openai':
                    result = request(settings["llm_url"] + "/models", timeout=5)
                    status["standalone"] = {"available": True, "models": [m["id"] for m in result.get("data", [])], 'configured': llm.configured(settings)}
                else:
                    status['standalone'] = llm.backend(settings).inspect(settings)
            except (ValueError, RuntimeError, KeyError) as exc:
                status["standalone"]["reason"] = str(exc)
        return status
    def image_path(self, image):
        return self.store.root / "images" / image["project"] / (image["id"] + image["extension"])
    def image_info(self, image):
        path = self.image_path(image)
        stat = path.stat()
        key = (image["id"], image["digest"], stat.st_size, stat.st_mtime_ns)
        if key not in self.metadata_cache:
            self.metadata_cache[key] = metadata(path)
        return dict(self.metadata_cache[key])
    def import_image(self, pid, value):
        self.store.assert_editable(pid)
        name = bounded_text(value.get("name", "image"), 255, "Filename")
        try:
            data = base64.b64decode(value["data"], validate=True)
        except (ValueError, KeyError, TypeError):
            raise ValueError("Invalid image payload")
        if not 8 <= len(data) <= 20 * 1024 * 1024:
            raise ValueError("Images must be smaller than 20 MB")
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            ext, mime = ".png", "image/png"
        elif data.startswith(b"\xff\xd8\xff") and data.endswith(b"\xff\xd9"):
            ext, mime = ".jpg", "image/jpeg"
        else:
            raise ValueError("Import PNG or JPEG images. The browser can convert WebP to PNG.")
        dimensions = metadata(io.BytesIO(data))
        with self.lock:
            self.store.assert_editable(pid)
            digest = hashlib.sha256(data).hexdigest()
            for existing in self.store.items("images", pid):
                if existing["digest"] == digest:
                    return {"image": existing, "duplicate": True}
            item = {"id": uuid.uuid4().hex, "project": pid, "name": name, "extension": ext, "mime": mime,
                    "digest": digest, "caption": "", "observations": "", "warnings": [], "included": True,
                    "reviewed": False, "manual": False, "revision": 1, "history": [], "created": time.time(),
                    **dimensions}
            path = self.image_path(item)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.store.put_item("images", item)
        return {"image": item, "duplicate": False}
    def dataset(self, pid):
        project = self.store.project(pid)
        images = self.store.items("images", pid)
        for item in images:
            if "width" not in item:
                try:
                    item.update(self.image_info(item))
                except (ValueError, OSError):
                    item["preparation_error"] = "Cannot decode original image. Re-import a valid image or exclude this one."
        for index, item in enumerate(images):
            candidates = []
            if item.get("dhash"):
                for other in images[:index]:
                    if other.get("dhash") and abs(item["width"] / item["height"] - other["width"] / other["height"]) < .1:
                        if (int(item["dhash"], 16) ^ int(other["dhash"], 16)).bit_count() <= 4:
                            candidates.append(other["id"])
                            other.setdefault("duplicate_candidates", []).append(item["id"])
            item.setdefault("duplicate_candidates", []).extend(candidates)
        images = [decorate(project, item) if "width" in item else item for item in images]
        for item in images:
            if 'plan' in item:
                item['visual_review'] = verification_state(project, item)
        return images

    def preview(self, image_id, options=None, object_target=None):
        item = self.store.item("images", image_id)
        item.update(self.image_info(item))
        if options is not None:
            item["preparation"] = normalize_options(options, inherit=True)
        project = self.store.project(item['project'])
        if object_target is not None:
            if project['kind'] != 'object':
                raise ValueError('Object targeting applies only to Object projects')
            item['object_target'] = objects.options(object_target)
        geometry = plan(project, item)
        result = {"plan": geometry, "png": base64.b64encode(png_bytes(self.image_path(item), geometry)).decode()}
        if project['kind'] == 'object' and objects.options(item.get('object_target'))['mask'] != 'none':
            buffer = io.BytesIO()
            objects.render_mask(self.image_path(item), project, item, geometry).save(buffer, format='PNG')
            result['mask_png'] = base64.b64encode(buffer.getvalue()).decode()
        return result

    def find_objects(self, pid, *, _job=None, _settings=None):
        project = self.store.project(pid)
        if project['kind'] != 'object':
            raise ValueError('Object targeting applies only to Object projects')
        settings = _settings or self.settings()
        images = [i for i in self.dataset(pid) if i['included'] and objects.selection(project, i)['status'] != 'ready']
        def work(job, stop):
            bridge, connection = None, None
            if images and settings['integration'] != 'standalone':
                for attempt in range(2):
                    try:
                        bridge = PromptStudio(settings['prompt_studio_url']); connection = bridge.connection()
                        capability = request(bridge.url + '/promptstudio/prompt-studio/vision-capability', connection)
                        if not capability.get('available'):
                            raise ValueError('Select a vision-capable connected model to locate objects, or draw rectangles manually.')
                        break
                    except RuntimeError:
                        if attempt == 0:
                            if stop.wait(.5):return
                            continue
                        if settings['integration'] == 'promptstudio' or not llm.configured(settings):
                            raise
                        bridge = None
            job.update(phase='Locating training objects', completed=0, total=len(images))
            self.store.put_item('jobs', job)
            for image in images:
                if stop.is_set():
                    return
                data = objects.image_bytes(self.image_path(image))
                prompt = objects.instruction(project, image)
                raw = bridge.caption(prompt, data, 'object.png', connection, temperature=0, max_tokens=1000) if bridge else standalone_caption(settings, prompt, data, 'image/png', temperature=0, max_tokens=1000)
                found = objects.parse(raw, project, image)
                with self.lock:
                    current = self.store.item('images', image['id'])
                    if stop.is_set():
                        return
                    if current['revision'] != image['revision'] or self.store.project(pid)['revision'] != project['revision']:
                        raise Conflict('The target or image changed during object selection. Completed selections are preserved; retry preparation.')
                    current.setdefault('object_history', []).append({'detection': current.get('object_detection'), 'target': current.get('object_target'), 'at': time.time()})
                    current.update(object_detection=found, revision=current['revision'] + 1)
                    self.store.put_item('images', current)
                job['completed'] += 1
                self.store.put_item('jobs', job)
            unresolved = sum(i['included'] and objects.selection(project, i)['status'] != 'ready' for i in self.dataset(pid))
            if unresolved:
                job['messages'].append(f'{unresolved} image(s) need target review. Refine the hint, draw a rectangle, keep the entire image explicitly, or exclude them.')
        return work(*_job) if _job is not None else self.new_job(pid, 'objects', work)

    def check_people(self, pid):
        if self.store.project(pid)['kind'] != 'character':
            raise ValueError('Person counts apply only to Person / character projects. Use the selected category visual checks instead.')
        settings = self.settings()
        executable = settings.get("detector_python") or settings.get("toolkit_python")
        if not executable or not Path(executable).is_file():
            raise ValueError("Set a Python executable with Torch and Torchvision in Settings (or configure AI Toolkit Python).")
        images = self.store.items("images", pid)
        if not images:
            raise ValueError("Import images first")
        def work(job, stop):
            root = self.store.root / "analysis" / job["id"]
            root.mkdir(parents=True)
            request_path, output_path = root / "request.json", root / "results.jsonl"
            request_path.write_text(json.dumps([{"id": i["id"], "path": str(self.image_path(i))} for i in images]), encoding="utf-8")
            job["total"] = len(images)
            job["messages"].append("CPU person detection only; counts are advisory. First use downloads approximately 14 MB of public weights. No images leave this computer.")
            self.store.put_item("jobs", job)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", TORCH_HOME=str(self.store.root / "models"), PYTHONUNBUFFERED="1")
            with (root / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen([executable, str(Path(__file__).with_name("person_detector.py")),
                    str(request_path), str(output_path), str(self.store.root / "models")],
                    env=environment, cwd=root, stdout=log, stderr=log,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), start_new_session=os.name != "nt")
                self.processes[job["id"]] = process
                try:
                    seen = 0
                    started = time.monotonic()
                    while True:
                        done = process.poll() is not None
                        if output_path.exists():
                            lines = output_path.read_text(encoding="utf-8").splitlines(keepends=True)
                            for line in lines[seen:]:
                                if not line.endswith("\n"):
                                    break
                                result = json.loads(line)
                                with self.lock:
                                    current = self.store.item("images", result["id"])
                                    current["detection"] = result["detection"]
                                    current["revision"] += 1
                                    self.store.put_item("images", current)
                                seen += 1
                                job["completed"] = seen
                                self.store.put_item("jobs", job)
                        if done:
                            break
                        if stop.wait(.4) or time.monotonic() - started > max(300, len(images) * 30):
                            terminate_tree(process)
                            if not stop.is_set():
                                raise RuntimeError("Person detector timed out. Review images manually or retry with a smaller dataset.")
                            return
                    if process.returncode or seen != len(images):
                        raise RuntimeError("Person detector failed. Check that the selected Python has compatible torch, torchvision and Pillow, and can download the weights on first use. Images remain usable; review counts manually.")
                finally:
                    if process.poll() is None:
                        terminate_tree(process)
        return self.new_job(pid, "detect", work)
    def update_image(self, image_id, patch):
        with self.lock:
            item = self.store.item("images", image_id)
            self.store.assert_editable(item['project'])
            if patch.get("revision") != item["revision"]:
                raise Conflict("This caption changed while you edited it. Reopen the image to review the latest version.")
            if 'object_target' in patch:
                if self.store.project(item['project'])['kind'] != 'object':
                    raise ValueError('Object targeting applies only to Object projects')
                target = objects.options(patch['object_target'])
                if target != item.get('object_target'):
                    item.setdefault('object_history', []).append({'target': item.get('object_target'), 'detection': item.get('object_detection'), 'at': time.time()})
                    item['object_target'] = target
            if "preparation" in patch:
                options = normalize_options(patch["preparation"], inherit=True)
                if options != item.get("preparation"):
                    item.setdefault("preparation_history", []).append({"preparation": item.get("preparation"), "at": time.time()})
                    item["preparation"] = options
            if "person_count" in patch:
                if self.store.project(item['project'])['kind'] != 'character':
                    raise ValueError('Person count overrides apply only to Person / character projects.')
                count = patch["person_count"]
                if count is not None and (isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 100):
                    raise ValueError("Person count must be unknown or a whole number from 0 to 100")
                item.setdefault("count_history", []).append({"person_count": item.get("person_count"), "at": time.time()})
                item["person_count"] = count
            if "coverage_override" in patch:
                kind = self.store.project(item['project'])['kind']
                if patch.get('coverage_kind', kind) != kind:
                    raise Conflict('The project category changed. Reopen the image before saving coverage.')
                value = normalize_coverage(patch["coverage_override"], kind)
                previous = item.get('coverage_override', {}) if kind == 'character' else item.get('coverage_overrides', {}).get(kind, {})
                if value != previous:
                    item.setdefault("coverage_history", []).append({"coverage_override": item.get("coverage_override", {}), "at": time.time()})
                    item['coverage_history'][-1].update(coverage_override=previous, coverage_kind=kind)
                    if kind == 'character':
                        item['coverage_override'] = value
                    else:
                        item.setdefault('coverage_overrides', {})[kind] = value
            for key in ("included", "reviewed"):
                if key in patch:
                    if not isinstance(patch[key], bool):
                        raise ValueError("Image flags must be true or false")
                    item[key] = patch[key]
            if "caption" in patch:
                caption = bounded_text(patch["caption"], 6000, "Caption")
                if item["caption"] != caption:
                    item["history"].append({"caption": item["caption"], "at": time.time(), "source": "before manual edit"})
                    item["caption"] = caption
                    item["manual"] = True
                    item["reviewed"] = False
                    item.pop("intent_digest", None)
            if patch.get("reviewed"):
                project = self.store.project(item["project"])
                if not item["caption"] or (project["trigger"] and project["trigger"].casefold() not in item["caption"].casefold()):
                    raise ValueError("A reviewed caption must include the trigger and visible content")
                item["intent_digest"] = intent_digest(project)
                item["reviewed"] = True
                if "width" not in item:
                    item.update(metadata(self.image_path(item)))
                item["caption_preparation_digest"] = plan(project, item)["digest"]
            if "accept_advisories" in patch:
                if not isinstance(patch["accept_advisories"], bool):
                    raise ValueError("Override must be true or false")
                check_item = next(i for i in self.dataset(item["project"]) if i["id"] == item["id"])
                check_item.update(item)
                check_item = decorate(self.store.project(item["project"]), check_item)
                item["advisory_override"] = check_item["advisory_digest"] if patch["accept_advisories"] else None
            if 'accept_verification' in patch:
                if not isinstance(patch['accept_verification'],bool):
                    raise ValueError('Visual check override must be true or false')
                if 'width' not in item:item.update(self.image_info(item))
                check_item = decorate(self.store.project(item['project']), item)
                if patch['accept_verification'] and verification_state(self.store.project(item['project']),check_item)['status'] in {'unchecked','stale'}:
                    raise ValueError('This visual check is outdated. Save the edit and run a new check if needed.')
                item['verification_override'] = item.get('verification',{}).get('id') if patch['accept_verification'] else None
            item["revision"] += 1
            self.store.put_item("images", item)
            return item
    def new_job(self, pid, kind, callback):
        self.store.assert_editable(pid)
        if any(j['status'] in {'running', 'queued', 'stopping'} for j in self.store.items('jobs')):
            raise Conflict('Another job is active. Wait for it to finish or pause it.')
        if not self.job_lock.acquire(blocking=False):
            raise Conflict("Another caption or training job is active. Wait for it to finish.")
        try:
            # Deletion may have completed between the first check and reservation.
            self.store.assert_editable(pid)
        except Exception:
            self.job_lock.release()
            raise
        job = {"id": uuid.uuid4().hex, "project": pid, "kind": kind, "status": "queued", "created": time.time(),
               "completed": 0, "total": 0, "messages": []}
        event = threading.Event()
        self.cancel[job["id"]] = event
        self.store.put_item("jobs", job)
        def run():
            try:
                job["status"] = "running"
                self.store.put_item("jobs", job)
                callback(job, event)
                job["status"] = "cancelled" if event.is_set() else "complete"
            except Exception as exc:
                job.update(status="cancelled" if event.is_set() else "failed", error=str(exc))
            finally:
                job["finished"] = time.time()
                self.sync_training_lock(job)
                self.store.put_item("jobs", job)
                self.cancel.pop(job["id"], None)
                self.processes.pop(job["id"], None)
                self.job_lock.release()
        threading.Thread(target=run, daemon=True).start()
        return dict(job)
    def caption(self, pid, reuse_observations=False, only_outdated=False, *, _job=None, _settings=None):
        if not isinstance(reuse_observations, bool):
            raise ValueError("Reuse observations must be true or false")
        if not isinstance(only_outdated, bool) or only_outdated and not reuse_observations:
            raise ValueError("Only outdated captions requires text-only reprocessing")
        project = self.store.project(pid)
        settings = _settings if _settings is not None else self.settings()
        if not project["trigger"]:
            raise ValueError("Set the trigger in Base info first")
        if project['kind'] == 'object' and any(i['included'] and objects.selection(project, i)['status'] != 'ready' for i in self.dataset(pid)):
            raise ValueError('Locate the training objects first with Prepare dataset or Find objects, or select them manually in image review.')
        images = [i for i in self.dataset(pid) if i["included"] and not i["manual"] and
                  (reuse_observations or not i["caption"] or i.get("intent_digest") != intent_digest(project) or i.get("preparation_stale") or
                   project['kind'] == 'object' and i.get('object_caption_version') != 1 or
                   project['kind'] in {'style', 'concept'} and not caption_text_current(project, i) or
                   i.get('caption_source') in {'joycaption', 'custom'} and i.get('observation_kind', 'character') != project['kind'])]
        if reuse_observations:
            images = [i for i in images if i.get("observations") and (i.get("caption_source") in {"joycaption", "custom"} or project["kind"] in {"style", "concept"} and i.get("observation_kind") == project["kind"])]
            if any(i.get('observation_kind', 'character') != project['kind'] for i in images):
                raise ValueError('Saved observations use another category. Prepare fresh captions for the selected category first.')
            if only_outdated:
                images = [i for i in images if not caption_text_current(project, i)]
            if any(i.get("observation_preparation_digest", i.get("caption_preparation_digest")) != i["plan"]["digest"] for i in images):
                raise ValueError("Saved observations describe an older training crop. Prepare new visual observations before reprocessing text.")
        if not images:
            raise ValueError("No eligible saved observations. Manual captions are preserved." if reuse_observations else "No missing or outdated automatic captions. Manual captions are preserved.")
        source = settings["caption_source"]
        if source == "joycaption" and not reuse_observations:
            joy_files(settings)
        elif source == "custom" and not reuse_observations and not settings["caption_model"]:
            raise ValueError("Choose a custom caption model ID in Settings → Captioning.")
        def work(job, stop):
            bridge, connection = None, None
            if settings["integration"] != "standalone":
                try:
                    bridge = PromptStudio(settings["prompt_studio_url"])
                    connection = bridge.connection()
                    if source == "connected" and not reuse_observations:
                        capability = request(bridge.url + "/promptstudio/prompt-studio/vision-capability", connection)
                        if not capability.get("available"):
                            raise ValueError(capability.get("reason") or "The selected Prompt Studio model is not vision-capable")
                except RuntimeError:
                    if settings["integration"] == "promptstudio":
                        raise
                    bridge = None
            if bridge is None and not llm.configured(settings):
                raise ValueError("Configure the connected local LLM in Settings. Dedicated captioners use it to prepare the final intent-aware text.")
            job["total"] = len(images)
            job["connection"] = "Prompt Studio" if bridge else "Standalone local LLM"
            job["completed"] = 0
            job["caption_source"] = source
            job["reuse_observations"] = reuse_observations
            self.store.put_item("jobs", job)
            raw_observations = {i["id"]: i["observations"] for i in images} if reuse_observations else {}
            if reuse_observations:
                job.update(phase="Reprocessing saved observations", observed=len(images))
                self.store.put_item("jobs", job)
            if source != "connected" and not reuse_observations:
                folder = self.store.root / "caption-jobs" / job["id"]
                folder.mkdir(parents=True, exist_ok=True)
                if source == "joycaption" and settings["integration"] != "standalone":
                    try:
                        PromptStudio(settings["prompt_studio_url"]).free_generation_memory("captioning")
                    except RuntimeError:
                        if settings["integration"] == "promptstudio":
                            raise
                runtime = joy_runtime(settings, folder, stop, self.processes, job["id"]) if source == "joycaption" else nullcontext(
                    {"llm_url": settings["caption_url"], "llm_model": settings["caption_model"]})
                job.update(phase="Reading images", observed=0)
                self.store.put_item("jobs", job)
                with runtime as endpoint:
                    for image in images:
                        if stop.is_set():
                            return
                        if image.get("preparation_error"):
                            raise ValueError(image["preparation_error"])
                        if source == "joycaption" and settings["integration"] != "standalone":
                            # Yield instead of competing with newly queued generations.
                            try:
                                queue = request(settings["prompt_studio_url"] + "/queue")
                                if queue.get("queue_running") or queue.get("queue_pending"):
                                    raise ValueError("ComfyUI has new work. Captioning stopped; completed observations are retained in the caption job folder.")
                            except RuntimeError:
                                if settings["integration"] == "promptstudio":
                                    raise
                        data = png_bytes(self.image_path(image), image["plan"])
                        raw = standalone_caption(endpoint, raw_caption_instruction(project) + objects.caption_context(project, image), joy_image(data) if source == "joycaption" else data,
                                                 "image/png", temperature=0, max_tokens=900, system="You are a helpful image captioner.")
                        raw = bounded_text(raw, 12000, "Caption observations")
                        if not raw.strip():
                            raise ValueError("The caption model returned no observations. No caption was replaced.")
                        raw_observations[image["id"]] = raw
                        (folder / (image["id"] + ".json")).write_text(json.dumps({"image": image["id"], "revision": image["revision"],
                            "observations": raw, "source": source, "model": settings["joycaption_model"] if source == "joycaption" else settings["caption_model"]}), encoding="utf-8")
                        job["observed"] += 1
                        self.store.put_item("jobs", job)
                # Release the dedicated runtime before the connected LLM rewrites text.
                job["phase"] = "Preparing caption text"
                self.store.put_item("jobs", job)
            for image in images:
                if stop.is_set():
                    break
                if image.get("preparation_error"):
                    raise ValueError(image["preparation_error"])
                instruction = caption_instruction(project) + objects.caption_context(project, image)
                if source == "connected" and not reuse_observations:
                    data = png_bytes(self.image_path(image), image["plan"])
                    response = bridge.caption(instruction, data, "prepared.png", connection) if bridge else standalone_caption(settings, instruction, data, "image/png")
                    parsed = parse_caption(response, project)
                    if project['kind'] in {'object', 'style', 'concept'}:
                        # Apply the same evidence/field policy even when one provider
                        # supplies both vision and text. Free-form vision captions can
                        # otherwise leak KEEP traits that the dedicated path rejects.
                        current = self.store.item('images', image['id'])
                        if stop.is_set():
                            return
                        if current['revision'] != image['revision'] or intent_digest(self.store.project(pid)) != intent_digest(project):
                            if _job is not None:
                                raise Conflict('The target or image changed during captioning. Retry preparation.')
                            job['messages'].append('Skipped an image changed during captioning; its edits are preserved.')
                            continue
                        def refine(prompt):
                            if stop.is_set():
                                raise InterruptedError('Caption text processing stopped.')
                            return bridge.text(prompt, connection) if bridge else standalone_caption(settings, prompt, None, 'image/png', max_tokens=3000, temperature=0)
                        visual = parsed
                        parsed = format_observations(project, visual['observations'], refine)
                        parsed['warnings'] = list(dict.fromkeys(visual['warnings'] + parsed['warnings']))
                        if project['kind'] in {'style', 'concept'}:
                            parsed['observation_warnings'] = visual['warnings']
                        parsed['coverage'] = visual['coverage'] or parsed.get('coverage', {})
                        if project['kind'] == 'concept':
                            parsed['coverage_intent_digest'] = intent_digest(project)
                        parsed['observation_preparation_digest'] = image['plan']['digest']
                        parsed['observation_kind'] = project['kind']
                else:
                    def complete_text(prompt):
                        if stop.is_set():
                            raise InterruptedError("Caption text processing stopped.")
                        return bridge.text(prompt, connection) if bridge else standalone_caption(settings, prompt, None, "image/png", max_tokens=3000, temperature=0)
                    parsed = format_observations(project, raw_observations[image["id"]], complete_text)
                    parsed["observation_preparation_digest"] = image["plan"]["digest"]
                    parsed['observation_kind'] = project['kind']
                    if reuse_observations and project['kind'] in {'style', 'concept'} and image.get('coverage_kind') == project['kind']:
                        parsed['coverage'] = image.get('coverage', {})
                        if project['kind'] == 'concept':
                            parsed['coverage_intent_digest'] = image.get('coverage_intent_digest')
                            if parsed['coverage_intent_digest'] != intent_digest(project):
                                parsed['coverage'] = {}
                                parsed['warnings'].append('Learning target changed. Saved observations were reused; run a fresh visual check to assess the current concept.')
                    if reuse_observations and project['kind'] in {'style', 'concept'}:
                        parsed['observation_warnings'] = image.get('observation_warnings', [])
                        parsed['warnings'] = list(dict.fromkeys(parsed['observation_warnings'] + parsed['warnings']))
                parsed["caption_source"] = image["caption_source"] if reuse_observations else source
                parsed["caption_preparation_digest"] = image["plan"]["digest"]
                with self.lock:
                    current = self.store.item("images", image["id"])
                    if current["revision"] != image["revision"] or intent_digest(self.store.project(pid)) != intent_digest(project) or stop.is_set():
                        if _job is not None and not stop.is_set():
                            raise Conflict('An image changed during automatic preparation. Completed captions are saved; click Train again after reviewing the change.')
                        job["messages"].append(f"Skipped {image['name']}: changed during captioning or stopped.")
                    else:
                        current["history"].append({"caption": current["caption"], "observations": current.get("observations", ""),
                            "warnings": current.get('warnings', []), "observation_warnings": current.get('observation_warnings', []),
                            "caption_source": current.get("caption_source", "connected"),
                            "caption_facts": current.get("caption_facts", []), "caption_formatter_version": current.get("caption_formatter_version"),
                            "observation_kind": current.get('observation_kind', 'character'),
                            "coverage": current.get('coverage', {}), "coverage_kind": current.get('coverage_kind', 'character'),
                            "coverage_intent_digest": current.get('coverage_intent_digest'),
                            "concept_caption_version": current.get('concept_caption_version'),
                            "caption_omitted_facts": current.get("caption_omitted_facts", []),
                            "at": time.time(), "source": "before text reprocessing" if reuse_observations else "before vision caption"})
                        current.update(parsed, reviewed=False, revision=current["revision"] + 1)
                        self.store.put_item("images", current)
                job["completed"] += 1
                self.store.put_item("jobs", job)
        if _job is not None:
            return work(*_job)
        return self.new_job(pid, "caption", work)

    def save_verification(self, project, image, findings, *, review_notes=None, focus=''):
        """Store advisory evidence only, with the same revision protections as captioning."""
        with self.lock:
            current = self.store.item('images',image['id'])
            if 'width' not in current:current.update(self.image_info(current))
            fresh_project = self.store.project(project['id'])
            fresh = decorate(fresh_project,current)
            if current['revision'] != image['revision'] or evidence_digest(fresh_project,fresh) != evidence_digest(project,image):
                raise Conflict('Image or learning rules changed during the visual check. The previous review is preserved; retry the check.')
            previous=current.get('verification')
            notes=review_notes if review_notes is not None else (previous.get('review_notes',[]) if previous and previous.get('digest')==evidence_digest(project,image) else [])
            if previous:current.setdefault('verification_history',[]).append(previous)
            current['verification']={'id':uuid.uuid4().hex,'at':time.time(),'digest':evidence_digest(project,image),
                'source':'connected','caption':image['caption'],'findings':findings,'review_notes':notes,'focus':focus}
            current['verification_override']=None
            current['revision']+=1
            self.store.put_item('images',current)

    def verify(self, pid, image_ids, checks, focus=''):
        checks=normalize_checks(checks, self.store.project(pid)['kind'])
        focus=bounded_text(focus,600,'Visual question')
        if not isinstance(image_ids,list) or not image_ids or len(image_ids)>1000 or any(not isinstance(i,str) for i in image_ids):
            raise ValueError('Choose images to check.')
        project,settings=self.store.project(pid),self.settings()
        available={i['id']:i for i in self.dataset(pid)}
        if any(i not in available for i in image_ids):raise ValueError('An image does not belong to this project.')
        images=[available[i] for i in dict.fromkeys(image_ids)]
        if any(not i['caption'] or i.get('preparation_error') for i in images):raise ValueError('Prepare captions and valid training images before checking them.')
        def work(job,stop):
            bridge,connection=None,None
            if settings['integration']!='standalone':
                try:
                    bridge=PromptStudio(settings['prompt_studio_url']);connection=bridge.connection()
                    capability=request(bridge.url+'/promptstudio/prompt-studio/vision-capability',connection)
                    if not capability.get('available'):raise ValueError('The connected LLM needs image support for visual checks. Captions remain usable without this optional check.')
                except RuntimeError:
                    if settings['integration']=='promptstudio':raise
                    bridge=None
            job.update(total=len(images),phase='Checking visible details',checks=checks)
            self.store.put_item('jobs',job)
            for image in images:
                if stop.is_set():return
                data=png_bytes(self.image_path(image),image['plan'])
                prompt=verification_instruction(image,checks,focus,project)
                raw=bridge.caption(prompt,data,'prepared.png',connection,temperature=0,max_tokens=2500) if bridge else standalone_caption(settings,prompt,data,'image/png',temperature=0,max_tokens=2500)
                findings=parse_verification(raw,checks)
                if stop.is_set():return
                self.save_verification(project,image,findings,focus=focus)
                job['completed']+=1
                self.store.put_item('jobs',job)
        return self.new_job(pid,'verify',work)

    def automatic_train(self,pid):
        project=self.store.project(pid)
        if project.get('training_mode','review')!='automatic':
            return self.prepare(pid,True)
        return self.automatic_prepare(pid,train=True)

    def automatic_prepare(self,pid,*,train=False):
        """Prepare and freeze in one job. Training is an explicit, separate opt-in."""
        if not isinstance(train,bool):raise ValueError('Train must be true or false')
        project,settings=self.store.project(pid),self.settings()
        def work(job,stop):
            images=self.dataset(pid)
            expected={i['id']:i['revision'] for i in images}
            check=preflight(project,images,settings,automatic=True,preparing=True)
            if check['errors']:raise ValueError(' '.join(check['errors']))
            if project['kind'] == 'object':
                locating = [i for i in images if i['included'] and objects.selection(project, i)['status'] != 'ready']
                if locating:
                    self.find_objects(pid, _job=(job,stop), _settings=settings)
                    if stop.is_set():return
                    for i in locating:expected[i['id']] += 1
                    images = self.dataset(pid)
                    if self.store.project(pid)['revision'] != project['revision'] or {i['id']:i['revision'] for i in images} != expected:
                        raise Conflict('The dataset changed during object selection. Retry preparation after reviewing changes.')
                    if any(i['included'] and objects.selection(project, i)['status'] != 'ready' for i in images):
                        raise ValueError('Some training objects are ambiguous or absent. Review their target selections before preparing; no dataset was exported.')
            pending=[i for i in images if i['included'] and not i['manual'] and
                     (not i['caption'] or i.get('intent_digest')!=intent_digest(project) or i.get('preparation_stale') or
                      project['kind'] == 'object' and i.get('caption_source') == 'connected' and i.get('object_caption_version') != 1 or
                      project['kind'] in {'style', 'concept'} and i.get('caption_source') == 'connected' and not caption_text_current(project, i) or
                      i.get('caption_source') in {'joycaption','custom'} and not caption_text_current(project,i))]
            # Fail configuration/manual-caption errors before spending time on inference.
            check=preflight(project,images,settings,automatic=True,preparing=True)
            if check['errors']:raise ValueError(' '.join(check['errors']))
            job.update(automatic=True,phase='Preparing dataset')
            self.store.put_item('jobs',job)
            if pending:
                visual=[i for i in pending if not i['caption'] or i.get('intent_digest')!=intent_digest(project) or i.get('preparation_stale') or
                        project['kind'] == 'object' and i.get('caption_source') == 'connected' and i.get('object_caption_version') != 1 or
                      project['kind'] in {'style', 'concept'} and i.get('caption_source') == 'connected' and not caption_text_current(project, i) or
                        i.get('caption_source') in {'joycaption', 'custom'} and i.get('observation_kind', 'character') != project['kind']]
                if visual:self.caption(pid,_job=(job,stop),_settings=settings)
                if not stop.is_set() and any(i['included'] and not i['manual'] and i.get('caption_source') in {'joycaption','custom'} and
                        not caption_text_current(project,i) for i in self.dataset(pid)):
                    self.caption(pid,reuse_observations=True,only_outdated=True,_job=(job,stop),_settings=settings)
                for i in pending:expected[i['id']]+=1
            if stop.is_set():return
            with self.lock:
                fresh=self.store.project(pid)
                current=self.dataset(pid)
                if fresh['revision']!=project['revision'] or {i['id']:i['revision'] for i in current}!=expected:
                    raise Conflict('The project or dataset changed during preparation. Captions are preserved; click Train again to use the updated dataset.')
                job.update(phase='Freezing dataset',completed=0,total=0)
                self.store.put_item('jobs',job)
                # Prepare captures immutable input values before releasing the lock.
                launch=self.prepare(pid,train,_job=(job,stop),_settings=settings,_defer=True,automatic=True)
            launch()
        return self.new_job(pid,'train' if train else 'prepare',work)

    def prepare(self, pid, train=False, *, _job=None, _settings=None, _defer=False, automatic=False):
        project, settings = self.store.project(pid), _settings if _settings is not None else self.settings()
        images = self.dataset(pid)
        check = preflight(project, images, settings,automatic=automatic)
        if check["errors"]:
            raise ValueError(" ".join(check["errors"]))
        # A server restart cannot safely assert whether an old trainer has stopped.
        if train and any(j["kind"] == "train" and j["status"] == "interrupted" for j in self.store.items("jobs")):
            raise ValueError("An earlier training run was interrupted by a Studio restart. Check its process before launching another run; use export for now.")
        def work(job, stop):
            if stop.is_set():return
            job['review_policy']='automatic drafts and advisory overrides' if automatic else 'review first'
            job['messages'].extend(check['warnings'])
            job['phase'] = 'Preparing preview prompts'
            self.store.put_item('jobs', job)
            suite = self.resolve_preview_suite(project, images, settings)
            if stop.is_set():return
            if train:
                with self.lock:
                    self.store.lock_preparation(pid, job['id'], expected=project, images=images)
            root = export_run(self.store, {**project, '_preview_suite': suite}, images, job["id"], settings)
            job["bundle"] = True
            job["snapshot"] = {"name": project["name"], "model": project["model"], "steps": project["training"]["steps"]}
            self.store.put_item("jobs", job)
            if not train or stop.is_set():
                job['phase']='Dataset prepared' if not stop.is_set() else 'Stopped'
                return
            job['phase']='Training'
            if settings["integration"] != "standalone" and settings["unload_comfy"]:
                bridge = PromptStudio(settings["prompt_studio_url"])
                # Auto mode may work without ComfyUI, but a reachable busy instance must block.
                try:
                    request(bridge.url + "/system_stats", timeout=5)
                except RuntimeError:
                    if settings["integration"] == "promptstudio":
                        raise
                else:
                    bridge.free_generation_memory()
                    job["messages"].append("Requested ComfyUI model unload; the server remains running. Avoid new generations while training.")
            self.launch_training(job, stop, root, settings, project)
        if _job is not None:
            return (lambda:work(*_job)) if _defer else work(*_job)
        return self.new_job(pid, "train" if train else "export", work)

    @property
    def monitor(self):
        with self.lock:
            if self._monitor is None:
                from .monitoring import TrainingMonitor
                self._monitor = TrainingMonitor(self, self.owner_url)
            return self._monitor

    def launch_training(self, job, stop, root, settings, project, *, resume=False):
        used_backend = llm._BACKENDS.get(settings.get('_llm_root'))
        if used_backend is not None:
            used_backend.release_used(settings)
        from .monitoring import control_database, write_json, run_config
        config_path = root / 'config.json'
        config = run_config(root, job)
        native = Path(settings['toolkit_path']) / 'extensions_built_in/sd_trainer/DiffusionTrainer.py'
        # Preserve the frozen training recipe; attempt files contain only launch/control differences.
        attempts = root / 'attempts'
        attempts.mkdir(exist_ok=True)
        attempt = len(list(attempts.glob('*.json'))) + 1
        if native.is_file():
            config['config']['process'][0].update(type='diffusion_trainer',
                sqlite_db_path=str(control_database(root, job['id']).resolve()))
        config_path = attempts / f'{attempt:03d}.json'
        write_json(config_path, config)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(project['training']['gpu']),
                           PYTHONUNBUFFERED='1', SEED=str(project['training']['seed']), AITK_JOB_ID=job['id'])
        self.monitor
        with (root / 'training.log').open('a' if resume else 'w', encoding='utf-8') as log:
            if resume:
                log.write(f'\nSTUDIO_RESUME attempt={attempt} total={config["config"]["process"][0]["train"]["steps"]} at={time.time()}\n')
                log.flush()
            log_offset = (root / 'training.log').stat().st_size
            process = subprocess.Popen([settings['toolkit_python'], 'run.py', str(config_path.resolve())],
                cwd=settings['toolkit_path'], env=environment, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), start_new_session=os.name != 'nt')
            self.processes[job['id']] = process
            job.pop('process_identity', None)
            job.update(pid=process.pid, phase='Training', attempt=attempt, reconnected=False)
            self.store.put_item('jobs', job)
            from .processes import identity
            try:
                job['process_identity'] = identity(process.pid)
            except ValueError:
                # Keep supervising the live Popen even if OS identity access is
                # unavailable; recovery will require exact command verification.
                job['process_identity'] = None
            self.store.put_item('jobs', job)
            while process.poll() is None:
                self.sync_training_lock(job)
                if stop.wait(.5):
                    terminate_tree(process)
                    break
                if resume:
                    with (root / 'training.log').open('rb') as reader:
                        reader.seek(log_offset)
                        recent = reader.read().decode('utf-8', errors='replace')
                    if 'Failed to load optimizer state' in recent:
                        terminate_tree(process)
                        raise RuntimeError('The backend could not restore optimizer state. Resume stopped; inspect the run log.')
            if process.returncode and not stop.is_set():
                raise RuntimeError(f'AI Toolkit exited with code {process.returncode}. Open the run log.')
        if not stop.is_set() and not list((root / 'output').glob('**/*.safetensors')):
            raise RuntimeError('AI Toolkit exited without a LoRA file. Inspect the log; this run is not marked successful.')
        if not stop.is_set() and job.get('continuations'):
            from .monitoring import parse_log, resume_checkpoint
            from .results import checkpoint_info
            process_config = config['config']['process'][0]
            total = process_config['train']['steps']
            parsed = parse_log((root / 'training.log').read_text(encoding='utf-8', errors='replace'), total,
                               initial_total=project['training']['steps'])
            weights, _ = resume_checkpoint(root, parsed, total, validate=True, allow_complete=True,
                output=Path(process_config['training_folder']) / config['config']['name'])
            if parsed['saved_step'] != total or checkpoint_info(weights)[0] != total:
                raise RuntimeError('AI Toolkit exited before saving the continuation target. Inspect the run log.')

    def resume_training(self, job_id, *, additional_checkpoints=None):
        from .monitoring import write_json, parse_log, resume_checkpoint, run_config
        from .results import checkpoint_info
        extending = additional_checkpoints is not None
        if extending and (type(additional_checkpoints) is not int or not 1 <= additional_checkpoints <= 1000):
            raise ValueError('Additional checkpoints must be a whole number from 1 to 1000')
        with self.lock:
            job = self.store.item('jobs', job_id)
            if job['kind'] != 'train':
                raise ValueError('Only training runs can be continued')
            if extending and job['status'] != 'complete':
                raise ValueError('Only a completed run can be continued for additional checkpoints')
            if not extending and job['status'] not in {'paused', 'cancelled', 'failed', 'interrupted'}:
                raise ValueError('Only a stopped run can be resumed')
            if job.get('pid') and running(job['pid']):
                raise ValueError('The previous trainer is still running')
            if any(j['status'] in {'running', 'queued', 'stopping'} for j in self.store.items('jobs')):
                raise Conflict('Another job is active')
            root = self.store.root / 'runs' / job_id
            manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
            project, settings = manifest['project'], self.settings()
            frozen = run_config(root, job)
            process = frozen['config']['process'][0]
            total = process['train']['steps']
            parsed = parse_log((root / 'training.log').read_text(encoding='utf-8', errors='replace'), total,
                               initial_total=project['training']['steps'])
            weights, optimizer = resume_checkpoint(root, parsed, total, validate=True, allow_complete=extending,
                output=Path(process['training_folder']) / frozen['config']['name'])
            if extending:
                if parsed['saved_step'] != total or checkpoint_info(weights)[0] != total:
                    raise ValueError('The final checkpoint does not contain the completed training step; continuation is blocked.')
                interval = process['save'].get('save_every')
                if type(interval) is not int or interval < 1:
                    raise ValueError('This run has no positive checkpoint interval to continue with.')
            if not self.job_lock.acquire(blocking=False):
                raise Conflict('Another job is active')
            event = threading.Event()
            self.cancel[job_id] = event
            job.setdefault('attempt_history', []).append({**{k: job.get(k) for k in ('status', 'finished', 'pid', 'error')}, 'total': total})
            for key in ('finished', 'error'):
                job.pop(key, None)
            job.update(status='queued', phase='Continuing training' if extending else 'Resuming saved checkpoint')
            try:
                if extending:
                    job.setdefault('continuations', []).append({'from_step': total, 'total': total + interval * additional_checkpoints,
                        'checkpoints': additional_checkpoints, 'created': time.time()})
                    destination = root / 'output' / f"continuation-{len(job['continuations'])}" / frozen['config']['name']
                    destination.mkdir(parents=True, exist_ok=True)
                    # Seed a numbered checkpoint so the next final save has a fresh backend creation time.
                    seed = destination / f'{frozen["config"]["name"]}_{total:09d}.safetensors'
                    shutil.copy2(weights, seed)
                    shutil.copy2(optimizer, destination / 'optimizer.pt')
                    with (root / 'training.log').open('a', encoding='utf-8') as log:
                        log.write('\nSTUDIO_SEED ' + json.dumps({'step': total, 'path': str(seed)}) + '\n')
                write_json(root / 'monitor-control.json', {})
                self.store.put_item('jobs', job)
            except Exception:
                self.cancel.pop(job_id, None)
                self.job_lock.release()
                raise
        def work():
            try:
                job['status'] = 'running'
                self.store.put_item('jobs', job)
                if settings['unload_comfy'] and settings['integration'] != 'standalone':
                    try:
                        request(settings['prompt_studio_url'] + '/system_stats', timeout=5)
                    except RuntimeError:
                        if settings['integration'] == 'promptstudio':
                            raise
                    else:
                        PromptStudio(settings['prompt_studio_url']).free_generation_memory()
                self.launch_training(job, event, root, settings, project, resume=True)
                job['status'] = 'cancelled' if event.is_set() else 'complete'
            except Exception as exc:
                job.update(status='cancelled' if event.is_set() else 'failed', error=str(exc))
            finally:
                job['finished'] = time.time()
                self.store.put_item('jobs', job)
                self.cancel.pop(job_id, None)
                self.processes.pop(job_id, None)
                self.job_lock.release()
        threading.Thread(target=work, daemon=True).start()
        return dict(job)
    def stop(self, job_id):
        if job_id in self.reconnected:
            from .recovery import stop
            return stop(self, job_id)
        event = self.cancel.get(job_id)
        if not event:
            raise ValueError("This job is no longer running")
        event.set()
        return {"requested": True}
    def recover(self, job_id):
        with self.lock:
            job = self.store.item("jobs", job_id)
            if job["status"] != "interrupted" or job["kind"] != "train":
                raise ValueError("This run does not need process recovery")
            if running(job.get("pid")):
                raise ValueError("The earlier trainer process is still running. Wait for it to finish before starting another run.")
            job.update(status="failed", error="Studio restarted during this run. The recorded trainer process has now exited; inspect its files before starting a new run.")
            self.store.put_item("jobs", job)
            self.sync_training_lock(job)
            return job
    def run_files(self, job_id):
        self.store.item("jobs", job_id)
        root = self.store.root / "runs" / job_id
        files = []
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".safetensors", ".png", ".jpg", ".json", ".log", ".zip", ".pt", ".yaml", ".jsonl"} and "dataset" not in path.relative_to(root).parts:
                    stat = path.stat()
                    files.append({"path": path.relative_to(root).as_posix(), "size": stat.st_size, 'modified': stat.st_mtime})
        return files
