import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

from .domain import caption_instruction, intent, intent_digest, parse_caption, bounded_text
from .integrations import PromptStudio, local_url, standalone_caption, request
from .store import Store, Conflict
from .trainer import backend_status, export_run, preflight
from .processes import running, terminate_tree

DEFAULT_SETTINGS = {"integration": "auto", "prompt_studio_url": "http://127.0.0.1:8188",
                    "unload_comfy": True, "llm_url": "http://127.0.0.1:8080/v1", "llm_model": "",
                    "toolkit_path": "", "toolkit_python": ""}

class Service:
    def __init__(self, root):
        self.store = Store(root)
        self.lock = threading.RLock()
        self.job_lock = threading.Lock()
        self.cancel = {}
        self.processes = {}
        for job in self.store.items("jobs"):
            if job["status"] in {"queued", "running", "stopping"}:
                job.update(status="interrupted", error="Studio restarted. Completed captions and exported files are preserved.")
                self.store.put_item("jobs", job)
    def settings(self):
        return {**DEFAULT_SETTINGS, **self.store.settings()}
    def save_settings(self, value):
        data = {**self.settings(), **{k: v for k, v in value.items() if k in DEFAULT_SETTINGS}}
        if data["integration"] not in {"auto", "promptstudio", "standalone"}:
            raise ValueError("Unknown integration mode")
        for key in ("prompt_studio_url", "llm_url"):
            data[key] = local_url(bounded_text(data[key], 500, key))
        for key in ("llm_model", "toolkit_path", "toolkit_python"):
            data[key] = bounded_text(data[key], 2000, key)
        if not isinstance(data["unload_comfy"], bool):
            raise ValueError("Unload setting must be true or false")
        self.store.save_settings(data)
        return data
    def connection(self):
        settings = self.settings()
        status = {"backend": backend_status(settings), "promptstudio": {"available": False}, "standalone": {"available": False}}
        if settings["integration"] != "standalone":
            try:
                status["promptstudio"] = PromptStudio(settings["prompt_studio_url"]).inspect()
            except (ValueError, RuntimeError, KeyError) as exc:
                status["promptstudio"]["reason"] = str(exc)
        if not status["promptstudio"]["available"] and settings["integration"] != "promptstudio":
            try:
                result = request(settings["llm_url"] + "/models", timeout=5)
                status["standalone"] = {"available": True, "models": [m["id"] for m in result.get("data", [])]}
            except (ValueError, RuntimeError, KeyError) as exc:
                status["standalone"]["reason"] = str(exc)
        return status
    def image_path(self, image):
        return self.store.root / "images" / image["project"] / (image["id"] + image["extension"])
    def import_image(self, pid, value):
        self.store.project(pid)
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
        with self.lock:
            digest = hashlib.sha256(data).hexdigest()
            for existing in self.store.items("images", pid):
                if existing["digest"] == digest:
                    return {"image": existing, "duplicate": True}
            item = {"id": uuid.uuid4().hex, "project": pid, "name": name, "extension": ext, "mime": mime,
                    "digest": digest, "caption": "", "observations": "", "warnings": [], "included": True,
                    "reviewed": False, "manual": False, "revision": 1, "history": [], "created": time.time()}
            path = self.image_path(item)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.store.put_item("images", item)
        return {"image": item, "duplicate": False}
    def update_image(self, image_id, patch):
        with self.lock:
            item = self.store.item("images", image_id)
            if patch.get("revision") != item["revision"]:
                raise Conflict("This caption changed while you edited it. Reopen the image to review the latest version.")
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
            item["revision"] += 1
            self.store.put_item("images", item)
            return item
    def new_job(self, pid, kind, callback):
        self.store.project(pid)
        if not self.job_lock.acquire(blocking=False):
            raise Conflict("Another caption or training job is active. Wait for it to finish.")
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
                job.update(status="failed", error=str(exc))
            finally:
                job["finished"] = time.time()
                self.store.put_item("jobs", job)
                self.cancel.pop(job["id"], None)
                self.processes.pop(job["id"], None)
                self.job_lock.release()
        threading.Thread(target=run, daemon=True).start()
        return dict(job)
    def caption(self, pid):
        project = self.store.project(pid)
        settings = self.settings()
        if not project["description"] or not project["trigger"]:
            raise ValueError("Describe the learning target and set a trigger first")
        images = [i for i in self.store.items("images", pid) if i["included"] and not i["manual"] and
                  (not i["caption"] or i.get("intent_digest") != intent_digest(project))]
        if not images:
            raise ValueError("No missing or outdated automatic captions. Manual captions are preserved.")
        def work(job, stop):
            bridge, connection = None, None
            if settings["integration"] != "standalone":
                try:
                    bridge = PromptStudio(settings["prompt_studio_url"])
                    connection = bridge.connection()
                    capability = request(bridge.url + "/promptstudio/prompt-studio/vision-capability", connection)
                    if not capability.get("available"):
                        raise ValueError(capability.get("reason") or "The selected Prompt Studio model is not vision-capable")
                except RuntimeError:
                    if settings["integration"] == "promptstudio":
                        raise
                    bridge = None
            job["total"] = len(images)
            job["connection"] = "Prompt Studio" if bridge else "Standalone local LLM"
            self.store.put_item("jobs", job)
            for image in images:
                if stop.is_set():
                    break
                data = self.image_path(image).read_bytes()
                instruction = caption_instruction(project)
                response = bridge.caption(instruction, data, image["name"], connection) if bridge else standalone_caption(settings, instruction, data, image["mime"])
                parsed = parse_caption(response, project)
                with self.lock:
                    current = self.store.item("images", image["id"])
                    if current["revision"] != image["revision"] or stop.is_set():
                        job["messages"].append(f"Skipped {image['name']}: changed during captioning or stopped.")
                    else:
                        current["history"].append({"caption": current["caption"], "at": time.time(), "source": "before vision caption"})
                        current.update(parsed, reviewed=not bool(parsed["warnings"]), revision=current["revision"] + 1)
                        self.store.put_item("images", current)
                job["completed"] += 1
                self.store.put_item("jobs", job)
        return self.new_job(pid, "caption", work)
    def prepare(self, pid, train=False):
        project, settings = self.store.project(pid), self.settings()
        images = self.store.items("images", pid)
        check = preflight(project, images, settings)
        if check["errors"]:
            raise ValueError(" ".join(check["errors"]))
        # A server restart cannot safely assert whether an old trainer has stopped.
        if train and any(j["kind"] == "train" and j["status"] == "interrupted" for j in self.store.items("jobs")):
            raise ValueError("An earlier training run was interrupted by a Studio restart. Check its process before launching another run; use export for now.")
        def work(job, stop):
            root = export_run(self.store, project, images, job["id"], settings)
            job["bundle"] = True
            job["snapshot"] = {"name": project["name"], "model": project["model"], "steps": project["training"]["steps"]}
            self.store.put_item("jobs", job)
            if not train or stop.is_set():
                return
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
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(project["training"]["gpu"]), PYTHONUNBUFFERED="1", SEED=str(project["training"]["seed"]))
            with (root / "training.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen([settings["toolkit_python"], "run.py", str((root / "config.json").resolve())],
                    cwd=settings["toolkit_path"], env=environment, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), start_new_session=os.name != "nt")
                self.processes[job["id"]] = process
                job["pid"] = process.pid
                self.store.put_item("jobs", job)
                while process.poll() is None:
                    if stop.wait(1):
                        terminate_tree(process)
                        break
                if process.returncode and not stop.is_set():
                    raise RuntimeError(f"AI Toolkit exited with code {process.returncode}. Open the run log.")
            if not stop.is_set() and not list((root / "output").glob("**/*.safetensors")):
                raise RuntimeError("AI Toolkit exited without a LoRA file. Inspect the log; this run is not marked successful.")
        return self.new_job(pid, "train" if train else "export", work)
    def stop(self, job_id):
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
            return job
    def run_files(self, job_id):
        self.store.item("jobs", job_id)
        root = self.store.root / "runs" / job_id
        files = []
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".safetensors", ".png", ".jpg", ".json", ".log", ".zip"} and "dataset" not in path.relative_to(root).parts:
                    files.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size})
        return files
