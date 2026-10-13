"""Local-only HTTP application. Run: python server.py --open"""
import argparse
import json
import mimetypes
from pathlib import Path
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse
import urllib.request
import webbrowser

from studio.domain import catalog, intent, intent_digest
from studio.service import Service
from studio.store import Conflict, PreparationLocked
from studio.trainer import preflight
from studio import llm
from studio import results
from studio import checkpoint_actions
from studio.comparisons import Comparisons

ROOT = Path(__file__).resolve().parent

def make_server(port=8787, data=None, owner_url=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def respond(self, value, status=200):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        def file(self, path, download=False):
            if not path.is_file():
                raise KeyError("File not found")
            self.send_response(200)
            mime = {".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}.get(path.suffix, mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            if download:
                self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + urllib.parse.quote(path.name, safe=''))
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'self'")
            self.end_headers()
            with path.open("rb") as file:
                while block := file.read(128 * 1024):
                    self.wfile.write(block)
        def dispatch(self):
            try:
                host = self.headers.get("Host", "")
                if host not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
                    return self.respond({"error": "Invalid host"}, 403)
                if self.command != "GET":
                    origin = self.headers.get("Origin")
                    if origin and origin not in {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}:
                        return self.respond({"error": "Cross-origin writes are not allowed"}, 403)
                    if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                        raise ValueError("Use JSON for requests")
                    size = int(self.headers.get("Content-Length", 0))
                    if not 0 < size <= 29 * 1024 * 1024:
                        raise ValueError("Request size is invalid")
                    value = json.loads(self.rfile.read(size))
                    if not isinstance(value, dict):
                        raise ValueError("JSON body must be an object")
                else:
                    value = {}
                url = urllib.parse.urlsplit(self.path)
                parts = url.path.strip("/").split("/")
                if parts[0] != "api":
                    relative = urllib.parse.unquote(url.path).lstrip("/") or "index.html"
                    path = (ROOT / "web" / relative).resolve()
                    if not path.is_relative_to((ROOT / "web").resolve()):
                        raise ValueError("Invalid path")
                    return self.file(path)
                route = parts[1:]
                if route == ['comparison-bridge'] and self.command == 'POST':
                    return self.respond(service.comparisons.capability(value.get('token')))
                if len(route) >= 2 and route[0] == 'comparisons':
                    identity = route[1]
                    if len(route) == 2 and self.command == 'GET':
                        return self.respond(service.comparisons.public(service.comparisons.load(identity)))
                    if len(route) == 3 and self.command == 'POST':
                        return self.respond(service.comparisons.action(identity, route[2]), 202)
                    if len(route) == 3 and route[2] == 'grid' and self.command == 'GET':
                        return self.file(service.comparisons.composite(identity), download=True)
                    if len(route) == 4 and route[2] == 'images' and self.command == 'GET':
                        return self.file(service.comparisons.image(identity, route[3]))
                    raise KeyError('Comparison route not found')
                if route == ["bootstrap"]:
                    return self.respond({"app": "lora-studio", "catalog": catalog(), "projects": service.store.projects(), "settings": service.settings(), 'presets': service.store.presets(), 'onboarding': service.setup.status()})
                if route == ['setup'] and self.command == 'GET':
                    return self.respond(service.setup.status())
                if route == ['setup', 'check'] and self.command == 'POST':
                    return self.respond(service.setup.check())
                if route == ['setup', 'vision'] and self.command == 'POST':
                    return self.respond(service.setup.vision())
                if route == ['setup', 'install'] and self.command == 'POST':
                    return self.respond(service.setup.start(), 202)
                if route == ['setup', 'stop'] and self.command == 'POST':
                    return self.respond(service.setup.stop(), 202)
                if route == ['setup', 'dismiss'] and self.command == 'POST':
                    return self.respond(service.setup.dismiss(value.get('disposition')))
                if route == ['setup', 'promptstudio'] and self.command == 'GET':
                    return self.respond(service.setup.manager_status())
                if route == ['setup', 'promptstudio', 'install'] and self.command == 'POST':
                    return self.respond(service.setup.install_promptstudio(value.get('acknowledged'), value.get('expected_url')), 202)
                if route == ['llm', 'profiles'] and self.command == 'GET':
                    return self.respond(llm.backend(service.settings()).profiles())
                if route == ['llm', 'profile'] and self.command == 'POST':
                    return self.respond(service.llm_profile(value))
                if route == ['llm', 'pick-file'] and self.command == 'POST':
                    return self.respond(llm.backend(service.settings()).runtime._pick_llamacpp_file(value.get('kind'), value.get('current_path', '')))
                if len(route) == 3 and route[:2] == ['llm', 'server'] and self.command == 'POST':
                    return self.respond(service.llm_process(route[2]))
                if route == ['activity'] and self.command == 'GET':
                    keys = ('id', 'project', 'kind', 'status', 'phase', 'created', 'pid')
                    return self.respond({'jobs': [{k: j[k] for k in keys if k in j}
                                                 for j in service.store.items('jobs') if j['kind'] == 'train']})
                if route == ['presets'] and self.command == 'GET':
                    return self.respond({'presets': service.store.presets()})
                if route == ["settings"] and self.command == "PUT":
                    return self.respond(service.save_settings(value))
                if route == ["connection"]:
                    return self.respond(service.connection())
                if route == ["projects"] and self.command == "POST":
                    return self.respond(service.store.save_project(value), 201)
                if len(route) >= 2 and route[0] == "projects":
                    pid = route[1]
                    project = service.store.project(pid)
                    if len(route) == 2:
                        if self.command == "PUT":
                            return self.respond(service.store.save_project(value, pid))
                        images = service.dataset(pid)
                        return self.respond({"project": project, "images": images, "jobs": service.store.items("jobs", pid),
                                             "intent": intent(project), "intent_digest": intent_digest(project),
                                             "preflight": preflight(project, images, service.settings()),
                                             "automatic_preflight": preflight(project, images, service.settings(),automatic=True,preparing=True)})
                    action = route[2]
                    if action == 'sample-prompts' and self.command == 'GET':
                        return self.respond(service.preview_suite(project, service.dataset(pid)))
                    if self.command == "POST":
                        if action == 'delete':
                            return self.respond(service.delete_project(pid, value))
                        if action == 'copy':
                            return self.respond(service.copy_project(pid), 201)
                        service.store.assert_editable(pid)
                        if action == 'preset':
                            return self.respond(service.apply_training_preset(pid, value))
                        if action == 'sample-prompts':
                            return self.respond(service.generate_preview_suite(pid), 202)
                        if action == "images":
                            return self.respond(service.import_image(pid, value), 201)
                        if action == "caption":
                            return self.respond(service.caption(pid, value.get("reuse_observations", False), value.get("only_outdated", False)), 202)
                        if action == "detect":
                            return self.respond(service.check_people(pid), 202)
                        if action == 'objects':
                            return self.respond(service.find_objects(pid), 202)
                        if action == 'verify':
                            return self.respond(service.verify(pid,value.get('image_ids'),value.get('checks'),value.get('focus','')),202)
                        if action == 'train':
                            return self.respond(service.automatic_train(pid),202)
                        if action == 'prepare':
                            return self.respond(service.automatic_prepare(pid),202)
                        if action in {"export", "train"}:
                            return self.respond(service.prepare(pid, action == "train"), 202)
                if len(route) >= 2 and route[0] == "images":
                    item = service.store.item("images", route[1])
                    if len(route) == 3 and route[2] == "preview" and self.command == "POST":
                        return self.respond(service.preview(route[1], value.get("preparation"), value.get('object_target')))
                    if self.command == "PUT":
                        return self.respond(service.update_image(route[1], value))
                    return self.file(service.image_path(item))
                if len(route) >= 2 and route[0] == "jobs":
                    job = service.store.item("jobs", route[1])
                    if len(route) == 3 and self.command == 'POST':
                        if route[2] == 'checkpoint-delete-plan':
                            return self.respond(checkpoint_actions.delete_checkpoint(service, job['id'], value.get('path')))
                        if route[2] == 'delete-checkpoint':
                            if not isinstance(value.get('token'), str):
                                raise ValueError('Review the checkpoint deletion before confirming it')
                            return self.respond(checkpoint_actions.delete_checkpoint(service, job['id'], value.get('path'), value['token']))
                    if len(route) == 3 and route[2] == 'comparison-options' and self.command == 'GET':
                        return self.respond(service.comparisons.catalog(job['id']))
                    if len(route) == 3 and route[2] == 'comparisons':
                        if self.command == 'POST':
                            return self.respond(service.comparisons.start(job['id'], value), 202)
                        if self.command == 'GET':
                            return self.respond(service.comparisons.list(job['id']))
                    if len(route) == 3 and route[2] == 'results' and self.command == 'GET':
                        return self.respond({**results.results(service.store, job['id']),
                                             'comparisons': service.comparisons.list(job['id'])})
                    if len(route) == 3 and route[2] == 'open-location' and self.command == 'POST':
                        return self.respond(results.open_location(service.store, job['id'], value.get('path')))
                    if len(route) == 3 and route[2] == 'result-file' and self.command == 'GET':
                        _, root = results.run_root(service.store, job['id'])
                        name = urllib.parse.parse_qs(url.query).get('path', [''])[0]
                        path = results.artifact_path(root, name)
                        return self.file(path, download=path.suffix.lower() == '.safetensors')
                    if len(route) == 3 and route[2] == 'preset' and self.command == 'POST':
                        return self.respond(service.save_training_preset(job['id'], value), 201)
                    if len(route) == 3 and route[2] == 'monitor' and self.command == 'GET':
                        return self.respond(service.monitor.snapshot(job['id']))
                    if len(route) == 3 and route[2] in {'pause', 'resume', 'cancel', 'continue', 'save', 'sample', 'extend'} and self.command == 'POST':
                        return self.respond(service.monitor.action(job['id'], route[2], value), 202)
                    if len(route) == 2:
                        return self.respond({"job": job, "files": service.run_files(job["id"])})
                    if route[2] == "stop" and self.command == "POST":
                        return self.respond(service.stop(job["id"]))
                    if route[2] == "recover" and self.command == "POST":
                        return self.respond(service.recover(job["id"]))
                    if route[2] == "file" and self.command == "GET":
                        name = urllib.parse.parse_qs(url.query).get("path", [""])[0]
                        if name not in {f["path"] for f in service.run_files(job["id"])}:
                            raise KeyError("Run file not found")
                        root = (service.store.root / "runs" / job["id"]).resolve()
                        path = (root / name).resolve()
                        if not path.is_relative_to(root):
                            raise ValueError("Invalid file")
                        return self.file(path, download=path.suffix in {".zip", ".safetensors", ".pt"})
                raise KeyError("Route not found")
            except Conflict as exc:
                self.respond({"error": str(exc), "code": 'preparation_locked' if isinstance(exc, PreparationLocked) else 'conflict'}, 409)
            except KeyError as exc:
                self.respond({"error": str(exc)}, 404)
            except (ValueError, TypeError) as exc:
                self.respond({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self.respond({"error": str(exc)}, 500)
        do_GET = do_POST = do_PUT = dispatch
    class StudioServer(ThreadingHTTPServer):
        def server_close(self):
            service.comparisons.close()
            if service._setup is not None:
                service._setup.close()
            settings = service.settings()
            owned = llm._BACKENDS.get(settings['_llm_root'])
            if owned is not None:
                owned.close()
            if service._monitor is not None:
                service._monitor.close()
            super().server_close()
    server = StudioServer(("127.0.0.1", port), Handler)
    # Bind first: a duplicate launcher must not reconcile the live owner's jobs.
    service = Service(data or ROOT / "data", owner_url=owner_url)
    service.comparisons = Comparisons(service, f'http://127.0.0.1:{server.server_port}')
    server.service = service
    settings = service.settings()
    training_active = any(j['kind'] == 'train' and j['status'] in {'queued', 'running', 'stopping', 'interrupted'}
                          for j in service.store.items('jobs'))
    if not owner_url and not training_active and settings['integration'] == 'standalone' and settings['llm_provider'] == 'llamacpp' and settings['llamacpp_autostart']:
        import threading
        def autostart():
            try:
                service.llm_process('start')
            except Exception as exc:
                llm.backend(settings).startup_error = str(exc)
        threading.Thread(target=autostart, daemon=True).start()
    return server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--open", action="store_true")
    parser.add_argument('--owner-url', help='Attach to an existing Studio server without interrupting its active jobs')
    args = parser.parse_args()
    # Reopening the launcher should reveal the existing app instead of starting a second server.
    if args.open and args.port and args.data is None:
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f"http://127.0.0.1:{args.port}/api/bootstrap", timeout=2) as response:
                existing = json.load(response)
            if existing.get("app") == "lora-studio":
                webbrowser.open(f"http://127.0.0.1:{args.port}")
                raise SystemExit(0)
        except (OSError, ValueError):
            pass
    server = make_server(args.port, args.data, args.owner_url)
    print(f"LoRA Studio: http://127.0.0.1:{server.server_port}", flush=True)
    if args.open:
        webbrowser.open(f"http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
