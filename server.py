"""Local-only, dependency-free HTTP application. Run: python server.py --open"""
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
from studio.store import Conflict
from studio.trainer import preflight

ROOT = Path(__file__).resolve().parent

def make_server(port=8787, data=None):
    service = Service(data or ROOT / "data")
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
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
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
                if route == ["bootstrap"]:
                    return self.respond({"app": "lora-studio", "catalog": catalog(), "projects": service.store.projects(), "settings": service.settings()})
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
                        images = service.store.items("images", pid)
                        return self.respond({"project": project, "images": images, "jobs": service.store.items("jobs", pid),
                                             "intent": intent(project), "intent_digest": intent_digest(project),
                                             "preflight": preflight(project, images, service.settings())})
                    action = route[2]
                    if self.command == "POST":
                        if action == "images":
                            return self.respond(service.import_image(pid, value), 201)
                        if action == "caption":
                            return self.respond(service.caption(pid), 202)
                        if action in {"export", "train"}:
                            return self.respond(service.prepare(pid, action == "train"), 202)
                if len(route) >= 2 and route[0] == "images":
                    item = service.store.item("images", route[1])
                    if self.command == "PUT":
                        return self.respond(service.update_image(route[1], value))
                    return self.file(service.image_path(item))
                if len(route) >= 2 and route[0] == "jobs":
                    job = service.store.item("jobs", route[1])
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
                        return self.file(path, download=path.suffix in {".zip", ".safetensors"})
                raise KeyError("Route not found")
            except Conflict as exc:
                self.respond({"error": str(exc)}, 409)
            except KeyError as exc:
                self.respond({"error": str(exc)}, 404)
            except (ValueError, TypeError) as exc:
                self.respond({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self.respond({"error": str(exc)}, 500)
        do_GET = do_POST = do_PUT = dispatch
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.service = service
    return server

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--open", action="store_true")
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
    server = make_server(args.port, args.data)
    print(f"LoRA Studio: http://127.0.0.1:{server.server_port}", flush=True)
    if args.open:
        webbrowser.open(f"http://127.0.0.1:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
