"""Optional HTTP bridge. Prompt Studio owns its provider transport and scheduling."""
import base64
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid

PS = "/promptstudio/prompt-studio"
PROVIDER_KEYS = {"llm_provider", "kobold_url", "ollama_url", "ollama_model", "llamacpp_url", "llamacpp_model",
                 "llamacpp_executable", "llamacpp_config_profile", "llamacpp_autostart", "keep_models_loaded"}

def local_url(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"} or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Use a local HTTP endpoint (localhost or 127.0.0.1)")
    return url.rstrip("/")

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("Local integration redirects are not allowed")

class IntegrationHTTPError(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status

def request(url, payload=None, timeout=15, raw=None, content_type=None, accept_text=False, headers=None):
    local_url(url.split("?", 1)[0])
    data = raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
    headers = {**(headers or {}), **({"Content-Type": content_type or "application/json"} if data is not None else {})}
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect()).open(req, timeout=timeout) as response:
            body = response.read(16 * 1024 * 1024 + 1)
        if len(body) > 16 * 1024 * 1024:
            raise ValueError("Integration response is too large")
        if accept_text:
            return body.decode('utf-8', errors='replace')
        return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read(2048).decode(errors="replace")
        try:
            body = json.loads(body).get("error", body)
        except ValueError:
            pass
        raise IntegrationHTTPError(exc.code, f"Service returned {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("Local service is unreachable or timed out. Check that it is running.") from exc

class PromptStudio:
    def __init__(self, url):
        self.url = local_url(url)
    def connection(self):
        summary = request(self.url + PS + "/chats?summaries=1&limit=1", timeout=8)
        active = summary.get("activeChatId")
        if not active:
            raise ValueError("Open a Prompt Studio chat and select a vision LLM first")
        details = request(self.url + PS + "/chats?chat_id=" + urllib.parse.quote(active), timeout=8)
        settings = details["chats"][0].get("studioSettings", {})
        filtered = {k: v for k, v in settings.items() if k in PROVIDER_KEYS}
        if not filtered.get("llm_provider"):
            raise ValueError("The active Prompt Studio chat has no saved LLM connection")
        return filtered
    def inspect(self):
        connection = self.connection()
        capability = request(self.url + PS + "/vision-capability", connection, timeout=12)
        return {"available": True, "provider": connection.get("llm_provider"),
                "model": connection.get("llamacpp_model") or connection.get("ollama_model") or "Loaded model",
                "vision": capability, "source": "Active Prompt Studio chat"}
    def upload(self, data, filename):
        boundary = "lora" + uuid.uuid4().hex
        raw = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="image.jpg"\r\n'
               'Content-Type: application/octet-stream\r\n\r\n').encode() + data + f"\r\n--{boundary}--\r\n".encode()
        return request(self.url + PS + "/import-image", raw=raw,
                       content_type="multipart/form-data; boundary=" + boundary, timeout=45)["image"]
    def caption(self, instruction, data, filename, connection, *, temperature=0.1, max_tokens=1800):
        image = self.upload(data, filename)
        result = request(self.url + PS + "/chat", {**connection, "thinking_mode": "Disabled",
            "temperature": temperature, "max_response_tokens": max_tokens, "request_timeout": 300,
            "messages": [{"role": "user", "text": instruction, "images": [image]}]}, timeout=330)
        if result.get("warning"):
            raise ValueError("Prompt Studio reported an incomplete response; the caption was not saved")
        return result.get("message", "")
    def text(self, instruction, connection):
        result = request(self.url + PS + "/chat", {**connection, "thinking_mode": "Disabled",
            "temperature": 0, "max_response_tokens": 3000, "request_timeout": 300,
            "messages": [{"role": "user", "text": instruction}]}, timeout=330)
        if result.get("warning"):
            raise ValueError("Prompt Studio returned an incomplete text response; the caption was not saved")
        return result.get("message", "")
    def free_generation_memory(self, purpose="training"):
        queue = request(self.url + "/queue")
        jobs = request(self.url + "/promptstudio/jobs")
        if queue.get("queue_running") or queue.get("queue_pending"):
            raise ValueError(f"ComfyUI has active or queued generations. Wait before starting {purpose}.")
        if any(j.get("state") in {"queued", "running"} for j in jobs.get("jobs", [])):
            raise ValueError(f"Prompt Studio has active work. Wait before starting {purpose}.")
        request(self.url + "/free", {"unload_models": True, "free_memory": True})

def standalone_caption(settings, instruction, data, mime, *, temperature=0.1, max_tokens=1800, system=None):
    if settings.get('llm_provider', 'openai') != 'openai':
        from .llm import backend
        return backend(settings).generate(settings, instruction, data, mime, temperature=temperature, max_tokens=max_tokens, system=system)
    base = local_url(settings["llm_url"])
    if not settings.get("llm_model"):
        raise ValueError("Choose a local vision model in Connections")
    content = [{"type": "text", "text": instruction}]
    if data is not None:
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode()}})
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    response = request(base + "/chat/completions", {
        "model": settings["llm_model"], "temperature": temperature, "max_tokens": max_tokens,
        "messages": messages}, timeout=330)
    try:
        choice = response["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("The caption exceeded the response limit; no caption was saved")
        return choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("The endpoint did not return an OpenAI-compatible completion")
