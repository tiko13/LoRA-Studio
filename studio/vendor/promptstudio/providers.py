# Derived from tiko13/ComfyUI_PromptStudio; GPL-3.0-only. See NOTICE.md.


import functools, inspect, ipaddress, json, math, os, re, threading, urllib.error, urllib.parse, urllib.request
from . import provider_transport as _provider_transport
from . import llm_coordinator as _llm_scheduling

CHAT_SYSTEM_MESSAGE = (
    "You are an expert image-generation prompt editor. Follow the requested transformation and "
    "output-format constraints precisely. Preserve the user's requested subject matter without "
    "sanitizing it or introducing more extreme content. Keep analysis in the model's private reasoning channel. "
    "The final answer must contain only the requested prompt output, without commentary or markdown."
)

DEFAULT_CONTINUATION_STOPS = ()

JSON_OBJECT_GBNF = r'''root ::= ws object ws
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
string ::= "\"" chars "\""
chars ::= ([^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" hex hex hex hex))*
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
hex ::= [0-9a-fA-F]
ws ::= [ \t\n\r]*
'''

def _allowed_kobold_hosts():
    return {"localhost", "127.0.0.1", "::1"}

def _allowed_ollama_hosts():
    return {"localhost", "127.0.0.1", "::1"}

def _allowed_llamacpp_hosts():
    return {"localhost", "127.0.0.1", "::1"}

def _clean_service_base_url(url, default_url, service_name, allowed_hosts, allowed_hosts_env):
    cleaned = (url or default_url).strip()
    if not cleaned:
        cleaned = default_url
    if "://" not in cleaned:
        cleaned = "http://" + cleaned
    cleaned = cleaned.rstrip("/")
    parsed = urllib.parse.urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"{service_name} URL must use http or https")
    if not parsed.hostname:
        raise ValueError(f"{service_name} URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{service_name} URL must not contain credentials")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError(f"{service_name} URL contains an invalid port") from exc
    if parsed.query or parsed.fragment:
        raise ValueError(f"{service_name} URL must not contain a query string or fragment")

    hostname = parsed.hostname.casefold()
    if "*" not in allowed_hosts and hostname not in allowed_hosts:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            raise ValueError(
                f"{service_name} host '{parsed.hostname}' is not allowed. "
                f"Add it to {allowed_hosts_env} or use '*' to allow remote hosts."
            )
    return cleaned

def _clean_base_url(url):
    return _clean_service_base_url(
        url,
        "http://localhost:5001",
        "KoboldCpp",
        _allowed_kobold_hosts(),
        "PROMPT_STUDIO_KOBOLD_ALLOWED_HOSTS",
    )

def _clean_ollama_base_url(url):
    return _clean_service_base_url(
        url,
        "http://localhost:11434",
        "Ollama",
        _allowed_ollama_hosts(),
        "PROMPT_STUDIO_OLLAMA_ALLOWED_HOSTS",
    )

def _ollama_api_url(base_url, endpoint):
    base = base_url.rstrip("/")
    endpoint = str(endpoint or "").strip("/")
    if urllib.parse.urlsplit(base).path.rstrip("/").endswith("/api"):
        return f"{base}/{endpoint}"
    return f"{base}/api/{endpoint}"

def _coordinated_native_llm(provider):
    """Wrap whole native operations and provider helpers without route imports."""
    def decorate(operation):
        signature = inspect.signature(operation)
        @functools.wraps(operation)
        def scheduled(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            settings = {"llm_provider": provider,
                        provider.replace("koboldcpp", "kobold") + "_url": bound.arguments.get(
                            provider.replace("koboldcpp", "kobold") + "_url")}
            coordinator = _llm_scheduling.SHARED_COORDINATOR
            nested = coordinator.current_token() is not None
            resources = [_llm_scheduling.endpoint_identity(settings)]
            if not nested:
                resources.append(("shared-gpu",))
            token = coordinator.current_token() or _llm_scheduling.CancellationToken(
                bound.arguments.get("cancellation_check"),
            )
            def run():
                if not nested and coordinator.native_prepare is not None:
                    coordinator.native_prepare(settings)
                token.check()
                return operation(*args, **kwargs)
            return coordinator.run(resources, run, token=token)
        return scheduled
    return decorate

def _provider_url_validator(url, service_name=""):
    name = service_name.casefold()
    if "ollama" in name:
        allowed, env = _allowed_ollama_hosts(), "PROMPT_STUDIO_OLLAMA_ALLOWED_HOSTS"
    elif "llama" in name:
        allowed, env = _allowed_llamacpp_hosts(), "PROMPT_STUDIO_LLAMACPP_ALLOWED_HOSTS"
    else:
        allowed, env = _allowed_kobold_hosts(), "PROMPT_STUDIO_KOBOLD_ALLOWED_HOSTS"
    parsed = urllib.parse.urlsplit(url)
    # Query strings are valid on provider endpoints; origin/credentials are
    # validated with the same policy used for the original configured URL.
    origin = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
    return _clean_service_base_url(origin, origin, service_name or "Provider", allowed, env)

def _open_provider_response(request, timeout, service_name=""):
    path = urllib.parse.urlsplit(request.full_url).path
    generation_path = next((suffix for suffix in ("/v1/chat/completions", "/api/v1/generate", "/api/chat")
                            if path.endswith(suffix)), None)
    inference = request.get_method() == "POST" and generation_path is not None
    activity_check = None
    if inference:
        base_url = request.full_url[:-len(generation_path)]
        if service_name == "Llama.cpp":
            model = json.loads(request.data).get("model", "")
            query = urllib.parse.urlencode({"model": model, "autoload": "false"})
            def activity_check():
                slots = _get_json(base_url + "/slots?" + query, min(3, timeout / 3), "Llama.cpp")
                return isinstance(slots, list) and any(
                    isinstance(slot, dict) and slot.get("is_processing") is True for slot in slots)
        elif service_name == "KoboldCpp":
            def activity_check():
                perf = _get_json(base_url + "/api/extra/perf", min(3, timeout / 3))
                return isinstance(perf, dict) and perf.get("idle") == 0
    return _provider_transport.open_response(
        request, timeout, lambda url: _provider_url_validator(url, service_name),
        **({"inference": True, "activity_check": activity_check} if inference else {}),
    )

def _post_json(
    url,
    payload,
    timeout,
    service_name="KoboldCpp",
    response_hook=None,
    headers=None,
):
    data = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method="POST",
    )
    try:
        with _open_provider_response(request, timeout, service_name) as response:
            if response_hook is not None:
                response_hook(response)
            try:
                if service_name == "Ollama" and payload.get("stream") is True:
                    return _read_ollama_chat_stream(response)
                body = response.read().decode("utf-8")
                getattr(response, "note_activity", lambda: None)()
            finally:
                if response_hook is not None:
                    response_hook(None)
    except urllib.error.HTTPError as exc:
        detail = str(exc)
        raise RuntimeError(f"{service_name} request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {service_name} at {url}: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{service_name} returned invalid JSON: {body[:500]}") from exc

def _read_ollama_chat_stream(response):
    content, thinking = [], []
    for line in response:
        if not line.strip():
            continue
        chunk = json.loads(line)
        if chunk.get("error"):
            raise RuntimeError(f"Ollama reported an error: {chunk['error']}")
        message = chunk.get("message") or {}
        if message.get("content") or message.get("thinking"):
            getattr(response, "note_activity", lambda: None)()
        content.append(str(message.get("content") or ""))
        thinking.append(str(message.get("thinking") or ""))
        if chunk.get("done") is True:
            return {**chunk, "message": {**message, "content": "".join(content), "thinking": "".join(thinking)}}
    raise RuntimeError("Ollama stream ended before completion")

def _clean_llamacpp_base_url(url):
    return _clean_service_base_url(
        url,
        "http://localhost:8080",
        "Llama.cpp",
        _allowed_llamacpp_hosts(),
        "PROMPT_STUDIO_LLAMACPP_ALLOWED_HOSTS",
    )

def _list_ollama_models(ollama_url, request_timeout=10):
    base_url = _clean_ollama_base_url(ollama_url)
    url = _ollama_api_url(base_url, "tags")
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with _open_provider_response(request, request_timeout, "Ollama") as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = str(exc)
        raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {url}: {exc.reason}") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama returned invalid JSON: {body[:500]}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise RuntimeError(f"Unexpected Ollama model-list response: {data}")

    models = []
    for item in data["models"]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("model") or item.get("name") or "").strip()
        if name and name not in models:
            models.append(name)
    return models

def _list_llamacpp_models(llamacpp_url, request_timeout=10):
    base_url = _clean_llamacpp_base_url(llamacpp_url)
    # /models lists both the active single model and router-mode available models.
    url = urllib.parse.urljoin(base_url + "/", "models")
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with _open_provider_response(request, request_timeout, "Llama.cpp") as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = str(exc)
        raise RuntimeError(f"Llama.cpp request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Llama.cpp at {url}: {exc.reason}") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Llama.cpp returned invalid JSON: {body[:500]}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise RuntimeError(f"Unexpected Llama.cpp model-list response: {data}")
    models = []
    for item in data["data"]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("id") or "").strip()
        if name and name not in models:
            models.append(name)
    return models

_LLAMACPP_ACTIVE_RESPONSES = {}

_LLAMACPP_ACTIVE_STREAMS = {}

_LLAMACPP_ABORTED_RESPONSES = set()

_LLAMACPP_RESPONSE_LOCK = threading.Lock()

def _post_llamacpp_chat(base_url, payload, timeout, response_hook=None, cancellation_check=None):
    """Read llama-server's SSE chat stream while keeping it externally cancellable."""
    data = json.dumps({**payload, "stream": True}).encode("utf-8")
    request = urllib.request.Request(
        urllib.parse.urljoin(base_url + "/", "v1/chat/completions"),
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    response = None
    response_id = None
    content_parts = []
    reasoning_parts = []
    finish_reason = None
    stream_complete = False
    chunks = []
    chat_template_kwargs = payload.get("chat_template_kwargs")
    thinking_enabled = bool(
        chat_template_kwargs.get("enable_thinking")
        if isinstance(chat_template_kwargs, dict) else False
    )
    try:
        response = _open_provider_response(request, timeout, "Llama.cpp")
        response_id = id(response)
        with _LLAMACPP_RESPONSE_LOCK:
            _LLAMACPP_ACTIVE_RESPONSES.setdefault(base_url, set()).add(response)
            _LLAMACPP_ACTIVE_STREAMS.setdefault(base_url, {})[response_id] = {
                "generation_phase": "thinking" if thinking_enabled else "generating",
            }
        if response_hook is not None:
            response_hook(response)
        for raw_line in response:
            if cancellation_check is not None and cancellation_check():
                with _LLAMACPP_RESPONSE_LOCK:
                    _LLAMACPP_ABORTED_RESPONSES.add(response_id)
                response.close()
                raise RuntimeError("Llama.cpp request was cancelled")
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            event = line[5:].strip()
            if event == "[DONE]":
                stream_complete = True
                break
            try:
                chunk = json.loads(event)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Llama.cpp returned invalid stream JSON: {event[:500]}") from exc
            if isinstance(chunk, dict) and chunk.get("error"):
                raise RuntimeError(f"Llama.cpp reported an error: {chunk['error']}")
            # Retain bounded diagnostic metadata, not every streamed token.
            if len(chunks) < 32:
                chunks.append(chunk)
            choices = chunk.get("choices") if isinstance(chunk, dict) else None
            choice = choices[0] if isinstance(choices, list) and choices else None
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                content_received = False
                if delta.get("content") is not None:
                    content = str(delta["content"])
                    content_parts.append(content)
                    content_received = bool(content)
                reasoning = delta.get("reasoning_content")
                if reasoning is None:
                    reasoning = delta.get("reasoning")
                reasoning_received = False
                if reasoning is not None:
                    reasoning = str(reasoning)
                    reasoning_parts.append(reasoning)
                    reasoning_received = bool(reasoning)
                if content_received or reasoning_received:
                    getattr(response, "note_activity", lambda: None)()
                    with _LLAMACPP_RESPONSE_LOCK:
                        stream = _LLAMACPP_ACTIVE_STREAMS.get(base_url, {}).get(response_id)
                        if stream is not None:
                            if content_received:
                                stream["generation_phase"] = "generating"
                            elif stream["generation_phase"] != "generating":
                                stream["generation_phase"] = "thinking"
            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason")
        with _LLAMACPP_RESPONSE_LOCK:
            aborted = response_id in _LLAMACPP_ABORTED_RESPONSES
        if aborted:
            raise RuntimeError("Llama.cpp request was cancelled")
        if not stream_complete and finish_reason is None:
            raise RuntimeError("Llama.cpp stream ended before completion")
    except urllib.error.HTTPError as exc:
        detail = str(exc)
        raise RuntimeError(f"Llama.cpp request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Llama.cpp at {request.full_url}: {exc.reason}") from exc
    except (OSError, ValueError) as exc:
        with _LLAMACPP_RESPONSE_LOCK:
            aborted = response_id in _LLAMACPP_ABORTED_RESPONSES
        if aborted:
            raise RuntimeError("Llama.cpp request was cancelled") from exc
        raise RuntimeError(f"Llama.cpp stream failed: {exc}") from exc
    finally:
        if response_hook is not None and response is not None:
            response_hook(None)
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        if response_id is not None:
            with _LLAMACPP_RESPONSE_LOCK:
                active = _LLAMACPP_ACTIVE_RESPONSES.get(base_url)
                if active is not None:
                    active.discard(response)
                    if not active:
                        _LLAMACPP_ACTIVE_RESPONSES.pop(base_url, None)
                streams = _LLAMACPP_ACTIVE_STREAMS.get(base_url)
                if streams is not None:
                    streams.pop(response_id, None)
                    if not streams:
                        _LLAMACPP_ACTIVE_STREAMS.pop(base_url, None)
                _LLAMACPP_ABORTED_RESPONSES.discard(response_id)
    return {
        "choices": [{
            "message": {
                "content": "".join(content_parts),
                "reasoning_content": "".join(reasoning_parts),
            },
            "finish_reason": finish_reason,
        }],
        "stream_chunks": chunks,
    }

def _get_json(url, timeout, service_name="KoboldCpp"):
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with _open_provider_response(request, timeout, service_name) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, RuntimeError, OSError):
        return None

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None

def _server_context_length(base_url, timeout):
    data = _get_json(
        urllib.parse.urljoin(base_url + "/", "api/extra/true_max_context_length"),
        timeout,
    )
    if not isinstance(data, dict):
        return None
    try:
        return int(data["value"])
    except (KeyError, TypeError, ValueError):
        return None

def _server_capabilities(base_url, timeout):
    data = _get_json(
        urllib.parse.urljoin(base_url + "/", "api/extra/version"),
        timeout,
    )
    return data if isinstance(data, dict) else {}

def _llamacpp_props(base_url, timeout, model="", *, autoload=True):
    params = {"model": str(model).strip()} if str(model).strip() else {}
    if not autoload:
        params["autoload"] = "false"
    query = urllib.parse.urlencode(params)
    url = urllib.parse.urljoin(base_url + "/", "props")
    if query:
        url = f"{url}?{query}"
    data = _get_json(url, timeout, "Llama.cpp")
    return data if isinstance(data, dict) else {}

def _llamacpp_context_length(base_url, timeout, model=""):
    settings = _llamacpp_props(base_url, timeout, model).get("default_generation_settings")
    if not isinstance(settings, dict):
        return None
    try:
        return int(settings.get("n_ctx"))
    except (TypeError, ValueError):
        return None

def _llamacpp_token_count(base_url, timeout, messages, model, reasoning_effort, enable_thinking):
    chat_template_kwargs = {"enable_thinking": bool(enable_thinking)}
    if reasoning_effort != "none":
        # Named instruct modes also need their native effort in the template.
        chat_template_kwargs["reasoning_effort"] = reasoning_effort
    try:
        result = _post_json(
            urllib.parse.urljoin(base_url + "/", "v1/chat/completions/input_tokens"),
            {
                "model": model,
                "messages": messages,
                "reasoning_effort": reasoning_effort,
                "chat_template_kwargs": chat_template_kwargs,
            },
            timeout,
            service_name="Llama.cpp",
        )
    except RuntimeError:
        return None
    if not isinstance(result, dict):
        return None
    for key in ("input_tokens", "tokens"):
        try:
            return int(result[key])
        except (KeyError, TypeError, ValueError):
            pass
    return None

def _kobold_vision_unavailable_reason(capabilities):
    if capabilities.get("vision") is True:
        if capabilities.get("jinja") is False:
            return (
                "KoboldCpp vision is active, but Use Jinja is disabled. Enable Use Jinja and "
                "restart KoboldCpp before captioning an image."
            )
        return ""
    if "vision" in capabilities:
        return (
            "KoboldCpp vision is not active. Load a vision-capable text model with its matching "
            "MMProj file, then restart KoboldCpp."
        )
    return (
        "KoboldCpp did not report whether vision is active. Check that the server is reachable "
        "and update KoboldCpp to a version whose /api/extra/version response includes 'vision'."
    )

def _ollama_model_info(base_url, model, timeout):
    result = _post_json(
        _ollama_api_url(base_url, "show"),
        {"model": model},
        timeout,
        service_name="Ollama",
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected Ollama model-info response: {result}")
    return result

def _ollama_model_capabilities(base_url, model, timeout):
    result = _ollama_model_info(base_url, model, timeout)
    capabilities = result.get("capabilities")
    return [str(value).strip().casefold() for value in capabilities] if isinstance(capabilities, list) else []

def _ollama_model_context_length(model_info):
    metadata = model_info.get("model_info") if isinstance(model_info, dict) else None
    if not isinstance(metadata, dict):
        return None
    architecture = str(metadata.get("general.architecture") or "").strip()
    if architecture:
        try:
            primary = int(metadata.get(f"{architecture}.context_length"))
        except (TypeError, ValueError):
            primary = 0
        if primary > 0:
            return primary
    values = []
    for key, value in metadata.items():
        if not str(key).casefold().endswith(".context_length"):
            continue
        try:
            length = int(value)
        except (TypeError, ValueError):
            continue
        if length > 0:
            values.append(length)
    # If an older model omits general.architecture, choose the most conservative
    # advertised context rather than accidentally selecting an auxiliary encoder.
    return min(values) if values else None

def _ollama_vision_unavailable_reason(capabilities, model):
    if "vision" in capabilities:
        return ""
    if capabilities:
        return (
            f"The selected Ollama model '{model}' does not support vision. "
            "Select an Ollama model whose capabilities include vision."
        )
    return (
        f"Ollama did not advertise capabilities for '{model}'. Update Ollama or select a model "
        "that reports the vision capability."
    )

def _llamacpp_vision_unavailable_reason(props, model):
    modalities = props.get("modalities") if isinstance(props, dict) else None
    if isinstance(modalities, dict) and modalities.get("vision") is True:
        return ""
    label = f" for '{model}'" if model else ""
    if isinstance(modalities, dict) and "vision" in modalities:
        return (
            f"Llama.cpp vision is not active{label}. Start llama-server with a supported "
            "multimodal model and its matching mmproj file."
        )
    return (
        f"Llama.cpp did not advertise vision capability{label}. Update llama.cpp or select a "
        "server/model whose /props response includes modalities.vision."
    )

def _llm_vision_capability(
    llm_provider,
    *,
    kobold_url="http://localhost:5001",
    ollama_url="http://localhost:11434",
    ollama_model="",
    llamacpp_url="http://localhost:8080",
    llamacpp_model="",
    request_timeout=10,
):
    provider = str(llm_provider or "koboldcpp").strip().casefold()
    timeout = int(request_timeout)
    if provider == "koboldcpp":
        base_url = _clean_base_url(kobold_url)
        capabilities = _server_capabilities(base_url, timeout)
        reason = _kobold_vision_unavailable_reason(capabilities)
        return {"available": not reason, "provider": "KoboldCpp", "reason": reason}
    if provider == "ollama":
        base_url = _clean_ollama_base_url(ollama_url)
        model = str(ollama_model or "").strip()
        if not model:
            return {
                "available": False,
                "provider": "Ollama",
                "reason": "Select an Ollama model before dropping an image.",
            }
        capabilities = _ollama_model_capabilities(base_url, model, timeout)
        reason = _ollama_vision_unavailable_reason(capabilities, model)
        return {"available": not reason, "provider": "Ollama", "model": model, "reason": reason}
    if provider == "llamacpp":
        base_url = _clean_llamacpp_base_url(llamacpp_url)
        model = str(llamacpp_model or "").strip()
        if not model:
            models = _list_llamacpp_models(base_url, timeout)
            model = models[0] if len(models) == 1 else ""
        if not model:
            return {
                "available": False,
                "provider": "Llama.cpp",
                "reason": "Select a Llama.cpp model before dropping an image.",
            }
        props = _llamacpp_props(base_url, timeout, model)
        reason = _llamacpp_vision_unavailable_reason(props, model)
        return {"available": not reason, "provider": "Llama.cpp", "model": model, "reason": reason}
    raise ValueError("llm_provider must be koboldcpp, ollama, or llamacpp")

def _kobold_token_count(
    base_url,
    timeout,
    *,
    prompt=None,
    messages=None,
    reasoning_effort=None,
    enable_thinking=None,
):
    payload = {"special": True}
    if messages is not None:
        payload["messages"] = messages
    else:
        payload["prompt"] = str(prompt or "")
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}

    try:
        data = _post_json(
            urllib.parse.urljoin(base_url + "/", "api/extra/tokencount"),
            payload,
            timeout,
        )
    except RuntimeError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return int(data["value"])
    except (KeyError, TypeError, ValueError):
        return None

def _context_safe_generation_limit(context_length, prompt_tokens):
    if context_length is None or context_length <= 0:
        return None
    limit = max(1, int(context_length * 0.8) - 1)
    if prompt_tokens is not None and prompt_tokens >= 0:
        limit = min(limit, max(1, int(context_length) - int(prompt_tokens) - 16))
    return limit

def _requested_response_tokens(max_response_tokens, default_max_response_tokens):
    requested = int(max_response_tokens)
    if requested <= 0:
        requested = int(default_max_response_tokens)
    return max(1, requested)

def _chat_generation_budget(
    response_tokens,
    thinking_mode,
    safe_limit=None,
    fixed_reasoning_budgets=False,
):
    """Return total completion tokens and an optional explicit thinking cap.

    ``response_tokens`` remains the workflow's final-answer allowance. KoboldCpp counts
    native reasoning and final content in one completion limit. Minimal, Low, and
    Medium receive fixed reasoning allowances while preserving the requested final-answer
    allowance. High and XHigh have no reasoning cap and may use the remaining context window.
    """
    response_tokens = max(1, int(response_tokens))
    effort = _reasoning_effort(thinking_mode)
    thinking_budget = None
    if fixed_reasoning_budgets and effort == "minimal":
        desired_reasoning = 200
        total = response_tokens + desired_reasoning
        thinking_budget = desired_reasoning
    elif fixed_reasoning_budgets and effort == "low":
        desired_reasoning = 500
        total = response_tokens + desired_reasoning
        thinking_budget = desired_reasoning
    elif fixed_reasoning_budgets and effort == "medium":
        desired_reasoning = 1000
        total = response_tokens + desired_reasoning
        thinking_budget = desired_reasoning
    elif fixed_reasoning_budgets and effort in {"high", "xhigh"}:
        # Normal requests supply safe_limit from KoboldCpp's actual context window.
        # Keep a generous fallback for direct calls if that capability is unavailable.
        total = int(safe_limit) if safe_limit is not None and safe_limit > 0 else response_tokens + 65536
    elif effort == "minimal":
        total = (response_tokens * 10 + 8) // 9
    elif effort == "low":
        total = (response_tokens * 10 + 6) // 7
    elif effort == "medium":
        total = (response_tokens * 5 + 1) // 2
    elif effort in {"high", "xhigh"}:
        desired_reasoning = 4096
        total = response_tokens + desired_reasoning
        thinking_budget = desired_reasoning
    else:
        total = response_tokens

    if safe_limit is not None and safe_limit > 0:
        total = min(total, int(safe_limit))
    total = max(1, total)

    if thinking_budget is not None:
        # Preserve the requested final allowance whenever the context window permits it.
        thinking_budget = min(thinking_budget, max(0, total - min(response_tokens, total)))
    return total, thinking_budget

def _llamacpp_generation_budget(
    response_tokens,
    thinking_mode,
    safe_limit=None,
    reasoning_budget_tokens=0,
):
    """Return llama.cpp's combined output limit and optional hard reasoning cap.

    Qwen-style reasoning effort is a qualitative chat-template control, not a
    token budget. With no explicit cap, allow the model to use the available
    context window. A positive cap opts into llama.cpp's forced end-of-thinking
    mechanism and reserves the requested final-answer allowance when possible.
    """
    response_tokens = max(1, int(response_tokens))
    effort = _reasoning_effort(thinking_mode)
    safe_limit = int(safe_limit) if safe_limit is not None and safe_limit > 0 else None
    if effort == "none":
        total = response_tokens
        if safe_limit is not None:
            total = min(total, safe_limit)
        return max(1, total), None

    requested_cap = max(0, int(reasoning_budget_tokens or 0))
    if requested_cap > 0:
        total = response_tokens + requested_cap
        if safe_limit is not None:
            total = min(total, safe_limit)
        total = max(1, total)
        final_allowance = min(response_tokens, total)
        applied_cap = min(requested_cap, max(0, total - final_allowance))
        return total, applied_cap

    total = safe_limit if safe_limit is not None else response_tokens + 65536
    return max(1, total), None

def _split_stop_sequences(value):
    if not value:
        return []
    return [item.strip() for item in value.splitlines() if item.strip()]

def _with_default_continuation_stops(stop_sequences):
    out = list(stop_sequences)
    for stop in DEFAULT_CONTINUATION_STOPS:
        if stop not in out:
            out.append(stop)
    return out

@_coordinated_native_llm("koboldcpp")
def _generate_kcpp(
    prompt,
    kobold_url,
    max_response_tokens,
    default_max_response_tokens,
    temperature,
    top_p,
    top_k,
    min_p,
    rep_pen,
    rep_pen_range,
    sampler_seed,
    thinking_mode,
    stop_sequence,
    request_timeout,
    include_default_continuation_stops=False,
    image_data_uri=None,
    messages_override=None,
    response_hook=None,
    cancellation_check=None,
    presence_penalty=0.0,
    response_schema=None,
):
    def ensure_active():
        if cancellation_check is not None and cancellation_check():
            raise RuntimeError("KoboldCpp request was cancelled")

    ensure_active()
    base_url = _clean_base_url(kobold_url)
    timeout = int(request_timeout)
    capabilities = _server_capabilities(base_url, timeout)
    if capabilities.get("jinja") is False:
        raise RuntimeError(
            "KoboldCpp Chat Completions requires Use Jinja for reliable model-native formatting. "
            "Enable Use Jinja in KoboldCpp, restart the server, and run the workflow again."
        )
    if messages_override is not None:
        messages = messages_override
        has_images = any(
            isinstance(message, dict)
            and isinstance(message.get("content"), list)
            and any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in message["content"]
            )
            for message in messages
        )
        if has_images:
            vision_reason = _kobold_vision_unavailable_reason(capabilities)
            if vision_reason:
                raise RuntimeError(vision_reason)
    elif image_data_uri:
        vision_reason = _kobold_vision_unavailable_reason(capabilities)
        if vision_reason:
            raise RuntimeError(vision_reason)
        user_content = [
            {"type": "text", "text": str(prompt or "")},
            {"type": "image_url", "image_url": {"url": str(image_data_uri)}},
        ]
    else:
        user_content = str(prompt or "")
    if messages_override is None:
        messages = [
            {"role": "system", "content": CHAT_SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
        ]
    response_tokens = _requested_response_tokens(max_response_tokens, default_max_response_tokens)
    context_length = _server_context_length(base_url, timeout)
    stop_sequences = _split_stop_sequences(stop_sequence)
    # Native reasoning may legitimately use labels such as "Response:" while moving
    # from analysis to final content. Legacy textual stops can cut that transition off,
    # so thinking requests rely on the model's Jinja end-of-turn markers instead.
    if include_default_continuation_stops and _reasoning_effort(thinking_mode) == "none":
        stop_sequences = _with_default_continuation_stops(stop_sequences)

    def generate_once(request_thinking_mode):
        ensure_active()
        effort = _reasoning_effort(request_thinking_mode)
        enable_thinking = effort != "none"
        prompt_tokens = _kobold_token_count(
            base_url,
            timeout,
            messages=messages,
            reasoning_effort=effort,
            enable_thinking=enable_thinking,
        )
        if (
            context_length is not None
            and prompt_tokens is not None
            and prompt_tokens >= context_length - 32
        ):
            raise RuntimeError(
                f"The Jinja-formatted request uses {prompt_tokens} tokens, leaving no usable space "
                f"in KoboldCpp's {context_length}-token context window. Shorten the instructions or "
                "increase the KoboldCpp context size."
            )
        safe_limit = _context_safe_generation_limit(context_length, prompt_tokens)
        max_length, thinking_budget = _chat_generation_budget(
            response_tokens,
            request_thinking_mode,
            safe_limit,
            fixed_reasoning_budgets=True,
        )
        chat_template_kwargs = {"enable_thinking": enable_thinking}
        payload = {
            "model": "koboldcpp",
            "messages": messages,
            "max_tokens": max_length,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "min_p": float(min_p),
            "presence_penalty": float(presence_penalty),
            "rep_pen": float(rep_pen),
            "rep_pen_range": int(rep_pen_range),
            "seed": int(sampler_seed),
            "chat_template_kwargs": chat_template_kwargs,
            "stop": stop_sequences,
            "encapsulate_thinking": True,
            "continue_assistant_turn": False,
            "stream": False,
        }
        if isinstance(response_schema, dict) and effort == "none":
            payload["grammar"] = JSON_OBJECT_GBNF
        if thinking_budget is not None:
            # KoboldCpp's top-level Minimal/Low/Medium efforts enforce percentage caps
            # before consulting thinking_budget_tokens. Keep the effort as a template
            # hint and omit it at the top level so the explicit fixed cap takes effect.
            chat_template_kwargs["reasoning_effort"] = effort
        else:
            payload["reasoning_effort"] = effort
        if thinking_budget is not None:
            payload["thinking_budget_tokens"] = thinking_budget

        ensure_active()
        result = _post_json(
            urllib.parse.urljoin(base_url + "/", "v1/chat/completions"),
            payload,
            timeout,
            response_hook=response_hook,
        )
        ensure_active()
        try:
            choice = result["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            finish_reason = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected KoboldCpp response: {result}") from exc
        if finish_reason == "error":
            raise RuntimeError("KoboldCpp reported an error while processing the chat completion.")
        return str(content), message, finish_reason, max_length, result

    content, message, finish_reason, max_length, result = generate_once(thinking_mode)
    if finish_reason == "length":
        raise RuntimeError(
            f"KoboldCpp exhausted the {max_length}-token completion budget before finishing. "
            "Increase max_response_tokens or increase the KoboldCpp context size."
        )
    if not content.strip():
        if message.get("reasoning_content"):
            raise RuntimeError(
                "KoboldCpp returned reasoning but no final answer. Increase max_response_tokens "
                "or increase the KoboldCpp context size."
            )
        raise RuntimeError(f"KoboldCpp returned an empty chat completion: {result}")
    return content

@_coordinated_native_llm("llamacpp")
def _generate_llamacpp(
    prompt,
    llamacpp_url,
    llamacpp_model,
    max_response_tokens,
    default_max_response_tokens,
    temperature,
    top_p,
    top_k,
    min_p,
    rep_pen,
    rep_pen_range,
    sampler_seed,
    thinking_mode,
    stop_sequence,
    request_timeout,
    include_default_continuation_stops=False,
    image_data_uri=None,
    messages_override=None,
    response_hook=None,
    cancellation_check=None,
    presence_penalty=0.0,
    reasoning_budget_tokens=0,
    response_schema=None,
):
    def ensure_active():
        if cancellation_check is not None and cancellation_check():
            raise RuntimeError("Llama.cpp request was cancelled")

    ensure_active()
    base_url = _clean_llamacpp_base_url(llamacpp_url)
    timeout = int(request_timeout)
    model = str(llamacpp_model or "").strip()
    if not model:
        models = _list_llamacpp_models(base_url, timeout)
        if len(models) == 1:
            model = models[0]
        elif not models:
            raise ValueError("Llama.cpp did not report a loaded or available model")
        else:
            raise ValueError("Select a Llama.cpp model in Prompt Studio settings")

    if messages_override is not None:
        messages = messages_override
        has_images = any(
            isinstance(message, dict)
            and isinstance(message.get("content"), list)
            and any(
                isinstance(part, dict) and part.get("type") == "image_url"
                for part in message["content"]
            )
            for message in messages
        )
    else:
        has_images = bool(image_data_uri)
        user_content = str(prompt or "")
        if image_data_uri:
            user_content = [
                {"type": "text", "text": str(prompt or "")},
                {"type": "image_url", "image_url": {"url": str(image_data_uri)}},
            ]
        messages = [
            {"role": "system", "content": CHAT_SYSTEM_MESSAGE},
            {"role": "user", "content": user_content},
        ]
    if has_images:
        reason = _llamacpp_vision_unavailable_reason(
            _llamacpp_props(base_url, timeout, model),
            model,
        )
        if reason:
            raise RuntimeError(reason)

    effort, enable_thinking = _llamacpp_reasoning_options(thinking_mode)
    prompt_tokens = _llamacpp_token_count(
        base_url,
        timeout,
        messages,
        model,
        effort,
        enable_thinking,
    )
    context_length = _llamacpp_context_length(base_url, timeout, model)
    if context_length is not None and prompt_tokens is not None and prompt_tokens >= context_length - 32:
        raise RuntimeError(
            f"The formatted request uses {prompt_tokens} tokens, leaving no usable space in "
            f"Llama.cpp's {context_length}-token context window. Shorten the instructions or "
            "increase llama-server's context size."
        )
    response_tokens = _requested_response_tokens(max_response_tokens, default_max_response_tokens)
    max_length, thinking_budget = _llamacpp_generation_budget(
        response_tokens,
        thinking_mode,
        _context_safe_generation_limit(context_length, prompt_tokens),
        reasoning_budget_tokens,
    )
    stop_sequences = _split_stop_sequences(stop_sequence)
    if include_default_continuation_stops and not enable_thinking:
        stop_sequences = _with_default_continuation_stops(stop_sequences)
    chat_template_kwargs = {"enable_thinking": enable_thinking}
    if effort != "none":
        # Keep the native effort for both reasoning and instruct variants.
        # Top-level "none" would make llama.cpp erase the template effort.
        chat_template_kwargs["reasoning_effort"] = effort
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_length,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "min_p": float(min_p),
        "presence_penalty": float(presence_penalty),
        "repeat_penalty": float(rep_pen),
        "repeat_last_n": int(rep_pen_range),
        "seed": int(sampler_seed),
        "reasoning_effort": effort,
        "reasoning_format": "auto",
        "chat_template_kwargs": chat_template_kwargs,
        "stop": stop_sequences,
    }
    if isinstance(response_schema, dict):
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "prompt_studio_response",
                "schema": response_schema,
                "strict": True,
            },
        }
    if thinking_budget is not None:
        payload["thinking_budget_tokens"] = thinking_budget
    ensure_active()
    result = _post_llamacpp_chat(
        base_url,
        payload,
        timeout,
        response_hook=response_hook,
        cancellation_check=cancellation_check,
    )
    ensure_active()
    try:
        choice = result["choices"][0]
        message = choice["message"]
        content = str(message.get("content") or "")
        finish_reason = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Llama.cpp response: {result}") from exc
    if finish_reason == "length":
        raise RuntimeError(
            f"Llama.cpp exhausted the {max_length}-token completion budget before finishing. "
            "Increase max_response_tokens or llama-server's context size."
        )
    if not content.strip():
        if message.get("reasoning_content"):
            raise RuntimeError(
                "Llama.cpp returned reasoning but no final answer. Increase max_response_tokens "
                "or llama-server's context size."
            )
        raise RuntimeError(f"Llama.cpp returned an empty chat completion: {result}")
    return content

def _ollama_thinking_value(thinking_mode):
    effort = _reasoning_effort(thinking_mode)
    if effort == "none":
        return False
    if effort in {"minimal", "low"}:
        return "low"
    return effort

OLLAMA_MAX_REQUEST_CONTEXT = 65536

def _ollama_context_estimate(messages):
    """Conservatively estimate formatted text and vision tokens for Ollama.

    Ollama does not expose the model's exact formatted input-token count through
    its native chat API.  A conservative estimate is still preferable to
    silently relying on a small server default that may discard system or user
    context.  Image payload bytes are excluded; each image receives a separate
    vision-token allowance.
    """
    text_chars = 0
    image_count = 0
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            text_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text_chars += len(str(part.get("text") or ""))
                elif part.get("type") == "image_url":
                    image_count += 1
        images = message.get("images")
        if isinstance(images, list):
            image_count += len(images)
    # English prompt text is commonly near four characters per token.  Two
    # characters per token leaves headroom for JSON/chat-template overhead and
    # less compact languages.  Vision encoders vary, so reserve a generous
    # fixed allowance per sanitized image.
    return math.ceil(text_chars / 2) + image_count * 4096 + 256

def _ollama_request_context(messages, completion_tokens, model_context_length=None):
    required = _ollama_context_estimate(messages) + max(1, int(completion_tokens))
    if model_context_length is not None and required > int(model_context_length):
        raise RuntimeError(
            f"The Ollama request needs approximately {required} context tokens, but the selected "
            f"model advertises a {int(model_context_length)}-token context window. Shorten the "
            "conversation or reduce the response budget."
        )
    if required > OLLAMA_MAX_REQUEST_CONTEXT:
        raise RuntimeError(
            f"The Ollama request needs approximately {required} context tokens, exceeding "
            f"Prompt Studio's {OLLAMA_MAX_REQUEST_CONTEXT}-token safety limit. Shorten the "
            "conversation or Director context."
        )
    # Stable power-of-two buckets avoid needless Ollama runner reloads between
    # closely related routing, rewrite, and discussion stages.
    context = 2048
    while context < required:
        context *= 2
    return min(context, int(model_context_length)) if model_context_length is not None else context

class _GenerationText(str):
    """A generated string with a non-fatal warning for interactive callers."""

    def __new__(cls, value, warning=""):
        instance = super().__new__(cls, str(value or ""))
        instance.warning = str(warning or "").strip()
        return instance

def _ollama_completion_warning(max_length, retried_without_thinking=False, partial=False):
    if retried_without_thinking:
        return (
            f"Ollama did not finish its first attempt within the {max_length}-token response "
            "budget, so Prompt Studio retried once with Thinking disabled. Review the result; "
            "increase Max response tokens if it looks incomplete."
        )
    if partial:
        return (
            f"Ollama reached the {max_length}-token response limit. Prompt Studio kept the partial "
            "result instead of failing, so it may be incomplete. Increase Max response tokens or "
            "lower/disable Thinking for this model."
        )
    return ""

def _unload_ollama_model(ollama_url, ollama_model, request_timeout=15):
    """Release a loaded Ollama model before ComfyUI starts diffusion work."""
    base_url = _clean_ollama_base_url(ollama_url)
    model = str(ollama_model or "").strip()
    if not model:
        raise ValueError("Select an Ollama model in Prompt Studio settings")
    result = _post_json(
        _ollama_api_url(base_url, "chat"),
        {
            "model": model,
            "messages": [],
            "stream": False,
            "keep_alive": 0,
        },
        int(request_timeout),
        service_name="Ollama",
    )
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected Ollama unload response: {result}")
    if result.get("error"):
        raise RuntimeError(f"Ollama reported an unload error: {result['error']}")
    return result

@_coordinated_native_llm("ollama")
def _generate_ollama(
    prompt,
    ollama_url,
    ollama_model,
    max_response_tokens,
    default_max_response_tokens,
    temperature,
    top_p,
    top_k,
    min_p,
    rep_pen,
    rep_pen_range,
    sampler_seed,
    thinking_mode,
    stop_sequence,
    request_timeout,
    include_default_continuation_stops=False,
    image_base64=None,
    messages_override=None,
    response_hook=None,
    cancellation_check=None,
    allow_partial=True,
    keep_alive=30,
    presence_penalty=0.0,
    response_schema=None,
):
    def ensure_active():
        if cancellation_check is not None and cancellation_check():
            raise RuntimeError("Ollama request was cancelled")

    ensure_active()
    base_url = _clean_ollama_base_url(ollama_url)
    model = str(ollama_model or "").strip()
    if not model:
        raise ValueError("Select an Ollama model in Prompt Studio settings")
    ensure_active()
    model_info = _ollama_model_info(base_url, model, int(request_timeout))
    model_context_length = _ollama_model_context_length(model_info)
    has_images = bool(image_base64) or (
        messages_override is not None
        and any(
            isinstance(message, dict) and bool(message.get("images"))
            for message in messages_override
        )
    )
    if has_images:
        capabilities = model_info.get("capabilities")
        capabilities = (
            [str(value).strip().casefold() for value in capabilities]
            if isinstance(capabilities, list)
            else []
        )
        vision_reason = _ollama_vision_unavailable_reason(capabilities, model)
        if vision_reason:
            raise RuntimeError(vision_reason)
    ensure_active()

    stop_sequences = _split_stop_sequences(stop_sequence)
    if include_default_continuation_stops and _reasoning_effort(thinking_mode) == "none":
        stop_sequences = _with_default_continuation_stops(stop_sequences)
    response_tokens = _requested_response_tokens(max_response_tokens, default_max_response_tokens)
    max_length, _thinking_budget = _chat_generation_budget(response_tokens, thinking_mode)
    user_message = {"role": "user", "content": str(prompt or "")}
    if image_base64:
        user_message["images"] = [str(image_base64)]
    messages = messages_override if messages_override is not None else [
        {"role": "system", "content": CHAT_SYSTEM_MESSAGE},
        user_message,
    ]
    request_context = _ollama_request_context(messages, max_length, model_context_length)
    def generate_once(request_thinking_mode, prediction_limit):
        options = {
            "num_predict": int(prediction_limit),
            "num_ctx": request_context,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "min_p": float(min_p),
            "presence_penalty": float(presence_penalty),
            "repeat_penalty": float(rep_pen),
            "repeat_last_n": int(rep_pen_range),
            "stop": stop_sequences,
        }
        if int(sampler_seed) >= 0:
            options["seed"] = int(sampler_seed)
        payload = {
            "model": model,
            "messages": messages,
            "options": options,
            "think": _ollama_thinking_value(request_thinking_mode),
            "stream": True,
            # Reuse the runner across Prompt Studio's routing and rewrite stages. The
            # frontend explicitly unloads it immediately before queueing ComfyUI.
            "keep_alive": keep_alive,
        }
        if isinstance(response_schema, dict):
            payload["format"] = response_schema
        ensure_active()
        result = _post_json(
            _ollama_api_url(base_url, "chat"),
            payload,
            int(request_timeout),
            service_name="Ollama",
            response_hook=response_hook,
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected Ollama response: {result}")
        if result.get("error"):
            raise RuntimeError(f"Ollama reported an error: {result['error']}")
        try:
            message = result["message"]
            content = str(message.get("content") or "")
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response: {result}") from exc
        return content, message, result.get("done_reason"), result

    content, message, done_reason, result = generate_once(thinking_mode, max_length)
    if done_reason == "length" and content.strip() and allow_partial:
        return _GenerationText(
            content,
            _ollama_completion_warning(max_length, partial=True),
        )

    effort = _reasoning_effort(thinking_mode)
    retry_without_thinking = effort != "none" and (
        done_reason == "length" or not content.strip()
    )
    if retry_without_thinking:
        content, message, done_reason, result = generate_once("Disabled", max_length)
        warning = _ollama_completion_warning(
            max_length,
            retried_without_thinking=True,
        )
        if done_reason == "length":
            if content.strip() and allow_partial:
                return _GenerationText(
                    content,
                    f"{warning} {_ollama_completion_warning(max_length, partial=True)}",
                )
            raise RuntimeError(
                f"Ollama exhausted the {max_length}-token completion budget twice, including "
                "a retry with Thinking disabled. Increase max_response_tokens."
            )
        if content.strip():
            return _GenerationText(content, warning)

    if done_reason == "length":
        raise RuntimeError(
            f"Ollama exhausted the {max_length}-token completion budget before finishing. "
            "Increase max_response_tokens."
        )
    if not content.strip():
        if message.get("thinking"):
            raise RuntimeError(
                "Ollama returned reasoning but no final answer, including after a retry with "
                "Thinking disabled. Increase max_response_tokens."
            )
        raise RuntimeError(f"Ollama returned an empty chat completion: {result}")
    return str(content)

def _strip_response(text):
    text = text.strip()

    final_marker_matches = list(
        re.finditer(
            r"(?:^|\n)\s*(?:\*\*)?final\s+prompt\s*:?\s*(?:\*\*)?\s*:?\s*",
            text,
            flags=re.IGNORECASE,
        )
    )
    has_final_marker = bool(final_marker_matches)
    if final_marker_matches:
        text = text[final_marker_matches[0].end():]

    text = re.sub(r"^\s*<\|channel\>thought\b.*?<channel\|>\s*", "", text, count=1, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^\s*<\|channel\>analysis\b.*?<channel\|>\s*", "", text, count=1, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^\s*<thinking>.*?</thinking>\s*", "", text, count=1, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, count=1, flags=re.IGNORECASE | re.DOTALL)
    if not has_final_marker:
        text = re.sub(r"^.*?</thinking>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"^.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</(?:thinking|think)>", "", text, flags=re.IGNORECASE)
    output_match = re.fullmatch(r"\s*<output>(.*?)</output>\s*", text, flags=re.IGNORECASE | re.DOTALL)
    if output_match:
        text = output_match.group(1)
    fence_match = re.fullmatch(r"\s*```(?:text|prompt)?\s*\n?(.*?)\n?```\s*", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    text = re.sub(r"</?(?:output|final_prompt)>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(?:rewritten prompt|amplified prompt|prompt)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()

def _reasoning_effort(thinking_mode):
    mode = str(thinking_mode or "Disabled").strip().lower()
    # Generic prompt policies and token budgets use standard effort levels.
    # llama.cpp receives the model-native name via _llamacpp_reasoning_options.
    mode = {"spoon": "xhigh", "einstein": "high"}.get(mode, mode)
    if mode == "disabled":
        return "none"
    if mode in {"minimal", "low", "medium", "high", "xhigh"}:
        return mode
    return "none"

def _llamacpp_reasoning_options(thinking_mode):
    """Return the GGUF template's effort and independent thinking switch.

    Twin Turbo's embedded template accepts bare effort names in both modes;
    the i-prefixed forms belong to its in-message {REASON:...} syntax only.
    """
    mode = str(thinking_mode or "Disabled").strip().lower()
    if mode in {"spoon", "einstein"}:
        return mode, True
    if mode.startswith("instruct "):
        native = mode.removeprefix("instruct ")
        if native in {"spoon", "einstein", "xhigh", "medium", "low"}:
            return native, False
    effort = _reasoning_effort(thinking_mode)
    return effort, effort != "none"
