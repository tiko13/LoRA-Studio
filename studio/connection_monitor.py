"""Read-only connection monitoring; never load models or reserve inference capacity."""
from urllib.parse import urlencode

from .integrations import PromptStudio, PS, request, local_url
from . import llm
from .trainer import backend_status
from .captioners import captioner_status


def optional(url, payload=None):
    try:
        return request(url, payload, timeout=3)
    except (ValueError, RuntimeError):
        return None


def standalone(settings):
    provider = settings['llm_provider']
    if provider != 'openai':
        settings = llm.backend(settings).effective(settings)
    status = {'provider': provider, 'reachable': True, 'busy': None, 'vision': None}
    if provider == 'koboldcpp':
        base = local_url(settings['kobold_url'])
        perf = request(base + '/api/extra/perf', timeout=3)
        model = optional(base + '/api/v1/model') or {}
        version = optional(base + '/api/extra/version') or {}
        status.update(busy=perf['idle'] == 0 if perf.get('idle') in (0, 1) else None, queue=perf.get('queue', 0),
                      model=model.get('result'), vision=version.get('vision'))
        status['model_installed'] = bool(status['model']) and not llm.template._kobold_model_is_inactive(status['model'])
        return status
    if provider == 'ollama':
        base = local_url(settings['ollama_url'])
        catalog = request(base + '/api/tags', timeout=3)
        model = settings.get('ollama_model', '')
        status.update(model=model, model_installed=model in [m.get('name') for m in catalog.get('models', [])])
        if status['model_installed']:
            info = optional(base + '/api/show', {'model': model}) or {}
            if isinstance(info.get('capabilities'), list):
                status['vision'] = 'vision' in info['capabilities']
        status['message'] = 'Connected. Ollama does not report live processing activity.'
        return status
    if provider == 'openai':
        base = local_url(settings['llm_url'])
        catalog = request(base + '/models', timeout=3)
        model = settings.get('llm_model', '')
        status.update(model=model, model_installed=model in [m.get('id') for m in catalog.get('data', [])],
                      message='Connected. This endpoint does not report live processing activity.')
        return status
    base = local_url(settings['llamacpp_url']).removesuffix('/v1')
    request(base + '/health', timeout=3)
    catalog = request(base + '/models', timeout=3)
    entries = catalog.get('data', [])
    model = settings.get('llamacpp_model') or (entries[0].get('id') if len(entries) == 1 else '')
    entry = next((m for m in entries if m.get('id') == model or model in m.get('aliases', [])), {})
    model = entry.get('id', model)
    status.update(model=model, model_installed=bool(entry))
    model_status = entry.get('status') or {}
    model_state = model_status.get('value') if isinstance(model_status, dict) else None
    status['model_state'] = model_state
    if model_state in {'unloaded', 'sleeping', 'loading'}:
        status['busy'] = model_state == 'loading'
        modalities = (entry.get('architecture') or {}).get('input_modalities')
        status['vision'] = 'image' in modalities if isinstance(modalities, list) else None
        status['message'] = 'Loading model…' if status['busy'] else 'Model unloaded; server ready for the next request.'
        if model_status.get('failed'):
            status['handoff_error'] = 'Model failed to load. Check the Llama.cpp server log.'
        return status
    if not model or not entry:
        return status
    # Router probes must never autoload weights during training or rendering.
    query = '?' + urlencode({'model': model, 'autoload': 'false'})
    slots = optional(base + '/slots' + query)
    props = optional(base + '/props' + query) or {}
    status['vision'] = (props.get('modalities') or {}).get('vision')
    if isinstance(slots, list):
        active = [s for s in slots if s.get('is_processing') is True]
        status.update(busy=bool(active), active_slots=len(active))
        counts = []
        for slot in active:
            next_token = slot.get('next_token')
            values = next_token if isinstance(next_token, list) else [next_token or slot]
            decoded = [v.get('n_decoded') for v in values if isinstance(v, dict)]
            counts.extend([max(v for v in decoded if isinstance(v, int) and v >= 0)]
                          if any(isinstance(v, int) and v >= 0 for v in decoded) else [])
        if counts:
            status['generated_tokens'] = sum(counts)
    else:
        status['message'] = 'Connected. Enable --slots on llama-server to monitor processing.'
    return status


def snapshot(settings):
    ps = {'enabled': settings['integration'] != 'standalone', 'reachable': False}
    result = {'promptstudio': ps, 'source': 'standalone', 'settings_revision': settings['settings_revision'],
              'backend': backend_status(settings), 'captioner': captioner_status(settings)}
    if ps['enabled']:
        bridge = PromptStudio(settings['prompt_studio_url'])
        try:
            summary = request(bridge.url + PS + '/chats?summaries=1&limit=1', timeout=3)
            ps['reachable'] = True
            connection = bridge.connection(summary)
            ps['configured'] = True
        except (ValueError, RuntimeError, KeyError, IndexError) as exc:
            ps['message'] = str(exc)
        else:
            result['source'] = 'promptstudio'
            try:
                result['llm'] = request(bridge.url + PS + '/llm/status', connection, timeout=15)
            except (ValueError, RuntimeError) as exc:
                result['llm'] = {'provider': connection['llm_provider'], 'reachable': False, 'message': str(exc)}
            return result
        if settings['integration'] == 'promptstudio':
            result.update(source='promptstudio', llm={'reachable': False, 'message': ps['message']})
            return result
    try:
        result['llm'] = standalone(settings)
    except (ValueError, RuntimeError, KeyError, TypeError) as exc:
        result['llm'] = {'provider': settings['llm_provider'], 'reachable': False, 'message': str(exc)}
    return result
