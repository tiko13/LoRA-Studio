"""LoRA Studio adapter for the vendored Prompt Studio native LLM backend."""
import base64
import copy
import importlib.util
from pathlib import Path
import threading
import uuid

from .vendor.promptstudio import providers as native, runtime as template
from .vendor.promptstudio.llm_coordinator import SHARED_COORDINATOR, endpoint_identity

PROVIDERS = {'llamacpp', 'ollama', 'koboldcpp', 'openai'}
DEFAULTS = {
    'llm_provider': 'llamacpp', 'kobold_url': 'http://127.0.0.1:5001',
    'ollama_url': 'http://127.0.0.1:11434', 'ollama_model': '',
    'llamacpp_url': 'http://127.0.0.1:8080', 'llamacpp_model': '',
    'llamacpp_executable': '', 'llamacpp_config_profile': '', 'llamacpp_autostart': False,
    'keep_models_loaded': False, 'llm_profiles': {}, 'llm_profile_library': {}, 'llm_profile_selection': {},
}
_BACKENDS = {}
_LOCK = threading.RLock()


def configured(settings):
    provider = settings.get('llm_provider', 'openai')
    return provider == 'koboldcpp' or bool(settings.get({'ollama': 'ollama_model', 'llamacpp': 'llamacpp_model'}.get(provider, 'llm_model'))) or (
        provider == 'llamacpp' and bool(settings.get('llamacpp_config_profile')))


def validate(settings):
    from .integrations import local_url
    if settings['llm_provider'] not in PROVIDERS:
        raise ValueError('Choose Llama.cpp, Ollama or KoboldCpp')
    for key in ('kobold_url', 'ollama_url', 'llamacpp_url'):
        settings[key] = local_url(settings[key])
    for key in ('ollama_model', 'llamacpp_model', 'llamacpp_executable', 'llamacpp_config_profile'):
        if not isinstance(settings[key], str) or len(settings[key]) > 2000 or '\x00' in settings[key]:
            raise ValueError(f'Invalid {key}')
        settings[key] = settings[key].strip()
    for key in ('llamacpp_autostart', 'keep_models_loaded'):
        if not isinstance(settings[key], bool):
            raise ValueError(f'{key} must be true or false')
    if settings['llamacpp_config_profile']:
        template._llamacpp_config_profile(settings['llamacpp_config_profile'])
    profiles = settings['llm_profiles']
    if not isinstance(profiles, dict) or any(k not in PROVIDERS for k in profiles):
        raise ValueError('Invalid provider profiles')
    settings['llm_profiles'] = {k: template._normalize_llamacpp_llm_profile(v) for k, v in profiles.items()}
    library = settings['llm_profile_library']
    if not isinstance(library, dict) or len(library) > 100:
        raise ValueError('Keep at most 100 LLM profiles')
    normalized = {}
    for key, value in library.items():
        if not isinstance(key, str) or not 1 <= len(key) <= 100 or not isinstance(value, dict):
            raise ValueError('Invalid LLM profile')
        name = value.get('name')
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError('Give the LLM profile a name of 1–80 characters')
        normalized[key] = {**template._normalize_llamacpp_llm_profile(value), 'name': name.strip()}
    settings['llm_profile_library'] = normalized
    selection = settings['llm_profile_selection']
    if not isinstance(selection, dict) or any(k not in PROVIDERS or not isinstance(v, str) or v and v not in library for k, v in selection.items()):
        raise ValueError('Select an existing LLM profile')


def backend(settings):
    root = settings.get('_llm_root')
    if not root:
        raise ValueError('The standalone backend needs a Studio data directory')
    key = str(Path(root).resolve())
    with _LOCK:
        if key not in _BACKENDS:
            _BACKENDS[key] = Backend(Path(key))
        return _BACKENDS[key]


class Backend:
    def __init__(self, root):
        self.root = root
        self.used = {}
        self.lock = threading.RLock()
        self.startup_error = ''
        root.mkdir(parents=True, exist_ok=True)
        spec = importlib.util.spec_from_file_location('studio.vendor.promptstudio.runtime_' + uuid.uuid4().hex, template.__file__)
        self.runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.runtime)
        r = self.runtime
        r.LLAMACPP_CONFIG_DIRECTORY = str(root / 'profiles')
        r.LLAMACPP_OUTPUT_LOG_PATH = str(root / 'server.log')
        r.LLAMACPP_PROCESS_STATE_PATH = str(root / 'process.json')
        r.LLAMACPP_AUTOSTART_PATH = str(root / 'autostart.json')
        r.LLAMACPP_WATCHDOG_PATH = str(Path(template.__file__).with_name('llamacpp_watchdog.py'))

    def serial(self, settings, fn):
        resources = [endpoint_identity(settings)]
        if settings['llm_provider'] == 'llamacpp':
            resources.append(('managed-llama', str(self.root)))
        return SHARED_COORDINATOR.run(resources, fn, exclusive=True)

    @staticmethod
    def validate_config(config):
        if str(config.get('host', '127.0.0.1')).strip() not in {'127.0.0.1', 'localhost', '::1'}:
            raise ValueError('The managed Llama.cpp server must listen on loopback.')
        if any(str(arg).split('=')[0] in {'--host', '-host', '--port', '-port'} for arg in config.get('extra_args', [])):
            raise ValueError('Set host and port in their config fields, not extra arguments.')

    def effective(self, settings):
        if settings['llm_provider'] == 'llamacpp' and settings.get('llamacpp_executable') and settings.get('llamacpp_config_profile'):
            document, _, _ = self.runtime._read_llamacpp_config_document(settings['llamacpp_config_profile'])
            self.validate_config(document)
            launcher = self.runtime._load_llamacpp_launcher_config(settings)
            return {**settings, 'llamacpp_url': launcher['url'], 'llamacpp_model': Path(launcher['model']).name}
        return settings

    def inspect(self, settings):
        settings = self.effective(settings)
        provider = settings['llm_provider']
        from .integrations import request
        if provider == 'openai':
            data = request(settings['llm_url'] + '/models', timeout=5)
            return {'available': True, 'provider': provider, 'models': [v['id'] for v in data.get('data', [])],
                    'configured': configured(settings), 'vision': {'available': False, 'reason': 'Test image support explicitly.'}}
        model_fn = {'ollama': lambda: native._list_ollama_models(settings['ollama_url'], 8),
                    'llamacpp': lambda: native._list_llamacpp_models(settings['llamacpp_url'], 8),
                    'koboldcpp': lambda: [self.runtime._kobold_model_name(settings['kobold_url'])]}
        models = model_fn[provider]()
        if not models or models == [None] or provider == 'koboldcpp' and self.runtime._kobold_model_is_inactive(models[0]):
            raise ValueError('The provider did not report an available model. Start the server and load a model, then refresh.')
        selected = provider == 'koboldcpp' or settings.get(provider + '_model') in models
        if provider == 'llamacpp':
            # Upstream capability discovery may load a router model via /props.
            with self.lock:
                previous = self.used.get(self.identity(settings), {})
                self.used[self.identity(settings)] = {**copy.deepcopy(settings),
                    '_managed_by_studio': previous.get('_managed_by_studio', False)}
        vision = native._llm_vision_capability(provider, **{k: settings[k] for k in
                    ('kobold_url', 'ollama_url', 'ollama_model', 'llamacpp_url', 'llamacpp_model')}, request_timeout=8)
        return {'available': True, 'provider': provider, 'models': models, 'configured': selected, 'vision': vision}

    def profiles(self):
        return {'profiles': self.runtime._list_llamacpp_config_profiles(), 'directory': self.runtime.LLAMACPP_CONFIG_DIRECTORY,
                'defaults': copy.deepcopy(template.LLAMACPP_LLM_PROFILE_DEFAULTS), 'thinking_modes': list(template.LLAMACPP_THINKING_MODES)}

    def profile(self, data):
        from .store import Conflict
        try:
            if data.get('action') == 'save' and isinstance(data.get('config'), dict):
                self.validate_config(data['config'])
            if data.get('action') == 'delete':
                return self.runtime._delete_llamacpp_config_profile(data.get('llamacpp_config_profile'))
            if data.get('action') == 'inspect':
                return self.runtime._inspect_llamacpp_config_profiles()
            return self.runtime._edit_llamacpp_config(data)
        except template.LlamacppConfigConflict as exc:
            raise Conflict(str(exc)) from exc
        except self.runtime.LlamacppConfigConflict as exc:
            raise Conflict(str(exc)) from exc

    def process(self, action, settings):
        if settings['llm_provider'] != 'llamacpp':
            raise ValueError('Managed server controls require Llama.cpp')
        if action in {'start', 'restart'}:
            settings = self.effective(settings)
        fn = {'status': self.runtime._llamacpp_managed_process_status, 'start': self.runtime._start_llamacpp_server,
              'stop': self.runtime._stop_llamacpp_server, 'restart': self.runtime._restart_llamacpp_server}.get(action)
        if fn is None:
            raise ValueError('Unknown server action')
        def run():
            result = fn(settings)
            self.startup_error = ''
            if action in {'start', 'restart'} and result.get('managed'):
                self.note_managed(settings)
            if action == 'stop' and result.get('stopped'):
                with self.lock:
                    self.used = {k: v for k, v in self.used.items() if not v.get('_managed_by_studio')}
            return {**result, **({'startup_error': self.startup_error} if self.startup_error else {})}
        if action == 'status':
            return {**fn(settings), 'startup_error': self.startup_error}
        return self.serial(settings, run)

    @staticmethod
    def identity(settings):
        provider = settings['llm_provider']
        return (endpoint_identity(settings), provider, settings.get(provider + '_model', ''))

    def note_managed(self, settings):
        with self.lock:
            self.used = {k: v for k, v in self.used.items() if not v.get('_managed_by_studio')}
            self.used[self.identity(settings)] = {**copy.deepcopy(settings), '_managed_by_studio': True}

    def generate(self, settings, instruction, data, mime, *, temperature, max_tokens, system=None):
        settings = dict(self.effective(settings))
        provider = settings['llm_provider']
        def run():
            if provider == 'llamacpp' and settings.get('llamacpp_config_profile') and settings.get('llamacpp_executable'):
                state = self.runtime._llamacpp_managed_process_status(settings)
                if state.get('running') and state.get('config_changed'):
                    result = self.runtime._restart_llamacpp_server(settings)
                else:
                    result = self.runtime._start_llamacpp_server(settings)
                if result.get('managed'):
                    settings['_managed_by_studio'] = True
                    self.note_managed(settings)
            if provider == 'koboldcpp':
                self.runtime._reload_kobold_model_if_needed(settings)
            profile = copy.deepcopy(settings.get('llm_profiles', {}).get(provider) or {
                **template.LLAMACPP_LLM_PROFILE_DEFAULTS, 'max_response_tokens': 0,
                'temperature': temperature, 'request_timeout': 300})
            if provider == 'llamacpp' and settings.get('llamacpp_config_profile'):
                profile = self.runtime._llamacpp_configured_generation_data(settings)
            thinking = native._reasoning_effort(profile['thinking_mode']) != 'none'
            common = {k: profile[k] for k in ('top_p','top_k','min_p','rep_pen','rep_pen_range','sampler_seed','thinking_mode','stop_sequence','request_timeout','presence_penalty')}
            if thinking:
                common.update({key: profile.get('thinking_'+key, common[key]) for key in ('top_p','top_k','min_p','rep_pen','rep_pen_range','presence_penalty')})
            # Callers bound response size for their task; user profiles can tighten it.
            limit = min(max_tokens, profile.get('max_response_tokens') or max_tokens)
            common.update(prompt=instruction, temperature=profile.get('thinking_temperature' if thinking else 'temperature', profile.get('temperature', temperature)),
                          max_response_tokens=limit, default_max_response_tokens=limit)
            encoded = base64.b64encode(data).decode() if data else None
            messages = [{'role': 'system', 'content': system}] if system else []
            # A failed request can still load weights, so retain its cleanup target.
            with self.lock:
                self.used[self.identity(settings)] = copy.deepcopy(settings)
            if provider == 'ollama':
                messages.append({'role':'user', 'content':instruction, **({'images':[encoded]} if encoded else {})})
                response = native._generate_ollama(**common, ollama_url=settings['ollama_url'], ollama_model=settings['ollama_model'],
                    messages_override=messages, allow_partial=False, keep_alive=-1 if settings['keep_models_loaded'] else 30)
            else:
                content = [{'type':'text','text':instruction}]
                if encoded:
                    content.append({'type':'image_url','image_url':{'url':f'data:{mime};base64,'+encoded}})
                messages.append({'role':'user','content':content})
                if provider == 'llamacpp':
                    response = native._generate_llamacpp(**common, llamacpp_url=settings['llamacpp_url'], llamacpp_model=settings['llamacpp_model'],
                        messages_override=messages, reasoning_budget_tokens=profile.get('llamacpp_reasoning_budget_tokens',0))
                else:
                    response = native._generate_kcpp(**common, kobold_url=settings['kobold_url'], messages_override=messages)
            if getattr(response, 'warning', ''):
                raise ValueError('The provider returned an incomplete response; no caption was saved. Adjust its response limit.')
            return native._strip_response(str(response))
        return self.serial(settings, run)

    def release(self, settings):
        if settings.get('keep_models_loaded') or settings.get('llm_provider') == 'openai':
            return
        def release():
            with self.lock:
                if self.identity(settings) not in self.used:
                    return
            provider = settings['llm_provider']
            if provider == 'ollama':
                native._unload_ollama_model(settings['ollama_url'], settings['ollama_model'])
            elif provider == 'koboldcpp':
                self.runtime._unload_kobold_model(settings)
            else:
                state = self.runtime._llamacpp_managed_process_status(settings)
                if settings.get('_managed_by_studio') and state.get('running'):
                    self.runtime._stop_llamacpp_server(settings)
                elif settings.get('_managed_by_studio'):
                    pass  # Already stopped; never try to unload a dead owned endpoint.
                else:
                    self.runtime._unload_llamacpp_model(settings['llamacpp_url'], settings['llamacpp_model'])
            with self.lock:
                self.used.pop(self.identity(settings), None)
        return self.serial(settings, release)

    def release_used(self, current):
        with self.lock:
            targets = list(self.used.values())
        for settings in targets:
            if settings['llm_provider'] == current['llm_provider']:
                settings = {**settings, 'keep_models_loaded': current['keep_models_loaded']}
            self.release(settings)

    def close(self):
        # Upstream ownership checks never terminate an external server.
        self.runtime._stop_llamacpp_server()
