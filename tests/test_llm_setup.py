"""Synthetic provider contracts; no installed LLM, ComfyUI or GPU required."""
import base64
import copy
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from studio import llm
from studio.service import Service
from studio.store import Conflict
from studio.setup import Setup

ROOT = Path(__file__).resolve().parents[1] / 'internal-tests/artifacts/unit'
ROOT.mkdir(parents=True, exist_ok=True)


class BackendCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.service = Service(Path(self.temp.name) / 'data')
        self.settings = self.service.save_settings({'integration': 'standalone', 'llm_provider': 'llamacpp'})
        self.backend = llm.backend(self.settings)
        self.addCleanup(lambda: llm._BACKENDS.pop(self.settings['_llm_root'], None))

    def test_legacy_endpoint_and_hidden_provider_choices_survive(self):
        other = Service(Path(self.temp.name) / 'legacy')
        other.store.save_settings({'llm_url': 'http://127.0.0.1:1234/v1', 'llm_model': 'synthetic'})
        self.assertEqual(other.settings()['llm_provider'], 'openai')
        other.save_settings({'llm_provider': 'ollama', 'ollama_model': 'native-synthetic'})
        restored = other.save_settings({'integration': 'promptstudio'})
        self.assertEqual(restored['llm_model'], 'synthetic')
        self.assertEqual(restored['ollama_model'], 'native-synthetic')
        self.assertNotIn('_llm_root', other.store.settings())

    def test_profiles_conflicts_unknown_fields_and_delete_clear_autostart(self):
        model = Path(self.temp.name) / 'synthetic.gguf'
        model.touch()
        doc = {'model_gguf': str(model), 'custom_future_field': {'retain': True}}
        created = self.service.llm_profile({'action': 'save', 'llamacpp_config_profile': 'synthetic.json', 'config': doc})
        loaded = self.service.llm_profile({'action': 'load', 'llamacpp_config_profile': 'synthetic.json'})
        self.assertEqual(loaded['config']['custom_future_field'], {'retain': True})
        self.assertEqual(loaded['revision'], created['revision'])
        with self.assertRaises(Conflict):
            self.service.llm_profile({'action': 'save', 'llamacpp_config_profile': 'synthetic.json', 'config': doc})
        self.service.save_settings({'llamacpp_config_profile': 'synthetic.json', 'llamacpp_autostart': True})
        self.service.llm_profile({'action': 'delete', 'llamacpp_config_profile': 'synthetic.json'})
        self.assertFalse(self.service.settings()['llamacpp_autostart'])
        self.assertEqual(self.service.settings()['llamacpp_config_profile'], '')
        self.assertTrue(model.is_file())

    def test_profile_paths_and_nonlocal_managed_hosts_are_rejected(self):
        for profile in ('../bad.json', 'C:/bad.json', 'bad.exe'):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                self.backend.profile({'action': 'load', 'llamacpp_config_profile': profile})
        for config in ({'host': '0.0.0.0'}, {'extra_args': ['--host=0.0.0.0']}, {'extra_args': ['--port', '1234']}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.backend.profile({'action': 'save', 'llamacpp_config_profile': 'bad.json', 'config': config})
        with self.assertRaises(ValueError):
            self.service.save_settings({'ollama_url': 'https://example.com'})

    def test_managed_profile_uses_its_endpoint_and_model(self):
        model = Path(self.temp.name) / 'synthetic.gguf'
        model.touch()
        executable = Path(self.temp.name) / 'llama-server.exe'
        executable.touch()
        self.backend.profile({'action': 'save', 'llamacpp_config_profile': 'synthetic.json',
                              'config': {'model_gguf': str(model), 'port': 19234}})
        settings = {**self.settings, 'llamacpp_executable': str(executable), 'llamacpp_config_profile': 'synthetic.json'}
        effective = self.backend.effective(settings)
        self.assertEqual(effective['llamacpp_url'], 'http://127.0.0.1:19234')
        self.assertEqual(effective['llamacpp_model'], 'synthetic.gguf')
        self.assertEqual(settings['llamacpp_url'], 'http://127.0.0.1:8080')

    def test_generation_honors_saved_thinking_profile_and_task_budget(self):
        profile = {**llm.template.LLAMACPP_LLM_PROFILE_DEFAULTS, 'thinking_mode': 'High', 'thinking_temperature': .42,
                   'max_response_tokens': 0}
        settings = {**self.settings, 'llm_provider': 'ollama', 'ollama_model': 'synthetic', 'llm_profiles': {'ollama': profile}}
        with patch.object(llm.native, '_generate_ollama', return_value='Red') as generate:
            self.assertEqual(self.backend.generate(settings, 'Name the color', b'png', 'image/png', temperature=.1, max_tokens=1800), 'Red')
        self.assertEqual(generate.call_args.kwargs['temperature'], .42)
        self.assertEqual(generate.call_args.kwargs['max_response_tokens'], 1800)
        self.assertFalse(generate.call_args.kwargs['allow_partial'])
        self.assertEqual(generate.call_args.kwargs['messages_override'][0]['images'], [base64.b64encode(b'png').decode()])

    def test_llama_config_default_thinking_mode_is_not_overridden(self):
        model = Path(self.temp.name) / 'synthetic.gguf'
        model.touch()
        profile = {**llm.template.LLAMACPP_LLM_PROFILE_DEFAULTS, 'thinking_mode': 'High', 'thinking_temperature': .37}
        self.backend.profile({'action': 'save', 'llamacpp_config_profile': 'synthetic.json',
                              'config': {'model_gguf': str(model), 'llm_profile': profile}})
        settings = {**self.settings, 'llamacpp_config_profile': 'synthetic.json'}
        with patch.object(llm.native, '_generate_llamacpp', return_value='Red') as generate:
            self.backend.generate(settings, 'Color', None, None, temperature=.1, max_tokens=50)
        self.assertEqual(generate.call_args.kwargs['thinking_mode'], 'High')
        self.assertEqual(generate.call_args.kwargs['temperature'], .37)

    def test_release_only_used_models_and_retain_failures_for_retry(self):
        settings = {**self.settings, 'llm_provider': 'ollama', 'ollama_model': 'synthetic'}
        with patch.object(llm.native, '_unload_ollama_model') as unload:
            self.backend.release(settings)
            unload.assert_not_called()
            self.backend.used[self.backend.identity(settings)] = settings
            self.backend.release({**settings, 'keep_models_loaded': True})
            unload.assert_not_called()
            unload.side_effect = RuntimeError('busy')
            with self.assertRaises(RuntimeError):
                self.backend.release(settings)
            self.assertTrue(self.backend.used)
            unload.side_effect = None
            self.backend.release_used(self.settings)
            self.assertFalse(self.backend.used)

    def test_switching_owned_profiles_releases_current_server_once(self):
        with patch.object(self.backend.runtime, '_start_llamacpp_server', return_value={'managed': True, 'running': True}):
            self.backend.process('start', {**self.settings, 'llamacpp_model': 'model-a'})
            self.backend.process('start', {**self.settings, 'llamacpp_model': 'model-b'})
        with patch.object(self.backend.runtime, '_llamacpp_managed_process_status', return_value={'managed': True, 'running': True}), \
             patch.object(self.backend.runtime, '_stop_llamacpp_server') as stop, \
             patch.object(self.backend.runtime, '_unload_llamacpp_model') as unload:
            self.backend.release_used(self.settings)
        stop.assert_called_once()
        unload.assert_not_called()
        self.assertFalse(self.backend.used)

    def test_manager_install_requires_ack_and_idle_comfyui(self):
        setup = self.service.setup
        status = {'url': 'http://127.0.0.1:8188', 'comfyui': True, 'manager': True, 'promptstudio': False}
        with self.assertRaisesRegex(ValueError, 'Confirm'):
            setup.install_promptstudio()
        with self.assertRaises(Conflict):
            setup.install_promptstudio(True, 'http://127.0.0.1:9999')
        with patch.object(setup, 'manager_status', return_value=status), \
             patch('studio.setup.request', return_value={'queue_running': [1], 'queue_pending': []}), \
             patch('studio.setup.threading.Thread') as thread:
            with self.assertRaisesRegex(ValueError, 'active or queued'):
                setup.install_promptstudio(True, status['url'])
            thread.assert_not_called()
        with patch.object(setup, 'manager_status', return_value=status), \
             patch('studio.setup.request', return_value={'queue_running': [], 'queue_pending': []}), \
             patch('studio.setup.threading.Thread'):
            self.assertEqual(setup.install_promptstudio(True, status['url'])['promptstudio_install']['status'], 'running')
        with patch('studio.setup.request', return_value='') as request:
            setup._install_promptstudio(status['url'])
        self.assertEqual(request.call_count, 1)
        self.assertEqual(request.call_args.args[1], {'url': 'https://github.com/tiko13/ComfyUI_PromptStudio'})
        self.assertEqual(setup.status()['promptstudio_install']['status'], 'restart_required')

    def test_manager_failure_and_restart_do_not_retry_automatically(self):
        setup = self.service.setup
        setup.state['promptstudio_install'] = {'status': 'running'}
        setup._save()
        self.assertEqual(Setup(self.service).status()['promptstudio_install']['status'], 'interrupted')
        with patch('studio.setup.request', side_effect=RuntimeError('rejected')):
            setup._install_promptstudio('http://127.0.0.1:8188')
        self.assertEqual(setup.status()['promptstudio_install']['status'], 'needs_attention')

    def test_named_profiles_and_active_job_server_guard(self):
        profile = {**llm.template.LLAMACPP_LLM_PROFILE_DEFAULTS, 'name': 'Synthetic sampler', 'thinking_temperature': .42}
        saved = self.service.save_settings({'llm_profile_library': {'synthetic': profile}, 'llm_profile_selection': {'ollama': 'synthetic'}})
        self.assertEqual(saved['llm_profile_library']['synthetic']['thinking_temperature'], .42)
        with self.assertRaises(ValueError):
            self.service.save_settings({'llm_profile_library': {}})
        with self.service.job_lock, patch.object(self.backend, 'process') as process:
            with self.assertRaises(Conflict):
                self.service.llm_process('start')
            process.assert_not_called()

    def test_promptstudio_only_does_not_fall_back_to_standalone_preview(self):
        from studio.sample_prompts import generate
        from studio.domain import normalize_project
        settings = {**self.settings, 'integration': 'promptstudio', 'llm_provider': 'koboldcpp'}
        with patch('studio.sample_prompts.PromptStudio.connection', side_effect=RuntimeError('offline')), \
             patch('studio.sample_prompts.standalone_caption') as standalone:
            result = generate(normalize_project({'trigger': 'synthetic'}), [], settings)
        self.assertEqual(result['source'], 'rules')
        standalone.assert_not_called()


class NativeProtocolCase(unittest.TestCase):
    setUp = BackendCase.setUp
    def test_real_native_transports_against_synthetic_loopback_server(self):
        calls = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def respond(self, value, content_type='application/json'):
                body = value.encode() if isinstance(value, str) else json.dumps(value).encode()
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def do_GET(self):
                route = self.path.split('?')[0]
                data = {'/api/extra/version': {'vision': True, 'jinja': True},
                        '/api/v1/config/max_context_length': {'value': 32768},
                        '/api/v1/model': {'result': 'synthetic'},
                        '/api/tags': {'models': [{'name': 'synthetic'}]},
                        '/v1/models': {'data': [{'id': 'synthetic'}]},
                        '/models': {'data': [{'id': 'synthetic'}]},
                        '/props': {'modalities': {'vision': True}, 'default_generation_settings': {'n_ctx': 32768}}}
                self.respond(data.get(route, {}))
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append((self.path, data))
                if self.path == '/api/show':
                    return self.respond({'capabilities': ['vision', 'completion'], 'model_info': {'synthetic.context_length': 32768}})
                if self.path.endswith('/input_tokens'):
                    return self.respond({'input_tokens': 40})
                if self.path == '/api/extra/tokencount':
                    return self.respond({'value': 40})
                if self.path == '/api/chat':
                    return self.respond(json.dumps({'message': {'content': 'Red'}, 'done': True, 'done_reason': 'stop'}) + '\n', 'application/x-ndjson')
                if data.get('stream'):
                    return self.respond('data: ' + json.dumps({'choices': [{'delta': {'content': 'Red'}, 'finish_reason': 'stop'}]}) + '\n\ndata: [DONE]\n\n', 'text/event-stream')
                self.respond({'choices': [{'message': {'content': 'Red'}, 'finish_reason': 'stop'}]})
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        def close():
            server.shutdown()
            server.server_close()
            worker.join(2)
        self.addCleanup(close)
        url = f'http://127.0.0.1:{server.server_port}'
        for provider, key in [('llamacpp', 'llamacpp'), ('ollama', 'ollama'), ('koboldcpp', 'kobold')]:
            with self.subTest(provider=provider):
                settings = {**self.settings, 'llm_provider': provider, key + '_url': url,
                            'llamacpp_model': 'synthetic', 'ollama_model': 'synthetic'}
                status = self.backend.inspect(settings)
                self.assertTrue(status['vision']['available'])
                self.assertTrue(status['configured'])
                self.assertEqual(self.backend.generate(settings, 'Name the color', b'synthetic', 'image/png', temperature=.1, max_tokens=1800), 'Red')
        generations = [data for route, data in calls if route in {'/api/chat', '/v1/chat/completions'}]
        self.assertEqual(len(generations), 3)
        self.assertTrue(generations[0]['stream'])
        self.assertEqual(generations[1]['options']['num_predict'], 1800)
        self.assertFalse(generations[2]['stream'])


if __name__ == '__main__':
    unittest.main()
