import unittest
from unittest.mock import patch

from studio.connection_monitor import snapshot, standalone
from studio.service import DEFAULT_SETTINGS
from studio.integrations import PS


class MonitorTests(unittest.TestCase):
    def test_promptstudio_connection_survives_llm_failure(self):
        def request(url, payload=None, **kwargs):
            if '/chats?' in url:
                return {'activeChatId': 'synthetic'}
            raise RuntimeError('LLM status unavailable')
        with patch('studio.connection_monitor.request', side_effect=request), \
             patch('studio.integrations.PromptStudio.connection', return_value={'llm_provider': 'llamacpp'}):
            result = snapshot(DEFAULT_SETTINGS)
        self.assertTrue(result['promptstudio']['reachable'])
        self.assertTrue(result['promptstudio']['configured'])
        self.assertFalse(result['llm']['reachable'])

    def test_no_chat_is_connected_but_needs_setup(self):
        with patch('studio.connection_monitor.request', return_value={}):
            result = snapshot(DEFAULT_SETTINGS)
        self.assertTrue(result['promptstudio']['reachable'])
        self.assertFalse(result['llm']['reachable'])

    def test_standalone_does_not_contact_promptstudio(self):
        with patch('studio.connection_monitor.request', return_value={'data': [{'id': 'synthetic'}]}) as req:
            result = snapshot({**DEFAULT_SETTINGS, 'integration': 'standalone', 'llm_provider': 'openai', 'llm_model': 'synthetic'})
        self.assertFalse(result['promptstudio']['enabled'])
        self.assertTrue(result['llm']['model_installed'])
        self.assertIsNone(result['llm']['busy'])
        self.assertEqual(req.call_count, 1)
        self.assertNotIn(PS, req.call_args.args[0])

    def test_legacy_auto_fallback_is_explicit(self):
        with patch('studio.connection_monitor.request', side_effect=RuntimeError('Offline')), \
             patch('studio.connection_monitor.standalone', return_value={'reachable': True}):
            result = snapshot({**DEFAULT_SETTINGS, 'integration': 'auto'})
        self.assertEqual(result['source'], 'standalone')
        self.assertFalse(result['promptstudio']['reachable'])

    def llama(self, state=None):
        settings = {**DEFAULT_SETTINGS, 'llamacpp_model': 'synthetic'}
        entry = {'id': 'synthetic', **({'status': {'value': state}} if state else {})}
        def request(url, payload=None, **kwargs):
            if url.endswith('/health'):
                return {'status': 'ok'}
            if url.endswith('/models'):
                return {'data': [entry]}
            self.assertIn('autoload=false', url)
            if '/slots?' in url:
                return [{'is_processing': True, 'next_token': [{'n_decoded': 9}, {'n_decoded': 12}]}]
            return {'modalities': {'vision': True}}
        with patch('studio.connection_monitor.llm.backend') as backend, \
             patch('studio.connection_monitor.request', side_effect=request) as req:
            backend.return_value.effective.return_value = settings
            result = standalone(settings)
        return result, req

    def test_llama_activity_uses_non_loading_probes(self):
        result, _ = self.llama()
        self.assertTrue(result['busy'])
        self.assertTrue(result['vision'])
        self.assertEqual(result['generated_tokens'], 12)

    def test_unloaded_model_is_not_woken_by_polling(self):
        result, req = self.llama('unloaded')
        self.assertFalse(result['busy'])
        self.assertEqual(req.call_count, 2)
        self.assertEqual(result['model_state'], 'unloaded')

    def test_loading_is_working(self):
        result, _ = self.llama('loading')
        self.assertTrue(result['busy'])

    def test_ollama_reports_capability_without_claiming_idle(self):
        settings = {**DEFAULT_SETTINGS, 'llm_provider': 'ollama', 'ollama_model': 'synthetic'}
        with patch('studio.connection_monitor.llm.backend') as backend, \
             patch('studio.connection_monitor.request', side_effect=[
                 {'models': [{'name': 'synthetic'}]}, {'capabilities': ['completion', 'vision']}]) as req:
            backend.return_value.effective.return_value = settings
            result = standalone(settings)
        self.assertIsNone(result['busy'])
        self.assertTrue(result['vision'])
        self.assertTrue(result['model_installed'])
        self.assertTrue(req.call_args.args[0].endswith('/api/show'))

    def test_kobold_working_and_unknown_activity(self):
        settings = {**DEFAULT_SETTINGS, 'llm_provider': 'koboldcpp'}
        for perf, busy in [({'idle': 0}, True), ({}, None)]:
            with patch('studio.connection_monitor.llm.backend') as backend, \
                 patch('studio.connection_monitor.request', side_effect=[perf, {'result': 'synthetic'}, {'vision': True}]):
                backend.return_value.effective.return_value = settings
                result = standalone(settings)
            self.assertIs(result['busy'], busy)
            self.assertTrue(result['vision'])
