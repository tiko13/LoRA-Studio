import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from PIL import Image

from studio.comparisons import Comparisons, cell_graph, supported_profile
from studio.integrations import IntegrationHTTPError
from studio.service import Service
from studio.store import Conflict
from tests.test_results import fixture, weights, TEST_ROOT


def profile():
    return {'id': 'workflows/[PS] Synthetic.json', 'name': 'Synthetic Create', 'kind': 'create',
        'promptNodeId': '1', 'resultNodeIds': ['5'], 'loraNodes': [{'id': '3', 'loraType': 'Krea2'}],
        'modelNodes': [{'id': '2', 'modelType': 'Krea2', 'modelName': 'Krea2/synthetic.safetensors',
                        'choices': [{'name': 'Krea2/synthetic.safetensors'}]}],
        'samplingNodes': [{'id': '4'}], 'snapshot': {'output': {
            '1': {'class_type': 'KCPP_PromptAmplify', 'inputs': {'prompt': 'synthetic', 'slot_name': 'Test'}},
            '2': {'class_type': 'KCPP_PromptStudioModelLoader', 'inputs': {'model_type': 'Krea2', 'unet_name': 'Krea2/synthetic.safetensors'}},
            '3': {'class_type': 'KCPP_PromptStudioLoraLoader', 'inputs': {'model': ['2', 0], 'lora_type': 'Krea2', 'lora_stack_json': '[{"name":"support","strength":0.7}]'}},
            '4': {'class_type': 'KCPP_PromptStudioSampler', 'inputs': {'model': ['3', 0], 'positive': ['1', 0], 'seed': 1, 'steps': 20, 'cfg': 1}},
            '5': {'class_type': 'SaveImage', 'inputs': {'images': ['4', 0]}}}}}


class ComparisonCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.service = Service(Path(self.temp.name))
        self.job, self.run, self.output = fixture(self.service.store)
        (self.run/'manifest.json').write_text(json.dumps({'project': {'model': 'krea2', 'trigger': 'synthetic'}}))
        weights(self.output/'lora_results-run_25.safetensors', 25)
        weights(self.output/'lora_results-run.safetensors', 100)
        self.manager = Comparisons(self.service, 'http://127.0.0.1:8787')
        self.catalog = {'workflows': [supported_profile(profile(), 'krea2')]}
        self.value = {'workflow': profile()['id'], 'checkpoints': ['output/lora_results-run/lora_results-run_25.safetensors',
            'output/lora_results-run/lora_results-run.safetensors'], 'prompt': 'A synthetic blue square',
            'seeds': [42, 123], 'strength': 0.75, 'baseline': True}

    def create(self):
        with patch.object(self.manager, 'catalog', return_value=self.catalog), patch.object(self.manager, 'launch'):
            data = self.manager.start(self.job['id'], self.value)
        self.service.job_lock.release()
        return self.manager.load(data['id'])

    def test_filter_requires_create_matching_family_and_connected_targets(self):
        self.assertIsNotNone(supported_profile(profile(), 'krea2'))
        self.assertIsNone(supported_profile(profile(), 'qwen21'))
        for change in ('edit', 'orphan', 'stale', 'reference', 'seed'):
            p = profile()
            if change == 'edit': p['kind'] = 'edit'
            if change == 'orphan': p['snapshot']['output']['4']['inputs']['model'] = ['2', 0]
            if change == 'stale': p['stale'] = True
            if change == 'reference': p['snapshot']['output']['1']['class_type'] = 'LoadImage'
            if change == 'seed': p['snapshot']['output']['4']['inputs']['seed'] = ['10', 0]
            self.assertIsNone(supported_profile(p, 'krea2'), change)

    def test_frozen_graph_keeps_support_loras_and_exact_prompt_seed(self):
        data = self.create()
        before = copy.deepcopy(data['profile'])
        for cell in data['cells']:
            graph = cell_graph(data, cell, None if cell['column'] == 0 else 'registered')
            self.assertEqual(graph['1']['class_type'], 'KCPP_PromptSlot')
            self.assertEqual(graph['1']['inputs']['prompt'], self.value['prompt'])
            self.assertEqual(graph['4']['inputs']['seed'], cell['seed'])
            self.assertEqual(graph['3']['inputs']['lora_stack_json'], before['snapshot']['output']['3']['inputs']['lora_stack_json'])
            target = graph['4']['inputs']['model'][0]
            if cell['column']:
                self.assertEqual(graph[target]['inputs']['checkpoint_id'], 'registered')
                self.assertEqual(graph[target]['inputs']['model'], ['3', 0])
                self.assertEqual(graph[target]['inputs']['strength'], 0.75)
            else:
                self.assertEqual(target, '3')
        self.assertEqual(data['profile'], before)

    def test_snapshots_survive_source_retention_and_capabilities_are_one_use(self):
        data = self.create()
        self.manager.freeze(data)
        self.output.joinpath('lora_results-run_25.safetensors').unlink()
        frozen = self.manager.load(data['id'])
        digest = frozen['columns'][1]['digest']
        token = self.manager.grant(data['id'], digest)
        capability = self.manager.capability(token)
        self.assertTrue(Path(capability['path']).is_file())
        self.assertTrue(Path(capability['path']).is_relative_to(self.service.store.root))
        with self.assertRaises(KeyError): self.manager.capability(token)
        self.manager.freeze(frozen)  # Reuses frozen files, never needs deleted training output.
        with self.assertRaises(ValueError): self.manager.directory('../escape')
        with self.assertRaises(KeyError): self.manager.image(data['id'], '../../studio.sqlite3')

    def test_validation_and_training_exclusion_release_reservation(self):
        for key, val in [('seeds', [-1]), ('seeds', [True]), ('strength', float('nan')),
                         ('checkpoints', ['../outside.safetensors']), ('width', 513), ('seeds', list(range(129)))]:
            with self.subTest(key=key, val=val), patch.object(self.manager, 'catalog', return_value=self.catalog):
                invalid = {**self.value, key: val}
                with self.assertRaises(ValueError): self.manager.start(self.job['id'], invalid)
                self.assertFalse(self.service.job_lock.locked())
        self.job['status'] = 'running'
        self.service.store.put_item('jobs', self.job)
        with self.assertRaises(Conflict): self.manager.reserve()

    def test_worker_order_persistence_failure_and_offline_export(self):
        data = self.create()
        submitted = []
        reservations = []
        def remote(url, payload=None, **kwargs):
            if url.endswith('/register'): return {'checkpoint_id': 'registered'}
            if '/generation/' in url:
                reservations.append(url.rsplit('/', 1)[1])
                return {'token': 'reservation', 'active': True}
            if url.endswith('/prompt'):
                self.assertEqual(kwargs['headers'], {'X-LoRA-Studio-Reservation': 'reservation'})
                submitted.append(payload['prompt']['4']['inputs']['seed'])
                return {'prompt_id': str(len(submitted))}
            self.fail('Unexpected request')
        def completed(manifest, cell):
            if cell['seed'] == 123 and cell['column'] == 1:
                raise ValueError('Synthetic failure')
            name = cell['id'] + '-0.png'
            Image.new('RGB', (64, 32), 'blue').save(self.manager.directory(data['id'])/'images'/name)
            cell.update(status='complete', images=[name])
        self.service.job_lock.acquire()
        with patch('studio.comparisons.request', side_effect=remote), patch.object(self.manager, 'wait_cell', side_effect=completed):
            self.manager.worker(data['id'], True)
        saved = self.manager.load(data['id'])
        self.assertEqual(submitted, [42, 123, 42, 123, 42, 123])
        self.assertEqual(reservations, ['begin', 'end'])
        self.assertEqual(saved['status'], 'failed')
        self.assertEqual(sum(c['status'] == 'complete' for c in saved['cells']), 5)
        self.assertFalse(self.service.job_lock.locked())
        with Image.open(self.manager.composite(data['id'])) as img:
            self.assertEqual(img.size, (868, 660))
        reopened = Comparisons(self.service, 'http://127.0.0.1:8787')
        self.assertEqual(reopened.list(self.job['id'])[0]['cells'], saved['cells'])

    def test_lost_submit_response_is_not_blindly_retried(self):
        data = self.create()
        data['cells'][0]['status'] = 'submitting'
        self.manager.save(data)
        reopened = Comparisons(self.service, 'http://127.0.0.1:8787')
        self.assertEqual(reopened.load(data['id'])['status'], 'interrupted')
        with self.assertRaises(Conflict): reopened.action(data['id'], 'retry')

    def test_reservation_loss_leaves_cell_retryable_and_releases_local_lock(self):
        data = self.create()
        calls = []
        def remote(url, payload=None, **kwargs):
            calls.append(url.rsplit('/', 1)[1])
            if url.endswith('/begin'): return {'token': 'lease', 'active': True}
            if url.endswith('/end'): return {}
            if url.endswith('/prompt'): raise IntegrationHTTPError(409, 'Expired lease')
            self.fail('Unexpected request')
        self.service.job_lock.acquire()
        with patch('studio.comparisons.request', side_effect=remote):
            self.manager.worker(data['id'], True)
        saved = self.manager.load(data['id'])
        self.assertEqual(saved['status'], 'interrupted')
        self.assertEqual(saved['cells'][0]['status'], 'pending')
        self.assertEqual(calls, ['begin', 'prompt', 'end'])
        self.assertFalse(self.service.job_lock.locked())

    def test_stopping_while_waiting_ends_reservation_without_submitting(self):
        data = self.create()
        calls = []
        def remote(url, payload=None, **kwargs):
            action = url.rsplit('/', 1)[1]
            calls.append(action)
            if action == 'begin':
                saved = self.manager.load(data['id'])
                saved['status'] = 'stopping'
                self.manager.save(saved)
                return {'token': 'lease', 'active': False}
            if action == 'end': return {}
            self.fail('Unexpected request')
        self.service.job_lock.acquire()
        with patch('studio.comparisons.request', side_effect=remote):
            self.manager.worker(data['id'], True)
        self.assertEqual(self.manager.load(data['id'])['status'], 'stopped')
        self.assertEqual(calls, ['begin', 'end'])
        self.assertFalse(self.service.job_lock.locked())

    def test_rerun_uses_retained_snapshots_after_sources_are_deleted(self):
        data = self.create()
        self.manager.freeze(data)
        for path in self.output.glob('*.safetensors'):
            path.unlink()
        with patch.object(self.manager, 'launch'):
            copied = self.manager.action(data['id'], 'rerun')
        self.service.job_lock.release()
        repeated = self.manager.load(copied['id'])
        self.manager.freeze(repeated)
        self.assertEqual(repeated['settings'], data['settings'])
        self.assertNotEqual(repeated['id'], data['id'])
        self.assertTrue(all(c['status'] == 'pending' for c in repeated['cells']))
        self.assertTrue(all((self.manager.directory(repeated['id'])/'weights'/(c['digest']+'.safetensors')).is_file()
                            for c in repeated['columns'] if not c.get('baseline')))


class BridgeCase(unittest.TestCase):
    def test_http_capability_registers_only_frozen_file_and_detects_replacement(self):
        from server import make_server
        path = Path(__file__).resolve().parents[1]/'integrations/promptstudio/lora_studio_bridge.py'
        spec = importlib.util.spec_from_file_location('synthetic_bridge_http', path)
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as temporary:
            server = make_server(0, temporary)
            job, run, output = fixture(server.service.store)
            weights(output/'lora_results-run.safetensors', 100)
            manager = server.service.comparisons
            config = {'workflow': profile()['id'], 'checkpoints': ['output/lora_results-run/lora_results-run.safetensors'],
                      'prompt': 'Synthetic shape', 'seeds': [42]}
            with patch.object(manager, 'catalog', return_value={'workflows':[supported_profile(profile(), 'krea2')]}), patch.object(manager, 'launch'):
                data = manager.start(job['id'], config)
            server.service.job_lock.release()
            frozen = manager.load(data['id'])
            manager.freeze(frozen)
            digest = frozen['columns'][0]['digest']
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                identity = bridge.register_checkpoint(manager.origin, manager.grant(data['id'], digest))
                self.assertEqual(bridge.LoraStudioCheckpoint().load('synthetic-model', identity, 0), ('synthetic-model',))
                self.assertTrue(bridge.REGISTRY[identity]['path'].is_relative_to(Path(temporary).resolve()))
                with self.assertRaises(ValueError): bridge.register_checkpoint(manager.origin, str(bridge.REGISTRY[identity]['path']))
                with bridge.REGISTRY[identity]['path'].open('ab') as stream: stream.write(b'changed')
                with self.assertRaises(ValueError): bridge.LoraStudioCheckpoint().load('synthetic-model', identity, 0)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_only_managed_capabilities_no_direct_paths_or_remote_origins(self):
        path = Path(__file__).resolve().parents[1]/'integrations/promptstudio/lora_studio_bridge.py'
        spec = importlib.util.spec_from_file_location('synthetic_bridge', path)
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        for origin, token in [('https://example.com', 'a'*64), ('http://localhost:1/other', 'a'*64),
                              ('http://127.0.0.1:1', '../weights')]:
            with self.assertRaises(ValueError): bridge.register_checkpoint(origin, token)
        with self.assertRaises(ValueError): bridge.LoraStudioCheckpoint().load(object(), 'unregistered', 1)


if __name__ == '__main__':
    unittest.main()
