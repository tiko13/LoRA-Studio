import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.parse
import urllib.request

from server import make_server
from studio.results import artifact_path, results, open_location
from studio.store import Store

TEST_ROOT = Path(__file__).resolve().parents[1] / 'internal-tests/artifacts/unit'
TEST_ROOT.mkdir(parents=True, exist_ok=True)


def fixture(store):
    project = store.save_project({'name': 'Synthetic results'})
    job = {'id': 'results-run', 'project': project['id'], 'kind': 'train',
           'status': 'complete', 'created': time.time(), 'messages': []}
    store.put_item('jobs', job)
    root = store.root / 'runs' / job['id']
    output = root / 'output/lora_results-run'
    (output / 'samples').mkdir(parents=True)
    (root / 'config.json').write_text(json.dumps({'config': {'name': 'lora_results-run', 'process': [{
        'train': {'steps': 100}, 'sample': {'samples': [{'prompt': 'A synthetic blue square'}]}}]}}))
    return job, root, output


def weights(path, step=None):
    header = {'tensor': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}
    if step is not None:
        header['__metadata__'] = {'training_info': json.dumps({'step': step})}
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, 'little') + encoded + b'\0' * 4)


class ResultsCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name))
        self.job, self.root, self.output = fixture(self.store)

    def test_exact_steps_final_metadata_baseline_and_retention(self):
        weights(self.output / 'lora_results-run_000000025.safetensors')
        weights(self.output / 'lora_results-run.safetensors', 100)
        for step in (0, 25, 50, 100):
            (self.output / 'samples' / f'123__{step:09d}_0.png').write_bytes(b'synthetic')
        for hidden in ('.thumbs', '.tmp'):
            (self.output / 'samples' / hidden).mkdir()
            (self.output / 'samples' / hidden / '123__000000100_0.png').write_bytes(b'derived preview')
        # Model-like files outside output must not be offered as results.
        (self.root / 'private.safetensors').write_bytes(b'not output')
        data = results(self.store, self.job['id'])
        self.assertEqual([c['step'] for c in data['checkpoints']], [100, 25])
        self.assertTrue(data['checkpoints'][0]['final'])
        self.assertTrue(all(c['ready'] for c in data['checkpoints']))
        self.assertEqual([p['step'] for p in data['other_previews']], [0, 50])
        self.assertEqual(data['checkpoints'][1]['previews'][0]['prompt'], 'A synthetic blue square')
        self.assertEqual(len(data['checkpoints'][0]['previews']), 1)
        # Editing current project settings must not affect frozen results.
        p = self.store.project(self.job['project'])
        p['training']['steps'] = 900
        self.store.save_project(p, p['id'])
        self.assertEqual(results(self.store, self.job['id'])['total_steps'], 100)

    def test_incomplete_unknown_and_noncompleted_final(self):
        self.job['status'] = 'failed'
        self.store.put_item('jobs', self.job)
        weights(self.output / 'lora_results-run.safetensors', 73)
        (self.output / 'broken.safetensors').write_bytes(b'partial')
        (self.output / 'samples/unlabelled.webp').write_bytes(b'synthetic')
        data = results(self.store, self.job['id'])
        self.assertEqual(data['checkpoints'][0]['step'], 73)
        self.assertIsNone(data['checkpoints'][1]['step'])
        self.assertFalse(data['checkpoints'][1]['ready'])
        self.assertEqual(len(data['other_previews']), 1)
        self.assertEqual(data['job']['status'], 'failed')

    def test_paths_and_open_location(self):
        path = self.output / 'lora_results-run.safetensors'
        weights(path)
        name = path.relative_to(self.root).as_posix()
        with patch('studio.results.sys.platform', 'win32'), patch('studio.results.os.startfile', create=True) as opened:
            self.assertTrue(open_location(self.store, self.job['id'], name)['opened'])
            opened.assert_called_once_with(str(self.output.resolve()))
            for invalid in ('../private.safetensors', 'output/../../private.safetensors',
                            'config.json', 'output/optimizer.pt', str(path.resolve()), 'output\\bad.safetensors'):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    open_location(self.store, self.job['id'], invalid)
            self.assertEqual(opened.call_count, 1)
        outside = self.root / 'private.safetensors'
        weights(outside)
        link = self.output / 'escape.safetensors'
        try:
            link.symlink_to(outside)
        except OSError:
            return  # Windows may not grant symlink privileges.
        with self.assertRaises(ValueError):
            artifact_path(self.root.resolve(), link.relative_to(self.root).as_posix())
        self.assertEqual(len(results(self.store, self.job['id'])['checkpoints']), 1)

    def test_no_output_and_non_training_job(self):
        self.job['id'] = 'no-files'
        self.store.put_item('jobs', self.job)
        data = results(self.store, 'no-files')
        self.assertEqual(data['checkpoints'], [])
        self.assertFalse(data['output_available'])
        self.job['kind'] = 'export'
        self.store.put_item('jobs', self.job)
        with self.assertRaises(ValueError):
            results(self.store, 'no-files')


class ResultsHTTPCase(unittest.TestCase):
    def test_download_results_and_folder_origin_guard(self):
        with tempfile.TemporaryDirectory(dir=TEST_ROOT) as directory:
            server = make_server(0, Path(directory))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                job, root, output = fixture(server.service.store)
                path = output / 'lora_results-run.safetensors'
                weights(path, 100)
                url = f'http://127.0.0.1:{server.server_port}/api/jobs/{job["id"]}'
                with urllib.request.urlopen(url + '/results') as response:
                    self.assertEqual(len(json.load(response)['checkpoints']), 1)
                with urllib.request.urlopen(url + '/result-file?path=' + urllib.parse.quote(path.relative_to(root).as_posix())) as response:
                    self.assertIn('attachment;', response.headers['Content-Disposition'])
                    self.assertEqual(response.read(), path.read_bytes())
                with patch('studio.results.os.startfile', create=True) as opened, patch('studio.results.sys.platform', 'win32'):
                    request = urllib.request.Request(url + '/open-location', data=b'{}', headers={
                        'Content-Type': 'application/json', 'Origin': 'https://untrusted.invalid'})
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(request)
                    self.assertEqual(caught.exception.code, 403)
                    caught.exception.close()
                    opened.assert_not_called()
                    request.remove_header('Origin')
                    with urllib.request.urlopen(request) as response:
                        self.assertTrue(json.load(response)['opened'])
                    opened.assert_called_once_with(str((root / 'output').resolve()))
                for name in ('../config.json', 'output/../config.json', 'output/missing.safetensors'):
                    with self.assertRaises(urllib.error.HTTPError) as caught:
                        urllib.request.urlopen(url + '/result-file?path=' + urllib.parse.quote(name))
                    self.assertIn(caught.exception.code, (400, 404))
                    caught.exception.close()
                def post(action, value, origin=None):
                    headers = {'Content-Type': 'application/json'}
                    if origin:
                        headers['Origin'] = origin
                    request = urllib.request.Request(url + '/' + action, data=json.dumps(value).encode(), headers=headers)
                    with urllib.request.urlopen(request) as response:
                        return json.load(response)
                name = path.relative_to(root).as_posix()
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    post('checkpoint-delete-plan', {'path': name}, 'https://untrusted.invalid')
                self.assertEqual(caught.exception.code, 403)
                caught.exception.close()
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    post('delete-checkpoint', {'path': name})
                self.assertEqual(caught.exception.code, 400)
                caught.exception.close()
                plan = post('checkpoint-delete-plan', {'path': name})
                self.assertTrue(path.exists())
                self.assertTrue(post('delete-checkpoint', {'path': name, 'token': plan['token']})['deleted'])
                self.assertFalse(path.exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
