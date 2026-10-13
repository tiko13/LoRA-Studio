import base64
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from studio.domain import intent_digest
from studio.monitoring import parse_log, write_json
from studio.preparation import plan
from studio.sample_prompts import signature
from studio.service import Service
from studio.store import Conflict, PreparationLocked
from tests.fixtures import fixture

TEST_ROOT = Path(__file__).resolve().parents[1] / 'internal-tests/artifacts/unit'
TEST_ROOT.mkdir(parents=True, exist_ok=True)


class LifecycleCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.service = Service(self.temp.name)
        self.store = self.service.store
        self.project = self.store.save_project({'name': 'Synthetic', 'description': 'A shape', 'trigger': 'shape',
                                               'sample_prompts': 'shape on a table', 'training': {'steps': 100}})
        self.image = self.service.import_image(self.project['id'], {
            'name': 'synthetic.png', 'data': base64.b64encode(fixture()).decode()})['image']
        self.image.update(caption='shape on a table', manual=True, reviewed=True,
                          intent_digest=intent_digest(self.project), history=[{'caption': 'earlier synthetic caption'}],
                          observations='A synthetic shape.', preparation={'mode': 'pad', 'x': .3, 'y': .7})
        self.image['caption_preparation_digest'] = plan(self.project, self.image)['digest']
        self.store.put_item('images', self.image)
        self.job = {'id': 'synthetic-run', 'project': self.project['id'], 'kind': 'train',
                    'status': 'running', 'created': time.time(), 'messages': []}
        self.store.put_item('jobs', self.job)
        self.root = self.store.root / 'runs' / self.job['id']
        self.root.mkdir(parents=True)
        write_json(self.root / 'config.json', {'config': {'process': [{'train': {'steps': 100}}]}})

    def loss(self):
        (self.root / 'training.log').write_text('run: 0/100 [00:02<?, 2.00s/it, lr: 1e-4 loss: 0.4]\r')

    def test_first_loss_at_zero_locks_and_survives_every_terminal_state_and_restart(self):
        (self.root / 'training.log').write_text('Caching latents: 1/100 [00:02<?, 2.00s/it]\n')
        self.service.sync_training_lock(self.job)
        self.assertNotIn('preparation_lock', self.store.project(self.project['id']))
        self.loss()
        self.service.sync_training_lock(self.job)
        locked = self.store.project(self.project['id'])
        self.assertTrue(locked['preparation_lock']['started'])
        for status in ('paused', 'cancelled', 'failed', 'complete', 'interrupted'):
            self.job['status'] = status
            self.store.put_item('jobs', self.job)
            self.service.sync_training_lock(self.job)
            self.store.release_preparation(self.project['id'], self.job['id'])
            self.assertEqual(self.store.project(self.project['id']), locked)
        (self.root / 'training.log').unlink()
        reopened = Service(self.temp.name)
        self.assertEqual(reopened.store.project(self.project['id']), locked)

    def test_nonfinite_first_loss_and_baseline_do_not_confuse_start_detection(self):
        self.assertFalse(parse_log('Generating baseline\nimages: 1/100 [00:02<?, 2.00s/it]', 100)['training_started'])
        self.assertTrue(parse_log('run: 0/100 [00:02<?, 2.00s/it, loss: nan]', 100)['training_started'])

    def test_startup_discovers_older_finished_training(self):
        self.loss()
        self.job['status'] = 'failed'
        self.store.put_item('jobs', self.job)
        reopened = Service(self.temp.name)
        self.assertTrue(reopened.store.project(self.project['id'])['preparation_lock']['started'])

    def test_frozen_handoff_blocks_edits_but_failed_start_releases(self):
        self.store.lock_preparation(self.project['id'], self.job['id'], expected=self.project, images=[self.image])
        with self.assertRaises(PreparationLocked):
            self.store.save_project(self.project, self.project['id'])
        self.job['status'] = 'failed'
        self.service.sync_training_lock(self.job)
        current = self.store.project(self.project['id'])
        self.assertNotIn('preparation_lock', current)
        self.store.save_project({**current, 'description': 'Another shape'}, current['id'])

    def test_training_worker_releases_startup_failure_but_keeps_first_step_failure_locked(self):
        from studio.sample_prompts import effective
        self.job['status'] = 'cancelled'
        self.store.put_item('jobs', self.job)
        self.service.save_settings({'integration': 'standalone'})
        for had_step in (False, True):
            def launch(job, stop, root, settings, project):
                self.assertFalse(self.store.project(project['id'])['preparation_lock']['started'])
                if had_step:
                    (root / 'training.log').write_text('run: 0/100 [00:02<?, 2.00s/it, loss: 0.4]')
                raise RuntimeError('Synthetic trainer failure')
            with patch('studio.service.preflight', return_value={'errors': [], 'warnings': []}), \
                 patch.object(self.service, 'resolve_preview_suite', side_effect=lambda p, i, s: effective(p, i)), \
                 patch.object(self.service, 'launch_training', side_effect=launch):
                job = self.service.prepare(self.project['id'], train=True)
                deadline = time.time() + 5
                while time.time() < deadline:
                    result = self.store.item('jobs', job['id'])
                    if result['status'] not in {'queued', 'running'}:
                        break
                    time.sleep(.01)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['error'], 'Synthetic trainer failure')
            self.assertEqual(bool(self.store.project(self.project['id']).get('preparation_lock', {}).get('started')), had_step)
            if not had_step:
                self.assertNotIn('preparation_lock', self.store.project(self.project['id']))

    def test_handoff_rejects_project_or_dataset_changes(self):
        self.image['revision'] += 1
        self.store.put_item('images', self.image)
        with self.assertRaises(Conflict):
            self.store.lock_preparation(self.project['id'], self.job['id'], expected=self.project,
                                        images=[{**self.image, 'revision': 1}])
        self.store.save_project(self.project, self.project['id'])
        with self.assertRaises(Conflict):
            self.store.lock_preparation(self.project['id'], self.job['id'], expected=self.project, images=[self.image])
        self.assertNotIn('preparation_lock', self.store.project(self.project['id']))

    def test_all_preparation_write_paths_reject_locked_project(self):
        self.store.lock_preparation(self.project['id'], self.job['id'], started=True)
        value = self.store.project(self.project['id'])
        value.pop('preparation_lock')  # A client cannot clear the server-owned lock.
        calls = [lambda: self.store.save_project(value, value['id']),
                 lambda: self.store.put_item('images', self.image),
                 lambda: self.service.update_image(self.image['id'], {'revision': 1, 'caption': 'edited'}),
                 lambda: self.service.import_image(value['id'], {}),
                 lambda: self.service.new_job(value['id'], 'caption', lambda *_: None)]
        for call in calls:
            with self.assertRaises(PreparationLocked):
                call()
        self.assertEqual(self.store.item('images', self.image['id']), self.image)

    def test_copy_preserves_all_preparation_and_owns_independent_files(self):
        excluded = {**copy.deepcopy(self.image), 'id': 'excluded', 'included': False, 'revision': 3}
        self.service.image_path(excluded).write_bytes(self.service.image_path(self.image).read_bytes())
        self.store.put_item('images', excluded)
        cache = self.store.root / 'sample-prompts' / (self.project['id'] + '.json')
        cache.parent.mkdir()
        write_json(cache, {'signature': signature(self.project, [self.image, excluded]),
                           'prompts': [{'text': 'shape on a table'}], 'history': [{'note': 'older suite'}]})
        self.store.lock_preparation(self.project['id'], self.job['id'], started=True)
        source = self.store.project(self.project['id'])
        clone = self.service.copy_project(source['id'])
        self.assertNotEqual(clone['id'], source['id'])
        self.assertNotIn('preparation_lock', clone)
        self.assertEqual(clone['revision'], 1)
        for key in set(source) - {'id', 'revision', 'updated', 'name', 'preparation_lock'}:
            self.assertEqual(clone[key], source[key], key)
        self.assertEqual(self.store.items('jobs', clone['id']), [])
        self.assertFalse((self.store.root / 'runs' / clone['id']).exists())
        copied = self.store.items('images', clone['id'])
        self.assertEqual(len(copied), 2)
        for original, item in zip([self.image, excluded], copied):
            for key in set(original) - {'id', 'project', 'revision'}:
                self.assertEqual(item[key], original[key], key)
            self.assertNotEqual(self.service.image_path(item), self.service.image_path(original))
            self.assertEqual(self.service.image_path(item).read_bytes(), self.service.image_path(original).read_bytes())
        new_cache = json.loads((cache.parent / (clone['id'] + '.json')).read_text())
        self.assertEqual(new_cache['signature'], signature(clone, copied))
        self.assertEqual(new_cache['history'], [{'note': 'older suite'}])
        self.store.save_project({**clone, 'description': 'A new target'}, clone['id'])
        self.service.update_image(copied[0]['id'], {'revision': 1, 'caption': 'new caption'})
        self.service.image_path(copied[0]).write_bytes(b'independent synthetic file')
        self.assertEqual(self.store.project(source['id']), source)
        self.assertEqual(self.store.item('images', self.image['id']), self.image)
        self.assertNotEqual(self.service.image_path(self.image).read_bytes(), b'independent synthetic file')

    def test_failed_copy_leaves_no_partial_project_or_files(self):
        before = self.store.projects()
        with patch('studio.project_lifecycle.shutil.copyfile', side_effect=OSError('synthetic failure')):
            with self.assertRaises(OSError):
                self.service.copy_project(self.project['id'])
        self.assertEqual(self.store.projects(), before)
        self.assertEqual(len(list((self.store.root / 'images').iterdir())), 1)

    def test_new_projects_cannot_inherit_client_supplied_lock(self):
        clone = self.store.save_project({**self.project, 'preparation_lock': {'started': True}})
        self.assertNotIn('preparation_lock', clone)

    def test_http_blocks_all_preparation_routes_but_allows_copy_and_read(self):
        import threading
        import urllib.error
        import urllib.request
        from server import make_server

        self.loss()
        server = make_server(0, self.store.root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}/api'
        def request(path, method='GET', data=None):
            req = urllib.request.Request(url + path, method=method,
                data=json.dumps(data or {}).encode() if method != 'GET' else None,
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.load(response)
        try:
            pid = self.project['id']
            for endpoint in ['images', 'caption', 'detect', 'objects', 'verify', 'train', 'prepare', 'export', 'preset', 'sample-prompts']:
                with self.assertRaises(urllib.error.HTTPError) as error:
                    request(f'/projects/{pid}/{endpoint}', 'POST')
                self.assertEqual(error.exception.code, 409, endpoint)
                error.exception.close()
            for path, data in [(f'/projects/{pid}', self.project), (f'/images/{self.image["id"]}', {'revision': 1})]:
                with self.assertRaises(urllib.error.HTTPError) as error:
                    request(path, 'PUT', data)
                self.assertEqual(error.exception.code, 409)
                error.exception.close()
            self.assertTrue(request(f'/projects/{pid}')['project']['preparation_lock']['started'])
            clone = request(f'/projects/{pid}/copy', 'POST')
            copied = request(f'/projects/{clone["id"]}')
            self.assertEqual(len(copied['images']), 1)
            self.assertEqual(copied['jobs'], [])
            self.assertNotIn('preparation_lock', request(f'/projects/{clone["id"]}', 'PUT', clone))
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
