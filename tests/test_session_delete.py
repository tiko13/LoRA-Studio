import base64
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from server import make_server
from studio.comparisons import Comparisons
from studio.monitoring import write_json
from studio.service import Service
from studio.store import Conflict
from tests.fixtures import fixture


TEST_ROOT = Path(__file__).resolve().parents[1] / 'internal-tests/artifacts/unit'
TEST_ROOT.mkdir(parents=True, exist_ok=True)


class SessionDeleteCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.service = Service(self.temp.name)
        self.store = self.service.store
        self.project = self.store.save_project({'name': 'Synthetic session'})
        self.other = self.store.save_project({'name': 'Other session'})
        self.pid = self.project['id']
        self.image = self.service.import_image(self.pid, {'name': 'shape.png', 'data': base64.b64encode(fixture()).decode()})['image']
        self.job = {'id': 'synthetic-job', 'project': self.pid, 'kind': 'train', 'status': 'complete'}
        self.store.put_item('jobs', self.job)
        self.paths = [self.service.image_path(self.image)]
        for folder in ('runs', 'analysis', 'caption-jobs'):
            path = self.store.root / folder / self.job['id'] / 'synthetic.dat'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'synthetic artifact')
            self.paths.append(path)
        self.cache = self.store.root / 'sample-prompts' / (self.pid + '.json')
        self.cache.parent.mkdir()
        write_json(self.cache, {'prompts': []})
        self.paths.append(self.cache)
        self.service.comparisons = Comparisons(self.service, 'http://127.0.0.1:9999')
        self.comparison = {'id': 'a' * 32, 'job': self.job['id'], 'status': 'complete', 'created': 1}
        self.comparison_path = self.service.comparisons.root / self.comparison['id'] / 'comparison.json'
        self.comparison_path.parent.mkdir()
        write_json(self.comparison_path, self.comparison)
        self.paths.append(self.comparison_path)
        self.shared = self.store.root / 'models' / 'shared.dat'
        self.shared.parent.mkdir()
        self.shared.write_bytes(b'shared model')
        self.source = self.store.root / 'synthetic-source.png'
        self.source.write_bytes(fixture())

    def delete(self, files=True, revision=None):
        return self.service.delete_project(self.pid, {'revision': revision or self.project['revision'], 'delete_files': files})

    def test_session_only_keeps_every_file_but_removes_records_after_restart(self):
        result = self.delete(False)
        self.assertEqual(result['image_ids'], [self.image['id']])
        self.assertTrue(all(p.exists() for p in self.paths))
        reopened = Service(self.temp.name)
        self.assertEqual([p['id'] for p in reopened.store.projects()], [self.other['id']])
        self.assertEqual(reopened.store.items('jobs', self.pid), [])
        self.assertEqual(reopened.store.items('images', self.pid), [])
        with self.assertRaises(KeyError):
            self.store.save_project(self.project, self.pid)

    def test_delete_all_removes_owned_artifacts_and_preserves_other_session_and_shared_files(self):
        clone = self.service.copy_project(self.pid)
        clone_image = self.store.items('images', clone['id'])[0]
        self.store.lock_preparation(self.pid, self.job['id'], started=True)
        result = self.delete(revision=self.store.project(self.pid)['revision'])
        self.assertFalse(result['cleanup_pending'])
        self.assertFalse(any(p.exists() for p in self.paths))
        self.assertTrue(self.source.exists())
        self.assertTrue(self.shared.exists())
        self.assertTrue(self.service.image_path(clone_image).exists())
        self.assertEqual(self.store.project(self.other['id']), self.other)
        self.assertEqual(self.store.project(clone['id']), clone)

    def test_active_and_interrupted_work_blocks_both_choices(self):
        for status in ('queued', 'running', 'stopping', 'interrupted'):
            self.store.put_item('jobs', {**self.job, 'status': status})
            for files in (False, True):
                with self.assertRaises(Conflict):
                    self.delete(files)
        self.store.put_item('jobs', self.job)
        with patch('studio.project_lifecycle.running', return_value=True):
            self.store.put_item('jobs', {**self.job, 'pid': 123})
            with self.assertRaises(Conflict):
                self.delete()
        self.assertTrue(all(p.exists() for p in self.paths))

    def test_comparisons_and_reservations_block_deletion(self):
        write_json(self.comparison_path, {**self.comparison, 'status': 'interrupted'})
        with self.assertRaises(Conflict):
            self.delete(False)
        write_json(self.comparison_path, self.comparison)
        self.service.job_lock.acquire()
        try:
            with self.assertRaises(Conflict):
                self.delete()
        finally:
            self.service.job_lock.release()

    def test_interrupted_caption_without_live_process_can_be_deleted(self):
        self.store.put_item('jobs', {**self.job, 'kind': 'caption', 'status': 'interrupted'})
        self.assertTrue(self.delete()['deleted'])

    def test_revision_and_explicit_choice_are_required(self):
        self.store.save_project(self.project, self.pid)
        with self.assertRaises(Conflict):
            self.delete()
        for value in ({}, {'revision': 2, 'delete_files': 'false'}, {'revision': True, 'delete_files': True}):
            with self.assertRaises(ValueError):
                self.service.delete_project(self.pid, value)
        self.assertTrue(all(p.exists() for p in self.paths))

    def test_failed_move_restores_records_and_all_files(self):
        replace = Path.replace
        count = 0
        def fail_second(path, destination):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError('Synthetic move failure')
            return replace(path, destination)
        with patch.object(Path, 'replace', fail_second):
            with self.assertRaises(OSError):
                self.delete()
        self.assertEqual(self.store.project(self.pid), self.project)
        self.assertTrue(all(p.exists() for p in self.paths))

    def test_cleanup_failure_reports_retained_files(self):
        with patch('studio.project_lifecycle.shutil.rmtree', side_effect=OSError('Synthetic locked file')):
            result = self.delete()
        self.assertTrue(result['deleted'])
        self.assertTrue(result['cleanup_pending'])
        self.assertTrue(any((self.store.root / 'session-trash').iterdir()))

    def test_invalid_job_path_is_rejected_before_any_removal(self):
        self.store.put_item('jobs', {**self.job, 'id': '../models'})
        with self.assertRaises(ValueError):
            self.delete()
        self.assertTrue(self.shared.exists())
        self.assertTrue(all(p.exists() for p in self.paths))

    def test_nested_link_is_rejected_before_any_removal(self):
        link = self.paths[1].parent / 'linked.dat'
        try:
            link.symlink_to(self.shared)
        except OSError:
            self.skipTest('Creating symlinks requires platform permission')
        with self.assertRaises(ValueError):
            self.delete()
        self.assertTrue(self.shared.exists())
        self.assertTrue(all(p.exists() for p in self.paths))

    def test_http_route_deletes_locked_session_and_rejects_cross_origin(self):
        self.store.lock_preparation(self.pid, self.job['id'], started=True)
        server = make_server(0, self.store.root)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}/api/projects/{self.pid}/delete'
            data = json.dumps({'revision': self.store.project(self.pid)['revision'], 'delete_files': False}).encode()
            req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json', 'Origin': 'https://invalid.example'})
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(req)
            self.assertEqual(error.exception.code, 403)
            error.exception.close()
            req.remove_header('Origin')
            with urllib.request.urlopen(req) as response:
                self.assertTrue(json.load(response)['deleted'])
            self.assertTrue(all(p.exists() for p in self.paths))
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
