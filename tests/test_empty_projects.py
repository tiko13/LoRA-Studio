import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from studio.store import EmptyProjectExists, Store

TEST_ROOT = Path(__file__).resolve().parents[1] / 'internal-tests/artifacts/unit'


class EmptyProjectsCase(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.store = Store(self.temp.name)

    def test_existing_project_is_preserved_and_excluded_images_count(self):
        project = self.store.save_project({'name': 'Synthetic draft'}, guard_empty=True)
        with self.assertRaises(EmptyProjectExists) as error:
            self.store.save_project({'name': 'Replacement'}, guard_empty=True)
        self.assertEqual(error.exception.project, project)
        self.assertEqual(self.store.projects(), [project])
        self.store.put_item('images', {'id': 'synthetic', 'project': project['id'], 'included': False})
        second = self.store.save_project({}, guard_empty=True)
        self.assertNotEqual(second['id'], project['id'])
        self.assertEqual(len(self.store.projects()), 2)

    def test_selected_legacy_empty_project_takes_precedence(self):
        selected = self.store.save_project({})
        latest = self.store.save_project({})
        for preferred, expected in [(selected['id'], selected), (None, latest)]:
            with self.assertRaises(EmptyProjectExists) as error:
                self.store.save_project({}, guard_empty=True, selected_project_id=preferred)
            self.assertEqual(error.exception.project, expected)
        self.assertEqual(len(self.store.projects()), 2)

    def test_concurrent_creation_only_inserts_one_project(self):
        barrier = threading.Barrier(6)
        def create(_):
            barrier.wait()
            try:
                return self.store.save_project({}, guard_empty=True)['id']
            except EmptyProjectExists as error:
                return error.project['id']
        with ThreadPoolExecutor(max_workers=6) as pool:
            ids = list(pool.map(create, range(6)))
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(len(self.store.projects()), 1)

    def test_http_creation_returns_existing_project_conflict(self):
        from server import make_server
        server = make_server(0, self.temp.name)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def request(value):
            return urllib.request.urlopen(urllib.request.Request(
                f'http://127.0.0.1:{server.server_port}/api/projects',
                data=json.dumps(value).encode(), headers={'Content-Type': 'application/json'}), timeout=5)
        try:
            with request({}) as response:
                self.assertEqual(response.status, 201)
                project = json.load(response)
            with self.assertRaises(urllib.error.HTTPError) as error:
                request({'selected_project_id': project['id']})
            with error.exception as response:
                self.assertEqual(response.status, 409)
                result = json.load(response)
            self.assertEqual(result['code'], 'empty_project_exists')
            self.assertEqual(result['project'], project)
            self.assertEqual(len(self.store.projects()), 1)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
