"""Restart checks use isolated synthetic state and controlled local processes."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from server import make_server
from studio.processes import terminate_tree

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'internal-tests' / 'artifacts' / 'restart-tests'


def request(base, path, value=None, **headers):
    req = urllib.request.Request(base + path, data=json.dumps(value).encode() if value is not None else None,
                                 headers={'Content-Type': 'application/json', **headers})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=2) as response:
        return json.load(response)


class RestartHTTPCase(unittest.TestCase):
    def setUp(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=ARTIFACTS)
        self.addCleanup(temp.cleanup)
        self.server = make_server(0, temp.name, restart_enabled=True)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close)
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def close(self):
        self.server.shutdown()
        self.thread.join(5)
        self.server.server_close()

    def rejected(self, status=409, **headers):
        with self.assertRaises(urllib.error.HTTPError) as raised:
            request(self.url, '/api/restart', {}, **headers)
        self.assertEqual(raised.exception.code, status)
        raised.exception.close()
        self.assertFalse(self.server.lifecycle.restarting)

    def test_security_and_unmanaged_server(self):
        self.rejected(403, Origin='http://example.com')
        self.rejected(403, Host='example.com')
        self.server.lifecycle.enabled = False
        self.rejected()

    def test_active_job_reservation_and_pending_writes_block_restart(self):
        self.server.service.job_lock.acquire()
        self.rejected()
        self.server.service.job_lock.release()
        self.server.lifecycle.enter_write()
        try:
            self.rejected()
        finally:
            self.server.lifecycle.leave_write()

    def test_active_and_interrupted_training_block_restart(self):
        for kind, status in [('caption', 'running'), ('train', 'queued'), ('train', 'interrupted')]:
            with self.subTest(kind=kind, status=status), patch.object(
                self.server.service.store, 'items', return_value=[{'kind': kind, 'status': status}]
            ):
                self.rejected()
        self.assertFalse(self.server.service.job_lock.locked())

    def test_setup_and_comparisons_block_restart(self):
        setup = self.server.service.setup
        setup.state['install'] = {'status': 'running'}
        self.rejected()
        setup.state['install'] = {}
        setup.check_lock.acquire()
        self.rejected()
        setup.check_lock.release()
        self.server.service.comparisons.active.add('synthetic')
        self.rejected()

    def test_idle_restart_acknowledges_and_stops_accepting_writes(self):
        before = request(self.url, '/api/runtime')
        result = request(self.url, '/api/restart', {})
        self.assertEqual(result['boot_id'], before['boot_id'])
        self.assertTrue(result['restarting'])
        self.thread.join(5)
        self.assertFalse(self.thread.is_alive())
        from studio.store import Conflict
        with self.assertRaises(Conflict):
            self.server.lifecycle.enter_write()


class RestartProcessCase(unittest.TestCase):
    def test_launcher_restarts_on_same_port_and_preserves_saved_state(self):
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory:
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            base = f'http://127.0.0.1:{port}'
            flags = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}
            with (Path(directory) / 'server.log').open('w') as log:
                process = subprocess.Popen([sys.executable, str(ROOT / 'server.py'), '--port', str(port),
                                            '--data', str(Path(directory) / 'data')], cwd=ROOT,
                                           stdout=log, stderr=log, **flags)
                try:
                    def ready(previous=None):
                        deadline = time.monotonic() + 20
                        while time.monotonic() < deadline:
                            self.assertIsNone(process.poll(), 'Launcher exited unexpectedly')
                            try:
                                state = request(base, '/api/runtime')
                                if state['boot_id'] != previous and not state['restarting']:
                                    return state
                            except (OSError, ValueError):
                                pass
                            time.sleep(.1)
                        self.fail('Server did not reconnect')
                    first = ready()
                    project = request(base, '/api/projects', {'name': 'Synthetic restart fixture', 'kind': 'object'})
                    for _ in range(2):
                        request(base, '/api/restart', {})
                        first = ready(first['boot_id'])
                        saved = request(base, '/api/bootstrap')['projects']
                        self.assertEqual(saved[0]['id'], project['id'])
                finally:
                    terminate_tree(process)


if __name__ == '__main__':
    unittest.main()
