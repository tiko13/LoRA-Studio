import json
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from studio.service import Service
from studio import processes, recovery
from studio.monitoring import control_database, write_json
from tests.test_results import fixture, weights, TEST_ROOT


class RecoveryCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.service = Service(Path(self.temp.name))
        self.job, self.root, self.output = fixture(self.service.store)
        self.job.update(status='running', pid=12345, attempt=1, phase='Training')
        self.identity = {'executable': str(Path(sys.executable).resolve()), 'creation_marker': 'synthetic:123'}
        self.job['process_identity'] = self.identity
        self.service.store.put_item('jobs', self.job)
        control_database(self.root, self.job['id'])
        config = json.loads((self.root / 'config.json').read_text())
        config['config']['process'][0]['training_folder'] = str(self.root / 'output')
        write_json(self.root / 'config.json', config)

    def restart(self):
        with patch.object(threading.Thread, 'start'):
            return Service(self.service.store.root)

    def native(self, status):
        with closing(sqlite3.connect(self.root / 'toolkit-control.sqlite3')) as db, db:
            db.execute('UPDATE Job SET status=?', (status,))

    def test_repeated_restart_reconnects_and_blocks_duplicate_work(self):
        with patch('studio.processes.identity', return_value=self.identity):
            first = self.restart()
            second = self.restart()
            self.assertEqual(first.reconnected, {self.job['id']})
            self.assertEqual(second.reconnected, first.reconnected)
            job = second.store.item('jobs', self.job['id'])
            self.assertTrue(job['reconnected'])
            self.assertEqual(job['status'], 'running')
            with self.assertRaisesRegex(ValueError, 'Another job'):
                second.new_job(self.job['project'], 'caption', lambda *_: None)
            second.stop(self.job['id'])
            with closing(sqlite3.connect(self.root / 'toolkit-control.sqlite3')) as db:
                self.assertEqual(db.execute('SELECT stop FROM Job').fetchone()[0], 1)
        with patch('studio.processes.identity', return_value=None):
            recovery.refresh(second)
        self.assertEqual(second.store.item('jobs', self.job['id'])['status'], 'cancelled')

    def test_offline_completion_requires_backend_and_complete_final_weights(self):
        self.native('completed')
        final = self.output / 'lora_results-run.safetensors'
        weights(final, 100)
        with patch('studio.processes.identity', return_value=None):
            recovered = self.restart()
        self.assertEqual(recovered.store.item('jobs', self.job['id'])['status'], 'complete')
        self.assertFalse(recovered.reconnected)
        for native, step, truncate in [('running', 100, False), ('error', 100, False),
                                       ('completed', 75, False), ('completed', 100, True)]:
            with self.subTest(native=native, step=step, truncate=truncate):
                self.service.store.put_item('jobs', self.job)
                self.native(native)
                weights(final, step)
                if truncate:
                    final.write_bytes(final.read_bytes()[:-1])
                with patch('studio.processes.identity', return_value=None):
                    recovered = self.restart()
                self.assertEqual(recovered.store.item('jobs', self.job['id'])['status'], 'failed')

    def test_pid_reuse_never_controls_replacement(self):
        with patch('studio.processes.identity', return_value=self.identity):
            recovered = self.restart()
        replacement = {**self.identity, 'creation_marker': 'synthetic:456'}
        with patch('studio.processes.identity', return_value=replacement), patch('studio.recovery.write_json') as flag:
            with self.assertRaisesRegex(ValueError, 'already exited'):
                recovered.stop(self.job['id'])
            flag.assert_not_called()
        self.assertFalse(recovered.reconnected)

    def test_permission_failure_remains_interrupted_without_claiming_completion(self):
        with patch('studio.processes.identity', side_effect=ValueError('Cannot verify identity')):
            recovered = self.restart()
        self.assertEqual(recovered.store.item('jobs', self.job['id'])['status'], 'interrupted')
        self.assertFalse(recovered.reconnected)

    def test_legacy_command_must_match_exact_attempt_and_stable_identity(self):
        self.job.pop('process_identity')
        self.service.store.put_item('jobs', self.job)
        args = [sys.executable, 'run.py', str(self.root / 'config.json')]
        with patch('studio.processes.identity', return_value=self.identity), patch('studio.processes.command_args', return_value=args):
            recovered = self.restart()
        self.assertEqual(recovered.store.item('jobs', self.job['id'])['process_identity'], self.identity)
        self.service.store.put_item('jobs', self.job)
        args[-1] = str(self.root / 'other.json')
        with patch('studio.processes.identity', return_value=self.identity), patch('studio.processes.command_args', return_value=args):
            recovered = self.restart()
        self.assertEqual(recovered.store.item('jobs', self.job['id'])['status'], 'interrupted')

    def test_pause_request_survives_restart_and_waits_for_checkpoint(self):
        self.native('running')
        write_json(self.root / 'monitor-control.json', {'action': 'pause', 'checkpoint_confirmed': 25, 'stop_sent': True})
        with patch('studio.processes.identity', return_value=self.identity):
            recovered = self.restart()
        self.assertEqual(recovered.store.item('jobs', self.job['id'])['status'], 'running')
        with patch('studio.processes.identity', return_value=None):
            recovery.refresh(recovered)
        job = recovered.store.item('jobs', self.job['id'])
        self.assertEqual(job['status'], 'paused')
        self.assertEqual(job['paused_step'], 25)

    def test_real_process_identity_survives_service_restart(self):
        # A neutral, controlled child process; no local backend, models or GPU.
        worker = self.root / 'run.py'
        worker.write_text('import time\ntime.sleep(60)\n')
        flags = {'creationflags': subprocess.CREATE_NO_WINDOW} if sys.platform == 'win32' else {'start_new_session': True}
        process = subprocess.Popen([sys.executable, str(worker), str(self.root / 'config.json')], **flags)
        try:
            self.job['pid'] = process.pid
            self.job['process_identity'] = processes.identity(process.pid)
            self.assertIsNotNone(self.job['process_identity'])
            self.service.store.put_item('jobs', self.job)
            recovered = self.restart()
            self.assertIn(self.job['id'], recovered.reconnected)
            self.assertIsNone(process.poll())
            process.terminate()
            process.wait(timeout=10)
            recovery.refresh(recovered)
            self.assertEqual(recovered.store.item('jobs', self.job['id'])['status'], 'failed')
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


if __name__ == '__main__':
    unittest.main()
