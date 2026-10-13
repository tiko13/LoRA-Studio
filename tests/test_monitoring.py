import json
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import zipfile

from studio.domain import normalize_project
from studio.monitoring import TrainingMonitor, parse_log, resume_checkpoint, control_database, native_flag, write_json, run_config
from studio.service import Service
from studio.trainer import config_for

TEST_ROOT = Path(__file__).resolve().parents[1] / 'internal-tests/artifacts/unit'
TEST_ROOT.mkdir(parents=True, exist_ok=True)


class MonitoringCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.service = Service(Path(self.temp.name))
        self.project = self.service.store.save_project({'name': 'Synthetic', 'training': {'steps': 100}})
        self.job = {'id': 'test-run', 'project': self.project['id'], 'kind': 'train', 'status': 'running', 'created': time.time(), 'phase': 'Training', 'pid': 12345}
        self.service.store.put_item('jobs', self.job)
        self.root = self.service.store.root / 'runs/test-run'
        self.output = self.root / 'output/lora_test-run'
        self.output.mkdir(parents=True)
        write_json(self.root / 'config.json', config_for(self.project, 'dataset', str(self.root / 'output'), 'test-run'))
        write_json(self.root / 'manifest.json', {'project': self.project, 'images': []})
        self.log = self.root / 'training.log'
        self.log.write_text('')
        with patch.object(threading.Thread, 'start'):
            self.monitor = TrainingMonitor(self.service)
        self.service._monitor = self.monitor

    def checkpoint(self, step=25, complete=True):
        path = self.output / f'lora_test-run_{step:09d}.safetensors'
        header = json.dumps({'__metadata__': {'training_info': json.dumps({'step': step})},
                             'tensor': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
        path.write_bytes(len(header).to_bytes(8, 'little') + header + b'\0' * 4)
        with self.log.open('a') as f:
            f.write(f'\nSaving at step {step}\nSaved checkpoint to {path}\n')
        if complete:
            with zipfile.ZipFile(self.output / 'optimizer.pt', 'w') as z:
                z.writestr('optimizer/data.pkl', b'synthetic fixture; not unpickled')
            with self.log.open('a') as f:
                f.write(f'Saved optimizer to {self.output / "optimizer.pt"}\n')
        return path

    def test_repaints_ansi_nonfinite_and_nontraining_bars(self):
        result = parse_log('\x1b[32mrun: 1/100 [00:02<03:18, 2.00s/it, lr: 1e-4 loss: 0.5]\r'
                           'run: 1/100 [00:02<03:18, 2.00s/it, lr: 1e-4 loss: 0.4]\r'
                           'images: 3/30 [00:03<00:27, 1.0s/it]\n'
                           'run: 2/100 [00:04<03:16, 0.50it/s, lr: 1e-4 loss: nan]', 100)
        self.assertEqual(len(result['points']), 2)
        self.assertEqual(result['points'][0]['loss'], .4)
        self.assertEqual(result['points'][1]['seconds_per_step'], 2)
        self.assertNotIn('loss', result['points'][1])
        self.assertEqual(result['invalid_values'], 1)

    def test_pause_waits_for_optimizer_and_owner_exit(self):
        self.service.cancel['test-run'] = threading.Event()
        self.monitor.action('test-run', 'pause')
        self.checkpoint(25, complete=False)
        with patch('studio.monitoring.gpu_metrics', return_value={'gpus': []}):
            self.monitor.tick()
            self.assertFalse(self.service.cancel['test-run'].is_set())
            self.checkpoint(25)
            self.monitor.tick()
            self.assertTrue(self.service.cancel['test-run'].is_set())
            self.assertEqual(self.monitor.snapshot('test-run')['status'], 'pausing')
            with self.assertRaises(ValueError):
                self.monitor.action('test-run', 'continue')
            self.job['status'] = 'cancelled'
            self.service.store.put_item('jobs', self.job)
            self.monitor.tick()
        snap = self.monitor.snapshot('test-run')
        self.assertEqual(snap['status'], 'paused')
        self.assertTrue(snap['capabilities']['resume'])
        self.assertEqual(snap['job']['paused_step'], 25)

    def test_partial_next_checkpoint_blocks_resume(self):
        self.checkpoint(25)
        self.checkpoint(50, complete=False)
        with self.assertRaises(ValueError):
            resume_checkpoint(self.root, parse_log(self.log.read_text(), 100), 100, validate=True)

    def test_checkpoint_integrity_and_completed_run(self):
        path = self.checkpoint(25)
        parsed = parse_log(self.log.read_text(), 100)
        self.assertEqual(resume_checkpoint(self.root, parsed, 100, validate=True)[0], path)
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            resume_checkpoint(self.root, parsed, 100, validate=True)
        self.checkpoint(100)
        with self.assertRaises(ValueError):
            resume_checkpoint(self.root, parse_log(self.log.read_text(), 100), 100)

    def test_native_save_sample_and_withdraw_pause(self):
        db = control_database(self.root, 'test-run')
        self.monitor.action('test-run', 'sample')
        self.monitor.action('test-run', 'pause')
        with closing(sqlite3.connect(db)) as conn:
            self.assertEqual(conn.execute('SELECT save_now,sample_now FROM Job').fetchone(), (1, 1))
        self.monitor.action('test-run', 'continue')
        self.assertEqual(self.monitor.snapshot('test-run')['status'], 'running')
        with self.assertRaises(ValueError):
            native_flag(self.root, 'test-run', 'stop')

    def test_resume_uses_frozen_project_and_preserves_history(self):
        self.checkpoint(25)
        self.job.update(status='paused', finished=time.time(), pid=None)
        self.service.store.put_item('jobs', self.job)
        current = self.service.store.project(self.project['id'])
        current['training']['steps'] = 999
        self.service.store.save_project(current, current['id'])
        seen = {}
        def launch(job, stop, root, settings, project, **kwargs):
            seen.update(steps=project['training']['steps'], resume=kwargs['resume'])
        with patch.object(self.service, 'launch_training', side_effect=launch):
            self.service.save_settings({'integration': 'standalone'})
            self.service.resume_training('test-run')
            for _ in range(100):
                job = self.service.store.item('jobs', 'test-run')
                if job['status'] == 'complete':
                    break
                time.sleep(.01)
        self.assertEqual(seen, {'steps': 100, 'resume': True})
        self.assertEqual(job['attempt_history'][0]['status'], 'paused')
        self.assertFalse(self.service.job_lock.locked())

    def test_checkpoint_and_preview_cadence_and_disabled_sampling(self):
        p = normalize_project({'training': {'checkpoint_every': 71, 'sample_every': 83, 'keep_checkpoints': 4, 'sample_steps': 19}})
        process = config_for(p, 'dataset', 'output', 'cadence')['config']['process'][0]
        self.assertEqual(process['save']['save_every'], 71)
        self.assertEqual(process['save']['max_step_saves_to_keep'], 4)
        self.assertEqual(process['sample']['sample_every'], 83)
        self.assertEqual(process['sample']['sample_steps'], 19)
        self.assertFalse(process['train']['disable_sampling'])
        p['training']['sample_every'] = 0
        self.assertTrue(config_for(p, 'dataset', 'output', 'off')['config']['process'][0]['train']['disable_sampling'])
        for key, value in [('checkpoint_every', 0), ('sample_every', -1), ('keep_checkpoints', 0), ('sample_steps', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                normalize_project({'training': {key: value}})

    def finish_run(self, step=100):
        path = self.checkpoint(step)
        final = path.with_name('lora_test-run.safetensors')
        path.rename(final)
        self.log.write_text(self.log.read_text().replace(str(path), str(final)))
        self.job.update(status='complete', finished=time.time(), pid=None)
        self.service.store.put_item('jobs', self.job)
        self.service.save_settings({'integration': 'standalone'})
        return final

    def test_continue_defaults_and_repeated_continuations_preserve_originals(self):
        original = self.finish_run()
        recipe = (self.root / 'config.json').read_bytes()
        manifest = (self.root / 'manifest.json').read_bytes()
        original_weights = original.read_bytes()
        seen = []
        def launch(job, stop, root, settings, project, **kwargs):
            config = run_config(root, job)
            process = config['config']['process'][0]
            total = process['train']['steps']
            seen.append((total, kwargs['resume'], project['training']['steps']))
            self.output = Path(process['training_folder']) / config['config']['name']
            with self.log.open('a') as log:
                log.write(f'\nSTUDIO_RESUME attempt=2 total={total}\n')
            self.checkpoint(total)
        def wait_finished():
            for _ in range(200):
                if not self.service.job_lock.locked():
                    return
                time.sleep(.01)
            self.fail('Continuation did not finish')
        with patch.object(self.service, 'launch_training', side_effect=launch):
            self.assertTrue(self.monitor.snapshot('test-run')['capabilities']['extend'])
            self.monitor.action('test-run', 'extend')
            wait_finished()
            self.monitor.action('test-run', 'extend', {'checkpoints': 2})
            wait_finished()
        self.assertEqual(seen, [(350, True, 100), (850, True, 100)])
        self.assertEqual(self.monitor.snapshot('test-run')['total'], 850)
        self.assertEqual(self.monitor.snapshot('test-run')['saved_step'], 850)
        self.assertEqual(self.monitor.snapshot('test-run')['progress'], 100)
        self.assertEqual(original.read_bytes(), original_weights)
        self.assertEqual((self.root / 'config.json').read_bytes(), recipe)
        self.assertEqual((self.root / 'manifest.json').read_bytes(), manifest)
        self.assertEqual(len(self.service.store.item('jobs', 'test-run')['continuations']), 2)
        public = json.loads(run_config(self.root, self.service.store.item('jobs', 'test-run'))['meta']['lora_studio'])
        self.assertEqual(public['training']['train']['steps'], 850)

    def test_continue_rejects_bad_counts_busy_run_and_incomplete_state(self):
        final = self.finish_run()
        for value in [None, True, 0, -1, 1.5, '2', 1001]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.monitor.action('test-run', 'extend', {'checkpoints': value})
        with patch('studio.service.running', return_value=True):
            self.job['pid'] = 123
            self.service.store.put_item('jobs', self.job)
            with self.assertRaisesRegex(ValueError, 'still running'):
                self.service.resume_training('test-run', additional_checkpoints=1)
        self.job['pid'] = None
        self.job['status'] = 'paused'
        self.service.store.put_item('jobs', self.job)
        with self.assertRaisesRegex(ValueError, 'completed run'):
            self.service.resume_training('test-run', additional_checkpoints=1)
        self.job['status'] = 'complete'
        self.service.store.put_item('jobs', self.job)
        self.service.job_lock.acquire()
        try:
            with self.assertRaisesRegex(Exception, 'Another job'):
                self.service.resume_training('test-run', additional_checkpoints=1)
        finally:
            self.service.job_lock.release()
        final.write_bytes(final.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'damaged'):
            self.service.resume_training('test-run', additional_checkpoints=1)
        self.assertEqual(self.service.store.item('jobs', 'test-run')['status'], 'complete')

    def test_continuation_log_keeps_previous_final_step_and_metrics(self):
        log = ('run: 100/100 [00:02<00:00, 2.00s/it, loss: 0.4]\n'
               'Saved checkpoint to lora_test-run.safetensors\nSaved optimizer to optimizer.pt\n'
               'STUDIO_RESUME attempt=2 total=350\n'
               'run: 101/350 [00:02<08:18, 2.00s/it, loss: 0.3]\n')
        parsed = parse_log(log, 350, initial_total=100)
        self.assertEqual(parsed['saved_step'], 100)
        self.assertEqual([p['step'] for p in parsed['points']], [100, 101])

    def test_failed_continuation_resumes_same_target(self):
        self.finish_run()
        seen = []
        def launch(job, stop, root, settings, project, **kwargs):
            config = run_config(root, job)
            process = config['config']['process'][0]
            seen.append(process['train']['steps'])
            self.output = Path(process['training_folder']) / config['config']['name']
            with self.log.open('a') as log:
                log.write('\nSTUDIO_RESUME attempt=2 total=350\n')
            if len(seen) == 1:
                self.checkpoint(250)
                raise RuntimeError('Synthetic interruption')
            self.checkpoint(350)
        with patch.object(self.service, 'launch_training', side_effect=launch):
            self.service.resume_training('test-run', additional_checkpoints=1)
            for _ in range(200):
                if not self.service.job_lock.locked():break
                time.sleep(.01)
            self.assertEqual(self.service.store.item('jobs', 'test-run')['status'], 'failed')
            self.assertTrue(self.monitor.snapshot('test-run')['capabilities']['resume'])
            self.service.resume_training('test-run')
            for _ in range(200):
                if not self.service.job_lock.locked():break
                time.sleep(.01)
        self.assertEqual(seen, [350, 350])
        self.assertEqual(len(self.service.store.item('jobs', 'test-run')['continuations']), 1)

    def test_cancel_during_preparation_before_run_folder_exists(self):
        job = {**self.job, 'id': 'preparing', 'pid': None, 'phase': 'Preparing dataset'}
        self.service.store.put_item('jobs', job)
        self.service.cancel[job['id']] = threading.Event()
        snap = self.monitor.snapshot(job['id'])
        self.assertFalse(snap['capabilities']['pause'])
        self.assertEqual(snap['phase'], 'Preparing dataset')
        self.monitor.action(job['id'], 'cancel')
        self.assertTrue(self.service.cancel[job['id']].is_set())

    def test_resume_stops_if_backend_cannot_restore_optimizer(self):
        toolkit = self.root / 'synthetic-toolkit'
        toolkit.mkdir()
        (toolkit / 'run.py').write_text('import time\nprint("Failed to load optimizer state from synthetic", flush=True)\ntime.sleep(30)\n')
        settings = {**self.service.settings(), 'toolkit_path': str(toolkit), 'toolkit_python': sys.executable}
        with self.assertRaisesRegex(RuntimeError, 'could not restore optimizer'):
            self.service.launch_training(self.job, threading.Event(), self.root, settings, self.project, resume=True)
        self.assertIsNotNone(self.service.processes[self.job['id']].poll())
