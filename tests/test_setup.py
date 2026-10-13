import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from studio.service import Service
from studio.setup import Setup, runtime_check, python_for
from studio.store import Conflict

ROOT = Path(__file__).resolve().parents[1] / 'internal-tests/artifacts/unit'
ROOT.mkdir(parents=True, exist_ok=True)


class SetupCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.service = Service(Path(self.temp.name) / 'data')
        self.setup = self.service.setup
        hardware = patch('studio.setup.hardware_check', return_value={'available': False, 'reason': 'Synthetic hardware'})
        hardware.start()
        self.addCleanup(hardware.stop)

    def test_first_run_and_dismissal_persist_without_touching_settings(self):
        before = self.service.settings()
        self.assertEqual(self.setup.status()['disposition'], 'new')
        self.setup.dismiss('later')
        self.assertEqual(Setup(self.service).status()['disposition'], 'later')
        self.assertEqual(self.service.settings(), before)

    def test_revision_conflict_preserves_newer_settings(self):
        first = self.service.settings()
        self.service.save_settings({'llm_model': 'new-choice'})
        with self.assertRaises(Conflict):
            self.service.save_settings({**first, 'llm_model': 'stale-choice'})
        self.assertEqual(self.service.settings()['llm_model'], 'new-choice')

    def test_interrupted_install_is_not_restarted_or_killed(self):
        self.setup.state['install'] = {'status': 'running', 'pid': 123, 'target': str(self.setup.root / 'ai-toolkit-test')}
        self.setup._save()
        recovered = Setup(self.service)
        self.assertEqual(recovered.status()['install']['status'], 'interrupted')
        with patch('studio.setup.running', return_value=True), self.assertRaisesRegex(ValueError, 'may still be running'):
            recovered.start()

    def test_no_git_reports_recovery_without_starting_process(self):
        with patch('studio.setup.shutil.which', return_value=None), self.assertRaisesRegex(ValueError, 'Install Git'):
            self.setup.start()
        self.assertEqual(self.setup.status()['install'], {})

    def test_checks_distinguish_backend_files_from_usable_runtime(self):
        with patch('studio.setup.backend_status', return_value={'available': True, 'architectures': ['synthetic']}), \
             patch('studio.setup.runtime_check', return_value={'available': True, 'cuda': False}), \
             patch.object(self.service, 'connection', return_value={'standalone': {'available': True}}):
            report = self.setup.check()
        self.assertFalse(report['training_ready'])
        self.assertTrue(report['backend']['available'])

    def test_bad_probe_and_timeout_do_not_expose_raw_output(self):
        for result in (subprocess.CompletedProcess([], 1, 'sensitive payload', 'secret'), subprocess.TimeoutExpired([], 90)):
            kw = {'side_effect': result} if isinstance(result, Exception) else {'return_value': result}
            with patch('studio.setup.subprocess.run', **kw):
                report = runtime_check(sys.executable)
            self.assertFalse(report['available'])
            self.assertNotIn('sensitive', json.dumps(report))

    def test_probe_checks_dependencies_and_cuda_separately(self):
        payload = {'python': '3.12', 'missing': ['peft'], 'cuda': True, 'devices': []}
        result = subprocess.CompletedProcess([], 0, 'warning\nSTUDIO_SETUP ' + json.dumps(payload), '')
        with patch('studio.setup.subprocess.run', return_value=result):
            report = runtime_check(sys.executable)
        self.assertFalse(report['available'])
        self.assertTrue(report['cuda'])
        self.assertIn('peft', report['reason'])

    def test_managed_install_uses_upstream_manager_and_preserves_changed_selection(self):
        target = self.setup.root / 'ai-toolkit-synthetic'
        (target / '.git').mkdir(parents=True)
        (target / 'manager').mkdir()
        (target / 'manager/__main__.py').write_text('')
        python = target / '.venv/Scripts/python.exe'
        python.parent.mkdir(parents=True)
        python.touch()
        self.setup.state['install'] = {'status': 'queued'}
        baseline = self.service.settings()
        self.service.save_settings({'toolkit_path': 'user-changed-backend'})
        with patch.object(self.setup, '_command') as command, patch('studio.setup.runtime_check', return_value={'available': True}):
            self.setup._install(target, baseline)
        self.assertEqual(command.call_args.args[0], [sys.executable, '-m', 'manager', 'install'])
        self.assertEqual(self.service.settings()['toolkit_path'], 'user-changed-backend')
        self.assertEqual(self.setup.status()['install']['status'], 'complete')
        self.assertEqual(python_for(target), str(python.resolve()))

    def test_install_failure_preserves_existing_backend(self):
        self.service.save_settings({'toolkit_path': 'existing-backend'})
        self.setup.state['install'] = {'status': 'queued'}
        with patch.object(self.setup, '_command', side_effect=RuntimeError('Synthetic failure')):
            self.setup._install(self.setup.root / 'ai-toolkit-synthetic', self.service.settings())
        self.assertEqual(self.service.settings()['toolkit_path'], 'existing-backend')
        self.assertEqual(self.setup.status()['install']['status'], 'failed')

    def test_new_install_downloads_only_official_repository_and_selects_verified_runtime(self):
        from studio.setup import TOOLKIT_REPOSITORY
        target = self.setup.root / 'ai-toolkit-synthetic'
        self.setup.state['install'] = {'status': 'queued'}
        def command(args, cwd, log):
            if 'clone' in args:
                (target / '.git').mkdir(parents=True)
                (target / 'manager').mkdir()
                (target / 'manager/__main__.py').touch()
        with patch.object(self.setup, '_command', side_effect=command) as calls, \
             patch('studio.setup.python_for', return_value='synthetic-python'), \
             patch('studio.setup.runtime_check', return_value={'available': True}):
            self.setup._install(target, self.service.settings())
        self.assertIn(TOOLKIT_REPOSITORY, calls.call_args_list[0].args[0])
        self.assertIn('submodule', calls.call_args_list[1].args[0])
        self.assertEqual(self.service.settings()['toolkit_python'], 'synthetic-python')
        self.assertEqual(self.setup.status()['install']['status'], 'complete')

    def test_cancel_during_runtime_check_does_not_select_installation(self):
        target = self.setup.root / 'ai-toolkit-synthetic'
        (target / '.git').mkdir(parents=True)
        (target / 'manager').mkdir()
        (target / 'manager/__main__.py').touch()
        self.setup.state['install'] = {'status': 'running'}
        def probe(python):
            self.setup.cancel.set()
            return {'available': True}
        with patch.object(self.setup, '_command'), patch('studio.setup.runtime_check', side_effect=probe):
            self.setup._install(target, self.service.settings())
        self.assertEqual(self.setup.status()['install']['status'], 'cancelled')
        self.assertEqual(self.service.settings()['toolkit_path'], '')

    def test_vision_test_uses_synthetic_image_and_rejects_wrong_answer(self):
        self.service.save_settings({'integration': 'standalone', 'llm_model': 'synthetic-vision'})
        with patch('studio.setup.standalone_caption', return_value='Blue') as caption:
            result = self.setup.vision()
        self.assertFalse(result['passed'])
        self.assertTrue(caption.call_args.args[2].startswith(b'\x89PNG'))
        with patch('studio.setup.standalone_caption', return_value='Red.'):
            self.assertTrue(self.setup.vision()['passed'])


if __name__ == '__main__':
    unittest.main()
