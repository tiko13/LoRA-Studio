import json
from pathlib import Path
import tempfile
import unittest

from studio.domain import normalize_project
from studio.presets import from_run
from studio.service import Service
from studio.store import Conflict, Store
from studio.trainer import config_for

ROOT = Path(__file__).resolve().parents[1] / 'internal-tests/artifacts/unit'
ROOT.mkdir(parents=True, exist_ok=True)


class PresetCase(unittest.TestCase):
    def test_new_trigger_is_automatic_and_existing_choices_survive(self):
        p = self.service.store.save_project({})
        self.assertRegex(p['trigger'], r'^lora_[0-9a-f]{10}$')
        saved = self.service.store.save_project({**p, 'name': 'Renamed'}, p['id'])
        self.assertEqual(saved['trigger'], p['trigger'])
        self.assertEqual(self.service.store.save_project({'trigger': 'chosen'})['trigger'], 'chosen')
        self.assertEqual(self.service.store.save_project({'trigger': ''})['trigger'], '')

    def test_custom_prompt_replacements_have_restorable_history(self):
        p = self.service.store.save_project({'sample_prompts': 'first draft'})
        p = self.service.store.save_project({**p, 'sample_prompt_mode': 'automatic'}, p['id'])
        self.assertEqual(p['sample_prompts'], 'first draft')
        self.assertEqual(p['sample_prompt_history'], [])
        p = self.service.store.save_project({**p, 'sample_prompts': 'second draft'}, p['id'])
        self.assertEqual(p['sample_prompt_history'][0]['text'], 'first draft')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.service = Service(Path(self.temp.name))
        self.project = normalize_project({'model': 'krea2', 'trigger': 'synthetic', 'sample_prompts': 'synthetic test',
                                         'training': {'gpu': 3, 'rank': 32, 'steps': 2500, 'optimizer': 'automagic3'}})
        config = config_for(self.project, '/private/dataset', '/private/output', 'synthetic-run')
        self.preset = self.service.store.save_preset(from_run('Synthetic baseline', config,
                     {'project': self.project, 'run_id': 'synthetic-run', 'backend_commit': 'test'}, default=True))

    def test_default_is_model_specific_and_does_not_copy_private_target_or_gpu(self):
        p = self.service.store.save_project({'description': 'A new target'})
        self.assertEqual(p['training']['rank'], 32)
        self.assertEqual(p['training']['gpu'], 0)
        self.assertEqual(p['training']['optimizer'], 'automagic3')
        self.assertEqual(p['sample_prompts'], '')
        self.assertEqual(p['training_preset'], self.preset['id'])
        self.assertEqual(p['sample_prompt_mode'], 'automatic')
        self.assertEqual(self.service.store.save_project({'model': 'qwen21'})['training']['rank'], 16)
        self.assertEqual(self.service.store.save_project({'training': {'rank': 8}})['training']['rank'], 8)
        self.assertNotIn('/private/', json.dumps(self.preset))
        self.assertNotIn('synthetic test', json.dumps(self.preset))
        self.assertEqual(Store(self.temp.name).presets()[0]['id'], self.preset['id'])

    def test_apply_preserves_intent_manual_prompts_gpu_and_checks_revision(self):
        p = self.service.store.save_project({**self.project, 'training': {'gpu': 2, 'rank': 8}})
        original = dict(p)
        result = self.service.apply_training_preset(p['id'], {'revision': p['revision'], 'preset': self.preset['id']})
        self.assertEqual(result['training']['gpu'], 2)
        self.assertEqual(result['training']['rank'], 32)
        self.assertEqual(result['sample_prompts'], original['sample_prompts'])
        self.assertEqual(result['trigger'], original['trigger'])
        with self.assertRaises(Conflict):
            self.service.apply_training_preset(p['id'], {'revision': p['revision'], 'preset': self.preset['id']})
        q = self.service.store.save_project({'model': 'qwen21'})
        with self.assertRaises(ValueError):
            self.service.apply_training_preset(q['id'], {'revision': q['revision'], 'preset': self.preset['id']})

    def test_saving_again_is_idempotent_and_default_is_unique(self):
        saved = self.service.store.save_preset(self.preset)
        self.assertEqual(saved['id'], self.preset['id'])
        self.service.store.save_preset({**self.preset, 'name': 'Second baseline'})
        self.assertEqual(sum(p['default'] for p in self.service.store.presets()), 1)
