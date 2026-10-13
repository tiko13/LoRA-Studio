import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studio.domain import normalize_project
from studio import sample_prompts as prompts
from studio.service import Service
from studio.store import Conflict
from studio.trainer import config_for

ROOT = Path(__file__).resolve().parents[1] / 'internal-tests/artifacts/unit'
ROOT.mkdir(parents=True, exist_ok=True)


def response():
    return json.dumps({'prompts': [
        {'role': 'familiar', 'text': 'shape displayed beside a window, three-quarter view'},
        {'role': 'recombined', 'text': 'shape on a shelf in diffuse light, front view'},
        {'role': 'generalization', 'text': 'shape in a quiet courtyard, close-up view'}]})


class PreviewCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.service = Service(Path(self.temp.name))
        self.p = self.service.store.save_project({'kind': 'object', 'trigger': 'shape', 'description': 'A geometric object'})
        self.images = [{'id': 'image', 'project': self.p['id'], 'caption': 'shape resting on a display surface', 'included': True,
                        'caption_facts': [{'attribute': 'placement', 'text': 'resting on a display surface'}]}]
        self.service.store.put_item('images', self.images[0])
        self.settings = {**self.service.settings(), 'integration': 'standalone', 'llm_provider': 'openai', 'llm_model': 'synthetic'}

    def test_generation_is_text_only_and_rejects_caption_copies(self):
        with patch('studio.sample_prompts.standalone_caption', return_value=response()) as infer:
            result = prompts.generate(self.p, self.images, self.settings)
        self.assertEqual(result['source'], 'local_llm')
        self.assertIsNone(infer.call_args.args[2])
        self.assertIn('keep', infer.call_args.args[1])
        copied = json.loads(response());copied['prompts'][0]['text'] = self.images[0]['caption']
        with self.assertRaises(ValueError):
            prompts.parse(json.dumps(copied), self.p, self.images)
        with patch('studio.sample_prompts.standalone_caption', return_value=json.dumps(copied)):
            self.assertEqual(prompts.generate(self.p, self.images, self.settings)['source'], 'rules')

    def test_type_fallback_and_keep_rules(self):
        for kind in ('character', 'object', 'style', 'concept'):
            p = normalize_project({'kind': kind, 'trigger': 'target'})
            suite = prompts.fallback(p, [])
            self.assertEqual(len(suite), 3)
            self.assertTrue(all('target' in row['text'] for row in suite))
            if kind != 'character':
                self.assertTrue(all('looking toward the camera' not in row['text'] for row in suite))
        p = {**self.p, 'choices': {'background': 'keep', 'lighting': 'keep'}}
        self.assertTrue(all('courtyard' not in p['text'] and 'evening' not in p['text'] for p in prompts.fallback(p, self.images)))
        self.assertIn('resting on a display surface', prompts.fallback(self.p, self.images)[0]['text'])

    def test_person_suite_retries_for_portrait_coverage(self):
        p = normalize_project({'kind': 'character', 'trigger': 'shape', 'description': 'A person'})
        valid = json.loads(response())
        valid['prompts'][0]['text'] = 'A close-up portrait of shape near a window'
        with patch('studio.sample_prompts.standalone_caption', side_effect=[response(), json.dumps(valid)]) as infer:
            result = prompts.generate(p, [], self.settings)
        self.assertEqual(infer.call_count, 2)
        self.assertEqual(result['source'], 'local_llm')
        self.assertIn('close-up', result['prompts'][0]['text'])

    def test_cache_invalidation_history_and_manual_preservation(self):
        with patch('studio.sample_prompts.standalone_caption', return_value=response()) as infer:
            first = self.service.resolve_preview_suite(self.p, self.images, self.settings)
            self.assertEqual(self.service.resolve_preview_suite(self.p, self.images, self.settings), first)
            self.assertEqual(infer.call_count, 1)
            image = {**self.images[0], 'caption': 'shape on a wooden shelf'}
            self.service.store.put_item('images', image)
            self.assertTrue(self.service.preview_suite(self.p, [image])['needs_generation'])
            self.service.resolve_preview_suite(self.p, [image], self.settings)
            self.assertEqual(infer.call_count, 2)
        record = json.loads((self.service.store.root / 'sample-prompts' / (self.p['id'] + '.json')).read_text())
        self.assertEqual(len(record['history']), 1)
        manual = normalize_project({**self.p, 'sample_prompts': 'shape custom evaluation', 'sample_prompt_mode': 'manual'})
        with patch('studio.sample_prompts.generate') as generate:
            suite = self.service.resolve_preview_suite(manual, [image], self.settings)
            generate.assert_not_called()
        self.assertEqual(suite['prompts'][0]['text'], 'shape custom evaluation')
        automatic = normalize_project({**manual, 'sample_prompt_mode': 'automatic'})
        self.assertEqual(automatic['sample_prompts'], manual['sample_prompts'])

    def test_concurrent_edit_does_not_save_stale_generated_prompts(self):
        def infer(*args, **kwargs):
            self.service.store.save_project({**self.p, 'description': 'Changed target'}, self.p['id'])
            return response()
        with patch('studio.sample_prompts.standalone_caption', side_effect=infer), self.assertRaises(Conflict):
            self.service.resolve_preview_suite(self.p, self.images, self.settings)
        self.assertFalse((self.service.store.root / 'sample-prompts').exists())

    def test_generated_suite_reaches_config_and_manual_mode_is_backward_compatible(self):
        suite = {'source': 'local_llm', 'prompts': prompts.parse(response(), self.p, self.images)}
        cfg = config_for({**self.p, '_preview_suite': suite}, 'dataset', 'output', 'synthetic')
        self.assertEqual([s['prompt'] for s in cfg['config']['process'][0]['sample']['samples']], [p['text'] for p in suite['prompts']])
        self.assertEqual(normalize_project({'sample_prompts': 'target test'})['sample_prompt_mode'], 'manual')
        with self.assertRaises(ValueError):
            prompts.effective(normalize_project({'sample_prompt_mode': 'manual'}), [])
