"""Style policies and end-to-end preparation with synthetic data and mock inference."""
import base64
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from studio.caption_processing import process_response, caption_text_current
from studio.domain import normalize_project, intent_digest, catalog
from studio.preparation import decorate, dataset_advisories
from studio.service import Service
from studio import style_preparation as styles
from tests.fixtures import fixture

ROOT = Path(__file__).resolve().parents[1] / 'internal-tests/artifacts/unit'
ROOT.mkdir(parents=True, exist_ok=True)


def project(**values):
    return normalize_project({'kind': 'style', 'trigger': 'testlook', 'description': 'Synthetic look', **values})


def facts(*rows):
    return json.dumps({'subject': '', 'facts': [{'attribute': a, 'text': t, 'sentence': n} for a, t, n in rows]})


class StylePolicyCase(unittest.TestCase):
    def test_family_defaults_validation_and_intent_invalidation(self):
        self.assertEqual(project()['style_family'], 'general')
        self.assertEqual(set(catalog()['style_families']), set(styles.FAMILIES))
        with self.assertRaises(ValueError):
            project(style_family='guessed')
        for kind in ['character', 'object', 'concept']:
            self.assertEqual(intent_digest(project(kind=kind)), intent_digest(project(kind=kind, style_family='photo')))
        self.assertNotEqual(intent_digest(project()), intent_digest(project(style_family='photo')))

    def test_default_keeps_style_but_retains_object_colours_and_materials(self):
        raw = 'A red mug rests on a wooden table. The photograph has film grain and a white border.'
        r = process_response(facts(('subject', 'A red mug', 1), ('background', 'on a wooden table', 1),
                                   ('texture', 'film grain', 2), ('border', 'a white border', 2)), project(), raw)
        self.assertEqual(r['caption'], 'testlook style, A red mug, on a wooden table.')
        self.assertEqual(r['observations'], raw)
        self.assertEqual(r['style_caption_version'], styles.VERSION)

    def test_misclassified_rendering_is_rejected_across_fields(self):
        for word in ['watercolor', 'pixel art', 'cartoon', 'film grain', 'bold outlines', 'muted colours', 'white border']:
            for field in ['subject', 'background', 'lighting', 'composition']:
                raw = 'A bird rests on a branch. The image has ' + word + '.'
                with self.subTest(word=word, field=field), self.assertRaisesRegex(ValueError, 'KEEP'):
                    process_response(facts(('subject', 'A bird', 1), (field, word, 2)), project(), raw)

    def test_explicit_change_rules_allow_each_rendering_trait(self):
        for field, text in [('medium', 'watercolor'), ('texture', 'film grain'), ('linework', 'bold outlines'),
                            ('palette', 'muted colours'), ('border', 'white border')]:
            raw = 'A bird rests on a branch. The image has ' + text + '.'
            p = project(choices={field: 'change'})
            r = process_response(facts(('subject', 'A bird', 1), (field, text, 2)), p, raw)
            self.assertIn(text, r['caption'])

    def test_trigger_can_be_descriptive_without_false_style_leak(self):
        p = project(trigger='soft watercolor illustration')
        r = process_response(facts(('subject', 'A green bird', 1)), p, 'A green bird is on a rock.')
        self.assertTrue(r['caption'].startswith(p['trigger']))
        self.assertTrue(caption_text_current(p, {**r, 'observation_kind': 'style'}))

    def test_abstract_and_depicted_picture_are_content(self):
        raw = 'Concentric circles overlap a square.'
        self.assertIn('Concentric circles', process_response(facts(('subject', raw[:-1], 1)), project(), raw)['caption'])
        raw = 'A person is holding a photograph.'
        self.assertIn('holding a photograph', process_response(facts(('subject', raw[:-1], 1)), project(), raw)['caption'])
        raw = 'A woman is painting a wall.'
        self.assertIn('painting a wall', process_response(facts(('subject', raw[:-1], 1)), project(), raw)['caption'])

    def test_leading_style_wrappers_preserve_the_content_and_source(self):
        for raw in ['A cartoon man wears a blue cap.', 'A watercolor painting of a red mug.',
                    'A close-up of one cartoon human face.', 'A group of cartoon characters stands beside a wall.']:
            r = process_response(facts(('subject', raw[:-1], 1)), project(), raw)
            self.assertNotRegex(r['caption'], r'cartoon|watercolor|painting')
            self.assertTrue(r['caption_facts'][0]['source_repaired'])
            self.assertEqual(r['caption_facts'][0]['source_sentence'], raw)
        p = project(choices={'medium': 'change'})
        self.assertIn('cartoon', process_response(facts(('subject', 'A cartoon man', 1)), p, 'A cartoon man wears a hat.')['caption'])
        raw='A group of cartoon characters stands beside a wall.'
        self.assertIn('group of characters', process_response(facts(('subject', raw[:-1], 1)), project(), raw)['caption'])
        raw='A blue pixel-art figure stands beside a wall.'
        self.assertIn('blue figure', process_response(facts(('subject', raw[:-1], 1)), project(), raw)['caption'])

    def test_verbose_style_output_is_condensed_without_cutting_facts(self):
        raw = 'A red mug sits on a desk. A book is open. A pen rests nearby. A chair stands behind. A window is open. A tree is outside. Light enters.'
        rows = [('subject', 'A red mug sits on a desk', 1), ('pose', 'sits on a desk', 1), ('background', 'A book is open', 2),
                ('background', 'A pen rests nearby', 3), ('background', 'A chair stands behind', 4), ('background', 'A window is open', 5),
                ('background', 'A tree is outside', 6), ('lighting', 'Light enters', 7)]
        r = process_response(facts(*rows), project(), raw)
        self.assertEqual(r['caption'].count('sits on a desk'), 1)
        self.assertEqual(len(r['caption_facts']), 6)

    def test_checkerboard_is_not_evidence_of_transparency(self):
        with self.assertRaisesRegex(ValueError, 'checkerboard'):
            process_response(facts(('subject', 'A bird', 1), ('background', 'transparent checkerboard', 2)), project(),
                             'A bird stands. It has a transparent checkerboard background.')

    def test_source_qualifiers_cannot_be_removed(self):
        with self.assertRaisesRegex(ValueError, 'source'):
            process_response(facts(('subject', 'a bird', 1)), project(), 'Possibly a bird is on a rock.')

    def test_framing_only_caption_is_not_a_success(self):
        with self.assertRaisesRegex(ValueError, 'main depicted content'):
            process_response(facts(('framing', 'Overhead view', 1)), project(), 'Overhead view of a red mug.')

    def test_border_and_repeated_content_advisories_do_not_exclude(self):
        p = project(style_family='pixel')
        item = {'width': 640, 'height': 480, 'digest': 'synthetic', 'included': True,
                'coverage_kind': 'style', 'coverage': {'content': 'portrait', 'border': 'present', 'surface_detail': 'clear'}}
        r = decorate(p, item)
        self.assertTrue(r['included'])
        self.assertIn('border', ' '.join(r['advisories']))
        self.assertIn('pixel edges', ' '.join(r['advisories']))
        self.assertIn('content', ' '.join(dataset_advisories(p, [r, r])))


class StyleServiceCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.service = Service(Path(self.temp.name) / 'data')
        self.p = self.service.store.save_project(project(sample_prompt_mode='manual', sample_prompts='testlook style, a tree'))
        self.i = self.service.import_image(self.p['id'], {'name': 'synthetic.png', 'data': base64.b64encode(fixture()).decode()})['image']
        self.service.save_settings({'integration': 'standalone', 'llm_model': 'synthetic'})
        self.raw = 'A red mug rests on a wooden table. The image has film grain.'
        self.calls = []

    def wait(self, job):
        for _ in range(500):
            j = self.service.store.item('jobs', job['id'])
            if j['status'] not in {'queued', 'running'}:
                return j
            time.sleep(.01)
        self.fail('Synthetic job timed out')

    def complete(self, settings, prompt, data, mime, **kwargs):
        self.calls.append(bool(data))
        if data:
            return json.dumps({'observations': self.raw, 'caption': 'testlook photograph of a red mug',
                               'coverage': {'content': 'object'}, 'warnings': []})
        return facts(('subject', 'A red mug', 1), ('background', 'on a wooden table', 1), ('texture', 'film grain', 2))

    def test_connected_pipeline_refines_freeform_caption_and_freezes_export(self):
        with patch('studio.service.standalone_caption', side_effect=self.complete), patch('studio.trainer.backend_status', return_value={'available': True, 'models': {'krea2': True}}):
            j = self.wait(self.service.automatic_prepare(self.p['id']))
        self.assertEqual(j['status'], 'complete', j.get('error'))
        self.assertEqual(self.calls, [True, False])
        i = self.service.dataset(self.p['id'])[0]
        self.assertNotIn('photograph', i['caption'])
        self.assertEqual(i['observations'], self.raw)
        root = self.service.store.root / 'runs' / j['id']
        self.assertEqual((root / 'dataset' / (i['id'] + '.txt')).read_text(), i['caption'])
        frozen = json.loads((root / 'manifest.json').read_text())
        self.assertEqual(frozen['intent']['style_family'], 'general')
        self.assertEqual(frozen['intent']['style_policy_version'], styles.VERSION)
        self.assertEqual(self.service.image_path(i).read_bytes(), fixture())

    def test_connected_saved_observations_reprocess_and_preserve_manual_edits(self):
        with patch('studio.service.standalone_caption', side_effect=self.complete):
            self.assertEqual(self.wait(self.service.caption(self.p['id']))['status'], 'complete')
            saved = self.service.store.item('images', self.i['id'])
            saved['observation_warnings'] = ['The crop edge needs review.']
            self.service.store.put_item('images', saved)
            self.calls.clear()
            self.assertEqual(self.wait(self.service.caption(self.p['id'], reuse_observations=True))['status'], 'complete')
            self.assertEqual(self.calls, [False])
        i = self.service.dataset(self.p['id'])[0]
        self.assertEqual(len(i['history']), 2)
        self.assertEqual(i['coverage'], {'content': 'object'})
        self.assertIn('The crop edge needs review.', i['warnings'])
        self.service.update_image(i['id'], {'revision': i['revision'], 'caption': 'testlook, manual scene'})
        with self.assertRaisesRegex(ValueError, 'Manual captions'):
            self.service.caption(self.p['id'], reuse_observations=True)

    def test_family_change_during_inference_preserves_existing_image(self):
        def changing(*args, **kwargs):
            p = self.service.store.project(self.p['id'])
            self.service.store.save_project({**p, 'style_family': 'photo'}, p['id'])
            return self.complete(*args, **kwargs)
        with patch('studio.service.standalone_caption', side_effect=changing):
            j = self.wait(self.service.caption(self.p['id']))
        self.assertEqual(j['status'], 'complete')
        self.assertTrue(j['messages'])
        self.assertEqual(self.service.store.item('images', self.i['id'])['caption'], '')

    def test_style_family_survives_type_switch(self):
        p = self.service.store.save_project({**self.p, 'style_family': 'photo'}, self.p['id'])
        p = self.service.store.save_project({**p, 'kind': 'object'}, p['id'])
        p = self.service.store.save_project({**p, 'kind': 'style'}, p['id'])
        self.assertEqual(p['style_family'], 'photo')
