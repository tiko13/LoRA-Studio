"""Category routing and persistence using authored synthetic evidence only."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from studio.categories import CATEGORIES, MODEL_CAPTIONS
from studio.captioners import RAW_CAPTION_PROMPT, raw_caption_instruction
from studio.caption_processing import processing_instruction, process_response, format_observations
from studio.domain import catalog, caption_instruction, intent, intent_digest, normalize_project, parse_caption
from studio.preparation import decorate, effective_coverage, dataset_advisories, normalize_coverage
from studio.service import Service
from studio.store import Conflict
from studio.trainer import export_run, config_for
from studio.verification import checks_for, normalize_checks, instruction
from studio.sample_prompts import instruction as preview_instruction, signature
from tests.fixtures import fixture

ROOT = Path(__file__).resolve().parents[1] / 'internal-tests/artifacts/unit'
ROOT.mkdir(parents=True, exist_ok=True)


def project(kind, model='krea2', **values):
    return normalize_project({'kind': kind, 'model': model, 'description': 'Synthetic visual target', 'trigger': 'syntok', **values})


def response(*facts, subject=''):
    return json.dumps({'subject': subject, 'facts': [{'attribute': a, 'text': t, 'sentence': n} for a, t, n in facts], 'warnings': []})


class CategoryPolicyCase(unittest.TestCase):
    def test_all_category_model_combinations_route_instructions(self):
        for kind in CATEGORIES:
            for model in MODEL_CAPTIONS:
                with self.subTest(kind=kind, model=model):
                    p = project(kind, model)
                    raw = raw_caption_instruction(p)
                    connected = caption_instruction(p)
                    text = processing_instruction(p, 'Synthetic evidence.')
                    for prompt in (raw, connected, text):
                        self.assertIn(kind, prompt)
                        self.assertNotIn('hair_color', prompt)
                        self.assertNotIn('exposed breasts/nipples/genitals', prompt)
                        self.assertNotIn('Do not infer another person from foreshortened limbs', prompt)
                    self.assertIn(MODEL_CAPTIONS[model], connected)
                    self.assertIn(MODEL_CAPTIONS[model], text)
                    self.assertIn('also_keep', text)

    def test_person_prompt_and_digest_contract_remain_unchanged(self):
        p = project('character')
        self.assertEqual(raw_caption_instruction(p), RAW_CAPTION_PROMPT)
        old = hashlib.sha256(json.dumps({'intent': intent(p), 'model': p['model']}, sort_keys=True).encode()).hexdigest()
        self.assertEqual(intent_digest(p), old)
        self.assertIn('Full-body requires', caption_instruction(p))
        self.assertIn('exposure is a SPECIAL part of clothing', processing_instruction(p, 'A person stands.'))

    def test_nonperson_preview_instructions_and_cache_follow_category_and_model(self):
        for kind in CATEGORIES:
            p=project(kind)
            prompt=preview_instruction(p, [])
            self.assertNotIn('head-and-shoulders', prompt)
            self.assertIn(CATEGORIES[kind]['caption'], prompt)
            self.assertNotEqual(signature(p, []), signature({**p, 'model': 'sdxl'}, []))

    def test_object_filters_kept_traits_but_retains_variable_markings(self):
        p = project('object', choices={'markings': 'change'})
        raw = 'A red mug is on a shelf. Its label reads AB12.'
        r = process_response(response(('color', 'red', 1), ('placement', 'on a shelf', 1),
                                      ('markings', 'label reads AB12', 2), subject='mug'), p, raw)
        self.assertEqual(r['caption'], 'syntok, mug, on a shelf, label reads AB12.')
        self.assertNotIn('red', r['caption'])
        with self.assertRaises(ValueError):
            process_response(response(('markings', 'label reads AB13', 2), subject='mug'), p, raw)
        with self.assertRaisesRegex(ValueError, 'Object class'):
            process_response(response(('placement', 'on a shelf', 1), subject='airplane'), p, raw)

    def test_style_trigger_modifies_style_and_not_subject_identity(self):
        p = project('style')
        raw = 'A bird rests on a branch. The drawing has blue linework.'
        r = process_response(response(('subject', 'A bird', 1), ('pose', 'rests on a branch', 1),
                                      ('linework', 'blue linework', 2), subject='invented identity'), p, raw)
        self.assertEqual(r['caption'], 'syntok style, A bird, rests on a branch.')
        self.assertNotIn('invented', r['caption'])
        p['choices']['linework'] = 'change'
        r = process_response(response(('subject', 'A bird', 1), ('linework', 'blue linework', 2)), p, raw)
        self.assertIn('blue linework', r['caption'])

    def test_concept_keeps_roles_direction_and_nonhuman_participants(self):
        p = project('concept', choices={'action': 'change'})
        raw = 'A crane lifts a crate. The scene is outdoors.'
        r = process_response(response(('action', 'A crane lifts a crate', 1), ('background', 'outdoors', 2)), p, raw)
        self.assertEqual(r['caption'], 'syntok, A crane lifts a crate, outdoors.')
        p['choices']['action'] = 'keep'
        r = process_response(response(('action', 'A crane lifts a crate', 1), ('subject', 'A crane', 1), ('background', 'outdoors', 2)), p, raw)
        self.assertNotIn('lifts', r['caption'])
        self.assertIn('A crane', r['caption'])

    def test_nonperson_cannot_use_person_fields_or_extra_to_bypass_policy(self):
        for kind in CATEGORIES:
            for attribute in ('identity', 'hair_color', 'exposure', 'extra'):
                with self.subTest(kind=kind, attribute=attribute), self.assertRaisesRegex(ValueError, 'selected category'):
                    process_response(response((attribute, 'on a shelf', 1)), project(kind), 'An object is on a shelf.')

    def test_custom_rule_retry_uses_category_policy_without_person_rules(self):
        for kind, field in [('object', 'placement'), ('style', 'subject'), ('concept', 'background')]:
            calls = []
            p = project(kind, also_change='Visible shelf')
            def complete(prompt):
                calls.append(prompt)
                return response((field, 'on a shelf', 1), subject='object')
            format_observations(p, 'An object is on a shelf.', complete)
            self.assertEqual(len(calls), 2)
            self.assertIn(kind, calls[1])
            self.assertNotIn('Preserve explicit exposure', calls[1])

    def test_coverage_and_visual_checks_are_category_scoped(self):
        for kind, policy in CATEGORIES.items():
            p = project(kind)
            tags = {k: v[0] for k, v in policy['coverage'].items()}
            r = parse_caption(json.dumps({'caption': 'syntok on a shelf', 'observations': 'Object on a shelf', 'coverage': tags}), p)
            self.assertEqual(r['coverage'], tags)
            self.assertEqual(r['coverage_kind'], kind)
            with self.assertRaises(ValueError):
                normalize_coverage({'framing': 'full-body'}, kind)
            with self.assertRaises(ValueError):
                normalize_checks(['people'], kind)
            prompt = instruction({'caption': r['caption']}, list(checks_for(kind)), project=p)
            self.assertNotIn('foreshortened limbs', prompt)
            self.assertIn('KEEP details can intentionally be absent', prompt)
        types = {t['id']: t for t in catalog()['types']}
        self.assertEqual(types['object']['checks'], list(CATEGORIES['object']['checks']))
        self.assertNotIn('framing', types['concept']['coverage'])

    def test_geometry_and_nonperson_advisories_ignore_saved_person_detections(self):
        item = {'width': 1000, 'height': 700, 'digest': 'synthetic', 'person_count': 0,
                'detection': {'status': 'complete', 'count': 3}, 'included': True}
        for kind, tags in [('object', {'target_visibility': 'absent'}), ('style', {'surface_detail': 'too-small'}),
                           ('concept', {'relationship': 'unclear'})]:
            with self.subTest(kind=kind):
                p = project(kind)
                r = decorate(p, {**item, 'coverage_kind': kind, 'coverage': tags})
                self.assertTrue(r['advisories'])
                self.assertNotIn('person', ' '.join(r['advisories']).lower())
                self.assertNotIn('people', ' '.join(r['advisories']).lower())
                self.assertEqual(r['plan'], decorate(project('character'), {k:v for k,v in item.items() if k != 'detection'})['plan'])
                stale = {**r, 'preparation_stale': True}
                self.assertEqual(effective_coverage(p, stale)[0], {})

    def test_diversity_checks_use_known_included_tags_and_keep_choices(self):
        p = project('style')
        images = [{'included': True, 'coverage_kind': 'style', 'coverage': {'content': 'object'}} for _ in range(2)]
        self.assertTrue(dataset_advisories(p, images))
        p['choices']['subject'] = 'keep'
        self.assertFalse(dataset_advisories(p, images))
        p['choices']['subject'] = 'change'
        images[1]['included'] = False
        self.assertFalse(dataset_advisories(p, images))
        images[1]['included'] = True
        images[1]['preparation_stale'] = True
        self.assertFalse(dataset_advisories(p, images))


class CategoryServiceCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.service = Service(Path(self.temp.name) / 'data')
        self.p = self.service.store.save_project(project('object'))
        self.image = self.service.import_image(self.p['id'], {'name': 'synthetic.png', 'data': base64.b64encode(fixture()).decode()})['image']
        # Caption workflow tests explicitly keep the full synthetic scene; targeting has its own suite.
        self.image['object_target'] = {'mode': 'full'}
        self.service.store.put_item('images', self.image)
        self.service.save_settings({'integration': 'standalone', 'llm_model': 'mock', 'caption_source': 'custom', 'caption_model': 'mock-caption'})

    def wait(self, job):
        for _ in range(300):
            result = self.service.store.item('jobs', job['id'])
            if result['status'] not in {'queued', 'running'}:
                return result
            time.sleep(.01)
        self.fail('Synthetic job timed out')

    def switch(self, kind):
        current = self.service.store.project(self.p['id'])
        self.p = self.service.store.save_project({**current, 'kind': kind}, current['id'])

    def test_category_switch_preserves_separate_coverage_corrections(self):
        tags = {'object': {'viewpoint': 'top'}, 'style': {'content': 'abstract'}, 'concept': {'relationship': 'clear'},
                'character': {'framing': 'detail'}}
        for kind, coverage in tags.items():
            self.switch(kind)
            i = self.service.store.item('images', self.image['id'])
            self.service.update_image(i['id'], {'revision': i['revision'], 'coverage_kind': kind, 'coverage_override': coverage})
        for kind, coverage in tags.items():
            self.switch(kind)
            self.assertEqual(self.service.dataset(self.p['id'])[0]['effective_coverage'], coverage)
        self.switch('style')
        i = self.service.store.item('images', self.image['id'])
        with self.assertRaises(Conflict):
            self.service.update_image(i['id'], {'revision': i['revision'], 'coverage_kind': 'object', 'coverage_override': {}})

    def test_person_apis_rejected_for_nonperson(self):
        with self.assertRaisesRegex(ValueError, 'only to Person'):
            self.service.update_image(self.image['id'], {'revision': 1, 'person_count': 0})
        with self.assertRaisesRegex(ValueError, 'only to Person'):
            self.service.check_people(self.p['id'])
        with self.assertRaises(ValueError):
            self.service.verify(self.p['id'], [self.image['id']], ['people'])

    def test_every_category_uses_selected_raw_and_text_prompts(self):
        for kind, field in [('object', 'placement'), ('style', 'subject'), ('concept', 'background')]:
            self.switch(kind)
            calls = []
            def complete(settings, prompt, data, mime, **kwargs):
                calls.append((prompt, data is not None))
                return 'An object is on a shelf.' if data else response((field, 'An object is on a shelf' if kind == 'style' else 'on a shelf', 1), subject='object')
            with patch('studio.service.standalone_caption', side_effect=complete):
                result = self.wait(self.service.caption(self.p['id']))
            self.assertEqual(result['status'], 'complete', result.get('error'))
            self.assertEqual([visual for _, visual in calls], [True, False])
            self.assertTrue(all(kind in prompt for prompt, _ in calls))
            saved = self.service.store.item('images', self.image['id'])
            self.assertEqual(saved['observation_kind'], kind)
            self.assertFalse(saved['manual'])
            self.assertFalse(saved['reviewed'])

    def test_legacy_or_different_category_observations_require_new_vision(self):
        i = self.service.dataset(self.p['id'])[0]
        i.update(observations='An object is on a shelf.', caption='syntok on a shelf', caption_source='custom',
                 intent_digest=intent_digest(self.p), caption_preparation_digest=i['plan']['digest'])
        self.service.store.put_item('images', i)
        with self.assertRaisesRegex(ValueError, 'another category'):
            self.service.caption(self.p['id'], reuse_observations=True)

    def test_change_category_during_captioning_does_not_save_wrong_policy(self):
        def complete(settings, prompt, data, mime, **kwargs):
            if data:
                self.switch('style')
                return 'An object is on a shelf.'
            return response(('placement', 'on a shelf', 1), subject='object')
        with patch('studio.service.standalone_caption', side_effect=complete):
            result = self.wait(self.service.caption(self.p['id']))
        self.assertTrue(result['messages'])
        self.assertEqual(self.service.store.item('images', self.image['id'])['caption'], '')

    def test_export_freezes_category_policy_and_model_geometry(self):
        for kind in CATEGORIES:
            for model in MODEL_CAPTIONS:
                p = {**self.p, **project(kind, model)}
                i = {**self.image, 'caption': 'syntok on a shelf'}
                root = export_run(self.service.store, p, [i], kind + model, {'toolkit_path': str(Path(self.temp.name) / 'missing')})
                frozen = json.loads((root / 'manifest.json').read_text())
                self.assertEqual(frozen['caption_policy'], kind + '-v1')
                self.assertEqual(frozen['intent']['kind'], kind)
                config = config_for(p, 'dataset', 'output', 'test')['config']['process'][0]['datasets'][0]
                self.assertFalse(config['shuffle_tokens'])
                self.assertTrue(config['buckets'])
                self.assertFalse(config['flip_x'])
                self.assertEqual((root / 'dataset' / (i['id'] + '.txt')).read_text(), i['caption'])


if __name__ == '__main__':
    unittest.main()
