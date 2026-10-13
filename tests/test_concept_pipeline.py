"""Concept intent, evidence and frozen exports, with synthetic inputs only."""
import base64
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from studio.domain import normalize_project, intent, intent_digest, catalog, caption_instruction
from studio.caption_processing import process_response, caption_text_current
from studio.preparation import effective_coverage
from studio.service import Service
from studio import concept_preparation as concepts
from tests.fixtures import fixture

ROOT = Path(__file__).resolve().parents[1] / 'internal-tests/artifacts/unit'
ROOT.mkdir(parents=True, exist_ok=True)


def project(**values):
    return normalize_project({'kind': 'concept', 'trigger': 'syntok', 'description': 'Synthetic target', **values})


def facts(*rows):
    return json.dumps({'subject': '', 'facts': [{'attribute': a, 'text': t, 'sentence': n} for a, t, n in rows]})


class ConceptPolicyCase(unittest.TestCase):
    def test_presets_resolve_auto_and_preserve_explicit_and_hidden_choices(self):
        for name, profile in concepts.FAMILIES.items():
            p = project(concept_family=name)
            rules = {a['id']: a['resolved'] for a in intent(p)['attributes']}
            self.assertEqual({a for a, v in rules.items() if v == 'keep'}, set(profile['keep']))
            p['choices']['expression'] = 'keep'
            self.assertEqual(next(a['resolved'] for a in intent(p)['attributes'] if a['id'] == 'expression'), 'keep')
        self.assertEqual(set(catalog()['concept_families']), set(concepts.FAMILIES))
        with self.assertRaises(ValueError):
            project(concept_family='invented')
        self.assertNotEqual(intent_digest(project()), intent_digest(project(concept_family='pose')))
        for kind in ['character', 'object', 'style']:
            self.assertEqual(intent_digest(project(kind=kind)), intent_digest(project(kind=kind, concept_family='pose')))
        self.assertEqual(project(choices={'framing': 'keep'})['choices']['viewpoint'], 'keep')
        self.assertEqual(project(choices={'pose': 'keep'})['choices']['expression'], 'keep')
        self.assertNotIn('viewpoint', project(concept_family='camera_angle', choices={'framing': 'keep'})['choices'])

    def test_ten_concept_types_omit_only_the_learned_field(self):
        for name, field, detail in [
            ('pose', 'pose', 'kneeling'), ('expression', 'expression', 'smiling'),
            ('camera_angle', 'viewpoint', 'viewed from above'), ('action', 'action', 'lifting a box'),
            ('interaction', 'action', 'passing a box to a person'), ('relationship', 'relationship', 'above a box'),
            ('lighting', 'lighting', 'soft side lighting'), ('composition', 'composition', 'centered'),
            ('focus', 'focus', 'blurred background'), ('effect', 'effect', 'falling snow')]:
            with self.subTest(name=name):
                raw = 'A person wears a blue shirt. The image shows ' + detail + '.'
                response = facts(('subject', 'A person', 1), ('clothing', 'a blue shirt', 1), (field, detail, 2))
                p = project(concept_family=name)
                r = process_response(response, p, raw)
                self.assertNotIn(detail, r['caption'])
                self.assertIn('blue shirt', r['caption'])
                self.assertEqual(r['observations'], raw)
                p['choices'][field] = 'change'
                self.assertIn(detail, process_response(response, p, raw)['caption'])

    def test_narrow_cross_field_guards_and_explicit_overrides(self):
        for family, field, text in [('expression', 'expression', 'a smiling person'),
                                    ('pose', 'pose', 'a kneeling person'),
                                    ('camera_angle', 'viewpoint', 'viewed from above'),
                                    ('focus', 'focus', 'blurred background')]:
            p = project(concept_family=family)
            raw = 'A person is near a wall. The image shows ' + text + '.'
            response = facts(('subject', 'A person', 1), ('background', text, 2))
            with self.assertRaisesRegex(ValueError, 'KEEP'):
                process_response(response, p, raw)
            p['choices'][field] = 'change'
            self.assertIn(text, process_response(response, p, raw)['caption'])

    def test_relationships_are_grounded_without_reversing_roles(self):
        p = project(concept_family='expression')
        raw = 'A red robot pushes a blue cart. The robot is smiling.'
        good = process_response(facts(('action', 'A red robot pushes a blue cart', 1)), p, raw)
        self.assertIn('robot pushes a blue cart', good['caption'])
        bad = process_response(facts(('action', 'A blue cart pushes a red robot', 1)), p, raw)
        self.assertIn('robot pushes a blue cart', bad['caption'])
        self.assertNotIn('cart pushes', bad['caption'])
        with self.assertRaises(ValueError):
            process_response(facts(('subject', 'A second robot', 1)), p, raw)
        with self.assertRaises(ValueError):
            process_response(facts(('subject', 'A robot', 1)), p, 'There may be a robot.')

    def test_subject_cannot_be_lost_or_text_coverage_invented(self):
        p = project(concept_family='camera_angle')
        with self.assertRaisesRegex(ValueError, 'main visible'):
            process_response(facts(('lighting', 'soft light', 1)), p, 'A cup is in soft light.')
        response = json.loads(facts(('subject', 'A cup', 1)))
        response['coverage'] = {'relationship': 'clear'}
        r = process_response(json.dumps(response), p, 'A cup is on a shelf.')
        self.assertEqual(r['coverage'], {})
        self.assertTrue(caption_text_current(p, {**r, 'observation_kind': 'concept'}))
        self.assertFalse(caption_text_current(p, {**r, 'observation_kind': 'concept', 'concept_caption_version': 0}))
        self.assertIn('intent, not proof', caption_instruction(p))

    def test_literal_participant_backstop_retains_count_without_guessing(self):
        p = project(concept_family='pose')
        raw = 'Two people are kneeling near a fence. They wear blue coats.'
        r = process_response(facts(('clothing', 'blue coats', 2)), p, raw)
        self.assertIn('Two people', r['caption'])
        self.assertNotIn('kneeling', r['caption'])
        self.assertTrue(r['caption_facts'][0]['restored'])
        self.assertEqual(r['observations'], raw)
        self.assertIsNone(concepts.participant_anchor('A mannequin wears a coat.'))
        self.assertEqual(concepts.participant_anchor('The shadow of a person is on a wall.'), 'shadow of a person')
        self.assertEqual(concepts.participant_anchor("A person's shadow is on a wall."), "A person's shadow")
        with self.assertRaisesRegex(ValueError, 'main visible'):
            process_response(facts(('clothing', 'blue coats', 2)), p, 'There may be two people. They wear blue coats.')
        p['choices']['subject'] = 'keep'
        self.assertNotIn('people', process_response(facts(('clothing', 'blue coats', 2)), p, raw)['caption'])
        self.assertTrue(concepts.fact_errors('an East Asian person', 'subject', {}))
        self.assertFalse(concepts.fact_errors('a standing lamp', 'background', {'pose': 'keep'}))
        self.assertTrue(concepts.fact_errors('looking at the camera', 'subject', {}))
        self.assertTrue(concepts.fact_errors('centered on', 'composition', {}))
        self.assertFalse(concepts.fact_errors('head-on', 'framing', {}))
        self.assertFalse(concepts.fact_errors('a snowman beside a snowboard', 'subject', {'effect': 'keep'}))
        self.assertFalse(concepts.fact_errors('reflective vest', 'clothing', {'effect': 'keep'}))

    def test_other_concept_details_cannot_leak_through_background(self):
        for family, text in [('action', 'holding a cube'), ('relationship', 'stacked cubes'),
                             ('lighting', 'a silhouette'), ('composition', 'centered'), ('effect', 'falling snow')]:
            p = project(concept_family=family)
            raw = 'A robot is in a room. The image shows ' + text + '.'
            with self.subTest(family=family), self.assertRaisesRegex(ValueError, 'KEEP'):
                process_response(facts(('subject', 'A robot', 1), ('background', text, 2)), p, raw)

    def test_sparse_detailed_observation_retries_then_keeps_an_advisory_draft(self):
        p = project(concept_family='pose')
        raw = ('A person is kneeling near a fence. The person wears a blue coat with yellow buttons and a pair of black boots. '
               'A red bag is beside the fence, and green trees and a brick wall are visible in the background.')
        response = facts(('subject', 'A person', 1))
        with self.assertRaisesRegex(ValueError, 'Only one variable fact'):
            process_response(response, p, raw)
        draft = process_response(response, p, raw, partial=True)
        self.assertTrue(any('very little variable context' in w for w in draft['warnings']))
        self.assertEqual(draft['observations'], raw)
        self.assertTrue(concepts.fact_errors('Both arms are raised overhead', 'action', {'pose': 'keep'}))


class ConceptServiceCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.service = Service(Path(self.temp.name) / 'data')
        self.p = self.service.store.save_project(project(concept_family='camera_angle', sample_prompt_mode='manual', sample_prompts='syntok, a cup'))
        self.i = self.service.import_image(self.p['id'], {'name': 'synthetic.png', 'data': base64.b64encode(fixture()).decode()})['image']
        self.service.save_settings({'integration': 'standalone', 'llm_model': 'synthetic'})
        self.calls = []
        self.raw = 'A red cup is on a shelf. It is viewed from above.'

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
            return json.dumps({'observations': self.raw, 'caption': 'syntok, a red cup viewed from above',
                               'coverage': {'relationship': 'partial'}, 'warnings': ['Target evidence is partial.']})
        return facts(('subject', 'A red cup', 1), ('background', 'on a shelf', 1), ('viewpoint', 'viewed from above', 2))

    def test_connected_prepare_freezes_correct_caption_geometry_and_intent(self):
        with patch('studio.service.standalone_caption', side_effect=self.complete), patch('studio.trainer.backend_status', return_value={'available': True, 'models': {'krea2': True}}):
            j = self.wait(self.service.automatic_prepare(self.p['id']))
        self.assertEqual(j['status'], 'complete', j.get('error'))
        self.assertEqual(self.calls, [True, False])
        i = self.service.dataset(self.p['id'])[0]
        self.assertEqual(i['caption'], 'syntok, A red cup, on a shelf.')
        self.assertTrue(i['advisories'])
        root = self.service.store.root / 'runs' / j['id']
        self.assertEqual((root / 'dataset' / (i['id'] + '.txt')).read_text(), i['caption'])
        frozen = json.loads((root / 'manifest.json').read_text())
        self.assertEqual(frozen['intent']['concept_family'], 'camera_angle')
        self.assertEqual(self.service.image_path(i).read_bytes(), fixture())
        self.service.update_image(i['id'], {'revision': i['revision'], 'caption': 'syntok manual'})
        self.assertNotEqual((root / 'dataset' / (i['id'] + '.txt')).read_text(), 'syntok manual')

    def test_reprocess_history_manual_edits_and_target_coverage_staleness(self):
        with patch('studio.service.standalone_caption', side_effect=self.complete):
            self.assertEqual(self.wait(self.service.caption(self.p['id']))['status'], 'complete')
            self.calls.clear()
            self.assertEqual(self.wait(self.service.caption(self.p['id'], reuse_observations=True))['status'], 'complete')
            self.assertEqual(self.calls, [False])
            i = self.service.dataset(self.p['id'])[0]
            self.assertEqual(i['effective_coverage']['relationship'], 'partial')
            p = self.service.store.save_project({**self.p, 'description': 'Different synthetic target'}, self.p['id'])
            self.assertEqual(effective_coverage(p, i)[0], {})
            self.assertEqual(self.wait(self.service.caption(p['id'], reuse_observations=True))['status'], 'complete')
        i = self.service.dataset(self.p['id'])[0]
        self.assertEqual(i['coverage'], {})
        self.assertTrue(any('target changed' in w for w in i['warnings']))
        self.assertEqual(len(i['history']), 3)
        self.assertEqual(i['observations'], self.raw)
        self.service.update_image(i['id'], {'revision': i['revision'], 'caption': 'syntok manual'})
        with self.assertRaisesRegex(ValueError, 'Manual captions'):
            self.service.caption(self.p['id'], reuse_observations=True)

    def test_type_switch_and_conflict_do_not_overwrite_edits(self):
        p = self.service.store.save_project({**self.p, 'kind': 'style'}, self.p['id'])
        p = self.service.store.save_project({**p, 'kind': 'concept'}, p['id'])
        self.assertEqual(p['concept_family'], 'camera_angle')
        def changing(*args, **kwargs):
            current = self.service.store.project(p['id'])
            self.service.store.save_project({**current, 'concept_family': 'expression'}, p['id'])
            return self.complete(*args, **kwargs)
        with patch('studio.service.standalone_caption', side_effect=changing):
            j = self.wait(self.service.caption(p['id']))
        self.assertTrue(j['messages'])
        self.assertEqual(self.service.store.item('images', self.i['id'])['caption'], '')

    def test_legacy_read_retains_explicit_viewpoint_without_an_implicit_write(self):
        p = {**self.p, 'choices': {'framing': 'keep', 'pose': 'keep'}}
        p.pop('concept_family')
        with self.service.store.connect() as db:
            db.execute('UPDATE projects SET data=? WHERE id=?', (json.dumps(p), p['id']))
        loaded = self.service.store.project(p['id'])
        self.assertEqual(loaded['concept_family'], 'general')
        self.assertEqual(loaded['choices']['viewpoint'], 'keep')
        self.assertEqual(loaded['choices']['expression'], 'keep')
        self.assertEqual(loaded['revision'], p['revision'])
        self.assertEqual(self.service.store.projects()[0]['choices']['viewpoint'], 'keep')
        with self.service.store.connect() as db:
            saved = json.loads(db.execute('SELECT data FROM projects WHERE id=?', (p['id'],)).fetchone()[0])
        self.assertNotIn('concept_family', saved)
