import unittest

from studio.domain import catalog, normalize_project, intent, intent_digest, caption_instruction
from studio.learning_target import DEFAULT_TARGETS, effective_description
from studio.caption_processing import processing_instruction
from studio.categories import observation_instruction
from studio.object_preparation import instruction as object_instruction


class LearningTargetCase(unittest.TestCase):
    def test_blank_targets_reach_all_caption_paths_without_filling_saved_text(self):
        for kind in DEFAULT_TARGETS:
            with self.subTest(kind=kind):
                project = normalize_project({'kind': kind, 'description': ' \n ', 'trigger': 'syntok'})
                target = DEFAULT_TARGETS[kind]
                self.assertEqual(project['description'], '')
                self.assertEqual(intent(project)['description'], target)
                self.assertIn(target, caption_instruction(project))
                self.assertIn(target, processing_instruction(project, 'A subject beside a wall.'))
                if kind != 'character':
                    self.assertIn(target, observation_instruction(project))
                if kind == 'object':
                    self.assertIn(target, object_instruction(project, {}))
                self.assertEqual(project['description'], '')
                self.assertEqual(next(t for t in catalog()['types'] if t['id'] == kind)['default_description'], target)

    def test_person_likeness_defaults_and_explicit_overrides(self):
        project = normalize_project({'trigger': 'syntok'})
        rules = {a['id']: a['resolved'] for a in intent(project)['attributes']}
        for key in ('identity', 'body', 'hair_color', 'markings'):
            self.assertEqual(rules[key], 'keep')
        for key in ('clothing', 'pose', 'hair_style', 'accessories', 'background', 'lighting', 'framing', 'medium'):
            self.assertEqual(rules[key], 'change')
        overridden = normalize_project({**project, 'choices': {'clothing': 'keep', 'hair_color': 'change'}})
        rules = {a['id']: a['resolved'] for a in intent(overridden)['attributes']}
        self.assertEqual(rules['clothing'], 'keep')
        self.assertEqual(rules['hair_color'], 'change')
        self.assertNotEqual(intent_digest(project), intent_digest(overridden))

    def test_explicit_text_wins_and_switching_type_preserves_it(self):
        project = normalize_project({'description': 'A synthetic target with striped features'})
        for kind in DEFAULT_TARGETS:
            switched = {**project, 'kind': kind}
            self.assertEqual(effective_description(switched), project['description'])
            self.assertEqual(effective_description({**switched, 'description': ''}), DEFAULT_TARGETS[kind])
        self.assertNotEqual(intent_digest(project), intent_digest({**project, 'description': ''}))

    def test_concept_family_changes_policy_with_empty_description(self):
        project = normalize_project({'kind': 'concept', 'concept_family': 'expression'})
        rules = {a['id']: a['resolved'] for a in intent(project)['attributes']}
        self.assertEqual(rules['expression'], 'keep')
        self.assertEqual(rules['action'], 'change')
        self.assertEqual(rules['subject'], 'change')
        self.assertNotEqual(intent_digest(project), intent_digest({**project, 'concept_family': 'pose'}))
