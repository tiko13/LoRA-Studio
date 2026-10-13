import json
import unittest
from studio.domain import normalize_project
from studio.caption_processing import process_response, format_observations, required_exposure, unrelated_facts, grounded_phrase

class ProcessingCase(unittest.TestCase):
    def setUp(self):
        self.project=normalize_project({'description':'Synthetic person study','trigger':'subjectx'})
        self.raw='A person with brown hair and freckles stands near a wall. The breasts and nipples are exposed. The vulva is visible. The person wears a silver necklace.'
    def response(self, *facts):
        return json.dumps({'subject':'person','facts':[{'attribute':a,'text':t,'sentence':n} for a,t,n in facts], 'warnings':[]})
    def valid(self):
        return self.response(('pose','standing near a wall',1),('exposure','with breasts and nipples exposed',2),('exposure','vulva visible',3))
    def test_keep_categories_are_filtered_but_exposure_is_not_body_shape(self):
        r=process_response(self.response(('hair_color','with brown hair',1),('markings','with freckles',1),
            ('body','with a slender body',1),('exposure','with breasts and nipples exposed',2),('exposure','vulva visible',3),('pose','standing near a wall',1)),self.project,self.raw)
        self.assertNotIn('brown',r['caption']);self.assertNotIn('freckles',r['caption']);self.assertNotIn('slender',r['caption'])
        self.assertIn('vulva visible',r['caption']);self.assertEqual(r['observations'],self.raw)
    def test_unrelated_scene_triggers_retry_instead_of_accepting_valid_sentence_numbers(self):
        calls=[]
        def complete(prompt):
            calls.append(prompt)
            return self.response(('background','snowy mountains surround frozen lakes beneath storm clouds and distant pine forests',1)) if len(calls)==1 else self.valid()
        result=format_observations(self.project,self.raw,complete)
        self.assertEqual(len(calls),2);self.assertNotIn('mountains',result['caption'])
        self.assertIn('source vocabulary',calls[1])
    def test_lexical_guard_is_not_a_semantic_correctness_check(self):
        self.assertFalse(unrelated_facts([{'text':'lying next to the white chair'}],'A person lies beside a white chair.'))
        self.assertFalse(unrelated_facts([{'text':'right hand raised'}],'The left hand is raised.'))
    def test_nude_is_not_a_substitute_for_explicit_visibility(self):
        result=process_response(self.response(('exposure','nude',5)),self.project,self.raw+' The person is nude.')
        self.assertIn('vulva visible',result['caption'])
        self.assertEqual(set(result['caption_restored_exposure']),{'breasts','nipples','vulva'})
    def test_negation_and_uncertainty_never_force_exposure(self):
        for raw in ['The breasts are not visible.','The vulva may be exposed.','No nipples are visible.','The genitals are hidden.']:
            self.assertEqual(required_exposure(raw,self.project),[])
    def test_keep_clothing_is_respected(self):
        self.project['choices']['clothing']='keep'
        r=process_response(self.response(('exposure','with breasts exposed',2),('pose','standing near a wall',1)),self.project,self.raw)
        self.assertNotIn('breasts',r['caption'])
    def test_change_hair_color_and_markings_are_not_filtered(self):
        self.project['choices'].update(hair_color='change',markings='change')
        r=process_response(self.response(('hair_color','with brown hair',1),('markings','with freckles',1),('exposure','with breasts and nipples exposed',2),('exposure','vulva visible',3)),self.project,self.raw)
        self.assertIn('brown hair',r['caption']);self.assertIn('freckles',r['caption'])
    def test_misclassified_stable_traits_trigger_targeted_retry(self):
        calls=[]
        def complete(prompt):
            calls.append(prompt)
            return self.response(('hair_style','with straight brown hair',1),('exposure','with breasts and nipples exposed',2),('exposure','vulva visible',3)) if len(calls)==1 else self.valid()
        r=format_observations(self.project,self.raw,complete)
        self.assertEqual(len(calls),2);self.assertIn('Remove hair colour',calls[1]);self.assertNotIn('brown',r['caption'])
    def test_missing_or_fabricated_evidence_number_is_rejected(self):
        for number in [0,99,True,'1']:
            with self.subTest(number=number),self.assertRaisesRegex(ValueError,'numbered'):
                process_response(self.response(('pose','standing',number)),self.project,self.raw)
    def test_invalid_responses_stop_after_three_attempts(self):
        calls=[]
        def complete(prompt):calls.append(prompt);return self.response(('hair_style','brown hair',1))
        with self.assertRaisesRegex(ValueError,'Previous caption preserved'):
            format_observations(self.project,self.raw,complete)
        self.assertEqual(len(calls),3)
    def test_custom_rules_get_separate_pass_and_same_validation(self):
        self.project['also_keep']='Necklace'
        calls=[]
        def complete(prompt):
            calls.append(prompt)
            return self.response(('accessories','wearing a silver necklace',4),('exposure','with breasts and nipples exposed',2),('exposure','vulva visible',3)) if len(calls)==1 else self.valid()
        r=format_observations(self.project,self.raw,complete)
        self.assertEqual(len(calls),2);self.assertIn('Apply ONLY',calls[1]);self.assertNotIn('necklace',r['caption'])
    def test_custom_pass_cannot_drop_explicit_exposure(self):
        self.project['also_change']='Wall texture when known'
        calls=[]
        def complete(prompt):
            calls.append(prompt)
            return self.valid() if len(calls)==1 else self.response(('pose','standing',1))
        result=format_observations(self.project,self.raw,complete)
        self.assertIn('vulva visible',result['caption'])
        self.assertEqual(len(calls),2)
    def test_lettering_claim_is_rejected_but_texture_is_allowed(self):
        with self.assertRaisesRegex(ValueError,'lettering'):
            process_response(self.response(('clothing','wearing a shirt with the text HELLO',1)),self.project,'A shirt with text is visible.')
        result=process_response(self.response(('clothing','a textured shirt',1)),self.project,'A textured shirt is visible.')
        self.assertIn('textured shirt',result['caption'])
        with self.assertRaisesRegex(ValueError,'lettering'):
            process_response(self.response(('clothing','with text printed on the shirt',1)),self.project,'A shirt with printed text is visible.')
        with self.assertRaisesRegex(ValueError,'lettering'):
            process_response(self.response(('clothing','wearing a shirt with white block letters',1)),self.project,'White block letters are printed on the shirt.')

    def test_anatomical_appearance_cannot_leak_into_exposure(self):
        with self.assertRaisesRegex(ValueError,'size and colour'):
            process_response(self.response(('exposure','with small pink areolas and nipples exposed',1)),self.project,'Areolas and nipples are visible.')

    def test_hairstyle_fragment_keeps_its_subject_and_duplicate_nudity_is_removed(self):
        r=process_response(self.response(('hair_style','spread around the head',1),('clothing','nude',2),('exposure','naked',2)),self.project,'Hair is spread around the head. The person is nude.')
        self.assertIn('hair spread',r['caption']);self.assertNotIn('naked',r['caption'])

    def test_generic_genitals_do_not_invent_specific_anatomy(self):
        r=process_response(self.response(('pose','standing',1)),self.project,'A person stands with genitals exposed.')
        self.assertIn('genitals visible',r['caption']);self.assertNotIn('vulva',r['caption']);self.assertNotIn('penis',r['caption'])

    def test_scrotum_does_not_mean_visible_testicles(self):
        r=process_response(self.response(('pose','standing',1)),self.project,'A person stands with scrotum visible.')
        self.assertIn('scrotum visible',r['caption']);self.assertNotIn('testicles',r['caption'])

    def test_body_shape_cannot_hide_in_exposure_phrase(self):
        with self.assertRaisesRegex(ValueError,'body proportions'):
            process_response(self.response(('exposure','with flat stomach exposed',1)),self.project,'The abdomen is uncovered.')
        r=process_response(self.response(('exposure','with abdomen uncovered',1)),self.project,'The abdomen is uncovered.')
        self.assertIn('abdomen uncovered',r['caption'])

    def test_individual_invention_is_rejected_even_with_mostly_correct_scene(self):
        with self.assertRaisesRegex(ValueError,'cited source'):
            process_response(self.response(('pose','standing near a wall',1),('framing','full-body view',1)),self.project,self.raw)

    def test_wrong_laterality_or_reordered_relation_is_rejected(self):
        raw='The left hand rests on the red chair beside a blue table.'
        for phrase in ['right hand resting on the red chair','left hand resting on the green chair']:
            with self.subTest(phrase=phrase),self.assertRaisesRegex(ValueError,'cited source'):
                process_response(self.response(('pose',phrase,1)),self.project,raw)

    def test_exposure_cannot_merge_anatomy_from_different_sentences(self):
        with self.assertRaisesRegex(ValueError,'cited source'):
            process_response(self.response(('exposure','breasts and vulva exposed',2)),self.project,self.raw)

    def test_qualifiers_cannot_be_dropped(self):
        for raw in ['No breasts are exposed.','Possibly the breasts are exposed.','The breasts are partly exposed.']:
            with self.subTest(raw=raw):
                result=process_response(self.response(('exposure','breasts exposed',1)),self.project,raw)
                self.assertTrue(result['caption_facts'][0]['source_repaired'])
                self.assertTrue(any(q in result['caption'].lower() for q in ['no breasts','possibly','partly']))
        self.assertTrue(grounded_phrase('looking at the camera','Legs are slightly apart and she is looking at the camera.',{}))

    def test_partial_coverage_is_restored_without_inventing_visibility(self):
        raw='A person is reclining. A hand is partially covering the vulva.'
        result=process_response(self.response(('pose','reclining',1)),self.project,raw)
        self.assertIn('vulva partially covered',result['caption']);self.assertNotIn('vulva visible',result['caption'])
        self.assertEqual(result['caption_restored_exposure'],['vulva'])

    def test_partial_visibility_is_not_restored_as_full_exposure(self):
        raw='A person is standing. The breasts are partially visible.'
        result=process_response(self.response(('pose','standing',1)),self.project,raw)
        self.assertIn('breasts partially visible',result['caption'])

    def test_last_retry_keeps_grounded_facts_and_records_omissions_for_review(self):
        calls=[]
        def complete(prompt):
            calls.append(prompt)
            return self.response(('pose','standing near a wall',1),('framing','full-body view',1))
        result=format_observations(self.project,self.raw,complete)
        self.assertEqual(len(calls),3);self.assertNotIn('full-body',result['caption'])
        self.assertTrue(result['warnings']);self.assertEqual(len(result['caption_omitted_facts']),1)
        self.assertEqual(result['caption_facts'][0]['source_sentence'],self.raw.split('. ')[0]+'.')

    def test_no_grounded_facts_still_preserves_previous_caption(self):
        with self.assertRaisesRegex(ValueError,'Previous caption preserved'):
            format_observations(self.project,self.raw,lambda p:self.response(('framing','full-body view',1)))

    def test_last_retry_omits_miscategorized_keep_traits_without_losing_valid_pose(self):
        response=self.response(('pose','standing near a wall',1),('extra','brown hair',1))
        result=format_observations(self.project,self.raw,lambda p:response)
        self.assertNotIn('brown',result['caption']);self.assertIn('standing near a wall',result['caption'])
        self.assertIn('hair colour',result['caption_omitted_facts'][0]['reason'])
        self.assertTrue(result['warnings'])

    def test_harmless_reordering_uses_source_words_in_source_order(self):
        result=process_response(self.response(('lighting','bright lighting',1)),self.project,'The lighting is bright.')
        self.assertIn('lighting is bright',result['caption'])
        self.assertEqual(result['caption_facts'][0]['proposed_text'],'bright lighting')
        self.assertTrue(result['caption_facts'][0]['source_repaired'])

    def test_wrong_relation_is_replaced_by_source_relation_not_accepted_as_written(self):
        result=process_response(self.response(('pose','chair on the left hand',1)),self.project,'The left hand rests on the red chair.')
        self.assertNotIn('chair on',result['caption']);self.assertIn('left hand rests on the red chair',result['caption'])

    def test_anatomy_noun_alone_does_not_count_as_preserved_visibility(self):
        result=process_response(self.response(('exposure','nipples',2),('pose','standing',1)),self.project,self.raw)
        self.assertIn('nipples visible',result['caption']);self.assertNotIn(', nipples,',result['caption'])

    def test_predicate_body_proportion_is_kept_implicit(self):
        with self.assertRaisesRegex(ValueError,'body proportions'):
            process_response(self.response(('exposure','breasts are small with visible nipples',1)),self.project,'The breasts are small with visible nipples.')

    def test_unknown_attribute_is_omitted_after_retries_not_guessed(self):
        response=self.response(('pose','standing',1),('made_up_field','near a wall',1))
        result=format_observations(self.project,self.raw,lambda p:response)
        self.assertIn('standing',result['caption']);self.assertNotIn('near a wall',result['caption'])
        self.assertIn('Unknown attribute',result['caption_omitted_facts'][0]['reason'])

    def test_punctuation_does_not_bypass_keep_rules(self):
        with self.assertRaisesRegex(ValueError,'hair colour'):
            process_response(self.response(('hair_style','brown, wavy hair',1)),self.project,'The person has brown, wavy hair.')

    def test_adjacent_caption_fields_do_not_create_false_hair_colour_claim(self):
        result=process_response(self.response(('hair_style','long hair',1),('background','white wall',2)),self.project,'The person has long hair. A white wall is behind the person.')
        self.assertIn('long hair, white wall',result['caption'])

    def test_explicit_clothing_state_survives_text_omission_without_new_anatomy(self):
        result=process_response(self.response(('pose','lying on a bench',1)),self.project,'A nude person is lying on a bench.')
        self.assertIn('nude',result['caption']);self.assertNotIn('breasts',result['caption'])
        result=process_response(self.response(('pose','standing',1)),self.project,'A person is standing. The person may be nude.')
        self.assertNotIn('nude',result['caption'])
        result=process_response(self.response(('pose','standing',1)),self.project,'A person is standing in a nude-colored dress.')
        self.assertNotIn('nude',result['caption'])
