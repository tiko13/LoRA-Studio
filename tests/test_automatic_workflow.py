import base64,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch,MagicMock
from studio.service import Service
from studio.domain import normalize_project,intent_digest
from studio.trainer import preflight
from tests.fixtures import fixture
ROOT=Path(__file__).resolve().parents[1]/'internal-tests/artifacts/unit';ROOT.mkdir(parents=True,exist_ok=True)
class AutomaticCase(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.service=Service(self.root/'data')
        self.project=self.service.store.save_project({'description':'A geometric shape','trigger':'shape','kind':'object'})
        self.image=self.service.import_image(self.project['id'],{'name':'synthetic.png','data':base64.b64encode(fixture()).decode()})['image']
        # Caption workflow tests explicitly keep the full synthetic scene; targeting has its own suite.
        self.image['object_target'] = {'mode': 'full'}
        self.service.store.put_item('images', self.image)
        self.service.save_settings({'integration':'standalone','llm_model':'synthetic','toolkit_path':str(self.root),'toolkit_python':'synthetic-python'})
        self.addCleanup(patch.stopall)
        from studio.sample_prompts import effective
        patch('studio.sample_prompts.generate',side_effect=lambda p,i,s:effective(p,i)).start()
        patch('studio.trainer.backend_status',return_value={'available':True,'models':{'krea2':True}}).start()
        self.infer=patch('studio.service.standalone_caption',return_value=json.dumps({'caption':'shape beside a circle','observations':'An object beside a circle.','subject':'object','facts':[{'attribute':'placement','text':'beside a circle','sentence':1}],'warnings':['Ambiguous surface.']})).start()
        self.captured=[]
        def export(store,project,images,run_id,settings):
            self.captured.append((project,images,settings))
            root=self.root/'bundle';root.mkdir();(root/'output').mkdir()
            (root/'output/synthetic.safetensors').write_bytes(b'test only')
            (root/'config.json').write_text('{}');return root
        self.export=patch('studio.service.export_run',side_effect=export).start()
        process=MagicMock(pid=123,returncode=0);process.poll.return_value=0
        self.launch=patch('studio.service.subprocess.Popen',return_value=process).start()
    def wait(self,job):
        for _ in range(500):
            r=self.service.store.item('jobs',job['id'])
            if r['status'] not in {'queued','running'}:return r
            time.sleep(.01)
        self.fail('Job timed out')
    def test_one_job_prepares_then_freezes_then_launches_mock_trainer(self):
        result=self.wait(self.service.automatic_train(self.project['id']))
        self.assertEqual(result['status'],'complete',result.get('error'))
        self.assertEqual(len(self.service.store.items('jobs')),1)
        self.assertEqual(self.infer.call_count,2);self.assertEqual(self.launch.call_count,1)
        self.assertEqual(self.captured[0][1][0]['caption'],'shape, object, beside a circle.')
        self.assertFalse(self.captured[0][1][0]['reviewed'])
        self.assertTrue(self.captured[0][1][0]['warnings'])
        self.assertIn('automatic',result['review_policy'])
    def test_caption_failure_never_launches(self):
        self.infer.side_effect=ValueError('Synthetic inference failure')
        result=self.wait(self.service.automatic_train(self.project['id']))
        self.assertEqual(result['status'],'failed');self.launch.assert_not_called();self.export.assert_not_called()
    def test_empty_person_description_reaches_mock_training_and_stays_empty(self):
        self.project = self.service.store.save_project({**self.project, 'kind': 'character', 'description': ''}, self.project['id'])
        self.infer.return_value = json.dumps({'caption': 'shape person standing beside a wall',
            'observations': 'A person standing beside a wall.', 'warnings': [], 'coverage': {}})
        result = self.wait(self.service.automatic_train(self.project['id']))
        self.assertEqual(result['status'], 'complete', result.get('error'))
        self.launch.assert_called_once()
        self.assertIn('Learn the likeness', self.infer.call_args.args[1])
        self.assertEqual(self.captured[0][0]['description'], '')
        self.assertEqual(self.service.store.project(self.project['id'])['description'], '')

    def test_empty_description_allows_preview_generation(self):
        self.project = self.service.store.save_project({**self.project, 'description': ''}, self.project['id'])
        result = self.wait(self.service.generate_preview_suite(self.project['id']))
        self.assertEqual(result['status'], 'complete', result.get('error'))
    def test_prepare_only_runs_same_pipeline_without_trainer(self):
        result=self.wait(self.service.automatic_prepare(self.project['id']))
        self.assertEqual(result['status'],'complete',result.get('error'))
        self.assertEqual(result['kind'],'prepare')
        self.assertEqual(result['phase'],'Dataset prepared')
        self.assertTrue(result['bundle']);self.launch.assert_not_called()
        self.assertEqual(self.infer.call_count,2);self.assertEqual(len(self.captured),1)
        self.assertEqual(self.captured[0][1][0]['caption'],'shape, object, beside a circle.')
    def test_automatic_refreshes_old_text_from_observations_without_new_vision(self):
        image=self.service.dataset(self.project['id'])[0]
        raw='An object beside a circle.'
        image.update(caption='shape old draft',observations=raw,caption_source='custom',caption_formatter_version=1,observation_kind='object',
            intent_digest=intent_digest(self.project),caption_preparation_digest=image['plan']['digest'],
            observation_preparation_digest=image['plan']['digest'])
        self.service.store.put_item('images',image)
        self.infer.return_value=json.dumps({'subject':'object','facts':[{'attribute':'placement','text':'beside a circle','sentence':1}],'warnings':[]})
        result=self.wait(self.service.automatic_prepare(self.project['id']))
        self.assertEqual(result['status'],'complete',result.get('error'))
        self.launch.assert_not_called();self.assertEqual(self.infer.call_count,1)
        self.assertIsNone(self.infer.call_args.args[2])
        saved=self.captured[0][1][0]
        self.assertEqual(saved['observations'],raw);self.assertEqual(saved['caption'],'shape, object, beside a circle.')
        self.assertEqual(saved['history'][-1]['caption'],'shape old draft')

    def test_automatic_refreshes_legacy_person_oriented_observations_for_object(self):
        image=self.service.dataset(self.project['id'])[0]
        image.update(caption='shape old draft',observations='Old untyped observations.',caption_source='custom',
            intent_digest=intent_digest(self.project),caption_preparation_digest=image['plan']['digest'])
        self.service.store.put_item('images',image)
        self.service.save_settings({'caption_source':'custom','caption_model':'mock-captioner'})
        def complete(settings,prompt,data,mime,**kwargs):
            if data is not None:
                self.assertIn('target object class',prompt)
                return 'An object beside a circle.'
            return json.dumps({'subject':'object','facts':[{'attribute':'placement','text':'beside a circle','sentence':1}],'warnings':[]})
        self.infer.side_effect=complete
        result=self.wait(self.service.automatic_prepare(self.project['id']))
        self.assertEqual(result['status'],'complete',result.get('error'))
        self.assertEqual(self.infer.call_count,2)
        self.assertEqual(self.captured[0][1][0]['observation_kind'],'object')
        self.launch.assert_not_called()
        self.assertEqual(self.captured[0][1][0]['history'][-1]['caption'],'shape old draft')
    def test_cancel_during_caption_does_not_export_or_launch(self):
        def infer(*a,**kw):
            next(iter(self.service.cancel.values())).set()
            return json.dumps({'caption':'shape beside a circle','observations':'Two shapes.','warnings':[]})
        self.infer.side_effect=infer
        result=self.wait(self.service.automatic_train(self.project['id']))
        self.assertEqual(result['status'],'cancelled');self.launch.assert_not_called();self.export.assert_not_called()
    def test_concurrent_project_edit_prevents_launch(self):
        original=self.infer.return_value
        def infer(*a,**kw):
            self.service.store.save_project({**self.project,'description':'Changed target'},self.project['id']);return original
        self.infer.side_effect=infer
        result=self.wait(self.service.automatic_train(self.project['id']))
        self.assertEqual(result['status'],'failed');self.launch.assert_not_called();self.export.assert_not_called()
    def test_stale_manual_caption_is_not_rewritten_or_bypassed(self):
        self.service.update_image(self.image['id'],{'revision':1,'caption':'shape manual'})
        result=self.wait(self.service.automatic_train(self.project['id']))
        self.assertEqual(result['status'],'failed');self.infer.assert_not_called();self.launch.assert_not_called()
    def test_review_first_still_requires_caption_and_no_job_is_created(self):
        self.service.store.save_project({**self.project,'training_mode':'review'},self.project['id'])
        with self.assertRaisesRegex(ValueError,'need captions'):self.service.automatic_train(self.project['id'])
        self.assertEqual(self.service.store.items('jobs'),[])
    def test_legacy_mode_and_validation(self):
        self.assertEqual(normalize_project({'id':'existing'})['training_mode'],'review')
        self.assertEqual(normalize_project({})['training_mode'],'automatic')
        with self.assertRaises(ValueError):normalize_project({'training_mode':'anything'})
    def test_automatic_does_not_bypass_backend_or_trigger_errors(self):
        images=self.service.dataset(self.project['id'])
        with patch('studio.trainer.backend_status',return_value={'available':False}):
            errors=preflight(self.project,images,self.service.settings(),automatic=True,preparing=True)['errors']
        self.assertTrue(any('AI Toolkit' in e for e in errors))
        self.assertTrue(preflight({**self.project,'trigger':''},images,self.service.settings(),automatic=True,preparing=True)['errors'])
