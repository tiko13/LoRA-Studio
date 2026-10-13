import base64,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from studio.service import Service
from studio.store import Conflict
from studio.verification import parse_verification,verification_state,normalize_checks
from tests.fixtures import fixture

ROOT=Path(__file__).resolve().parents[1]/'internal-tests/artifacts/unit';ROOT.mkdir(parents=True,exist_ok=True)
class VerificationCase(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT);self.addCleanup(self.temp.cleanup)
        self.service=Service(Path(self.temp.name)/'data')
        self.project=self.service.store.save_project({'description':'A geometric shape','trigger':'shape','kind':'object'})
        self.image=self.service.import_image(self.project['id'],{'name':'synthetic.png','data':base64.b64encode(fixture()).decode()})['image']
        self.service.update_image(self.image['id'],{'revision':1,'caption':'shape next to a circle','reviewed':True})
        self.service.save_settings({'integration':'standalone','llm_model':'synthetic'})
        self.findings=[{'category':'setting','status':'uncertain','observation':'The surface is outside the frame.'}]
    def wait(self,job):
        for _ in range(300):
            r=self.service.store.item('jobs',job['id'])
            if r['status'] not in {'queued','running'}:return r
            time.sleep(.01)
        self.fail('Job timed out')
    def test_check_never_edits_manual_caption_or_observations(self):
        before=self.service.store.item('images',self.image['id'])
        with patch('studio.service.standalone_caption',return_value=json.dumps({'findings':self.findings})):
            job=self.wait(self.service.verify(self.project['id'],[self.image['id']],['setting']))
        self.assertEqual(job['status'],'complete')
        after=self.service.dataset(self.project['id'])[0]
        for k in ['caption','observations','manual','reviewed','history']:self.assertEqual(before[k],after[k])
        self.assertEqual(after['visual_review']['status'],'concerns')
        self.service.update_image(after['id'],{'revision':after['revision'],'accept_verification':True})
        self.assertEqual(self.service.dataset(self.project['id'])[0]['visual_review']['status'],'overridden')
    def test_edit_and_rule_change_expire_verification_and_override(self):
        i=self.service.dataset(self.project['id'])[0]
        self.service.save_verification(self.project,i,self.findings)
        saved=self.service.dataset(self.project['id'])[0]
        changed={**saved,'caption':'different claim'}
        self.assertEqual(verification_state(self.project,changed)['status'],'stale')
        p={**self.project,'choices':{'shape':'change'}}
        self.assertEqual(verification_state(p,saved)['status'],'stale')
        with self.assertRaisesRegex(ValueError,'outdated'):
            self.service.update_image(saved['id'],{'revision':saved['revision'],'caption':'shape new caption','accept_verification':True})
    def test_concurrent_edit_preserved(self):
        i=self.service.dataset(self.project['id'])[0]
        self.service.update_image(i['id'],{'revision':i['revision'],'caption':'shape changed manually'})
        with self.assertRaises(Conflict):self.service.save_verification(self.project,i,self.findings)
        self.assertNotIn('verification',self.service.store.item('images',i['id']))
    def test_invalid_output_preserves_previous_check(self):
        i=self.service.dataset(self.project['id'])[0];self.service.save_verification(self.project,i,self.findings)
        before=self.service.store.item('images',i['id'])
        with patch('studio.service.standalone_caption',return_value='not JSON'):
            job=self.wait(self.service.verify(self.project['id'],[i['id']],['setting']))
        self.assertEqual(job['status'],'failed');self.assertEqual(before,self.service.store.item('images',i['id']))
    def test_prior_review_concerns_are_not_cleared_by_model_agreement(self):
        i=self.service.dataset(self.project['id'])[0]
        self.service.save_verification(self.project,i,[{**self.findings[0],'status':'supported'}],review_notes=['Check the supporting surface.'])
        self.assertEqual(self.service.dataset(self.project['id'])[0]['visual_review']['status'],'concerns')
    def test_invalid_categories_and_duplicate_findings(self):
        for value in [None,[],['pose','pose'],['unknown']]:
            with self.assertRaises(ValueError):normalize_checks(value)
        with self.assertRaises(ValueError):parse_verification(json.dumps({'findings':self.findings*2}),['setting','pose'])

    def test_legacy_image_without_persisted_dimensions_can_be_checked(self):
        item=self.service.store.item('images',self.image['id']);item.pop('width')
        self.service.store.put_item('images',item)
        with patch('studio.service.standalone_caption',return_value=json.dumps({'findings':self.findings})):
            result=self.wait(self.service.verify(self.project['id'],[item['id']],['setting']))
        self.assertEqual(result['status'],'complete',result.get('error'))
