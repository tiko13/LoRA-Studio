import base64
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request
import zipfile

from server import make_server
from studio.domain import normalize_project, intent, intent_digest, parse_caption
from studio.integrations import PromptStudio, local_url
from studio.service import Service
from studio.store import Store, Conflict
from studio.trainer import config_for, export_run, preflight
from tests.live_vision import fixture

TEST_ROOT = Path(__file__).resolve().parents[1] / 'test-results'
TEST_ROOT.mkdir(exist_ok=True)

class StudioCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = Service(self.root/'data')
        self.project = self.service.store.save_project({'name':'Test LoRA','kind':'object','description':'The red shape','trigger':'redshape'})
        self.toolkit=self.root/'toolkit'
        self.toolkit.mkdir()
        (self.toolkit/'run.py').write_text('')
        arch=self.toolkit/'extensions_built_in/diffusion_models/krea2'
        arch.mkdir(parents=True)
        (arch/'krea2.py').write_text('arch = "krea2"')
        self.settings=self.service.save_settings({'integration':'standalone','llm_model':'vision','toolkit_path':str(self.toolkit),'toolkit_python':sys.executable})
    def image(self):
        return self.service.import_image(self.project['id'],{'name':'fixture.png','data':base64.b64encode(fixture()).decode()})['image']
    def ready_image(self):
        image=self.image()
        return self.service.update_image(image['id'],{'revision':1,'caption':'redshape next to a blue circle','reviewed':True})
    def wait(self, job):
        end=time.time()+10
        while time.time()<end:
            value=self.service.store.item('jobs',job['id'])
            if value['status'] not in {'queued','running'}:
                return value
            time.sleep(.02)
        self.fail('Job did not finish')
    def test_store_revision_conflict_and_hidden_choices(self):
        p={**self.project,'choices':{'hair_color':'change','shape':'keep'}}
        saved=self.service.store.save_project(p,p['id'])
        self.assertEqual(saved['choices']['hair_color'],'change')
        with self.assertRaises(Conflict):self.service.store.save_project(p,p['id'])
        self.assertEqual(Store(self.root/'data').project(p['id']),saved)
    def test_invalid_training_parameters(self):
        for value in [True,float('nan'),float('inf'),0,20001,100.5]:
            with self.subTest(value=value),self.assertRaises(ValueError):normalize_project({'training':{'steps':value}})
        with self.assertRaises(ValueError):normalize_project({'training':{'resolution':777}})
    def test_intent_digest_tracks_rules_not_name(self):
        p={**self.project,'name':'New display name'}
        self.assertEqual(intent_digest(p),intent_digest(self.project))
        p['choices']={'shape':'change'}
        self.assertNotEqual(intent_digest(p),intent_digest(self.project))
        self.assertEqual(next(a for a in intent(p)['attributes'] if a['id']=='shape')['resolved'],'change')
    def test_caption_json_and_trigger_validation(self):
        result=parse_caption('```json\n{"caption":"a red shape", "observations":"Two shapes", "warnings":[]}\n```',self.project)
        self.assertTrue(result['warnings'])
        for value in ['not json','[]','{"caption":"redshape","observations":""}']:
            with self.assertRaises(ValueError):parse_caption(value,self.project)
    def test_duplicate_import_and_revision_protection(self):
        image=self.image()
        self.assertEqual(image['id'],self.image()['id'])
        self.assertEqual(len(self.service.store.items('images',self.project['id'])),1)
        self.service.update_image(image['id'],{'revision':1,'included':False})
        with self.assertRaises(Conflict):self.service.update_image(image['id'],{'revision':1,'caption':'obsolete'})
    def test_preflight_stale_warning_and_rule_conflict(self):
        image=self.ready_image()
        self.assertFalse(preflight(self.project,[image],self.settings)['errors'])
        p={**self.project,'also_keep':'background','also_change':'BACKGROUND'}
        errors=preflight(p,[image],self.settings)['errors']
        self.assertTrue(any('both columns' in e for e in errors))
        self.assertTrue(any('older intent' in e for e in errors))
        image.update(warnings=['Ambiguous target'],reviewed=False)
        self.assertTrue(any('warnings' in e for e in preflight(self.project,[image],self.settings)['errors']))
    def test_manual_caption_is_not_regenerated(self):
        self.ready_image()
        with self.assertRaisesRegex(ValueError,'Manual captions are preserved'):self.service.caption(self.project['id'])
    def test_manual_edit_requires_fresh_review(self):
        image=self.ready_image()
        image=self.service.update_image(image['id'],{'revision':image['revision'],'caption':'redshape edited'})
        self.assertFalse(image['reviewed'])
        self.assertNotIn('intent_digest',image)
        image=self.service.update_image(image['id'],{'revision':image['revision'],'reviewed':True})
        self.assertTrue(image['reviewed'])
        self.assertEqual(image['intent_digest'],intent_digest(self.project))
    def test_interrupted_run_recovery_checks_process(self):
        job={'id':'old-run','project':self.project['id'],'kind':'train','status':'interrupted','pid':12345}
        self.service.store.put_item('jobs',job)
        with patch('studio.service.running',return_value=True),self.assertRaisesRegex(ValueError,'still running'):
            self.service.recover(job['id'])
        with patch('studio.service.running',return_value=False):
            self.assertEqual(self.service.recover(job['id'])['status'],'failed')
    def test_caption_does_not_overwrite_concurrent_manual_edit(self):
        image=self.image()
        entered,release=threading.Event(),threading.Event()
        def completion(*args):
            entered.set();release.wait(3)
            return json.dumps({'caption':'redshape on beige','observations':'Red shape','warnings':[]})
        with patch('studio.service.standalone_caption',side_effect=completion):
            job=self.service.caption(self.project['id'])
            self.assertTrue(entered.wait(2))
            self.service.update_image(image['id'],{'revision':1,'caption':'redshape manual caption','reviewed':True})
            with self.assertRaises(Conflict):self.service.new_job(self.project['id'],'export',lambda *a:None)
            release.set()
            result=self.wait(job)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(self.service.store.item('images',image['id'])['caption'],'redshape manual caption')
        self.assertTrue(result['messages'])
    def test_snapshot_is_immutable_and_zip_portable(self):
        image=self.ready_image()
        with patch('studio.trainer.subprocess.check_output',return_value='test-commit'):
            root=export_run(self.service.store,self.project,[image],'snapshot',self.settings)
        self.service.update_image(image['id'],{'revision':image['revision'],'caption':'redshape edited later','reviewed':True})
        self.assertEqual((root/'dataset'/f"{image['id']}.txt").read_text(),image['caption'])
        with zipfile.ZipFile(root/'training-bundle.zip') as archive:
            config=json.loads(archive.read('config.json'))
            self.assertEqual(config['config']['process'][0]['datasets'][0]['folder_path'],'./dataset')
        self.assertEqual(json.loads((root/'manifest.json').read_text())['backend_commit'],'test-commit')
    def test_primary_model_architectures(self):
        for model,arch in [('qwen21','qwen_image_2'),('krea2','krea2')]:
            p={**self.project,'model':model}
            cfg=config_for(p,'dataset','output','test')['config']['process'][0]
            self.assertEqual(cfg['model']['arch'],arch)
            self.assertEqual(cfg['train']['noise_scheduler'],'flowmatch')
    def test_fake_trainer_failure_is_reported(self):
        self.ready_image()
        (self.toolkit/'run.py').write_text('import sys\nprint("deliberate test failure")\nsys.exit(3)\n')
        result=self.wait(self.service.prepare(self.project['id'],train=True))
        self.assertEqual(result['status'],'failed')
        self.assertIn('code 3',result['error'])
        self.assertIn('deliberate test failure',(self.service.store.root/'runs'/result['id']/'training.log').read_text())
    def test_fake_trainer_empty_success_is_failure(self):
        self.ready_image()
        result=self.wait(self.service.prepare(self.project['id'],train=True))
        self.assertEqual(result['status'],'failed')
        self.assertIn('without a LoRA',result['error'])
    def test_trainer_output_is_exposed_after_completion(self):
        self.ready_image()
        (self.toolkit/'run.py').write_text('import json, sys\nfrom pathlib import Path\ncfg=json.load(open(sys.argv[1]))\nout=Path(cfg["config"]["process"][0]["training_folder"])\n(out/"test.safetensors").write_bytes(b"test fixture, not model weights")\n')
        result=self.wait(self.service.prepare(self.project['id'],train=True))
        self.assertEqual(result['status'],'complete')
        self.assertIn('output/test.safetensors',{f['path'] for f in self.service.run_files(result['id'])})
    def test_stop_terminates_owned_trainer(self):
        self.ready_image()
        (self.toolkit/'run.py').write_text('import time\ntime.sleep(60)\n')
        job=self.service.prepare(self.project['id'],train=True)
        deadline=time.time()+5
        while job['id'] not in self.service.processes and time.time()<deadline:
            time.sleep(.02)
        self.assertIn(job['id'],self.service.processes)
        process=self.service.processes[job['id']]
        self.service.stop(job['id'])
        self.assertEqual(self.wait(job)['status'],'cancelled')
        self.assertIsNotNone(process.poll())
    def test_restart_marks_active_work_interrupted(self):
        self.service.store.put_item('jobs',{'id':'interrupted','project':self.project['id'],'kind':'train','status':'running'})
        service=Service(self.root/'data')
        self.assertEqual(service.store.item('jobs','interrupted')['status'],'interrupted')
    def test_local_endpoints_only(self):
        for url in ['https://localhost:8188','http://example.com','http://localhost@evil.test','http://localhost:8188?x=1']:
            with self.assertRaises(ValueError):local_url(url)
    def test_comfy_busy_blocks_unload(self):
        with patch('studio.integrations.request',side_effect=[{'queue_running':[['x']]},{'jobs':[]}]) as request:
            with self.assertRaisesRegex(ValueError,'active or queued'):PromptStudio('http://localhost:8188').free_generation_memory()
            self.assertEqual(request.call_count,2)
    def test_comfy_idle_requests_unload(self):
        with patch('studio.integrations.request',side_effect=[{'queue_running':[],'queue_pending':[]},{'jobs':[]},{}]) as request:
            PromptStudio('http://localhost:8188').free_generation_memory()
            self.assertEqual(request.call_args.args,('http://localhost:8188/free',{'unload_models':True,'free_memory':True}))

class HTTPCase(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=TEST_ROOT)
        self.server=make_server(0,Path(self.temp.name))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}'
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.temp.cleanup()
    def test_routes_and_cross_origin_write(self):
        with urllib.request.urlopen(self.url+'/') as r:
            self.assertIn(b'LoRA Studio',r.read())
            self.assertIn("script-src 'self'",r.headers['Content-Security-Policy'])
        req=urllib.request.Request(self.url+'/api/projects',data=b'{}',headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req) as r:self.assertEqual(r.status,201)
        req.add_header('Origin','https://malicious.invalid')
        with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(req)
        self.assertEqual(error.exception.code,403)
        error.exception.close()
    def test_path_traversal(self):
        with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(self.url+'/%2e%2e/server.py')
        self.assertEqual(error.exception.code,400)
        error.exception.close()

if __name__=='__main__':unittest.main()
