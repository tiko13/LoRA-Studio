"""Synthetic object targeting, immutable geometry, masks and workflow checks."""
import base64
import io
import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from studio import object_preparation as obj
from studio.domain import intent_digest
from studio.preparation import plan, render
from studio.service import Service
from studio.store import Conflict
from studio.trainer import export_run, preflight
from studio.caption_processing import process_response
from tests.fixtures import fixture

ROOT = Path(__file__).resolve().parents[1] / 'internal-tests/artifacts/unit'
ROOT.mkdir(parents=True, exist_ok=True)


class ObjectPreparationCase(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.service = Service(Path(temp.name)/'data')
        self.project = self.service.store.save_project({'kind':'object','description':'The red rectangle','trigger':'synth',
                                                       'sample_prompt_mode':'manual','sample_prompts':'synth on a table'})
        self.image = self.service.import_image(self.project['id'],{'name':'synthetic.png','data':base64.b64encode(fixture()).decode()})['image']
        self.service.save_settings({'integration':'standalone','llm_model':'synthetic'})
        self.backend = patch('studio.trainer.backend_status',return_value={'available':True,'models':{'krea2':True}})
        self.backend.start();self.addCleanup(self.backend.stop)

    def wait(self,job):
        for _ in range(1000):
            current=self.service.store.item('jobs',job['id'])
            if current['status'] not in {'queued','running'}:return current
            time.sleep(.01)
        self.fail('Synthetic job timed out')

    def found(self,status='found'):
        return json.dumps({'status':status,'box':[137,234,459,716] if status=='found' else None,
                           'evidence':'The red rectangle at the left','occluded':False,'truncated':False,'other_objects_inside':False})

    def target(self,**value):
        item=self.service.store.item('images',self.image['id'])
        return self.service.update_image(item['id'],{'revision':item['revision'],'object_target':value})

    def test_bad_coordinates_and_masks_are_rejected(self):
        for b in [[0,0,0,1],[0,0,2,1],[False,0,1,1],[0,float('nan'),1,1],[0,1,1,0]]:
            with self.subTest(b=b),self.assertRaises(ValueError):obj.options({'mode':'manual','box':b})
        for value in [{'mask':'polygon','polygon':[[0,0],[1,1]]},{'background_weight':2},{'margin':float('inf')},{'query':[]},{'mask':'alpha','mode':'manual'}]:
            with self.subTest(value=value),self.assertRaises(ValueError):obj.options(value)
        with self.assertRaises(ValueError):obj.parse(self.found().replace('137','-137'),self.project,self.image)

    def test_crop_excludes_other_shape_and_is_identical_in_preview_and_export(self):
        image=self.target(mode='manual',box=[70/512,90/384,235/512,275/384],margin=.05)
        image=self.service.update_image(image['id'],{'revision':image['revision'],'caption':'synth rectangle on a plain surface','reviewed':True})
        preview=self.service.preview(image['id'])
        root=export_run(self.service.store,self.project,[image],'synthetic',self.service.settings())
        exported=(root/'dataset'/(image['id']+'.png')).read_bytes()
        self.assertEqual(base64.b64decode(preview['png']),exported)
        pixels=Image.open(io.BytesIO(exported)).convert('RGB')
        self.assertFalse(any(b>r+60 for r,g,b in pixels.get_flattened_data()))
        self.assertEqual((root/'originals'/(image['id']+'.png')).read_bytes(),fixture())

    def test_mask_is_aligned_and_portable_without_changing_caption_pixels(self):
        image=self.target(mode='manual',box=[.1,.1,.55,.85],mask='polygon',polygon=[[.14,.24],[.46,.24],[.46,.72],[.14,.72]],background_weight=.1)
        geometry=plan(self.project,image);source=self.service.image_path(image)
        pixels=render(source,geometry).tobytes()
        mask=obj.render_mask(source,self.project,image,geometry)
        self.assertEqual(mask.size,(geometry['width'],geometry['height']))
        self.assertEqual(mask.getextrema(),(26,255))
        self.assertEqual(render(source,geometry).tobytes(),pixels)
        root=export_run(self.service.store,self.project,[image],'masked',self.service.settings())
        config=json.loads((root/'config.json').read_text())['config']['process'][0]['datasets'][0]
        self.assertEqual(config['mask_min_value'],0)
        with zipfile.ZipFile(root/'training-bundle.zip') as bundle:
            portable=json.loads(bundle.read('config.json'))['config']['process'][0]['datasets'][0]
            self.assertEqual(portable['mask_path'],'./masks')
            self.assertIn('masks/'+image['id']+'.png',bundle.namelist())

    def test_empty_cropped_mask_is_rejected(self):
        image=self.target(mode='manual',box=[.1,.1,.3,.3],margin=0,mask='polygon',polygon=[[.7,.7],[.9,.7],[.9,.9]])
        with self.assertRaisesRegex(ValueError,'empty'):self.service.preview(image['id'])

    def test_alpha_and_mixed_mask_bundle_on_each_model_grid(self):
        source=Image.new('RGBA',(400,300),(0,0,0,0))
        source.paste((200,30,40,255),(100,60,300,240))
        buffer=io.BytesIO();source.save(buffer,format='PNG')
        alpha=self.service.import_image(self.project['id'],{'name':'alpha.png','data':base64.b64encode(buffer.getvalue()).decode()})['image']
        alpha=self.service.update_image(alpha['id'],{'revision':1,'object_target':{'mode':'manual','box':[.2,.1,.8,.9],'mask':'alpha','background_weight':0}})
        ordinary=self.target(mode='full')
        for model,multiple in [('krea2',16),('qwen21',32),('flux',16),('sdxl',64)]:
            p={**self.project,'model':model}
            geometry=plan(p,alpha)
            mask=obj.render_mask(self.service.image_path(alpha),p,alpha,geometry)
            self.assertEqual(mask.size,(geometry['width'],geometry['height']))
            self.assertEqual(mask.getextrema(),(0,255))
            self.assertTrue(all(d%multiple==0 for d in mask.size))
            root=export_run(self.service.store,p,[alpha,ordinary],'mixed-'+model,self.service.settings())
            with Image.open(root/'masks'/(ordinary['id']+'.png')) as full:
                self.assertEqual(full.getextrema(),(255,255))
            with Image.open(root/'dataset'/(alpha['id']+'.png')) as prepared:
                self.assertEqual(prepared.mode,'RGBA' if model=='qwen21' else 'RGB')

    def test_exif_orientation_is_applied_before_target_coordinates(self):
        source=Image.new('RGB',(600,300),'blue');source.paste('red',(0,0,300,300))
        exif=Image.Exif();exif[274]=6
        buffer=io.BytesIO();source.save(buffer,format='JPEG',exif=exif)
        item=self.service.import_image(self.project['id'],{'name':'oriented.jpg','data':base64.b64encode(buffer.getvalue()).decode()})['image']
        item=self.service.update_image(item['id'],{'revision':1,'object_target':{'mode':'manual','box':[.1,.05,.9,.45],'margin':0}})
        geometry=plan(self.project,item)
        self.assertEqual((geometry['source_width'],geometry['source_height']),(300,600))
        pixels=render(self.service.image_path(item),geometry).convert('RGB')
        self.assertFalse(any(b>r+60 for r,g,b in pixels.get_flattened_data()))

    def test_alpha_mask_requires_alpha_and_other_categories_ignore_saved_target(self):
        image=self.target(mode='manual',box=[.1,.1,.5,.7],mask='alpha')
        self.assertTrue(any('transparency' in e for e in preflight(self.project,[image],self.service.settings(),automatic=True,preparing=True)['errors']))
        for kind in ['character','style','concept']:
            self.assertNotIn('source_region',plan({**self.project,'kind':kind},image))
        self.service.store.save_project({**self.project,'kind':'style'},self.project['id'])
        with self.assertRaisesRegex(ValueError,'Object'):self.target(mode='full')
        self.assertEqual(self.service.store.item('images',image['id'])['object_target']['mask'],'alpha')

    def test_automatic_grounding_then_caption_uses_cropped_pixels(self):
        sizes=[]
        def complete(settings,prompt,data,*args,**kwargs):
            if data is None:
                return json.dumps({'subject':'rectangle','facts':[{'attribute':'background','text':'a plain surface','sentence':1}],'warnings':[]})
            sizes.append(Image.open(io.BytesIO(data)).size)
            if 'Locate the ONE' in prompt:return self.found()
            self.assertIn('selected object',prompt)
            return json.dumps({'observations':'A rectangle on a plain surface.','caption':'synth rectangle on a plain surface','warnings':[]})
        with patch('studio.service.standalone_caption',side_effect=complete):
            result=self.wait(self.service.automatic_prepare(self.project['id']))
        self.assertEqual(result['status'],'complete',result.get('error'))
        self.assertEqual(sizes[0],(512,384));self.assertLess(sizes[1][0],250)
        image=self.service.dataset(self.project['id'])[0]
        self.assertEqual(image['revision'],3)
        self.assertFalse(image['preparation_stale'])
        self.assertFalse(image['manual'])

    def test_absent_and_ambiguous_stop_before_caption_or_export(self):
        self.project = self.service.store.save_project({**self.project, 'description': ''}, self.project['id'])
        for status in ['absent','ambiguous']:
            with patch('studio.service.standalone_caption',return_value=self.found(status)) as infer,patch('studio.service.export_run') as export:
                result=self.wait(self.service.automatic_prepare(self.project['id']))
                self.assertEqual(result['status'],'failed')
                self.assertEqual(infer.call_count,1);export.assert_not_called()
                self.assertEqual(obj.selection(self.project,self.service.dataset(self.project['id'])[0])['status'],status)

    def test_saved_detection_invalidates_on_hint_or_description_change(self):
        item={**self.image,'object_detection':obj.parse(self.found(),self.project,self.image)}
        self.assertEqual(obj.selection(self.project,item)['status'],'ready')
        self.assertEqual(obj.selection({**self.project,'description':'Another object'},item)['status'],'pending')
        self.assertEqual(obj.selection(self.project,{**item,'object_target':{'query':'right one'}})['status'],'pending')

    def test_concurrent_edit_and_cancel_do_not_overwrite_selection(self):
        def complete(*args,**kwargs):
            self.target(mode='full');return self.found()
        with patch('studio.service.standalone_caption',side_effect=complete):
            result=self.wait(self.service.find_objects(self.project['id']))
        self.assertEqual(result['status'],'failed')
        self.assertEqual(self.service.dataset(self.project['id'])[0]['object_selection']['source'],'manual-full')
        self.target(mode='auto')
        def cancelled(*args,**kwargs):
            next(iter(self.service.cancel.values())).set();return self.found()
        with patch('studio.service.standalone_caption',side_effect=cancelled):
            result=self.wait(self.service.find_objects(self.project['id']))
        self.assertEqual(result['status'],'cancelled')
        self.assertNotIn('object_detection',self.service.store.item('images',self.image['id']))

    def test_manual_caption_is_preserved_after_crop_and_revision_conflict(self):
        item=self.service.update_image(self.image['id'],{'revision':1,'caption':'synth manual caption','reviewed':True})
        self.target(mode='manual',box=[.1,.1,.5,.7])
        self.assertTrue(self.service.dataset(self.project['id'])[0]['preparation_stale'])
        result=self.wait(self.service.automatic_prepare(self.project['id']))
        self.assertEqual(result['status'],'failed')
        self.assertEqual(self.service.store.item('images',item['id'])['caption'],'synth manual caption')
        with self.assertRaises(Conflict):self.service.update_image(item['id'],{'revision':1,'object_target':{'mode':'full'}})

    def test_tight_auto_fit_preserves_target_with_padding(self):
        image=self.target(mode='manual',box=[70/512,90/384,235/512,275/384],margin=0)
        geometry=plan(self.project,image)
        self.assertEqual(geometry['mode'],'pad')

    def test_colour_cannot_be_misclassified_as_condition(self):
        value={'subject':'teapot','facts':[{'attribute':'condition','text':'solid blue','sentence':1},
               {'attribute':'background','text':'a red table','sentence':2}],'warnings':[]}
        observation='The teapot is solid blue. It is on a red table.'
        with self.assertRaisesRegex(ValueError,'Condition'):
            process_response(json.dumps(value),self.project,observation)
        # A coloured prop is still valid background context, even with target colour KEEP.
        del value['facts'][0]
        parsed=process_response(json.dumps(value),self.project,observation)
        self.assertIn('red table',parsed['caption'])
        value['facts']=[{'attribute':'framing','text':'a single blue teapot','sentence':1}]
        with self.assertRaisesRegex(ValueError,'color'):
            process_response(json.dumps(value),self.project,'A single blue teapot is shown.')
        value['facts']=[{'attribute':'background','text':'a blue teapot on the shelf behind it','sentence':1}]
        parsed=process_response(json.dumps(value),self.project,'There is a blue teapot on the shelf behind it.')
        self.assertIn('blue teapot',parsed['caption'])

    def test_no_mask_configuration_without_installed_backend_support(self):
        item=self.target(mode='full',mask='box')
        result=preflight(self.project,[item],self.service.settings(),automatic=True,preparing=True)
        self.assertTrue(any('loss-mask contract' in e for e in result['errors']))

    def test_occluding_prop_cannot_lose_its_actor(self):
        value={'subject':'teapot','facts':[{'attribute':'framing','text':'overlaps the teapot','sentence':1}],'warnings':[]}
        source='An orange block overlaps the teapot.'
        with self.assertRaisesRegex(ValueError,'explicit actor'):
            process_response(json.dumps(value),self.project,source)
        value['facts']=[{'attribute':'background','text':'An orange block overlaps the teapot','sentence':1}]
        self.assertIn('orange block overlaps',process_response(json.dumps(value),self.project,source)['caption'])


if __name__=='__main__':unittest.main()
