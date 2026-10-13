import base64
from contextlib import contextmanager
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image
from studio.captioners import joy_image, joy_runtime
from studio.integrations import standalone_caption
from studio.service import Service
from tests.fixtures import fixture

ROOT = Path(__file__).resolve().parents[1] / "internal-tests/artifacts/unit"
ROOT.mkdir(parents=True, exist_ok=True)

class CaptionSourcesCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.service = Service(self.root / "data")
        self.project = self.service.store.save_project({"name":"Shapes", "kind":"object", "description":"A red shape", "trigger":"redshape"})
        self.image = self.service.import_image(self.project["id"], {"name":"shapes.png", "data":base64.b64encode(fixture()).decode()})["image"]
        # Caption workflow tests explicitly keep the full synthetic scene; targeting has its own suite.
        self.image['object_target'] = {'mode': 'full'}
        self.service.store.put_item('images', self.image)
        self.service.save_settings({"integration":"standalone", "llm_model":"formatter", "caption_source":"custom", "caption_model":"captioner"})

    def wait(self, job):
        for _ in range(300):
            result = self.service.store.item("jobs", job["id"])
            if result["status"] not in {"queued", "running"}:
                return result
            time.sleep(.01)
        self.fail("Caption job did not finish")

    def test_dedicated_captioner_keeps_original_evidence_and_unreviewed_draft(self):
        calls=[]
        def complete(settings, instruction, data, mime, **options):
            calls.append((settings["llm_model"], data, instruction))
            if data is not None:
                return "A red square next to a blue circle."
            return json.dumps({"subject":"shape","facts":[{"attribute":"placement","text":"next to a blue circle","sentence":1}],"warnings":[]})
        with patch("studio.service.standalone_caption", side_effect=complete):
            result=self.wait(self.service.caption(self.project["id"]))
        self.assertEqual(result["status"], "complete")
        self.assertEqual([x[0] for x in calls], ["captioner", "formatter"])
        self.assertIsNotNone(calls[0][1]); self.assertIsNone(calls[1][1])
        image=self.service.store.item("images",self.image["id"])
        self.assertEqual(image["observations"], "A red square next to a blue circle.")
        self.assertFalse(image["reviewed"])
        self.assertEqual(image["caption_source"], "custom")
        self.assertTrue(list((self.service.store.root / "caption-jobs" / result["id"]).glob("*.json")))

    def test_custom_endpoint_failure_does_not_silently_use_connected_llm(self):
        with patch("studio.service.standalone_caption", side_effect=RuntimeError("Endpoint unavailable")) as call:
            result=self.wait(self.service.caption(self.project["id"]))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(call.call_count,1)
        self.assertEqual(self.service.store.item("images",self.image["id"])["caption"], "")

    def test_manual_edit_during_observation_is_preserved(self):
        def complete(settings, instruction, data, mime, **options):
            if data is not None:
                self.service.update_image(self.image["id"], {"revision":1,"caption":"redshape manual"})
                return "A red square in view."
            return json.dumps({"subject":"shape","facts":[{"attribute":"placement","text":"in view","sentence":1}],"warnings":[]})
        with patch("studio.service.standalone_caption", side_effect=complete):
            result=self.wait(self.service.caption(self.project["id"]))
        self.assertEqual(result["status"],"complete")
        self.assertEqual(self.service.store.item("images",self.image["id"])["caption"],"redshape manual")
        self.assertTrue(result["messages"])

    def test_model_options_survive_switching_and_invalid_values_are_rejected(self):
        self.service.save_settings({"caption_source":"connected"})
        self.assertEqual(self.service.settings()["caption_model"],"captioner")
        for value in ({"caption_source":"unknown"},{"caption_source":[]},{"caption_gpu":True},{"caption_gpu":-1},{"caption_url":"https://example.com/v1"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.service.save_settings(value)

    def test_connected_object_uses_evidence_formatter_and_never_auto_reviews(self):
        self.service.save_settings({"caption_source":"connected"})
        with patch("studio.service.standalone_caption",return_value=json.dumps({"caption":"redshape next to a circle","observations":"A shape next to a circle.","subject":"shape","facts":[{"attribute":"placement","text":"next to a circle","sentence":1}],"warnings":[]})) as call:
            result=self.wait(self.service.caption(self.project["id"]))
        self.assertEqual(result["status"],"complete")
        self.assertEqual(call.call_count,2)
        self.assertFalse(self.service.store.item("images",self.image["id"])["reviewed"])

    def test_joy_runtime_closes_before_text_stage(self):
        sequence=[]
        @contextmanager
        def runtime(*args):
            sequence.append("start")
            try:yield {"llm_url":"http://127.0.0.1:9999/v1", "llm_model":"joy"}
            finally:sequence.append("stop")
        def complete(settings,instruction,data,mime,**options):
            sequence.append("vision" if data else "text")
            if data:
                self.assertEqual(Image.open(io.BytesIO(data)).size,(384,384))
                return "A shape next to a circle."
            return json.dumps({"subject":"shape","facts":[{"attribute":"placement","text":"next to a circle","sentence":1}],"warnings":[]})
        self.service.save_settings({"caption_source":"joycaption"})
        with patch("studio.service.joy_files"),patch("studio.service.joy_runtime",runtime),patch("studio.service.standalone_caption",side_effect=complete):
            result=self.wait(self.service.caption(self.project["id"]))
        self.assertEqual(result["status"],"complete")
        self.assertEqual(sequence,["start","vision","stop","text"])

    def test_runtime_is_stopped_after_inference_failure(self):
        files=[self.root/n for n in ["llama-server.exe","model.gguf","projector.gguf"]]
        for path in files:path.touch()
        settings={"caption_runtime":str(files[0]),"joycaption_model":str(files[1]),"joycaption_projector":str(files[2]),"caption_gpu":0}
        process=MagicMock();process.poll.return_value=None
        calls=[]
        def launch(command,**kwargs):calls.append(command);return process
        def response(url,**kwargs):
            return {"status":"ok"} if url.endswith("health") else {"data":[{"id":calls[0][-1]}]}
        sock=MagicMock();sock.__enter__.return_value.getsockname.return_value=("127.0.0.1",9999)
        processes={}
        with patch("studio.captioners.socket.socket",return_value=sock),patch("studio.captioners.subprocess.Popen",side_effect=launch),patch("studio.captioners.request",side_effect=response),patch("studio.captioners.terminate_tree") as terminate:
            with self.assertRaisesRegex(RuntimeError,"synthetic"):
                with joy_runtime(settings,self.root/'log',threading.Event(),processes,"job"):
                    raise RuntimeError("synthetic inference failure")
        terminate.assert_called_once_with(process)
        self.assertFalse(processes)
        self.assertEqual(calls[0][calls[0].index("--host")+1],"127.0.0.1")

    def test_text_formatter_request_contains_no_image(self):
        with patch("studio.integrations.request",return_value={"choices":[{"finish_reason":"stop","message":{"content":"text"}}]}) as call:
            standalone_caption({"llm_url":"http://127.0.0.1:9999/v1","llm_model":"formatter"},"Rules",None,"image/png")
        content=call.call_args.args[1]["messages"][0]["content"]
        self.assertEqual(content,[{"type":"text","text":"Rules"}])

    def test_joy_preprocessing_preserves_dataset_input(self):
        data=fixture();prepared=joy_image(data)
        self.assertEqual(Image.open(io.BytesIO(prepared)).size,(384,384))
        self.assertEqual(Image.open(io.BytesIO(data)).size,(512,384))

    def saved_observations(self):
        i=self.service.store.item('images',self.image['id'])
        i.update(observations='Two colored shapes next to each other.',caption='redshape previous',caption_source='joycaption',observation_kind='object',
                 caption_preparation_digest=self.service.dataset(self.project['id'])[0]['plan']['digest'])
        self.service.store.put_item('images',i)
        return i

    def test_reprocess_uses_text_only_preserves_evidence_and_history(self):
        before=self.saved_observations()
        self.service.save_settings({'caption_source':'connected'})
        response=json.dumps({'subject':'shape','facts':[{'attribute':'placement','text':'next to each other','sentence':1}]})
        with patch('studio.service.standalone_caption',return_value=response) as complete,patch('studio.service.joy_runtime') as runtime:
            job=self.wait(self.service.caption(self.project['id'],True))
        self.assertEqual(job['status'],'complete');runtime.assert_not_called()
        self.assertIsNone(complete.call_args.args[2])
        after=self.service.store.item('images',self.image['id'])
        self.assertEqual(after['observations'],before['observations']);self.assertEqual(after['caption_source'],'joycaption')
        self.assertEqual(after['history'][-1]['caption'],before['caption']);self.assertFalse(after['reviewed'])

    def test_reprocess_rejects_changed_crop_and_skips_manual_caption(self):
        image=self.saved_observations();image['caption_preparation_digest']='old';self.service.store.put_item('images',image)
        with self.assertRaisesRegex(ValueError,'older training crop'):self.service.caption(self.project['id'],True)
        image['manual']=True;self.service.store.put_item('images',image)
        with self.assertRaisesRegex(ValueError,'No eligible'):self.service.caption(self.project['id'],True)
        with self.assertRaisesRegex(ValueError,'true or false'):self.service.caption(self.project['id'],'yes')

    def test_reprocess_invalid_output_preserves_previous_caption(self):
        before=self.saved_observations()
        with patch('studio.service.standalone_caption',return_value='{}'):
            job=self.wait(self.service.caption(self.project['id'],True))
        self.assertEqual(job['status'],'failed')
        self.assertEqual(self.service.store.item('images',self.image['id']),before)

    def test_resume_skips_current_text_and_rejects_invalid_mode(self):
        from studio.domain import intent_digest
        image=self.saved_observations()
        from studio.caption_processing import FORMATTER_VERSION
        image.update(caption_formatter_version=FORMATTER_VERSION,object_caption_version=1,intent_digest=intent_digest(self.project))
        self.service.store.put_item('images',image)
        with self.assertRaisesRegex(ValueError,'No eligible'):
            self.service.caption(self.project['id'],True,True)
        with self.assertRaisesRegex(ValueError,'Only outdated'):
            self.service.caption(self.project['id'],False,True)
