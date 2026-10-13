"""Synthetic preparation and advisory invariants; no local model required."""
import base64
import io
import json
from pathlib import Path
import tempfile
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from PIL import Image

from studio.domain import normalize_project, parse_caption
from studio.preparation import bucket, metadata, plan, render, PROFILES
from studio.service import Service
from studio.store import Conflict
from studio.trainer import export_run, config_for, preflight

ROOT = Path(__file__).resolve().parents[1] / "internal-tests" / "artifacts" / "unit"
ROOT.mkdir(parents=True, exist_ok=True)


def png(width=600, height=900):
    image = Image.new("RGB", (width, height), "red")
    image.paste("blue", (0, 0, width // 2, height))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class PreparationCase(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(temp.cleanup)
        self.service = Service(Path(temp.name) / "data")
        self.project = self.service.store.save_project({"description": "Synthetic colour panels", "trigger": "panels"})
        self.image = self.service.import_image(self.project["id"], {"name": "panels.png", "data": base64.b64encode(png()).decode()})["image"]

    def test_model_grids_and_preserved_resolution(self):
        self.assertEqual(self.project["training"]["resolution"], 1024)
        for model, valid, invalid in [("krea2", 1040, 1032), ("qwen21", 1056, 1040)]:
            self.assertEqual(normalize_project({"model": model, "training": {"resolution": valid}})["training"]["resolution"], valid)
            with self.assertRaises(ValueError):
                normalize_project({"model": model, "training": {"resolution": invalid}})
        old = normalize_project({"model": "qwen21", "training": {"resolution": 768}})
        self.assertEqual(old["training"]["resolution"], 768)

    def test_buckets_match_known_backend_geometry_and_are_idempotent(self):
        self.assertEqual(bucket(2000, 3000, 1024, 16), (832, 1248))
        self.assertEqual(bucket(1920, 1080, 1024, 32), (1344, 768))
        for multiple in (16, 32, 64):
            for width, height in [(600, 900), (1920, 1080), (32, 8000), (16, 1000000), (2048, 2048)]:
                w, h = bucket(width, height, 1024, multiple)
                self.assertEqual(w % multiple, 0)
                self.assertEqual(h % multiple, 0)
                self.assertLessEqual(w * h, 1024 ** 2)
                self.assertEqual(bucket(w, h, 1024, multiple), (w, h))

    def test_export_matches_preview_and_keeps_original(self):
        image = self.service.update_image(self.image["id"], {"revision": 1, "caption": "panels in red and blue", "reviewed": True,
                                           "preparation": {"mode": "pad", "x": 0, "y": 1}})
        preview = self.service.preview(image["id"])
        with patch("studio.trainer.subprocess.check_output", return_value="synthetic"):
            root = export_run(self.service.store, self.project, [image], "export", {"toolkit_path": "synthetic-toolkit"})
        with Image.open(io.BytesIO(base64.b64decode(preview["png"]))) as expected, Image.open(root / "dataset" / (image["id"] + ".png")) as actual:
            self.assertEqual(actual.size, expected.size)
            self.assertEqual(actual.tobytes(), expected.tobytes())
        self.assertEqual((root / "originals" / (image["id"] + ".png")).read_bytes(), png())
        self.assertEqual(self.service.image_path(image).read_bytes(), png())

    def test_crop_position_changes_prepared_pixels_and_requires_review(self):
        item = self.service.update_image(self.image["id"], {"revision": 1, "caption": "panels", "reviewed": True})
        item = self.service.update_image(item["id"], {"revision": item["revision"], "preparation": {"mode": "crop", "x": 0}})
        self.assertTrue(self.service.dataset(self.project["id"])[0]["preparation_stale"])
        self.assertEqual(item["caption"], "panels")
        self.assertTrue(item["manual"])
        a = self.service.preview(item["id"], {"mode": "crop", "x": 0})
        b = self.service.preview(item["id"], {"mode": "crop", "x": 1})
        self.assertNotEqual(a["png"], b["png"])
        item = self.service.update_image(item["id"], {"revision": item["revision"], "reviewed": True})
        self.assertFalse(self.service.dataset(self.project["id"])[0]["preparation_stale"])

    def test_resolution_change_invalidates_caption_geometry(self):
        self.service.update_image(self.image["id"], {"revision": 1, "caption": "panels", "reviewed": True})
        project = {**self.project, "training": {**self.project["training"], "resolution": 512}}
        self.service.store.save_project(project, project["id"])
        item = self.service.dataset(project["id"])[0]
        self.assertTrue(item["preparation_stale"])
        self.assertTrue(any("crop" in e for e in preflight(project, [item], {})["errors"]))

    def test_person_override_persists_and_flags_are_advisory(self):
        item = self.service.update_image(self.image["id"], {"revision": 1, "caption": "panels", "reviewed": True, "person_count": 2, "accept_advisories": True})
        view = self.service.dataset(self.project["id"])[0]
        self.assertTrue(view["advisories_accepted"])
        self.assertTrue(any("Multiple people" in s for s in view["advisories"]))
        self.assertFalse(any("people" in s for s in preflight(self.project, [view], {})["errors"]))
        service = Service(self.service.store.root)
        self.assertEqual(service.store.item("images", item["id"])["person_count"], 2)
        with self.assertRaises(Conflict):
            service.update_image(item["id"], {"revision": 1, "person_count": 1})
        service.update_image(item["id"], {"revision": item["revision"], "person_count": 1})
        self.assertFalse(service.dataset(self.project["id"])[0]["advisories_accepted"])

    def test_corrupt_image_rejected_and_no_record_written(self):
        with self.assertRaises(ValueError):
            self.service.import_image(self.project["id"], {"name": "bad.png", "data": base64.b64encode(b'\x89PNG\r\n\x1a\ncorrupt').decode()})
        self.assertEqual(len(self.service.store.items("images", self.project["id"])), 1)

    def test_exif_orientation_is_applied_once(self):
        buffer = io.BytesIO()
        image = Image.new("RGB", (80, 120), "green")
        exif = image.getexif(); exif[274] = 6
        image.save(buffer, format="JPEG", exif=exif)
        dimensions = metadata(io.BytesIO(buffer.getvalue()))
        self.assertEqual((dimensions["width"], dimensions["height"]), (120, 80))

    def test_bad_count_and_geometry_are_rejected(self):
        for patch_value in ({"person_count": True}, {"person_count": -1}, {"preparation": {"x": float("nan")}}, {"preparation": {"mode": "stretch"}}):
            with self.subTest(patch_value=patch_value), self.assertRaises(ValueError):
                self.service.update_image(self.image["id"], {"revision": 1, **patch_value})

    def test_model_sample_shapes_and_no_hidden_augmentations(self):
        for model in ("krea2", "qwen21"):
            project = normalize_project({"model": model, "trigger": "panels", "description": "Colour panels"})
            config = config_for(project, "dataset", "output", "run")["config"]["process"][0]
            samples = config["sample"]["samples"]
            self.assertEqual(len({(s["width"], s["height"]) for s in samples}), 3)
            self.assertTrue(all(s["width"] % PROFILES[model]["multiple"] == 0 for s in samples))
            self.assertFalse(config["datasets"][0]["flip_x"])
            self.assertFalse(config["datasets"][0]["square_crop"])

    def test_old_image_metadata_is_derived_without_resetting_caption(self):
        item = self.service.store.item("images", self.image["id"])
        item.pop("width"); item.pop("height"); item.pop("dhash")
        item["caption"] = "old manual caption"; item["manual"] = True
        self.service.store.put_item("images", item)
        result = self.service.dataset(self.project["id"])[0]
        self.assertEqual(result["width"], 600)
        self.assertEqual(result["caption"], "old manual caption")
        self.assertTrue(result["preparation_stale"])

    def test_coverage_corrections_survive_caption_updates_and_type_changes(self):
        item = self.service.update_image(self.image["id"], {"revision": 1, "coverage_override": {"framing": "detail"}})
        item["coverage"] = {"framing": "other"}
        self.service.store.put_item("images", item)
        project = {**self.project, "kind": "object"}
        self.service.store.save_project(project, project["id"])
        self.assertEqual(self.service.dataset(project["id"])[0]["coverage_override"], {"framing": "detail"})
        with self.assertRaises(ValueError):
            self.service.update_image(item["id"], {"revision": item["revision"], "coverage_override": {"identity": "same"}})

    def test_unrecognized_optional_coverage_does_not_destroy_caption(self):
        result = parse_caption(json.dumps({"caption": "panels in red and blue", "observations": "Two colours", "warnings": [],
                                           "coverage": {"framing": "not a category"}}), self.project)
        self.assertEqual(result["caption"], "panels in red and blue")
        self.assertEqual(result["coverage"]["framing"], "uncertain")
        self.assertTrue(result["warnings"])

    def test_trigger_alone_is_not_a_usable_automatic_caption(self):
        for caption in ("panels", "panels."):
            with self.assertRaisesRegex(ValueError, "only the trigger"):
                parse_caption(json.dumps({"caption": caption, "observations": "Two colour panels", "warnings": []}), self.project)

    def test_unreviewed_auto_caption_warns_without_preventing_override(self):
        item = self.service.update_image(self.image["id"], {"revision": 1, "caption": "panels in red and blue", "reviewed": True})
        item.update(manual=False, reviewed=False)
        result = preflight(self.project, [item], {})
        self.assertTrue(any("have not been reviewed" in w for w in result["warnings"]))
        self.assertFalse(any("have not been reviewed" in e for e in result["errors"]))

    def test_detector_results_are_advisory_and_keep_manual_count(self):
        item = self.service.update_image(self.image["id"], {"revision": 1, "person_count": 1})
        self.service.save_settings({"detector_python": sys.executable})
        def worker(command, **kwargs):
            payload = {"id": item["id"], "detection": {"status": "complete", "count": 2, "threshold": .5,
                       "boxes": [{"box": [.1, .1, .4, .8], "score": .9}, {"box": [.5, .1, .9, .8], "score": .8}]}}
            Path(command[3]).write_text(json.dumps(payload) + "\n", encoding="utf-8")
            return SimpleNamespace(poll=lambda: 0, returncode=0, pid=1234)
        with patch("studio.service.subprocess.Popen", side_effect=worker):
            job = self.service.check_people(self.project["id"])
            for _ in range(100):
                result = self.service.store.item("jobs", job["id"])
                if result["status"] not in {"queued", "running"}:
                    break
                time.sleep(.01)
        self.assertEqual(result["status"], "complete")
        item = self.service.dataset(self.project["id"])[0]
        self.assertEqual(item["person_count"], 1)
        self.assertEqual(item["detection"]["count"], 2)
        self.assertFalse(any("Multiple people" in f for f in item["advisories"]))
