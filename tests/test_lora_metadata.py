import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from studio.domain import normalize_project, intent_digest, MODELS
from studio.lora_metadata import normalize_metadata, public_metadata
from studio.store import Store, Conflict
from studio.trainer import config_for, export_run


class MetadataTests(unittest.TestCase):
    def project(self, **changes):
        return normalize_project({"name": "Synthetic LoRA", "trigger": "shape_token",
            "description": "PRIVATE_LEARNING_TARGET", "also_keep": "PRIVATE_RULE",
            "sample_prompts": "PRIVATE_PREVIEW_PROMPT", **changes})

    def test_validation_and_legacy_defaults(self):
        defaults = normalize_metadata()
        self.assertEqual(self.project()["lora_metadata"], defaults)
        self.assertIsNone(defaults["recommended_strength"])
        self.assertEqual(normalize_metadata({"trigger_words": [" a b ", "", "a b", "c"]})["trigger_words"], ["a b", "c"])
        self.assertIsNone(normalize_metadata({"trigger_words": [" "]})["trigger_words"])
        for value in (True, "1", float("nan"), float("inf"), -11, 11):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_metadata({"recommended_strength": value})
        for value in ({"description": 12}, {"author": "x" * 201}, {"license": []},
                      {"trigger_words": "phrase"}, {"trigger_words": [True]},
                      {"trigger_words": ["x"] * 129}, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_metadata(value)
        self.assertEqual(normalize_metadata({"recommended_strength": 0})["recommended_strength"], 0)

    def test_persistence_revision_and_caption_intent(self):
        root = Path(__file__).resolve().parents[1] / "internal-tests/artifacts/unit"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            store = Store(Path(temp))
            initial = store.save_project(self.project())
            fields = {"description": "Public description", "author": "Synthetic author",
                      "license": "CC0-1.0", "trigger_words": ["public phrase"], "recommended_strength": .75}
            saved = store.save_project({**initial, "lora_metadata": fields}, initial["id"])
            self.assertEqual(Store(Path(temp)).project(saved["id"])["lora_metadata"], fields)
            self.assertEqual(intent_digest(initial), intent_digest(saved))
            with self.assertRaises(Conflict):
                store.save_project(initial, initial["id"])
            with patch("studio.trainer.subprocess.check_output", return_value="synthetic-commit"):
                run = export_run(store, saved, [], "metadata-test", {"toolkit_path": "unused"})
            before = json.loads((run / "config.json").read_text())["meta"]
            store.save_project({**saved, "lora_metadata": {"author": "Later author"}}, saved["id"])
            self.assertEqual(json.loads((run / "config.json").read_text())["meta"], before)
            with zipfile.ZipFile(run / "training-bundle.zip") as archive:
                self.assertEqual(json.loads(archive.read("config.json"))["meta"], before)

    def test_allowlist_excludes_private_data_and_unknown_future_config(self):
        project = self.project(lora_metadata={"description": "A public description [name]",
                                              "author": "Example", "license": "CC0-1.0"})
        config = config_for(project, "PRIVATE_DATASET_PATH", "PRIVATE_OUTPUT_PATH", "run")
        process = config["config"]["process"][0]
        process["train"]["future_prompt"] = "PRIVATE_FUTURE_PROMPT"
        process["train"]["optimizer_params"] = {"weight_decay": .01, "path": "PRIVATE_OPTIMIZER_PATH"}
        process["model"]["name_or_path"] = "PRIVATE_MODEL_PATH"
        process["datasets"][0]["captions"] = ["PRIVATE_CAPTION"]
        project["lora_metadata"]["unknown"] = "PRIVATE_UNKNOWN_FIELD"
        meta = public_metadata(project, process, MODELS[0])
        self.assertNotIn("PRIVATE_", json.dumps(meta))
        self.assertNotIn("intent_digest", meta)
        self.assertNotIn("[name]", meta["lora_studio"])
        decoded = json.loads(meta["lora_studio"])
        self.assertEqual(decoded["description"], "A public description [name]")
        self.assertEqual(decoded["trigger_words"], ["shape_token"])
        self.assertEqual(decoded["training"]["train"]["optimizer_params"], {"weight_decay": .01})

    def test_effective_training_settings_and_explicit_trigger_override(self):
        for model in MODELS:
            project = self.project(model=model["id"], training={"optimizer": "automagic3", "krea_weights": "comfy_bf16"},
                lora_metadata={"trigger_words": ["first phrase", "second"], "recommended_strength": -.5})
            config = config_for(project, "dataset", "output", "run")
            meta = json.loads(config["meta"]["lora_studio"])
            process = config["config"]["process"][0]
            self.assertEqual(meta["schema_version"], 1)
            self.assertEqual(meta["trigger_words"], ["first phrase", "second"])
            self.assertEqual(meta["recommended_strength"], -.5)
            self.assertEqual(process["trigger_word"], "shape_token")
            self.assertEqual(meta["training"]["train"], process["train"])
            self.assertEqual(meta["training"]["network"], process["network"])
            self.assertEqual(meta["training"]["save"], process["save"])
            self.assertEqual(meta["training"]["dataset"]["resolution"], [project["training"]["resolution"]])
            self.assertNotIn("gpu", meta["training"])
            self.assertNotIn("samples", meta["training"]["sample"])
            process["network"]["linear"] = 99
            self.assertEqual(meta["training"]["network"]["linear"], 16)
