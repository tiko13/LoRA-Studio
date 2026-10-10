import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from .domain import MODELS, conflicts, intent, intent_digest

def backend_status(settings):
    root = Path(settings.get("toolkit_path") or ".__not_configured__")
    python = Path(settings.get("toolkit_python") or ".__not_configured__")
    supported = set()
    if (root / "run.py").is_file():
        for path in (root / "extensions_built_in" / "diffusion_models").glob("**/*.py"):
            if path.name == path.parent.name + ".py" or path.name in {"flux.py", "sdxl.py"}:
                supported.update(re.findall(r'\barch\s*=\s*[\"\']([^\"\']+)', path.read_text(encoding="utf-8", errors="replace")))
        # Legacy models are selected explicitly by their model flags in BaseModel.
        if (root / "toolkit" / "stable_diffusion_model.py").is_file():
            supported.update({"flux", "sdxl"})
    return {"available": (root / "run.py").is_file() and python.is_file(), "architectures": sorted(supported),
            "models": {m["id"]: m["arch"] in supported for m in MODELS}}

def preflight(project, images, settings):
    included = [i for i in images if i.get("included", True)]
    errors, warnings = [], []
    if not project["name"] or not project["description"] or not project["trigger"]:
        errors.append("Add a name, learning description, and trigger in Setup.")
    if not included:
        errors.append("Import at least one training image.")
    if conflicts(project):
        errors.append("Resolve the additional rules that appear in both columns.")
    missing = sum(not i.get("caption", "").strip() for i in included)
    stale = sum(bool(i.get("caption")) and i.get("intent_digest") != intent_digest(project) for i in included)
    review = sum(bool(i.get("warnings")) and not i.get("reviewed") for i in included)
    if missing:
        errors.append(f"{missing} image(s) need captions.")
    if stale:
        errors.append(f"{stale} caption(s) use older intent rules. Re-caption or review them.")
    if review:
        errors.append(f"{review} image(s) have caption warnings to review.")
    if any(project["trigger"].casefold() not in i.get("caption", "").casefold() for i in included) and project["trigger"]:
        errors.append("Every included caption must identify the target with its trigger.")
    if 0 < len(included) < 10:
        warnings.append("Small dataset: check coverage and flexibility carefully.")
    warnings.append("These are starter settings, not a quality or VRAM guarantee. Compare saved checkpoints.")
    status = backend_status(settings)
    if not status["available"]:
        errors.append("Set AI Toolkit and its Python executable in Connections.")
    elif not status["models"].get(project["model"]):
        errors.append("This AI Toolkit installation does not support the selected architecture.")
    return {"errors": errors, "warnings": warnings, "included": len(included), "backend": status}

def config_for(project, dataset, output, run_id):
    model = next(m for m in MODELS if m["id"] == project["model"])
    t = project["training"]
    sdxl = model["id"] == "sdxl"
    model_config = {"name_or_path": model["base"], "arch": model["arch"], "quantize": not sdxl,
                    "quantize_te": not sdxl, "qtype": "qfloat8", "qtype_te": "qfloat8"}
    if model["id"] == "flux":
        model_config["is_flux"] = True
    if sdxl:
        model_config["is_xl"] = True
    prompts = [p.strip() for p in project["sample_prompts"].splitlines() if p.strip()]
    if not prompts:
        prompts = [f"{project['trigger']}, {project['description'][:200]}, studio setting",
                   f"{project['trigger']}, {project['description'][:200]}, outdoors in daylight"]
    return {"job": "extension", "config": {"name": "lora_" + run_id, "process": [{
        "type": "sd_trainer", "training_folder": str(output), "device": "cuda:0", "trigger_word": project["trigger"],
        "network": {"type": "lora", "linear": t["rank"], "linear_alpha": t["rank"]},
        "save": {"dtype": "float16", "save_every": min(250, t["steps"]), "max_step_saves_to_keep": 8},
        "datasets": [{"folder_path": str(dataset), "caption_ext": "txt", "caption_dropout_rate": 0,
                      "shuffle_tokens": False, "cache_latents_to_disk": True, "resolution": [t["resolution"]]}],
        "train": {"batch_size": 1, "steps": t["steps"], "gradient_accumulation": 1,
                  "train_unet": True, "train_text_encoder": False, "gradient_checkpointing": True,
                  "cache_text_embeddings": not sdxl, "noise_scheduler": "ddpm" if sdxl else "flowmatch",
                  "optimizer": "adamw8bit", "lr": t["lr"], "dtype": "bf16"},
        "model": model_config,
        "sample": {"sampler": "ddpm" if sdxl else "flowmatch", "sample_every": min(250, t["steps"]),
                   "width": t["resolution"], "height": t["resolution"], "prompts": prompts[:12],
                   "seed": t["seed"], "walk_seed": False, "guidance_scale": 5 if sdxl else 4,
                   "sample_steps": 30}}]},
        "meta": {"name": project["name"], "studio_version": "0.1.0", "intent_digest": intent_digest(project)}}

def export_run(store, project, images, run_id, settings):
    root = store.root / "runs" / run_id
    dataset, output = root / "dataset", root / "output"
    dataset.mkdir(parents=True)
    output.mkdir()
    for item in images:
        if not item.get("included", True):
            continue
        source = store.root / "images" / item["project"] / (item["id"] + item["extension"])
        shutil.copyfile(source, dataset / source.name)
        (dataset / (item["id"] + ".txt")).write_text(item["caption"], encoding="utf-8")
    config = config_for(project, dataset.resolve(), output.resolve(), run_id)
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    revision = "unknown"
    try:
        revision = subprocess.check_output(["git", "-C", settings["toolkit_path"], "rev-parse", "HEAD"], text=True, timeout=8, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    manifest = {"project": project, "intent": intent(project), "images": images, "backend_commit": revision,
                "caption_policy": "intent-v1", "run_id": run_id}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # ZIP config uses paths relative to the extracted bundle, not this machine.
    portable = config_for(project, "./dataset", "./output", run_id)
    with zipfile.ZipFile(root / "training-bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("config.json", json.dumps(portable, indent=2))
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2))
        for file in dataset.iterdir():
            bundle.write(file, "dataset/" + file.name)
        bundle.writestr("README.txt", "Extract this bundle. Activate your AI Toolkit environment and run from this folder:\npython /path/to/ai-toolkit/run.py config.json\nReview config.json before training.\n")
    return root
