import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from .domain import MODELS, conflicts, intent, intent_digest
from .preparation import PROFILES, metadata, plan, render, decorate, bucket, dataset_advisories
from .sample_prompts import effective, mode
from .lora_metadata import public_metadata
from .categories import CATEGORY_VERSION
from . import object_preparation as objects

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


def mask_support(settings, model):
    """Check the installed adapter contract, without importing GPU libraries."""
    root = Path(settings.get('toolkit_path') or '.__not_configured__')
    files = [root/'toolkit/models/base_model.py', root/'toolkit/dataloader_mixins.py',
             root/'extensions_built_in/sd_trainer/SDTrainer.py']
    try:
        base, loader, trainer = [p.read_text(encoding='utf-8') for p in files]
    except OSError:
        return False
    if not re.search(r'do_masked_loss\s*=\s*True', base) or 'dataset_config.mask_path' not in loader or 'batch.mask_tensor' not in trainer:
        return False
    architecture = next(m['arch'] for m in MODELS if m['id'] == model)
    for path in (root/'extensions_built_in/diffusion_models').glob('**/*.py'):
        source = path.read_text(encoding='utf-8', errors='replace')
        if re.search(r'arch\s*=\s*[\"\']' + re.escape(architecture) + r'[\"\']', source) and re.search(r'do_masked_loss\s*=\s*False', source):
            return False
    return True

def preflight(project, images, settings, *, automatic=False, preparing=False):
    included = [i for i in images if i.get("included", True)]
    errors, warnings = [], []
    warnings.extend(dataset_advisories(project, images))
    if not project["name"] or not project["trigger"]:
        errors.append("Add a name and trigger in Base info.")
    if not included:
        errors.append("Import at least one training image.")
    if mode(project) == 'manual' and not project.get('sample_prompts', '').strip():
        errors.append('Add manual preview prompts or switch to Automatic preview prompts.')
    if len([p for p in project.get('sample_prompts', '').splitlines() if p.strip()]) > 12 and mode(project) == 'manual':
        errors.append('Use at most 12 preview prompts.')
    if conflicts(project):
        errors.append("Resolve the additional rules that appear in both columns.")
    if project['kind'] == 'object':
        unresolved = sum(objects.selection(project, i)['status'] != 'ready' for i in included)
        if unresolved and not (automatic and preparing):
            errors.append(f'{unresolved} image(s) need an Object target selection. Prepare dataset, draw a rectangle, explicitly keep the entire image, or exclude them.')
        if any(objects.options(i.get('object_target'))['mask'] == 'alpha' and not i.get('has_alpha') for i in included):
            errors.append('An alpha loss mask requires source transparency. Choose a polygon/rectangle mask or disable masking.')
        if any(objects.options(i.get('object_target'))['mask'] != 'none' for i in included) and not mask_support(settings, project['model']):
            errors.append('This installed AI Toolkit adapter has no verified loss-mask contract. Update the backend or disable Object loss masking.')
    caption_checks = [i for i in included if not (automatic and preparing and not i.get('manual'))]
    missing = sum(not i.get("caption", "").strip() for i in caption_checks)
    stale = sum(bool(i.get("caption")) and i.get("intent_digest") != intent_digest(project) for i in caption_checks)
    review = sum(bool(i.get("warnings")) and not i.get("reviewed") for i in included)
    if missing:
        errors.append(f"{missing} image(s) need captions.")
    if stale:
        errors.append(f"{stale} caption(s) use older intent rules. Re-caption or review them.")
    if review:
        (warnings if automatic else errors).append(f"{review} image(s) have caption warnings to review.")
    visual = sum(i.get('visual_review',{}).get('status')=='concerns' for i in included)
    if visual:
        (warnings if automatic else errors).append(f'{visual} image(s) have visual-check concerns. Edit the caption, keep despite the findings, exclude the image, or use Automatic mode to proceed with advisory flags.')
    unchecked = sum(bool(i.get("caption")) and not i.get("manual") and not i.get("reviewed") for i in included)
    if unchecked:
        examples = 'people, poses and details' if project['kind'] == 'character' else 'objects, relationships and visual details'
        warnings.append(f"{unchecked} automatic caption(s) have not been reviewed. Vision models can invent {examples} even without reporting warnings; review or continue with these drafts knowingly.")
    preparation_stale = sum(bool(i.get("caption")) and "width" in i and
                            i.get("caption_preparation_digest") != plan(project, i)["digest"] for i in caption_checks)
    if preparation_stale:
        errors.append(f"{preparation_stale} caption(s) need review against the current training crop. Re-caption or keep the caption after review.")
    if any(i.get("preparation_error") for i in included):
        errors.append("An included image cannot be decoded. Re-import or exclude it.")
    pending = sum(bool(i.get("advisories")) and not i.get("advisories_accepted") for i in included)
    if pending:
        warnings.append(f"{pending} image(s) have preparation advisories. Review or keep anyway; these flags do not block training.")
    if project["training"]["resolution"] > 1024:
        warnings.append("Above 1024 is an experimental training target. Check source detail and available memory; 2K has four times the pixels of 1K.")
    if project["model"] == "krea2":
        warnings.append("Training uses Krea 2 Raw. Built-in samples use Raw too; evaluate saved adapters on Turbo before judging final quality.")
    if any(project["trigger"].casefold() not in i.get("caption", "").casefold() for i in caption_checks) and project["trigger"]:
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
    if model["id"] == "krea2" and t.get("krea_weights") == "comfy_bf16":
        model_config.update(name_or_path="Comfy-Org/Krea-2",
                            model_kwargs={"checkpoint_filename": "diffusion_models/krea2_raw_bf16.safetensors"})
    if model["id"] == "flux":
        model_config["is_flux"] = True
    if sdxl:
        model_config["is_xl"] = True
    prompts = [p['text'] for p in effective(project, [], project.get('_preview_suite'))['prompts']]
    shapes = {"square": (4096, 4096), "portrait": (4096, 6144), "landscape": (6144, 4096)}
    layout = t.get("sample_layout", "mixed")
    layouts = list(shapes) if layout == "mixed" else [layout]
    samples = []
    for index, prompt in enumerate(prompts[:12]):
        w, h = bucket(*shapes[layouts[index % len(layouts)]], t["resolution"], PROFILES[model["id"]]["multiple"])
        samples.append({"prompt": prompt, "width": w, "height": h})
    config = {"job": "extension", "config": {"name": "lora_" + run_id, "process": [{
        "type": "sd_trainer", "training_folder": str(output), "device": "cuda:0", "trigger_word": project["trigger"],
        "network": {"type": "lora", "linear": t["rank"], "linear_alpha": t["rank"]},
        "save": {"dtype": "float16", "save_every": t.get('checkpoint_every', 250), "max_step_saves_to_keep": t.get('keep_checkpoints', 8)},
        "datasets": [{"folder_path": str(dataset), "caption_ext": "txt", "caption_dropout_rate": 0,
                      "shuffle_tokens": False, "cache_latents_to_disk": True, "resolution": [t["resolution"]],
                      "buckets": True, "square_crop": False, "random_crop": False, "random_scale": False,
                      "flip_x": False, "flip_y": False}],
        "train": {"batch_size": 1, "steps": t["steps"], "gradient_accumulation": 1,
                  "train_unet": True, "train_text_encoder": False, "gradient_checkpointing": True,
                  "cache_text_embeddings": not sdxl, "noise_scheduler": "ddpm" if sdxl else "flowmatch",
                  "optimizer": t.get("optimizer", "adamw8bit"), "lr": t["lr"], "dtype": "bf16",
                  "disable_sampling": t.get('sample_every', 250) == 0,
                  **({"optimizer_params": {"weight_decay": 0.0001}, "timestep_type": "sigmoid"}
                     if t.get("optimizer") == "automagic3" else {})},
        "model": model_config,
        "sample": {"sampler": "ddpm" if sdxl else "flowmatch", "sample_every": t.get('sample_every', 250) or 250,
                   "width": t["resolution"], "height": t["resolution"], "samples": samples, "neg": "",
                   "seed": t["seed"], "walk_seed": False, "guidance_scale": 5 if sdxl else (1 if model["id"] == "qwen21" else 3.5),
                   "sample_steps": t.get('sample_steps', 40 if model["id"] == "qwen21" else 30)}}]},
        "meta": {}}
    config["meta"] = public_metadata(project, config["config"]["process"][0], model)
    return config

def export_run(store, project, images, run_id, settings):
    project = {**project, '_preview_suite': effective(project, images, project.get('_preview_suite'))}
    root = store.root / "runs" / run_id
    dataset, output = root / "dataset", root / "output"
    dataset.mkdir(parents=True)
    output.mkdir()
    originals = root / "originals"
    originals.mkdir()
    prepared_images = []
    masked = project['kind'] == 'object' and any(i.get('included', True) and objects.options(i.get('object_target'))['mask'] != 'none' for i in images)
    masks = root / 'masks'
    if masked:
        masks.mkdir()
    for item in images:
        if not item.get("included", True):
            continue
        source = store.root / "images" / item["project"] / (item["id"] + item["extension"])
        item = {**item, **metadata(source)}
        geometry = plan(project, item)
        if project['kind'] == 'object' and objects.selection(project, item)['status'] != 'ready':
            raise ValueError('Cannot export an unresolved Object target')
        shutil.copyfile(source, originals / source.name)
        render(source, geometry).save(dataset / (item["id"] + ".png"), format="PNG")
        if masked:
            objects.render_mask(source, project, item, geometry).save(masks / (item['id'] + '.png'))
        prepared_images.append(decorate(project, item))
        (dataset / (item["id"] + ".txt")).write_text(item["caption"], encoding="utf-8")
    config = config_for(project, dataset.resolve(), output.resolve(), run_id)
    if masked:
        config['config']['process'][0]['datasets'][0].update(mask_path=str(masks.resolve()), mask_min_value=0, invert_mask=False)
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    revision = "unknown"
    try:
        revision = subprocess.check_output(["git", "-C", settings["toolkit_path"], "rev-parse", "HEAD"], text=True, timeout=8, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    manifest = {"project": {k: v for k, v in project.items() if not k.startswith('_')}, "intent": intent(project), "images": prepared_images, "backend_commit": revision,
                'preview_suite': effective(project, images, project.get('_preview_suite')),
                "caption_policy": "prepared-v1" if project['kind'] == 'character' else f"{project['kind']}-v{CATEGORY_VERSION}",
                "preparation_version": 1, "run_id": run_id}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # ZIP config uses paths relative to the extracted bundle, not this machine.
    portable = config_for(project, "./dataset", "./output", run_id)
    if masked:
        portable['config']['process'][0]['datasets'][0].update(mask_path='./masks', mask_min_value=0, invert_mask=False)
    with zipfile.ZipFile(root / "training-bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("config.json", json.dumps(portable, indent=2))
        bundle.writestr("manifest.json", json.dumps(manifest, indent=2))
        for file in dataset.iterdir():
            bundle.write(file, "dataset/" + file.name)
        if masked:
            for file in masks.iterdir():
                bundle.write(file, 'masks/' + file.name)
        bundle.writestr("README.txt", "Extract this bundle. Activate your AI Toolkit environment and run from this folder:\npython /path/to/ai-toolkit/run.py config.json\nReview config.json before training.\n")
    return root
