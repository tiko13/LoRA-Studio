# LoRA Studio user guide

[Back to the overview](../README.md)

A local workspace for intent-aware dataset captioning and AI Toolkit training. The interface follows Prompt Studio's visual language. Prompt Studio integration is optional; it uses the existing HTTP API and requires no changes to ComfyUI or Prompt Studio.

Dataset thumbnails, prepared images, training previews and checkpoint comparisons open in an in-window image popup. Click outside the image, press Escape or use the close button to dismiss it. Use **Review image** on a dataset card to edit captions and preparation.

## Start

Open the status dot at the top right and choose **Restart LoRA Studio** to reload the server in the existing console. Studio saves pending project changes first and reconnects without reloading the page, preserving open forms. Restart waits until jobs, comparisons and installation work are finished; resolve any interrupted training run first. After installing this feature, start Studio once from its launcher to enable the new endpoint. Refresh the browser separately when you want to load updated interface code.

Double-click `start.bat` with Python 3.10 or newer installed. First launch creates an ignored `.venv` and installs Pillow for image decoding, orientation and preparation, then opens **http://127.0.0.1:8787** in your browser. Keep the console open while using Studio; press Ctrl+C to stop it. Reopening the launcher opens the existing Studio instance. From a terminal, use `start.bat --port 8788` for a different port or `start.bat --help` for all options. Manual setup: `python -m venv .venv`, `.venv\Scripts\python -m pip install -r requirements.txt`, then `.venv\Scripts\python server.py --open`. No npm build is required. AI Toolkit and the vision server keep their own environments and dependencies.

The initial setup wizard opens on first use and checks saved prerequisites in the background. Reopen it from **Settings → Run setup**. Choose an existing AI Toolkit folder (containing `run.py`) and its Python executable, or use **Download & install**. The latter clones the official [AI Toolkit repository](https://github.com/ostris/ai-toolkit) and runs its experimental `python -m manager install` interface in a separate folder under ignored `data/setup/`. Git and at least 10 GB of free space are required; model weights and training outputs require additional space. Installation does not start training or launch the Toolkit UI. The upstream manager provisions its hardware-specific dependencies; consult its installation guide for unsupported hardware or driver requirements.

Installation progress survives closing the wizard. Keep the Studio server running; cancellation preserves downloaded files, and retry reuses a completed managed clone. After a server restart, an unfinished installation is marked interrupted and cannot be retried while its recorded process is still alive. Diagnostics stay in `data/setup/install.log`. Existing Toolkit installations are never updated by this wizard. A successful managed install selects its own runtime unless the backend selection changed while installation was running.

The LLM step offers **Use Prompt Studio backend (recommended)** and **Use standalone backend**. Standalone uses a vendored copy of Prompt Studio's native Llama.cpp, Ollama and KoboldCpp providers; no ComfyUI installation is required. Both paths can send a generated color swatch for a basic vision check. Core-package, PyTorch/CUDA, architecture and endpoint checks are not proof of a successful training run, model access, caption accuracy or sufficient VRAM for a chosen model. Dedicated captioners remain in Settings. **Set up later** and **Save & open workspace** retain valid connection edits and remember dismissal; missing prerequisites remain visible when checks run. Concurrent settings edits are revision-checked.

Machine-specific paths and setup state are stored in the ignored `data/` directory. Keep this local app bound to loopback; it is not designed for remote hosting.

The top-right status dot opens connection details for the vision provider and AI Toolkit, with a refresh action. Green means connections are available, amber means attention is needed, red means unavailable or a failed check, and grey means checking. Standalone endpoint connectivity does not verify vision support.

## First workflow

**New LoRA** reuses any project with no dataset images. If that project is already selected, a red message explains that an empty LoRA project already exists. Otherwise, Studio opens it and shows a green message. Add at least one image before creating another project; excluded images still count as dataset images.

1. **Base info:** choose the LoRA type and base model, name it, and optionally describe the target. New projects receive an editable trigger automatically, shown beside the LoRA name. Keep/change attributes and extra rules are under **Optional · learning rules**.
2. **Dataset:** import PNG, JPEG, or WebP images. Square tiles show the whole image; duplicates are skipped. Continue to Create, or open an image to edit its caption, crop, inclusion or review overrides. Preparation, person counts, captioning and coverage controls are under the optional section.
3. **Create:** click **Create LoRA**. Automatic mode prepares missing or stale automatic captions, generates target-aware preview prompts, freezes the run and launches AI Toolkit. Manual captions and overrides remain preserved. Blocking errors are always shown. Presets, custom prompts, hyperparameters, review-first mode, prepare-only actions and export are optional controls.

An empty learning-target description uses a backend default for the selected LoRA type; the field stays empty and displays the default as a hint. Person / character learns likeness by default, while clothing or visible state of undress, pose, expression, hairstyle, accessories, background, lighting and framing remain changeable. Object learns the main object’s appearance; Style learns visual treatment; Action / concept follows the selected concept family. Explicit descriptions replace this fallback, and Keep/Change choices remain in effect. Ambiguous objects still require a hint or manual selection. These rules guide captions; dataset variety is still needed to avoid learning unwanted context.

**Dashboard** remains a top tab for training progress, checkpoints, test generations and pause/resume/cancel. **Runs & activity** is available from Create and Dashboard. The first training run may download model components; access and authentication belong to the AI Toolkit environment.

Preparation becomes permanently read-only when the trainer reports its first optimization step (the first loss can appear at counter 0). Base info, images, captions, preparation overrides, preview prompts, public metadata and training settings stay fixed after pause, cancellation, failure or completion. Frozen inputs are also held during the launch handoff; a failure before any optimization step releases that temporary hold. Existing runs with recorded training progress acquire the same persistent lock when loaded by the server.

Use **Copy to new run** in the locked project's banner to create and open a fresh editable project. It copies all saved preparation, including excluded source images, caption/observation histories, review overrides and preview prompts, with independent image files. Training jobs, checkpoints and outputs stay with the original project. Copying does not start training. Unsaved browser drafts remain available on the original project for recovery or download.

### Baseline presets and preview prompts

Save a run's frozen configuration from Dashboard's configuration section as a named local preset, optionally the default for new projects of that model. Presets capture supported training and preparation settings, not the target, dataset, prompts or GPU assignment. Applying one to an existing project is explicit and revision-checked. A saved baseline is an accepted starting point, not a universal optimum.

Automatic preview prompts use the target type, keep/change rules and a bounded selection of prepared captions. Three prompts cover familiar conditions, recombination and a new setting; person suites include a portrait when framing can change. Prompts must include the trigger and avoid caption copies. The local text LLM gets one correction attempt; if unavailable or invalid, type-aware starter prompts keep the one-stop path working. Custom rules can require review. This validates prompt structure, not generated-image quality.

Preview prompts are cached per learning intent and dataset revision, editable under Create's optional controls, and frozen with each run for checkpoint comparison. Switching automatic/custom preserves custom text; replacements retain restorable history. New generated prompts do not alter an existing run or its resume configuration.

The current model catalog includes **Krea 2 Raw**, **Qwen Image 2.1**, **FLUX.1 Dev**, and **SDXL**. Installed architecture discovery gates training. This is an extensible adapter, not a claim that every architecture or hardware configuration has been trained and validated.

## Optional Prompt Studio connection

### Caption source

Choose **Settings → Captioning**: **Connected local LLM** (the existing default), **JoyCaption Beta One**, or **Custom local caption model**. Choices and hidden configuration fields are retained when switching. Changing the source does not replace existing captions; **Prepare captions** fills missing or outdated automatic captions and preserves manual edits.

JoyCaption and custom captioners first describe the prepared image. The connected LLM then applies the project's learning rules to those observations using text only. Original observations are retained separately from the final caption and in caption history. Every generated caption remains a draft for review: text processing cannot correct incorrect visual observations.

Object and Style projects also apply that evidence-based text formatter to observations from **Connected local LLM**. This adds a text-only request after the visual pass and checks attribute assignments. Style captions must retain depicted content, with common rendering-trait leaks rejected across fields. Automatic Object localization requires a vision-capable connected model even when JoyCaption supplies the later observations; manual rectangles and explicit full-image selections can bypass localization.

The text stage extracts categorized facts linked to numbered observation sentences. Studio excludes categories set to **Keep consistent** and assembles the remaining phrases with the trigger and subject class. Exposure is treated as clothing state, separately from body proportions: retaining body appearance implicitly does not discard explicitly observed visible anatomy. If the LLM omits an affirmative exposure fact, Studio retains its neutral body-part name and visibility directly from the supporting observation sentence, recording that restoration in the fact metadata. Explicitly stated nude/topless subject states are also retained; nude-colored clothing does not establish nudity. Negated or uncertain source sentences never force exposure, and generic genital descriptions do not imply a specific body part. Validation checks common stable-trait leaks and requires each phrase to match its cited source sentence, allowing limited grammatical normalization and removal of kept appearance modifiers. When editor word order differs, a compact matching source excerpt can be used instead; the source wording and proposed phrase are retained for inspection. Partial coverage is retained without asserting complete exposure. After bounded retries, unsupported text-editor phrases may be omitted with an explicit review warning while supported facts remain; unusable or policy-invalid responses preserve the prior caption and report a failure. These checks establish source extraction, not visual accuracy. Optional custom keep/change rules get a separate refinement pass followed by the same checks; conflicts with the built-in policy should be surfaced as warnings.

Under **Dataset → Reprocess caption text**, **Reprocess saved observations** regenerates automatic captions using the saved JoyCaption/custom observations and current rules without loading the image captioner. Manual captions are skipped, revision checks protect concurrent edits, and old captions/observations remain in history. Changed training crops require fresh observations; saved evidence from an older crop cannot be reused. Reprocessed captions remain unreviewed drafts.

**Automatic · prepare and train** provides the short workflow: define the subject, upload images, click **Train**. It prepares missing or outdated automatic captions, runs final preflight, freezes the dataset and starts the maintained training backend in one cancellable job. Caption and visual-check warnings remain advisory in this mode; captions are not falsely marked human-reviewed. Invalid configuration, missing triggers, unreadable images, inference failures and stale manual captions still stop launch. Edits during preparation abort launch while preserving completed work. New projects default to Automatic; existing projects retain Review first until changed.

To stop at dataset preparation, open **Prepare without training → Prepare dataset only**. This uses the same automatic caption preparation and immutable export path, then stops before the trainer launch. It preserves manual captions and overrides, leaves automatic captions as drafts, and exports the current starter configuration for inspection. The API equivalent is `POST /api/projects/{id}/prepare`; this endpoint never starts training. A complete preparation job means files were prepared successfully, not that model-generated captions were independently verified or training settings were optimized.

Automatic preparation also refreshes outdated dedicated-captioner text from its saved observations when the crop remains current. Text validation rejects common lettering leaks and individual additions, changed relationships or lost qualifiers that do not match the cited source phrases. In image review, **Caption evidence** shows accepted phrases, their supporting observations and any omitted text-editor claims. A correct extraction can still repeat a visual-model error, including incorrect laterality or anatomy; text-only processing cannot settle that. Matching is deliberately conservative and can omit valid paraphrases, so omission warnings need review. The default remains one visual observation pass plus bounded text processing; additional visual checks are optional. One dataset is regression evidence, not a general caption-accuracy benchmark. Factual review flags remain overridable and separate from successful file preparation.

**Review first** keeps caption warnings and unresolved visual findings in preflight. In Dataset, filter to **Needs attention** or **Unreviewed drafts**, open an image, edit its caption or inclusion, and use **Save & next**. **Keep this caption despite the visual-check findings** is an explicit override tied to that particular result. It is invalidated by a new check; caption, crop or learning-rule changes make earlier evidence outdated. Manual captions, original observations and review history are preserved.

**Ask a focused visual question** is an optional check through the connected vision LLM (Prompt Studio or a standalone image-capable endpoint). Select categories and optionally ask one precise question. The model reports supported, contradicted, missing or uncertain evidence alongside the prepared image. These are fallible suggestions: agreement is not approval, earlier review concerns are not cleared by model agreement, and checks never silently rewrite captions, count overrides or inclusion. The check is not a mandatory stage in Automatic mode. Save caption/crop edits first; cancelled, malformed or conflicting results preserve the previous review.

For JoyCaption, supply a trusted local `llama-server` executable, a compatible Beta One GGUF, its matching vision projector and the CUDA device index. Studio starts a private loopback runtime for the observation stage and closes it before text processing, including on failure or cancellation. Its input uses the model's [official 384 × 384 processor](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava/blob/main/preprocessor_config.json); dataset originals and training dimensions are unchanged. Models are not downloaded automatically. Allow enough GPU memory for the captioner alongside any externally managed LLM. In connected mode, Studio checks ComfyUI and Prompt Studio activity before releasing idle generation models; it yields if new ComfyUI work appears between images. Standalone mode does not contact ComfyUI.

Custom mode accepts another locally hosted, image-capable OpenAI-compatible endpoint and model ID, including a compatible V4.5 or Gliese deployment. Studio does not manage that server's lifecycle or preprocessing. Endpoint discovery and file checks establish configuration, not image support or caption accuracy. Private observations and owned-runtime logs live in `data/caption-jobs/<id>/`.

### LLM connection

- **Use Prompt Studio backend (recommended)** reuses the active chat's saved provider and shared scheduling. Requests do not add chat messages. Uploaded image references remain in Prompt Studio's own image storage.
- **Check ComfyUI & Manager** detects the configured local installation. When Manager is available and Prompt Studio is missing, an installation panel explains the download and dependency changes before **Install Prompt Studio via Manager** submits the official repository. Installation requires an idle ComfyUI queue and Manager. LoRA Studio never changes Manager's security policy or restarts ComfyUI. A timeout/restart leaves a recovery message; inspect Manager's console before retrying. Manual installation instructions are always available.
- **Use standalone backend** offers **Llama.cpp**, **Ollama** and **KoboldCpp**, using the copied native provider protocols, streaming, capability checks, reasoning/context handling and endpoint scheduling. Backend selection is explicit. Previously saved Automatic and generic OpenAI-compatible connections remain supported without silently migrating their model choices.
- Standalone retains each provider's endpoint, model, thinking/instruct modes, separate thinking sampler, response budget, seed, stop sequences and timeout. Named LLM profiles are saved locally and can be reused across providers. Only enable modes supported by the selected model. Hidden provider values survive switching.
- Llama.cpp includes Prompt Studio's native **New/Edit config** and **Manage profiles** dialogs: model/projector paths, context/GPU settings, MTP and sampler settings. Writes use revision checks and preserve unknown config fields. A managed profile determines its endpoint and model; the next inference request applies profile edits to a Studio-owned server. Start/stop/restart and autostart manage only Studio-owned processes. Explicit server changes are blocked during Studio jobs and unresolved interrupted training. Existing external servers are preserved. Native file browsing is available on Windows; paths can be entered on other platforms.
- Unless **Keep models loaded** is enabled, training releases models this standalone backend used, including earlier provider selections. Ollama uses its unload API; KoboldCpp requires Admin Mode and an Admin Directory (optional password via `PROMPT_STUDIO_KOBOLD_ADMIN_PASSWORD` in Studio's environment). Studio stops its owned Llama.cpp server; an externally managed Llama.cpp server requires router unloading support. A failed release blocks training with a recovery error. Keep loaded is intended for separate LLM/training GPUs.
- When **Connected local LLM** reads images, missing vision capability is an error, not a silent provider switch. Dedicated captioners only require text support from the connected LLM. Settings are captured at the beginning of each caption job. Capability discovery is not caption-accuracy validation.
- Connected mode checks ComfyUI's queue and Prompt Studio jobs before requesting model release and `/free`. ComfyUI stays running. This is a cooperative handoff, not a global reservation: avoid starting new generations during training. Standalone mode does not contact ComfyUI.

Open **Settings → Backend settings** to return directly to this step. Only loopback HTTP endpoints are accepted. No API keys are required or stored by this version. Provider settings and named profiles live in the ignored settings store; Llama.cpp configs, process recovery and logs live under `data/llm/`. Managed servers are restricted to loopback even if a profile is edited outside the UI.

The GPL-3.0 provider subset and UI editors are attributed in [the vendor notice](../studio/vendor/promptstudio/NOTICE.md). `scripts/vendor_promptstudio.py <checkout>` refreshes only reviewed source dependencies from a Prompt Studio checkout; it never copies local profiles, settings, model files or user content. LoRA Studio's integration/lifecycle adapter is `studio/llm.py`.

## Action / concept preparation

Choose **Action / concept**, describe what should repeat, and select a concept type in Base info. Pose, facial expression, camera angle, action, interaction, spatial relationship, lighting, composition, focus and weather/effect presets adjust the Auto rules. Explicit Keep/Change choices survive preset and LoRA-type changes. The default General option preserves the original action/defining-prop policy.

The trigger carries the learned concept. Captions retain visible variable participants and context, with separate controls for body pose, facial expression, viewing direction and shot distance. Both connected vision and dedicated captioners use evidence-linked text extraction; original observations and prior drafts remain available. Review the prepared crop for defining joints, facial detail, contact points or composition. Concept presence suggestions and visual checks are advisory, not proof of the target or automatic image exclusion.

Saved Concept observations can be reprocessed without another image call. Manual captions are skipped, changed crops need fresh observations, and changing the learning target makes earlier concept-presence suggestions stale. A preparation job freezes matching image/caption pairs and the selected concept rules for export. It does not alter the source images or start training unless you choose Train.

## Persistence and reproducibility

`data/studio.sqlite3` stores projects, caption history, settings and jobs. Imported images live in `data/images/`. Each run freezes included originals in `data/runs/<id>/originals/`, prepared PNGs and captions in `dataset/`, and its configuration and geometry manifest. Preview, captioning and export share the same server-side renderer. Later edits do not affect that snapshot. Export bundles contain prepared images and relative dataset/output paths. Browser image-review drafts survive errors and conflicts; restore them against the latest revision and review before saving.

Project writes use revision checks. Browser drafts survive reloads and require an explicit restore/discard choice. Manual caption changes require review against the current rules. After a server restart, Studio reconnects to surviving trainers using their executable and OS creation marker; older runs require an exact frozen-config command match. The dashboard resumes monitoring and AI Toolkit controls without launching another trainer. Completion while disconnected requires the backend's completed status and a readable final LoRA at the planned step. Saved checkpoints and previews populate Results for existing runs too. Unverifiable processes and jobs interrupted before trainer launch remain explicitly interrupted. Restart only the Studio server, not its process tree or the computer; those stop the trainer too. Caption/preparation jobs do not reconnect.

Back up the entire `data/` directory with the application stopped, including SQLite sidecar files if present. Datasets, local settings, outputs and test artifacts are excluded from Git.

## Image preparation for every category

Without a saved default preset, new Krea 2 and Qwen Image 2.1 projects start at a **1024 pixel-area target** (approximately one megapixel), with aspect-ratio bucketing. Existing projects keep their saved resolution. Width and height align to **16 pixels for Krea 2** and **32 for Qwen 2.1**. These are model grids, not square-image requirements. The preparation panel supports 256–2048 targets for these profiles; targets above 1024 are experimental and need a memory/quality comparison. Smaller sources stay close to their native area; grid alignment may slightly enlarge or crop them. No AI upscaling is applied.

The default uses proportional resizing and minimal cropping. Open an image review, then click the original image or **Edit crop & mask**. Drag inside the crop to move it, or drag any of its eight edge/corner handles to resize. Choose a free or locked aspect ratio, center the crop, use the full image, or return to automatic cropping. Precise coordinates use original-image pixels; arrow keys move the crop or a focused edge by one pixel (Shift: ten). Use the mouse wheel to zoom around the cursor, or the + / − and Fit view buttons. Zoom changes only the editing view. Optional grey padding retains the entire selected region but becomes training content.

**Hide brush** and **Hide rectangle** cover unwanted visible content, such as watermarks, with neutral grey. **Restore brush** reveals original pixels again. The circular cursor shows the brush's footprint at the current zoom; the size slider displays its diameter in original-image pixels. Undo/redo includes both crop and mask edits; Clear mask removes all hiding. This is reversible content masking, not inpainting or an Object loss mask: grey remains visible training content. The exact training preview shows the backend's grid adjustment and masking. **Apply crop & mask** puts edits in the recoverable image-review draft; **Save image review** persists them with revision checks. Cancel leaves the review unchanged. Crop/mask edits make earlier caption evidence stale and require review or fresh observations. Original bytes remain intact; captioning and frozen training exports use the same prepared pixels.

Exported dimensions are already aligned, and AI Toolkit receives explicit bucketing with random crops, random scaling and flips disabled. Samples can cycle through square, portrait and landscape shapes or use one selected shape. Multi-resolution curriculum training is not enabled implicitly.

Krea trains on **Raw**; assess saved adapters on **Turbo** for intended inference quality. Current in-training samples still use Raw and are labelled in preflight as insufficient for that evaluation. Qwen's native 2K generation does not require 2K LoRA training. References: [Krea training guide](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_krea2.md), [Qwen training guide](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_qwenimage21.md), [Qwen model card](https://huggingface.co/Qwen/Qwen-Image-2.1).

**Check person counts** optionally runs Torchvision's SSDLite MobileNetV3 COCO detector on CPU. Configure a Python with compatible `torch`, `torchvision` and `Pillow` under Settings, or use the existing AI Toolkit Python. First use downloads approximately 14 MB of official weights into ignored `data/models/`; images are never uploaded by this detector. Results/logs stay under ignored `data/analysis/`. Interrupted or failed checks do not imply a count of zero. [Detector documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html).

Counts refer to the original oriented image. A confidence threshold of 0.5 counts detections; lower-confidence candidates are marked uncertain. Small boxes substantially contained in larger person boxes are reported as an ambiguous count range, retaining all boxes: geometry cannot distinguish a head/body duplicate from overlapping people. This assessment also applies to previously saved detector results. Close-ups, occlusions, reflections and unusual poses can cause misses or false positives. It does **not** match identities or compute face embeddings, and one detection is not proof of a single person. Correct any count manually, clear it to use the detector again, or keep an image despite flags. Preparation flags never automatically exclude images or block training; acknowledgement is tracked separately from caption review.

Coverage summaries report aspect ratios and optional framing/viewpoint/lighting tags. Vision tags are suggestions, manual corrections are retained, and stale crop suggestions are excluded from coverage. Perceptual-hash matches are duplicate candidates, not proof of duplicated content or identity. There are no fixed diversity quotas or automatic aesthetic/identity scores. Review meaningful face detail, poses, expressions, clothing and settings in the intended output domains. Passing preparation checks is not a trained-quality guarantee.

### Object, Style and Action / concept preparation

The selected category controls the connected vision prompt, JoyCaption/custom-model observation prompt, evidence-linked text processing, optional visual checks, coverage tags and dataset advisories. Person keeps its established instructions and checks. Person-count detection, overrides and anatomy-specific caption rules are unavailable for the other categories.

- **Object / product:** identify the target class; follow keep/change choices for structure, material, colour and integral markings. Check target visibility, occlusion, viewpoint, placement and background variety. Reflections, packaging and multiple instances require review. Relevant legible markings can remain when changeable; overlays and watermarks are omitted.
- **Style:** use the trigger as a style modifier and describe variable depicted content. Check content/composition coverage and usable texture/linework detail. Keep/change choices decide which rendering traits are implicit. Krea 2 guidance recommends a descriptive style trigger phrase; the application explains this without replacing a saved trigger.
- **Action / concept:** use the trigger for the concept, preserve variable participant roles and relationship direction, and check visible action evidence, props/contact, context and crop completeness. Participants may be objects, animals or people; a still image does not prove intent or an event sequence.

Coverage suggestions are fallible model output, not automatic recognition scores. Unknown, partial or narrow coverage produces advisories, never automatic exclusion or a training block. Optional visual checks inspect the selected category and respect intentional KEEP omissions. Manual coverage corrections are stored separately per category and survive switching types. Crop changes expire automatic suggestions; corrections remain until edited. Earlier observations without category provenance need a new visual pass for non-person projects; use **Prepare captions** or **Prepare dataset only**. Manual captions remain preserved and require explicit review if their learning policy changed.

Model guidance covers Krea 2 Raw, Qwen Image 2.1, FLUX.1 Dev and SDXL. Geometry, prepared PNG/caption pairs, caption order and frozen run snapshots remain shared. No category silently changes your resolution or hyperparameters. See [category and model guidance](category-guidance.md) for official and community sources, implementation decisions and validation limits.

### Object selection and preparation

For Object projects, **Prepare dataset only** and automatic **Create LoRA** locate the described object before captioning. The connected local vision model sees the original image and proposes a bounding rectangle. Preparation adds an 8% margin around it, aligns to the selected model's grid, and captions the resulting training image. Automatic fitting uses padding if grid alignment would cut the selected object's bounds. Original files remain untouched. The separate **Find objects** action stops after localization so selections can be inspected first.

If several objects match or the target is absent, preparation stops before export. Open the image, add a distinguishing hint (for example, “the striped teapot on the left”), draw a target rectangle, explicitly choose **Keep entire image**, or exclude it. Manual selections are preserved; changing a hint or project description expires automatic localization. Target choices survive switching categories and apply only to Object. Crop changes require fresh automatic captions or explicit review of a manual caption.

Image review also offers a margin control, precise coordinates, a polygon loss mask, a rectangular loss mask, and source-alpha masking. These are optional refinements, not automatic pixel segmentation. The ordinary path crops and captions without masking or another downloaded model. A rectangular mask includes background inside the rectangle; a polygon lets the user follow the visible object. Transparent source images can use their actual alpha channel. Background loss weight defaults to 0.1 when masking is enabled and can range from 0 to 1.

The mask preview and exported grayscale sidecar share the training image's crop, resize and padding. AI Toolkit receives `mask_path`, `mask_min_value: 0`, and matching filenames; each image's selected background weight is baked into the mask. Images without a mask get a full-weight sidecar in mixed masked datasets. Portable bundles contain the masks with relative paths. The installed backend must expose the supported loss-mask contract. Masks do not erase pixels, replace factual background captions, or guarantee disentanglement. Localization and captions remain fallible suggestions: inspect fine protrusions, overlap, reflections and repeated props, and use varied viewpoints and settings.

### Style dataset preparation

Select **Style**, describe the intended look, and import your images. **Create LoRA** prepares images and captions before training; **Dataset → Optional → Prepare dataset only** runs the same preparation and freezes an export without starting training. An optional family hint supports photography/instant film, cartoons, drawing/ink/pencil, watercolor/painting and pixel art. Any/mixed medium works without selecting a family. The hint changes instructions, never image pixels or saved triggers, and remains saved if you switch project types.

By default the trigger carries palette, texture, linework, medium and print borders; captions describe depicted subjects, actions and settings. A red mug or wooden table is content and can remain even when palette/texture are kept. Optional learning rules can make rendering traits explicit. Descriptive Krea 2 triggers remain supported. The formatter keeps source-linked facts, preserves uncertainty and rejects a framing-only caption with no subject. Narrow wording checks catch common style leaks; they cannot prove caption accuracy or complete separation of content and style.

Review the exact training preview for composition, print borders, thin strokes and pixel edges. Minimal crop and full-image padding use the existing model-specific buckets; no automatic restyling, sharpening, border removal or background replacement is performed. Visible borders and repeated content are advisory. Padding, watermarks and signatures still exist in the training pixels even when omitted from captions. Keep one coherent style and enough varied content for the flexibility you want.

Connected Style observations can be reused through **Reprocess saved observations**, alongside JoyCaption/custom observations. Original observations and previous captions are retained, manual captions are skipped, and a changed crop requires fresh visual evidence. Changing style family or rules marks old captions for review. Exports freeze the selected family, policy version, geometry, images and captions.

## Scope of this first version

Caption preparation, import, editable starter settings, immutable export, trainer launch, stop, logs and output downloads are implemented. Captions describe visible evidence without inventing hidden details or sanitizing relevant factual terminology. Keep/change choices guide language; they do not edit images or guarantee disentanglement. Review model output for factual accuracy.

Automatic dataset generation, semantic dataset curation, quality scoring, additional trainer adapters and automatic hyperparameter tuning are future work. With no source dataset, begin by collecting suitable images. Starter defaults are not hardware sizing or quality guarantees. Config parsing and a live vision request do not establish training quality; validate a real run on your own dataset.

Advanced training settings offer AdamW8Bit (the existing default) or AI Toolkit's experimental Automagic3. Automagic3 uses the selected learning rate as its initial value and adapts it during training; the adapter uses sigmoid timesteps, weight decay 0.0001, batch size 1 and gradient accumulation 1. These are selectable working settings, not an automatic quality optimum. Optimizer selection is saved with the project and each immutable run.

Krea Raw weights can come from the original gated Krea repository or the public Comfy-Org BF16 repack, under the same Krea license. Both choices train Raw; the selected source is recorded in the frozen configuration. The original source is the built-in default; a saved model preset can select another source. Model files download on first use and require sufficient disk space and network access.

## Training monitoring, checkpoints and previews

### Embedded LoRA metadata

In **Create → LoRA file metadata · optional**, edit the public description, author, license, trigger phrases (one per line) and recommended strength. The public description starts empty and is separate from the private learning-target description. Blank author/license/description mean unspecified. Blank trigger phrases follow the current training trigger as one phrase; custom phrases describe inference usage without changing captions or training. Blank strength is unspecified; zero and negative values are preserved (accepted range −10 to 10). Strength is a suggestion, not a tested optimum or a training parameter.

These fields use the existing project revision checks, autosave and browser draft recovery. Metadata edits do not invalidate captions. Each newly exported/prepared training run freezes the public fields and effective training settings in its configuration; AI Toolkit embeds them into each saved `.safetensors` checkpoint and final LoRA. Edits affect future runs only. Already exported/running/resumed runs and existing LoRA files keep their original metadata.

Only the LoRA name, explicitly authored public fields and allowlisted numeric/boolean/enum training settings are exported into the Studio metadata. Private learning instructions, captions, observations, preview prompts, dataset filenames, local model/dataset/output paths, endpoints, GPU selection and the intent fingerprint are excluded. The base model is identified by its public catalog ID, not a local path. The training bundle and manifest still contain the private material needed to train; this privacy boundary applies to the embedded LoRA metadata, not the whole bundle. Text deliberately entered into the public fields is included verbatim after outer whitespace trimming.

#### Reader contract for other applications and agents

Read the Safetensors header's `__metadata__["lora_studio"]`. Its value is a **JSON-encoded string**; parse it once. With Python's `safe_open`, `metadata()` returns that header dictionary without loading tensor values. The decoded v1 object contains:

| Field | JSON type | Meaning |
| --- | --- | --- |
| `schema_version` | integer | Currently `1`; check before interpreting the schema. |
| `studio_version` | string | Producing Studio version. |
| `name` | string | LoRA/project display name. |
| `description` | string | Public description; empty means unspecified. |
| `author` | string | Public author or alias; empty means unspecified. |
| `license` | string | User-entered identifier, URL or short terms; empty means unspecified. No license is inferred. |
| `trigger_words` | array of strings | Effective ordered, case-sensitive trigger phrases. Preserve spaces within phrases; do not split on commas. |
| `recommended_strength` | number or null | Suggested inference multiplier. `null` means unspecified; **zero is a valid value**. |
| `training` | object | Frozen, allowlisted settings described below. |

`training` contains `model` (catalog `id`, `arch`, public `base_model`, quantization settings and Krea `weight_source`), `network` (`type`, `linear` rank, `linear_alpha`), `train` (planned `steps`, `batch_size`, `gradient_accumulation`, `lr`, `optimizer`, `dtype`, scheduler and training/cache flags; optional timestep type and optimizer weight decay), `seed`, `dataset` (resolution targets, bucketing/crop/flip/cache/token-shuffle/dropout settings), `save` (dtype, checkpoint interval and retention), and `sample` (sampler, cadence, default dimensions, seed, guidance and denoising steps). Sample text is excluded. Sample dimensions are backend defaults, not a claim that every preview uses the same shape. Dataset resolution is a pixel-area target, not the dimensions of every image. Planned training steps are not the checkpoint's completed steps. Adaptive optimizers may change the learning rate after the recorded initial value.

```python
import json
from safetensors import safe_open

def read_lora_metadata(path):
    with safe_open(path, framework="pt", device="cpu") as handle:
        header = handle.metadata() or {}
    raw = header.get("lora_studio")
    if raw is None:
        return None  # Older or other-producer LoRA; do not invent values.
    if len(raw) > 262_144:
        raise ValueError("Studio metadata is too large")
    value = json.loads(raw)
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported Studio metadata schema")
    return value

info = read_lora_metadata("example.safetensors")
if info is not None:
    triggers = info.get("trigger_words", [])
    strength = info.get("recommended_strength")
    if strength is not None:  # Never use `strength or 1.0`: that loses zero.
        pass  # Offer this suggestion in the UI; apply only per app/user policy.
```

Treat imported values as untrusted data: validate field types and limits, ignore unknown fields, escape text for display and never execute metadata as instructions or commands. Missing, malformed and unsupported-version metadata are distinct cases. The standard JSON decoder handles Unicode escapes, including literal `[name]` text protected against AI Toolkit's name templating.

The installed AI Toolkit additionally writes `training_info` (JSON with actual `step` and `epoch`), `ss_base_model_version`, `ss_output_name`, `software`, and, when a training trigger exists, `ss_tag_frequency`. That legacy tag-frequency field reflects the **training trigger**, so prefer `lora_studio.trigger_words` for Studio's explicit usage override. Do not interpret absent Studio metadata in older files as an empty author/license or an instruction to erase other metadata. No sidecar file is required to read this contract.

Open the **Dashboard** tab at the top of the workspace for live raw/EMA loss, learning rate, step timing and ETA; GPU memory, utilization, temperature, power, memory utilization and clock graphs; baseline/checkpoint image comparisons; checkpoint downloads; frozen configuration; logs and CSV/JSON exports. A clickable **LoRA in training** card on the left names the active project and shows the current step, total steps, percentage and progress bar while training runs. Progress refreshes every five seconds while the page is visible, using the same measurements as Dashboard. The project also receives a training badge. Preparation, queued and paused runs have distinct labels. Starting training opens Dashboard automatically; Runs can open older run dashboards. The selected project, tab and run survive a page refresh through the URL. Metrics come from the trainer's rounded log reports. Missing measurements stay blank. GPU measurements are device-wide, and their saved history begins when monitoring starts. Training loss is not validation loss or an automatic quality score.

Under **Adjust training settings**, choose the checkpoint interval in steps, checkpoints to retain, preview interval, denoising steps, preview shapes and prompts. Defaults save and generate previews every 250 steps, retain eight step checkpoints, and use 30 denoising steps (40 for Qwen). Set the preview interval to zero to disable automatic previews. The backend also saves final weights. Prompts and seeds remain fixed for comparisons. These choices are frozen per run; editing the project affects future runs only. Frequent previews increase runtime and frequent checkpoints increase disk activity.

**Pause & save** asks the installed AI Toolkit DiffusionTrainer to save weights and optimizer state, then stops the owned process and releases its GPU memory. **Resume checkpoint** uses that run's frozen data and configuration. **Cancel run** stops immediately and retains files; work after the last complete checkpoint is lost. Resume is blocked for incomplete or structurally damaged saves, and stops if the backend reports that it cannot load optimizer state. The toolkit retains one optimizer state for its latest checkpoint: older weight files remain available for inference, but arbitrary older-step optimizer resume is not offered. Backend resume restores weights, optimizer state and step, but resets configured learning rate and does not promise identical random/data-loader order.

After completion, **Continue training** is available in Results and the training dashboard. It defaults to one additional checkpoint interval; choose 1–1,000 intervals. For example, a completed 1,000-step run with a 250-step checkpoint interval continues to 1,250 steps by default, or 1,500 for two intervals. The final weights must contain the completed step and have a complete optimizer save. Continuation reuses the frozen dataset, captions, prompts and settings, records the increased target per attempt, and preserves earlier outputs in their original folders. Each continuation writes to its own output folder and can be paused/resumed or continued again. The original configuration, manifest and export bundle remain unchanged; the dashboard displays the effective continuation target. Checkpoint retention applies separately within each continuation folder.

New runs using a compatible installed DiffusionTrainer support **Save checkpoint now** and **Generate previews**. Older SDTrainer runs can pause at their next scheduled checkpoint; resuming upgrades their launch adapter to native controls without changing the frozen recipe. A pending pause can be withdrawn before process stopping begins. If Studio restarts with a trainer still alive, reconnect to its owning server; do not launch another trainer over it. Developers can attach an updated UI without interrupting an older owner using `python server.py --port <free-port> --data <same-data-directory> --owner-url http://127.0.0.1:<owner-port>`. Stop requests are forwarded to that owner after matching run ownership. Keep the owning server running until the trainer exits.

## Development and checks

- `.venv\Scripts\python -m unittest discover -v` — persistence, caption concurrency, immutable export, trainer failure handling, recovery and HTTP checks.
- `node --test tests/domain.test.mjs` — browser state rules and text escaping.

The reproducible suite uses synthetic fixtures and creates disposable files under ignored `internal-tests/artifacts/unit/`. It does not require a live LLM, installed toolkit or existing internal scripts.

All real/local integration scripts and evidence belong in the single ignored `internal-tests/` folder and are not distributed in Git. On a development checkout where the local scripts are present:

- `python internal-tests/live_vision.py` — opt-in live Prompt Studio check with a synthetic image.
- `<toolkit-python> internal-tests/check_toolkit.py <toolkit-folder>` — parse configurations with that local AI Toolkit installation, without training.
- `node internal-tests/browser.mjs` — local browser workflow. Configure `PLAYWRIGHT_MODULE`, `BROWSER_EXECUTABLE`, `TOOLKIT_PATH` and `TOOLKIT_PYTHON` for the local installation. Generate the fixture with `internal-tests/live_vision.py` first. `LIVE_VISION=1` opts into an actual caption request.

All such scripts write evidence under `internal-tests/artifacts/`. Do not commit their data, images, captions, logs or reports. See [AGENTS.md](../AGENTS.md) for privacy, integration and development guidance.

Core adapter modules live under `studio/`; the UI lives under `web/`. AI Toolkit owns the training implementation and can be updated independently. Adapter schemas still need compatibility checks after major upstream changes.

## License and attribution

[GPL-3.0-only](../LICENSE). The visual token palette and interaction conventions follow [ComfyUI Prompt Studio](https://github.com/tiko13/ComfyUI_PromptStudio). AI Toolkit and Prompt Studio remain separate installations with their own licenses.

## Training results

The **Results** tab lists the final LoRA and every retained checkpoint for the selected run, with file sizes, **Download LoRA**, **Open location**, and full-size preview links. The run selector and selected tab survive a page refresh. When a run finishes while you are viewing its Dashboard or Create screen, Studio opens Results automatically; other editing screens stay in place. Older runs are available from Results and Runs & activity.

Each checkpoint also has **Delete checkpoint**. Review the confirmation before permanently deleting its weights, matching sidecars, exact-step previews and thumbnails. Previews shared with another retained checkpoint are preserved. Deleting the latest save in an output folder also removes its optimizer/SNR state, so that save can no longer be resumed. Entire saved comparisons containing the checkpoint (including their retained weights and images) are removed; the confirmation reports how many. Dataset images, captions, frozen configuration, run history and other training checkpoints are preserved. Deletion is blocked while work is active or the run/comparison has an unresolved interrupted state, and a changed file set requires a new confirmation.

**Compare checkpoints (Requires Prompt Studio)** opens a checkbox list with all complete checkpoints selected, Select all/none, a compatible **Create** workflow, model selectors when exposed, strength, prompt, and editable seeds. An optional baseline excludes the trained LoRA while preserving the workflow's supporting LoRAs. Optional resolution, steps and CFG overrides use workflow defaults when blank. The prompt is used directly, without LLM enhancement; each checkpoint uses the same seeds. Comparisons are limited to 128 cells.

The grid stays in Results, with progress, full-size images, local image/grid downloads, Stop after current image, Resume/retry failed, and Re-run comparison. Settings and results survive restarts. Re-run uses retained checkpoint snapshots, even after training retention removes originals. Only one local preparation/training/comparison job runs at a time. A lost queue response is reported as unconfirmed and is never silently resubmitted. Check ComfyUI history before creating a replacement in that case.

Use **Cell size** above the comparison grid to resize previews, matching Prompt Studio's 120–420 px slider in 20 px steps (220 px by default). The display preference is remembered in this browser across comparisons and refreshes. Generated image resolution and downloaded grids keep their original settings.

While a comparison generates, Prompt Studio shows **Generating in LoRA studio**. Image and Video generation actions queue until the whole comparison finishes or stops, including between checkpoints and seeds. Existing ComfyUI work finishes first. If LoRA Studio disconnects, the reservation expires automatically after any submitted comparison image finishes. No checkpoints are added to ComfyUI's LoRA folders.

Install the [optional Prompt Studio comparison bridge](../integrations/promptstudio/README.md) and restart both services after active jobs finish. It reads comparison snapshots directly from ignored `data/comparisons/`; it never copies checkpoints into ComfyUI's LoRA folders or creates a Prompt Studio plot/chat. Comparisons retain their own copies of selected weights and images, which consumes local disk space. Saved images remain viewable without Prompt Studio. Workflow discovery uses Prompt Studio's saved cache: refresh workflows there if no compatible Create workflow is listed. Unknown LoRA families and workflows requiring input images are excluded.

Previews appear beside weights from the exact same training step. Baseline images and previews with no retained checkpoint appear separately, including when saving and sampling intervals differ. Missing previews and incomplete weight files are labelled explicitly. Results lists files still on disk; it cannot restore checkpoints already removed by AI Toolkit's retention policy. Folder actions open the run's local output directory on the computer running Studio. Failed, cancelled and paused runs keep their saved outputs available too.
