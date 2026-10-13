# One click (almost) LoRA training

Select what you want to teach, import your images, and click **Create LoRA**. LoRA Studio prepares everything and starts training with AI Toolkit, with optional controls when you want to go deeper.

- **Go from images to training in one workspace.** Follow Base info → Dataset → Create, with automatic preparation and editable defaults.
- **Teach what matters.** Choose what should stay consistent and what should remain changeable, with captioning tailored to people, objects, styles and concepts.
- **Keep control of your captions.** Review the image, caption and supporting observations together. Manual edits and earlier versions stay recoverable.
- **Watch your LoRA take shape.** Follow live training progress and previews, save checkpoints, pause and resume, or continue a completed run.
- **Compare before you choose.** Test checkpoints with matching prompts and seeds in a saved image grid through the optional Prompt Studio connection.
- **Take your LoRA with you.** Download weights with embedded trigger phrases, public metadata and selected training settings.
- **Pick up where you left off.** Saved projects, recoverable drafts, frozen runs and reusable presets keep your work organised.

**Runs locally. ComfyUI is optional.** Use a standalone **Llama.cpp**, **Ollama** or **KoboldCpp** backend, or reuse your [Prompt Studio](https://github.com/tiko13/ComfyUI_PromptStudio) connection. [AI Toolkit](https://github.com/ostris/ai-toolkit) handles training in its own environment.

[Get started](#quick-start) · [The workflow](#from-images-to-a-lora) · [Captions and preparation](#prepare-images-with-purpose) · [Training and results](#watch-train-and-compare) · [Setup and models](#setup-and-models) · [User guide](docs/user-guide.md)

## Quick start

1. Download or clone this repository into its own folder and install **Python 3.10 or newer**, available as `python` in your terminal.
2. Double-click **`start.bat`** on Windows. It creates a local Python environment, installs the app's dependencies and opens [LoRA Studio](http://127.0.0.1:8787) in your browser. No npm build is required.
3. Follow the setup wizard: connect an existing **AI Toolkit** installation or choose **Download & install**, then choose **Prompt Studio** or a **standalone local LLM** for captioning.
4. In **Base info**, choose your LoRA type and base model, give it a name, and describe the target if needed.
5. In **Dataset**, import PNG, JPEG or WebP images. Exact duplicates are skipped; open any image to review its caption and training crop.
6. In **Create**, click **Create LoRA**. Automatic mode prepares missing or outdated automatic captions, creates preview prompts and starts training after prerequisite checks pass.

The “almost” is setup and a suitable dataset. You still choose the images and model, and resolve any blocking preparation issues. Caption suggestions are editable; automatic preparation is not a guarantee of caption accuracy or trained quality.

Keep the launcher console open while using Studio; press **Ctrl+C** to stop the server. You can reopen setup from **Settings → Run setup**, or choose **Set up later** while you organise a project.

The top-right status dot monitors your connections automatically. Open it to see separate indicators for Prompt Studio (when selected), the backend LLM, vision support or dedicated captioning, and AI Toolkit. Green means ready or connected, orange means working, and red means a problem needs attention; unknown states stay gray. LLM details include the model, processing phase and token count when reported by the backend. Ollama and generic OpenAI-compatible endpoints do not report live activity. Monitoring refreshes about every 5 seconds while open and 15 seconds while closed, pauses in hidden tabs, and never loads an unloaded Llama.cpp router model.

<details>
<summary>Terminal launch and manual installation</summary>

From the repository folder on Windows:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python server.py --open
```

Use `start.bat --port 8788` for another port, or `start.bat --help` for available options. AI Toolkit and your captioning backend use their own environments and dependencies.

</details>

## From images to a LoRA

### Define what you want to teach

Start with a name, model and target type. Studio supplies an editable trigger and category-specific defaults. Optional **Keep / Change** rules let you explain which traits belong to the LoRA and which should remain flexible.

| LoRA type | What the workflow focuses on |
| --- | --- |
| **Person / character** | Likeness, with optional review of person counts, framing and dataset coverage. |
| **Object / product** | The target object's appearance, with guided object selection, reversible crops and optional loss masks. |
| **Style** | Visual treatment, while captions describe the subjects and scenes. Optional family hints cover photography, cartoons, drawing, painting and pixel art. |
| **Action / concept** | Visible actions, participant roles, relationships and context, guided by the selected concept family. |

Keep/change rules guide captioning. Varied source images still matter: captions and masks cannot guarantee that a model separates your target from its surroundings.

### Choose how much to review

**Automatic · prepare and train** handles preparation and launch in one job. **Review first** lets you resolve caption warnings and visual findings before training. **Prepare dataset only** creates the prepared dataset and frozen export without starting the trainer.

Presets, learning rules, custom preview prompts, image checks and training settings are available when you need them. The ordinary path stays Base info → Dataset → Create.

## Prepare images with purpose

- **See the whole source and the actual training crop.** Full-image previews, aspect-ratio buckets and reversible positioning make preparation visible. Originals remain intact.
- **Find the intended object.** Automatic localization proposes a crop; ambiguous or missing targets need a hint, manual selection, explicit full-image choice or exclusion. Optional rectangle, polygon and source-alpha masks refine training loss.
- **Use your preferred captioning model.** Choose the connected local LLM, JoyCaption Beta One or a custom local image-capable endpoint. Dedicated captioners provide observations; the connected text LLM applies your learning rules.
- **Review the evidence.** Inspect saved observations beside the final caption, edit drafts, and use optional focused visual questions to investigate uncertain details.
- **Reuse work safely.** Reprocess saved observations when learning rules change and the image crop is still current. Manual captions are preserved; changed intent or crops flag earlier work for review.

Read more about [caption sources and review](docs/user-guide.md#caption-source), [image preparation](docs/user-guide.md#image-preparation-for-every-category) and [category-specific guidance](docs/category-guidance.md).

## Watch, train and compare

### Follow a run as it happens

The **Dashboard** brings together progress, loss, resource monitoring, logs, checkpoints and preview images. Adjust checkpoint and preview intervals before training, or use **Save checkpoint now** and **Generate previews** with a compatible installed backend.

**Pause & save** preserves a resumable checkpoint and releases the trainer's GPU memory. Resume uses the frozen run. **Continue training** extends a completed run when its final weights and optimizer state are available. **Cancel run** retains files already saved.

When a project's latest training run finishes successfully, its sidebar entry shows a green **Training complete** badge. The badge is restored from saved run history after reloading; continuing training shows the current training status until it finishes again.

Each run freezes its images, captions and configuration. Once training begins, preparation stays locked; **Copy to new run** gives you an independent, editable project for the next experiment. Save a run's settings as a named preset to reuse a starting point.

Each session in the sidebar has a **Delete** button. Choose **Delete session only** to remove its records from Studio and keep associated files on disk, or **Delete session and files** to also permanently remove Studio-owned image copies, analysis, exports, checkpoints, previews and comparisons. Original source images, shared models and saved presets are preserved. Active work and unresolved interrupted training/comparisons block deletion; changes in another window require reviewing the confirmation again. If disk cleanup fails, Studio reports retained files in its private `data/session-trash/` folder.

### Choose the checkpoint you like

**Results** puts final weights and retained checkpoints beside their matching previews, with full-size viewing, downloads and local folder access.

With [Prompt Studio and the optional comparison bridge](integrations/promptstudio/README.md), compare selected checkpoints using the same prompt and seeds. Add a baseline without the trained LoRA, adjust strength, and inspect a resizable grid. Comparison settings and generated images are saved for later review.

Comparisons require a compatible Prompt Studio **Create** workflow and ComfyUI on the same computer. Dataset preparation and training work independently of that connection.

### Export a LoRA with useful details

Add a public description, author, license, trigger phrases and recommended strength under **LoRA file metadata**. Studio embeds these alongside selected training settings in the Safetensors header. Private captions, preview prompts and local paths are excluded from that metadata.

See the [metadata reader contract](docs/user-guide.md#reader-contract-for-other-applications-and-agents) for integrations, and the [user guide](docs/user-guide.md#training-results) for checkpoint management and comparison details.

## Setup and models

The model catalog currently includes **Krea 2 Raw**, **Qwen Image 2.1**, **FLUX.1 Dev** and **SDXL**. Studio checks whether the installed AI Toolkit supports the selected architecture. Hardware, VRAM, model access and dependencies depend on that backend and model; a catalog entry is not a tested hardware or quality guarantee.

The setup wizard can connect an existing AI Toolkit installation or download a separate managed copy. Managed installation requires **Git** and at least **10 GB of free space**, with additional space needed for models, datasets and outputs. Existing Toolkit installations are not updated by the wizard.

For the LLM connection, choose explicitly between:

- **Prompt Studio backend:** reuse its saved provider and shared scheduling. Studio checks active work before requesting GPU model release.
- **Standalone backend:** connect Llama.cpp, Ollama or KoboldCpp without ComfyUI. Named profiles retain provider settings; Studio can manage its own Llama.cpp process.

Automatic image captioning needs a suitable vision model, or a dedicated captioner paired with a text LLM. The connection indicator helps diagnose availability; a reachable server alone does not establish image support.

Read the [setup guide](docs/user-guide.md#start) and [LLM connection details](docs/user-guide.md#llm-connection) for installation, lifecycle and shared-GPU behavior.

## Local storage and recovery

Projects, images, captions, histories, settings and training outputs live under ignored **`data/`**. Requests use the captioning and training backends you configure. Studio binds to loopback and is intended for local use.

Writes use revision checks, browser drafts survive reloads, and earlier caption versions remain recoverable. Back up the complete `data/` directory with the application stopped. See [persistence and recovery](docs/user-guide.md#persistence-and-reproducibility) before moving data or restarting during training.

## Development

The Python domain, persistence and training adapters live in `studio/`; the browser UI lives in `web/`. AI Toolkit provides the training engine.

```powershell
.venv\Scripts\python -m unittest discover -v
node --test tests/domain.test.mjs
git diff --check
```

The reproducible tests use synthetic inputs. Real integrations, private datasets and generated evidence belong in ignored `internal-tests/`. See [AGENTS.md](AGENTS.md) and the [development guide](docs/user-guide.md#development-and-checks).

## License and acknowledgements

LoRA Studio is licensed under **[GNU GPL v3.0 only](LICENSE)** (`GPL-3.0-only`), the same license as [Prompt Studio](https://github.com/tiko13/ComfyUI_PromptStudio).

Its visual language and interaction patterns follow Prompt Studio. The vendored provider code is attributed in the [Prompt Studio vendor notice](studio/vendor/promptstudio/NOTICE.md). [AI Toolkit](https://github.com/ostris/ai-toolkit) supplies the training backend. Third-party components and model weights retain their respective licenses.
