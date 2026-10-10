# LoRA Studio

A local workspace for intent-aware dataset captioning and AI Toolkit training. The interface follows Prompt Studio's visual language. Prompt Studio integration is optional; it uses the existing HTTP API and requires no changes to ComfyUI or Prompt Studio.

## Start

Run `start.bat`, or `python server.py --open` with Python 3.10 or newer. Open **http://127.0.0.1:8787**. The application itself uses the Python standard library and browser APIs; no npm build or pip install is required. AI Toolkit and the vision server keep their own environments and dependencies.

In **Connections**, select your AI Toolkit folder (containing `run.py`) and its Python executable. Machine-specific paths are stored in the ignored `data/` directory. Keep this local app bound to loopback; it is not designed for remote hosting.

## First workflow

1. Create a LoRA, choose a type and base model, and describe the target and trigger.
2. Choose **Auto**, **Keep consistent**, or **Make changeable** for each attribute. Auto's resolved choice is shown. Extra free-text rules are optional. Type changes preserve hidden choices.
3. Import PNG, JPEG, or WebP images (WebP is converted to PNG in the browser). Exact duplicates are skipped. Exclude unsuitable images from their review dialog.
4. Select **Prepare captions**. The local vision model returns visual observations, an intent-aware caption, and uncertainty warnings. Review flagged images; edits remain manual and are never overwritten by automatic captioning. Changing learning rules makes older captions stale.
5. Open **Training**. Use the starter settings or adjust steps, rank, resolution, learning rate, GPU, seed and preview prompts. Export a portable bundle or launch the installed AI Toolkit. The first run may download base model components; model access and Hugging Face authentication belong to AI Toolkit's environment.
6. Use **Runs** for logs, configuration, manifest, preview images, bundles and saved `.safetensors` files. Stop terminates the trainer and its child processes; work after the last checkpoint can be lost.

The current model catalog includes **Krea 2 Raw**, **Qwen Image 2.1**, **FLUX.1 Dev**, and **SDXL**. Installed architecture discovery gates training. This is an extensible adapter, not a claim that every architecture or hardware configuration has been trained and validated.

## Optional Prompt Studio connection

- **Automatic** uses the active Prompt Studio chat's saved LLM connection when available, with a standalone local OpenAI-compatible endpoint as fallback when Prompt Studio is unreachable.
- **Prompt Studio only** requires that integration; **Standalone** does not contact ComfyUI.
- A connected but non-vision model is an error, not a silent provider switch. Select the intended vision model in Prompt Studio first. The settings are captured at the beginning of each caption job.
- Caption requests use Prompt Studio's existing provider and scheduling code. They do not add messages to your chat. Uploaded reference images are retained in Prompt Studio's own image storage.
- With model unloading enabled, Studio checks ComfyUI's queue and Prompt Studio jobs, then requests `/free` before training. ComfyUI stays running. This is a cooperative handoff, not a global reservation: avoid starting new generations during training. The external vision LLM is not forcibly unloaded.

Only localhost HTTP endpoints are accepted. Standalone endpoints use the OpenAI-compatible `/models` and `/chat/completions` contracts and must support image input. No API keys are required or stored by this version.

## Persistence and reproducibility

`data/studio.sqlite3` stores projects, caption history, settings and jobs. Imported images live in `data/images/`. Each run copies included images and captions into `data/runs/<id>/`, alongside its configuration and manifest. Later edits do not affect that snapshot. Export bundles use relative dataset/output paths.

Project writes use revision checks. Browser drafts survive reloads and require an explicit restore/discard choice. Manual caption changes require review against the current rules. Jobs interrupted by a server restart are marked interrupted; training remains blocked until the recorded process is confirmed stopped from the run details. A missing process ID requires manual investigation. Keep the server open during training.

Back up the entire `data/` directory with the application stopped, including SQLite sidecar files if present. Datasets, local settings, outputs and test artifacts are excluded from Git.

## Scope of this first version

Caption preparation, import, editable starter settings, immutable export, trainer launch, stop, logs and output downloads are implemented. Captions describe visible evidence without inventing hidden details or sanitizing relevant factual terminology. Keep/change choices guide language; they do not edit images or guarantee disentanglement. Review model output for factual accuracy.

Automatic dataset generation, semantic dataset curation, quality scoring, checkpoint comparisons, resume controls, additional trainer adapters and automatic hyperparameter tuning are future work. With no source dataset, begin by collecting suitable images. Starter defaults are not hardware sizing or quality guarantees. Config parsing and a live vision request do not establish training quality; validate a real run on your own dataset.

## Development and checks

- `python -m unittest discover -v` — persistence, caption concurrency, immutable export, trainer failure handling, recovery and HTTP checks.
- `node --test tests/domain.test.mjs` — browser state rules and text escaping.
- `python tests/live_vision.py` — **opt-in live** Prompt Studio test with a synthetic image, written under `test-results/`.
- `<toolkit-python> tests/check_toolkit.py <toolkit-folder>` — parse all four configurations with installed AI Toolkit classes, without loading a model or training.
- `node tests/browser.mjs` — browser workflow using an installed Playwright. Set `PLAYWRIGHT_MODULE` to its `index.mjs` or install it locally, `BROWSER_EXECUTABLE` to Edge/Chrome, and `TOOLKIT_PATH` / `TOOLKIT_PYTHON` to the installed backend. Generate the fixture with `tests/live_vision.py` first. Set `LIVE_VISION=1` for the optional end-to-end vision call; otherwise the browser test uses a manual caption. Test data is isolated from the normal workspace.

Core adapter modules live under `studio/`; the UI lives under `web/`. AI Toolkit owns the training implementation and can be updated independently. Adapter schemas still need compatibility checks after major upstream changes.

## License and attribution

GPL-3.0-only. The visual token palette and interaction conventions follow [ComfyUI Prompt Studio](https://github.com/tiko13/ComfyUI_PromptStudio). AI Toolkit and Prompt Studio remain separate installations with their own licenses.
