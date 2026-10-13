"""Install the optional bridge into an explicitly selected Prompt Studio checkout."""
import argparse
from pathlib import Path
import shutil


def replace_once(text, before, after):
    if after in text:
        return text
    if text.count(before) != 1:
        raise ValueError('Prompt Studio frontend has changed; update the comparison integration before installing')
    return text.replace(before, after, 1)


def install_frontend(destination, *, video=False):
    path = destination / ('web/js/promptstudio_video_studio.js' if video else 'web/js/prompt_studio.js')
    text = path.read_text(encoding='utf-8')
    module = '/extensions/ComfyUI_PromptStudio/js/prompt-studio/ui/' if video else './prompt-studio/ui/'
    statement = f'import {{ watchExternalGeneration }} from "{module}external-generation.js";\n'
    if statement not in text:
        text = statement + text
    text = replace_once(text, 'return state.comfyQueueRemaining > 0',
                        'return state.loraStudioGeneration?.reserved || state.comfyQueueRemaining > 0')
    text = replace_once(text, 'const stateName = unhealthy ? "offline" : (processing ? "busy" : "idle");',
        'const stateName = state.apiConnected && state.loraStudioGeneration?.reserved ? "busy" : (unhealthy ? "offline" : (processing ? "busy" : "idle"));')
    text = replace_once(text, '  if (!state.apiConnected) {\n    comfyDetail.textContent',
        '  if (state.apiConnected && state.loraStudioGeneration?.reserved) label.textContent = state.loraStudioGeneration.label;\n\n  if (!state.apiConnected) {\n    comfyDetail.textContent')
    text = replace_once(text, '  } else if (comfyProcessing) {\n    const queued',
        '  } else if (state.loraStudioGeneration?.reserved) {\n    comfyDetail.textContent = state.loraStudioGeneration.label;\n    comfyDetail.dataset.state = "busy";\n  } else if (comfyProcessing) {\n    const queued')
    monitor = 'startKoboldStatusMonitor();' if video else 'startLlmStatusMonitor();'
    refresh = 'renderHeader();' if video else 'updateComposeMode();\n      syncBackgroundActivityIndicator();'
    text = replace_once(text, '    ' + monitor,
        '    ' + monitor + '\n    watchExternalGeneration((status) => {\n      state.loraStudioGeneration = status;\n      '
        + refresh + '\n      renderSystemStatusSummary();\n    }, api.fetchApi.bind(api));')
    if video:
        text = replace_once(text, 'generate.textContent = structuredExtension ? "Generate extension" : "Generate";',
            'generate.textContent = state.loraStudioGeneration?.reserved\n      ? (structuredExtension ? "Queue extension" : "Queue generation")\n      : (structuredExtension ? "Generate extension" : "Generate");')
    else:
        activity_path = destination / 'web/js/prompt-studio/ui/background-activity.js'
        activity = activity_path.read_text(encoding='utf-8')
        activity = replace_once(activity, 'function backgroundActivityLabel() {',
            'function backgroundActivityLabel() {\n  if (state.loraStudioGeneration?.reserved) return state.loraStudioGeneration.label;')
        activity = replace_once(activity, 'return pendingStudioGenerationCount() > 0',
            'return state.loraStudioGeneration?.reserved || pendingStudioGenerationCount() > 0')
        activity_path.write_text(activity, encoding='utf-8')
    path.write_text(text, encoding='utf-8')


def install(destination):
    destination = Path(destination).resolve(strict=True)
    init = destination / '__init__.py'
    if not (destination / 'backend/nodes.py').is_file() or not (destination / 'backend/routes.py').is_file():
        raise ValueError('Select the ComfyUI_PromptStudio extension directory')
    source = Path(__file__).resolve().parents[1] / 'integrations/promptstudio/lora_studio_bridge.py'
    text = init.read_text(encoding='utf-8')
    marker = 'from .backend.lora_studio_bridge import register as register_lora_studio_bridge'
    if marker not in text:
        if 'WEB_DIRECTORY =' not in text or 'NODE_CLASS_MAPPINGS' not in text:
            raise ValueError('Unrecognized Prompt Studio initialization; no changes made')
        text = text.replace('WEB_DIRECTORY =', marker + '\nregister_lora_studio_bridge(routes.PromptServer.instance, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)\n\nWEB_DIRECTORY =', 1)
    shutil.copyfile(source, destination / 'backend/lora_studio_bridge.py')
    shutil.copyfile(source.with_name('generation_reservation.py'), destination / 'backend/generation_reservation.py')
    shutil.copyfile(source.with_name('external-generation.js'), destination / 'web/js/prompt-studio/ui/external-generation.js')
    install_frontend(destination)
    init.write_text(text, encoding='utf-8')
    help_file = destination / 'docs/assistant-help/lora-studio-comparisons.md'
    help_file.parent.mkdir(parents=True, exist_ok=True)
    help_file.write_text('''---
{"id":"lora-studio.comparisons","topic":"plots","studio":"image","summary":"Compare LoRA Studio checkpoints with Create workflows without importing weights into the LoRA library."}
---

In LoRA Studio Results, choose Compare checkpoints. Select saved checkpoints, a compatible Create workflow, model when exposed, strength, prompt and seeds. The grid and saved images stay in LoRA Studio; no Prompt Studio chat or plot is created. Retry failed cells or re-run using retained snapshots, even if training retention removed the original files.

Requires the local comparison bridge and both apps on the same computer. The internal PS_LoraStudioCheckpoint node accepts only registered checkpoint IDs, loads immutable LoRA Studio snapshots, and never adds files or paths to ComfyUI's LoRA library. Restarting ComfyUI clears registrations; resume from LoRA Studio to register pending checkpoints again.

During a comparison the status reads Generating in LoRA studio. Image and Video generation requests queue until the entire comparison finishes or stops, including between checkpoints and seeds. Existing ComfyUI work finishes first. Shared-GPU LLM requests use the same reservation; dedicated keep-loaded providers retain their configured behavior. If LoRA Studio disconnects, the reservation expires automatically after any already submitted comparison image finishes.

Refresh workflows in Prompt Studio if a Create workflow is missing. Supported workflows need explicit matching LoRA families, connected seed controls and image results. Edit and required-image workflows are excluded. Video Studio does not expose checkpoint comparisons.
''', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-studio', required=True, type=Path)
    parser.add_argument('--video-studio', type=Path, help='Optional companion checkout for queue labels and shared status')
    args = parser.parse_args()
    install(args.prompt_studio)
    if args.video_studio:
        install_frontend(args.video_studio.resolve(strict=True), video=True)
        (args.video_studio / 'docs/assistant-help/lora-studio-queue.md').write_text('''---
{"id":"video.lora-studio-queue","topic":"video-workflows","studio":"video","summary":"Video generation queues while a LoRA Studio comparison uses ComfyUI."}
---

While LoRA Studio compares checkpoints, the status reads Generating in LoRA studio and Generate becomes Queue generation (or Queue extension). Queued video work starts after the comparison finishes or stops. Existing ComfyUI images and videos finish before a comparison begins. A disconnected comparison releases its reservation automatically after any submitted image finishes. The comparison and its images remain in LoRA Studio.
''', encoding='utf-8')
    print('Bridge installed. Restart idle ComfyUI and LoRA Studio to load the new backend.')
