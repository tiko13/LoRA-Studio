"""Refresh the GPL provider subset from a reviewed Prompt Studio checkout.

Copies only named source dependencies, never settings, profiles or user data.
Usage: python scripts/vendor_promptstudio.py <Prompt-Studio-checkout>
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

SOURCE = Path(sys.argv[1])
DEST = Path(__file__).resolve().parents[1] / 'studio/vendor/promptstudio'
DEST.mkdir(parents=True, exist_ok=True)
HEADER = '# Derived from tiko13/ComfyUI_PromptStudio; GPL-3.0-only. See NOTICE.md.\n'


def subset(file, roots, imports, replacements=None):
    source = (SOURCE / 'backend' / file).read_text(encoding='utf-8')
    lines = source.splitlines()
    definitions = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
    selected, pending = set(), list(roots)
    while pending:
        name = pending.pop()
        if name in selected or name not in definitions:
            continue
        selected.add(name)
        pending.extend(n.id for n in ast.walk(definitions[name]) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
    output = [HEADER, imports]
    for name in sorted(selected, key=lambda k: definitions[k].lineno):
        node = definitions[name]
        first = min([node.lineno] + [n.lineno for n in getattr(node, 'decorator_list', [])])
        output.append((replacements or {}).get(name, '\n'.join(lines[first-1:node.end_lineno])))
    return '\n\n'.join(output) + '\n'


providers = subset('nodes.py', [
    '_generate_kcpp', '_generate_llamacpp', '_generate_ollama', '_llm_vision_capability',
    '_list_ollama_models', '_list_llamacpp_models', '_unload_ollama_model', '_strip_response',
], '''import functools, inspect, ipaddress, json, math, os, re, threading, urllib.error, urllib.parse, urllib.request
from . import provider_transport as _provider_transport
from . import llm_coordinator as _llm_scheduling''', {
    name: f'def {name}():\n    return {{"localhost", "127.0.0.1", "::1"}}'
    for name in ('_allowed_kobold_hosts', '_allowed_ollama_hosts', '_allowed_llamacpp_hosts')
})
(DEST / 'providers.py').write_text(providers, encoding='utf-8')
enums = json.loads((SOURCE / 'web/js/prompt-studio/core/wire-contract-enums.json').read_text(encoding='utf-8'))
runtime = subset('routes.py', [
    '_start_llamacpp_server', '_stop_llamacpp_server', '_restart_llamacpp_server', '_edit_llamacpp_config',
    '_list_llamacpp_config_profiles', '_inspect_llamacpp_config_profiles', '_delete_llamacpp_config_profile',
    '_llamacpp_managed_process_status', '_llamacpp_configured_generation_data',
    '_unload_kobold_model', '_reload_kobold_model_if_needed', '_unload_llamacpp_model',
    '_pick_llamacpp_file',
], '''import hashlib, hmac, json, logging, math, os, re, secrets, subprocess, sys, threading, time, urllib.parse, uuid
from pathlib import Path
from .providers import _get_json, _reasoning_effort, _post_json, _clean_base_url, _clean_llamacpp_base_url, _list_llamacpp_models
BASE_DIR = str(Path(__file__).parent)
_WIRE_ENUMS = ''' + repr({'thinking_modes': enums['thinking_modes']}), {
    name: name + ' = None' for name in ['LLAMACPP_CONFIG_DIRECTORY', 'LLAMACPP_OUTPUT_LOG_PATH',
                                      'LLAMACPP_AUTOSTART_PATH', 'LLAMACPP_PROCESS_STATE_PATH', 'LLAMACPP_WATCHDOG_PATH']
})
(DEST / 'runtime.py').write_text(runtime, encoding='utf-8')
web = DEST.parents[2] / 'web/vendor/promptstudio'
web.mkdir(parents=True, exist_ok=True)
for name in ('llamacpp-config-editor.js', 'llamacpp-profile-manager.js', 'dialog-focus.js'):
    text = (SOURCE / 'web/js/prompt-studio/ui' / name).read_text(encoding='utf-8')
    text = text.replace('import { api } from "/scripts/api.js";', 'const api = {fetchApi: (url, options) => fetch(url, options)};')
    text = text.replace('import { LLAMACPP_CONFIG_BUILDER_ENDPOINT } from "../core/constants.js";', 'const LLAMACPP_CONFIG_BUILDER_ENDPOINT = "/api/llm/profile";')
    text = text.replace('/promptstudio/prompt-studio/llamacpp/pick-file', '/api/llm/pick-file')
    text = text.replace('/promptstudio/prompt-studio/llamacpp/config-profiles', '/api/llm/profile')
    text = text.replace('ComfyUI computer', 'LoRA Studio computer').replace('Start with ComfyUI', 'Start with LoRA Studio').replace('Restart ComfyUI', 'Restart LoRA Studio')
    text = text.replace('event.preventDefault(); if (busy || !loaded)', 'event.preventDefault(); event.stopPropagation(); if (busy || !loaded)')
    text = text.replace('A running managed server applies it when LLM work is idle.', 'The next LLM request applies changes to a Studio-managed server. You can also restart it explicitly.')
    (web / name).write_text('// Derived from Prompt Studio, GPL-3.0-only; see studio/vendor/promptstudio/NOTICE.md.\n' + text, encoding='utf-8')
for name in ('provider_transport.py', 'llm_coordinator.py', 'llamacpp_watchdog.py'):
    (DEST / name).write_text(HEADER + (SOURCE / 'backend' / name).read_text(encoding='utf-8'), encoding='utf-8')
(DEST / '__init__.py').write_text(HEADER, encoding='utf-8')
(DEST.parent / '__init__.py').write_text('"""Attributed third-party source subsets."""\n', encoding='utf-8')
hashes = {name: hashlib.sha256((SOURCE / 'backend' / name).read_bytes()).hexdigest()
          for name in ('nodes.py', 'routes.py', 'provider_transport.py', 'llm_coordinator.py', 'llamacpp_watchdog.py')}
for name in ('llamacpp-config-editor.js', 'llamacpp-profile-manager.js', 'dialog-focus.js'):
    hashes['ui/' + name] = hashlib.sha256((SOURCE / 'web/js/prompt-studio/ui' / name).read_bytes()).hexdigest()
(DEST / 'NOTICE.md').write_text('''# Prompt Studio provider backend

Source: https://github.com/tiko13/ComfyUI_PromptStudio (local source snapshot, 2026-10-13).
Copyright remains with the upstream contributors. Licensed under GPL-3.0-only;
the full license is included at the LoRA Studio repository root in LICENSE.

Generated by scripts/vendor_promptstudio.py. The subset retains upstream
provider protocols, vision checks, context budgeting, streaming, scheduling,
llama.cpp profile validation, process ownership and watchdog recovery.
ComfyUI routes, diffusion nodes, user profiles and local settings are excluded.
LoRA Studio restricts provider hosts to loopback and binds runtime paths to
its ignored data/llm folder. Its adapter supplies application lifecycle and UI.
No runtime import from an installed Prompt Studio checkout is needed.

SHA-256 of source files used for this snapshot:

''' + '\n'.join(f'- `{name}`: `{digest}`' for name, digest in hashes.items()) + '\n', encoding='utf-8')
