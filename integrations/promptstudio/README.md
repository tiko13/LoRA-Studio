# Checkpoint comparison bridge

LoRA Studio owns comparisons, snapshots, prompts and generated images. This small
bridge lets Prompt Studio's ComfyUI process read approved immutable snapshots
directly from LoRA Studio. It does not register extra LoRA folders, copy weights
into ComfyUI, or create Prompt Studio chats/plots. Both apps must run on the same
computer and their HTTP origins must be loopback addresses.

Install into an existing Prompt Studio extension with:

```powershell
python scripts/install_comparison_bridge.py --prompt-studio "path/to/ComfyUI_PromptStudio"
# If the Video Studio companion is installed, include its checkout:
python scripts/install_comparison_bridge.py --prompt-studio "path/to/ComfyUI_PromptStudio" --video-studio "path/to/PromptStudio_Video"
```

The installer copies the bridge, reservation and shared status modules, adds
registration to the extension's `__init__.py`, connects status and queue labels
in the existing UI, and adds assistant help cards. It never restarts
services or changes model paths. Restart ComfyUI after its active jobs finish,
and restart LoRA Studio to load its new backend. Re-run the installer after a
Prompt Studio update if that update removed the registration.

The capability endpoint reports protocol version 1. Registration accepts a
loopback LoRA Studio origin and a one-use, short-lived random token. LoRA Studio
resolves it to a comparison-owned snapshot; the bridge confines and verifies its
digest before issuing an in-memory checkpoint ID. The loader accepts that ID,
never an arbitrary filesystem path. Restarting ComfyUI clears registrations;
LoRA Studio registers pending cells again when resumed.

Generation uses one reservation for the whole comparison. Existing ComfyUI work
finishes first; then the status reads **Generating in LoRA studio**. Image and
Video actions show **Queue** and their generation waits until the comparison
finishes or stops. Server admission protects other tabs and the canvas too,
including gaps between cells. Shared-GPU LLM work waits on the same scheduler;
dedicated keep-loaded providers retain their configured behavior.

LoRA Studio renews the reservation every five seconds. If it disconnects, the
lease expires after thirty seconds; any already submitted comparison image
finishes before other work is admitted. No unrelated prompt is interrupted.
The capability response includes `generation_reservation: 1`; older bridges
must be updated before generating comparisons.

Create workflow filtering requires a matching explicit LoRA family descriptor,
connected Prompt Studio prompt/seed controls, and configured image output nodes.
Unknown types, stale caches, Edit/Upscale and required image inputs are excluded.
This bridge generates Image Create comparisons only. Video Studio consumes the
same reservation status and queues its video generation behind comparisons.

Synthetic regression sources are included here for the corresponding Prompt
Studio test suite: `test_generation_reservation.py` belongs in `tests/`, and
`lora-studio-queue.spec.mjs` belongs in `tests/browser/`. These use its existing
scheduler and browser fixture; they do not generate images or use private data.
