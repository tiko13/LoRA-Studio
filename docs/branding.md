# Application icon

`web/lora-studio-icon.svg` is the editable source for LoRA Studio's connected
LS monogram. Its dark rounded tile, purple gradient and rounded strokes follow
the [Prompt Studio icon](https://github.com/tiko13/ComfyUI_PromptStudio/blob/main/web/prompt-studio-icon.svg)
by tiko13 (GPL-3.0); the letter geometry is specific to LoRA Studio.

The workspace and training monitor use this SVG in their headers and as their
preferred favicon. Companion exports are `web/favicon.ico` (16, 32 and 48 px),
`web/apple-touch-icon.png` (180 px), and `web/lora-studio-icon.png` (512 px).
When changing the SVG, regenerate these raster exports from the same artwork.
The two PNGs have explicit source-asset exceptions in `.gitignore`.
