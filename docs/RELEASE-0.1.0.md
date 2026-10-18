# 0.1.0 preview

First independent Sigma Rescale Neo package for Forge Neo and ComfyUI.

- Basic Start/End mapping compatible with old Simple PNG metadata.
- Opt-in Strength, Curve, final-zero, pass selection and separate Hires controls.
- Native PNG Info restoration, including old Simple/Full field names.
- Removable runtime integration; no Forge source-file modifications or layout changes.
- Small ComfyUI SIGMAS node sharing the same tensor math.
- Installation, reproduction and controlled natural-texture comparison guides.

## Verification

On the inspected Forge Neo checkout (base commit `91e2881b`, with existing local modifications):

- 41 CPU tests passed, including native scheduler/Euler/PNG parser integration and real Gradio paste fields.
- 9 isolated browser checks passed at desktop and narrow widths, with no page errors.
- 80 private compatibility cases matched the installed Simple transform bit-for-bit.
- 96 private compatibility cases matched the installed Full transform bit-for-bit.

Reference plugin sources and private test images are not bundled. Tests do not start a GPU generation or load image models. **Full-model visual quality, clean-install GPU smoke tests and cross-application pixel equality are not claimed.** Treat this as a preview, not a validated skin-texture preset.

Loading an old PNG does not require rewriting the image. It requires intact metadata and the original model/LoRA/sampler/scheduler dependencies. Disable the older Sigma Rescale extension when moving to Neo to avoid both readers activating from the same PNG.
