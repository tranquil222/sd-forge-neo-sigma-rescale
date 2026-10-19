# Sigma Rescale Neo

**Start/End sigma range controls for Forge Neo, with optional Advanced controls and PNG settings restoration.** A small ComfyUI SIGMAS node is included too.

Version **0.2.0 — preview**. Independently authored implementation, inspired by the sigma tools in RES4LYF and lz-55's Forge workflow. Not an official RES4LYF or lz-55 release. See [credits and provenance](NOTICE.md).

## What it does

Remaps the sigma schedule **after the selected scheduler constructs it**:

- Incoming maximum → **Start**; incoming minimum → **End**.
- Basic mode preserves the familiar linear Start/End behavior, including nonzero End values. Main pass only; Hires untouched.
- Advanced is **off by default**: Strength, Curve, optional final-zero, pass selection, and separate Hires settings.
- PNG metadata stores the controls. PNG Info → Send to txt2img/img2img restores them.
- Imports existing `Sigma Rescale` and `Sigma Rescale Full` PNG fields.
- Includes Full's **Begin applying at (% of steps)** timing, including separate Hires timing. Old PNGs without timing fields retain the original full-run behavior.
- No Forge source edits, sampler replacement, button relocation, model downloads, or dependency installer.

This is a schedule experiment tool, **not a sharpening filter, skin enhancer, or guaranteed realism preset**. For natural texture without the overly contrasty/plastic look, see the [controlled comparison guide](docs/NATURAL-TEXTURE.md).

## Install in Forge Neo

1. Open **Extensions → Install from URL**.
2. Enter `https://github.com/tranquil222/sd-forge-neo-sigma-rescale` and install.
3. Restart Forge completely.
4. Open **Sigma Rescale Neo** in txt2img or img2img; enable it and set Start/End.

Alternatively, from your Forge directory:

```sh
git clone https://github.com/tranquil222/sd-forge-neo-sigma-rescale extensions/sd-forge-neo-sigma-rescale
```

**Moving from the old Simple/Full plugins:** disable those extensions and restart before using Neo. You can keep their folders. Loading an old PNG can enable its old panel as well as Neo if both extensions are active; Neo stops with a clear error instead of rescaling twice. Do not enable multiple sigma-rescale panels for the same generation.

Defaults are disabled, Start `0.9`, End `0`. Your existing PNG's values take precedence when loaded. Start/End are **sigma levels, not beginning/end percentages or step numbers**.

## Recreate an image

1. Install Neo and restart. Get the **original PNG with embedded generation metadata**, not a screenshot or recompressed copy.
2. Drop it into **PNG Info**, then **Send to txt2img** (or img2img for that workflow).
3. Check that Sigma Rescale Neo enabled/Start/End/Advanced match the PNG.
4. Match the model, LoRAs and weights, seed, sampler, scheduler, steps, CFG, size, precision and any other extensions used.
5. Generate. Check `Sigma Rescale Neo main applied: True` in the new PNG's parameters to confirm a main-pass transform actually ran.

Neo restores **its own controls only**. Hybrid schedulers, custom scheduler presets, model shifts and other extension settings need their own matching implementation. Their absence cannot be repaired by loading a PNG. Matching sigma controls alone does not guarantee pixel-identical images across machines or applications.

Basic Neo images also include old Simple Start/End fields for interoperability. Advanced images deliberately do **not** enable an old Simple reader, because that reader cannot reproduce Strength/Curve/Hires changes. Use Neo for Advanced PNGs. This is not a general Sigma Lab importer/exporter.

## Controls

| Control | Meaning |
| --- | --- |
| Start / End | Targets for the incoming maximum / minimum, including the terminal sigma. |
| Begin applying at (% of steps) | 0 = full run; 50 = first half unchanged, then fade in; 100 = no adjustment. Counts completed sampling intervals, rounded up—not internal model evaluations. |
| Strength | 1 = full remap; 0 = exact original schedule. Intermediate values blend the two schedules. Lower does **not** inherently mean less contrast. |
| Curve | 1 = linear; above 1 pulls interior levels toward End; below 1 toward Start. Endpoints stay fixed before blending/final-zero. |
| Force final sigma to zero | Overrides the last output value after mapping/blending. Off preserves your nonzero End behavior. |
| Apply to | Main only, Main + Hires, or Hires only. |
| Separate Hires settings | Uses its own Start/End/Strength/Curve/final-zero/begin percentage when a Hires pass actually runs. |

With Advanced off, all Advanced values are ignored. With Advanced on and Begin 0% / Strength 1 / Curve 1 / final-zero off / Main only, the math is the same as Basic mode. An empty/constant schedule or Strength 0 is passed through unchanged, including the final value.

Set Begin applying at to **0%** for the original full-run math. A nonzero start
leaves the initial schedule and boundary sigma untouched, then smoothly increases
the adjustment to your selected Strength. This can reduce early composition changes;
it is **not a composition lock** or a texture guarantee. No steps are added.
For img2img/Hires, the percentage counts the steps actually sampled after denoise
slicing, while Start/End still map the full constructed range. If rounding leaves
no steps after the boundary, the schedule stays unchanged—even with final-zero on.

Late application requires a descending schedule and a sampler that uses an
explicit sigma schedule (such as Euler and the usual RES4LYF schedule-based paths).
Endpoint-only/adaptive samplers cannot honor a counted-step boundary. A late
transform that would make sigma rise raises an error instead of silently changing
your settings. Adjust End/Strength/Curve or begin earlier in that case.

The broad endpoint range permits reversed/negative schedules for experimentation; this does not make those ranges appropriate for every model or sampler. Start from a known working image's settings. Neo does not reinterpret or clamp Start/End to a model-specific range.

## ComfyUI

Clone this same repository under `ComfyUI/custom_nodes`, restart, and add **Sigma Rescale Neo** from `sampling/sigmas`.

```text
Scheduler SIGMAS → Sigma Rescale Neo → SamplerCustomAdvanced sigmas
```

Use a sampler with an explicit SIGMAS input, not a regular KSampler node. The node operates on the tensor you connect; separate Hires control means using it in the appropriate pass of your graph. PNG Info restoration described above is a **Forge feature**, not automatic Forge-PNG-to-Comfy-workflow conversion.

Same input tensor, dtype and settings use the same math. Different model scaling, scheduler implementations or denoise slicing can give different inputs/results. In particular, Forge transforms the full constructed schedule before img2img slicing; a Comfy scheduler may have already sliced it for denoise. Cross-application image equivalence is not claimed.

The optional `begin_percent` input is active when Advanced is enabled. Comfy counts
intervals in the SIGMAS tensor you connect. Existing workflows default to 0%.

## Compatibility and verification

Targets Forge Neo's `KDiffusionSampler.get_sigmas` path, including samplers inheriting it. Samplers that bypass this path are not transformed. Native scheduler and sampler names are left intact. A zero Strength, unselected pass, constant schedule or unsupported path can correctly report `applied: False`.

The preview has CPU tests for the math, metadata/PNG restoration through Forge's parser and real Gradio fields, native schedule construction, native Euler execution with a small CPU test denoiser, Hires behavior, unload/reload and conflicting plugins. These are **not full-model image-quality tests**. See [integration notes](docs/INTEGRATION.md) and [testing](docs/TESTING.md).

Version 0.2.0 adds 768 compatibility cases against the installed Full implementation
across four dtypes, main/sliced schedules, curves, strengths and begin percentages.
Accepted schedules match bit-for-bit; incompatible rising schedules are rejected
by both. See [release notes](docs/RELEASE-0.2.0.md).

## Sharing on Civitai

Use the [copy-ready description](docs/CIVITAI.md). Share the original PNG or generation text alongside your exact model/LoRA dependencies. No private images, prompts, models or queue history are included in this project.

## License

New project code: **AGPL-3.0-only**, see [LICENSE](LICENSE). Third-party tools linked in the credits retain their own licenses. This repository does not bundle their source or relicense their plugins.
