# Integration contract

## Forge

- Loads through the ordinary `scripts/` extension entry point.
- Own native Gradio accordion in txt2img/img2img; no JavaScript/CSS, toolbar relocation, Generate override or progress-bar modification.
- Attaches an immutable Settings value to the current processing object. Native script arguments let an existing queue capture these controls just like other extension controls. The queue itself is not changed or required.
- Wraps `KDiffusionSampler.get_sigmas` in memory and calls the captured method once. The native scheduler runs first, including custom scheduler overrides and penultimate removal; the selected pass is then transformed.
- img2img's denoise slice remains downstream of the full-schedule mapping. A sampler subclass overriding the method without calling it is outside this integration.
- Known old Simple/Full processing flags are checked before schedule construction when Neo is enabled. Simultaneous enablement raises an actionable error. Other schedule-modifying plugins are not automatically identified; order can matter.
- Unloading restores the previous method only when Neo still owns the top-level slot. If another extension wrapped it later, Neo becomes a pass-through and leaves that wrapper intact. Reload closes the old owner before adding a new one.
- Disabling Neo returns the original tensor object unchanged. No sampler/scheduler registry modifications, model changes or global sampling options.
- The extension itself writes no settings/database files. Forge saves normal UI preferences and generation metadata through its existing mechanisms. No network requests or dependency installs.

## Metadata

Every control is recorded under `Sigma Rescale Neo`, including Advanced/pass/Hires choices and a version. Per-pass `applied`, sigma count and (when changed) first/last/hash values distinguish requested controls from actual transformation. Hash = first 12 hex characters of SHA-256 over the output schedule converted to contiguous float64 bytes; it is a schedule comparison aid, not an image or model identity.

`Sigma Rescale` / `Sigma Rescale start` / `Sigma Rescale end` aliases support the old Simple reader in Basic mode. Advanced sets the old enable alias false rather than claim compatibility. These fields are emitted after the main schedule is constructed; fields that Neo emitted are cleared on the next request. Unrelated extension metadata is not removed.

Restoration priority: Neo keys, then enabled Full keys, then Simple keys. Missing keys use defaults; a PNG without sigma metadata disables Neo and resets hidden Advanced values. PNG values can be strings. The native PNG parser and Gradio paste callbacks are covered by CPU tests.

If a sampler bypasses the hook, no transform is applied. The initial per-pass applied flags remain false, and postprocess reports that no schedule changed. Strength zero and constant input also correctly report false. Do not interpret the enabled checkbox alone as proof that a sampler path was transformed.

## Math

For incoming tensor `s`, its minimum `lo`, maximum `hi`, and nonzero `hi-lo`:

```text
u = (s - lo) / (hi - lo)
mapped = End + (Start - End) * u ** Curve
output = s + Strength * (mapped - s)
```

For Curve 1, the implementation uses the explicit arithmetic order
`((s-lo)*(Start-End))/(hi-lo)+End` to retain existing Simple-mode rounding. Strength 1 avoids an unnecessary blend. Optional final-zero is last. Original tensors are not modified; dtype/device/shape are preserved. Empty, single-element and constant schedules, or Strength 0, are returned unchanged. Nonfinite tensors/settings, invalid Strength/Curve and output overflow are rejected.

## ComfyUI

The root package exposes legacy `NODE_CLASS_MAPPINGS`, supported by the inspected ComfyUI loader. Only the tensor core is shared; no Forge modules are imported. No model, scheduler or sampling hooks are installed. A workflow serializes the node controls in the normal Comfy format; Forge infotext import is not a Comfy feature of this package.

## Removal

Disable the extension in Forge and restart, or remove its folder while Forge is stopped. Comfy users remove its custom-node folder and restart. There are no core-file patches to reverse.
