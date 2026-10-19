# 0.2.0 preview — Full late-start compatibility

- Adds **Begin applying at (% of steps)** and separate Hires timing.
- Restores these values from existing Sigma Rescale Full PNG metadata.
- 0% preserves original full-run behavior; 100% is a no-op. Older PNGs default to 0%.
- Preserves the prefix through the rounded-up boundary, then smoothly fades in.
- Counts the sampled suffix for img2img/Hires without changing Forge source files.
- Includes the same optional timing input in the ComfyUI SIGMAS node.
- Rejects unsupported endpoint-only samplers and late schedules that would rise.
- Includes updated installation and PNG/IIB restoration instructions.

## Verification

50 CPU tests passed with the Forge host and local Full reference supplied,
including 768 compatibility cases across four tensor dtypes. Accepted schedules
match the Full reference bit-for-bit; rejected rising schedules agree too.
The real Forge PNG parser and Gradio paste fields restore both main and Hires
timing, including old Full metadata. Native scheduler/Euler integration remains
covered. Eleven isolated browser checks passed, including main/Hires percentage
inputs and narrow layouts. Full-model GPU quality and cross-application pixel
equality are not claimed.

This release updates the standalone public extension. It does not replace an
existing local Full installation automatically. Disable old Simple/Full extensions
before switching to Neo, so two plugins do not transform the same schedule.
