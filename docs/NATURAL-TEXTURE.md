# Comparing for natural texture

The target here is less harsh contrast and less waxy/plastic-looking skin, **not maximum apparent sharpness or exaggerated pores**. Sigma controls change the sampling schedule. A setting that helps one model/prompt can hurt another; no skin preset has been validated for this preview.

## Keep a known-good baseline

If you already use Start `0.93` / End `0.03`, keep those exact values as the baseline. That is an existing-workflow example, not a universal recommendation. Leave Advanced off for the first comparison. Use the same seed, prompt, model/LoRAs, resolution, sampler, scheduler, CFG and steps.

Save the original PNG from each run. Compare at the same display size, with a face crop at 100% as well as the whole image. Look for gentle tonal transitions and believable variation, not merely more edge contrast, fake pore noise or blur that hides all texture.

## Small, one-variable experiments

These are deliberately modest **test points**, not proven optimal settings:

| Comparison | Values to try | Hold fixed |
| --- | --- | --- |
| Baseline | Advanced off | Your existing Start/End |
| Curve | 0.9, 1.0, 1.1 | Advanced on; Strength 1; same Start/End; final-zero off; Main only |
| Strength | 1.0, 0.9, 0.8 | Your preferred Curve and the same endpoints |
| Native control | Disable Neo | Everything else |

Strength blends back toward the original scheduler. If the original is what looked harsh, reducing Strength can **bring that look back**; it is not a "softness" slider. Curve changes intermediate sigma spacing, not skin detail directly.

Do not change End and Force final zero at the same time. A nonzero End also maps an incoming terminal zero to that nonzero value. Force final zero changes that behavior, so leave it off when reproducing older images with a nonzero End.

Keep Hires untouched during the first comparisons. If the base image looks right but the final Hires image does not, isolate the Hires stage before adding more main-pass changes. Separate Hires controls are for that comparison, not a required enhancement.

If none of the sigma comparisons helps, return to the baseline rather than layering more controls. Compare guidance, LoRA influence or postprocessing separately, within the model/workflow's documented ranges. There is no model-independent CFG number or sigma setting that guarantees natural skin.

## Sigma rescale is not CFG rescale

This extension remaps the **sigma schedule**; it does not rescale guidance predictions. Guidance rescaling is a separate method studied for overexposure in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://arxiv.org/abs/2305.08891). That paper does not validate these Start/End settings as a skin fix for Krea or every flow model. Neo's final-zero toggle is also not the paper's zero-terminal-SNR training-schedule change.

For sharing, report the actual settings and dependencies that produced the image. Prefer "these are the settings used" over promising a universal realism improvement.
