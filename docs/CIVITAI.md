# Copy-ready Civitai note

Some of these images use **Sigma Rescale**. To load my Start/End settings in Forge Neo, you can install **Sigma Rescale Neo**:

https://github.com/tranquil222/sd-forge-neo-sigma-rescale

In Forge: **Extensions → Install from URL**, paste the link, install, then restart. Load the original PNG in **PNG Info → Send to txt2img** (or IIB's normal **Send to txt2img**) and check the Sigma Rescale Neo panel. The extension reads my Simple and Full metadata, including Strength, Curve, and **Begin applying at (% of steps)**. Advanced turns on automatically when loading Full settings.

You still need the same model, LoRAs/weights, sampler, scheduler, seed and other generation settings. If an image uses a hybrid/custom scheduler, that is a separate dependency. A screenshot or compressed image may not contain the original parameters. When replacing an older Sigma Rescale plugin, disable the old extension so the transform is not applied twice.

Some images also use [Krea Seed Variance Enhancer for Forge Neo](https://github.com/tranquil222/sd-forge-neo-krea-seed-variance). Install that separately to restore its settings and actual variation seed too.

This is an independent tool inspired by [RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) and lz-55's Forge workflow—not an official release from those authors. A ComfyUI SIGMAS node is also included, but it does not automatically convert Forge PNG settings into a complete Comfy workflow. Version 0.2.0 is a preview.
