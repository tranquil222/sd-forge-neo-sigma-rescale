# Credits and provenance

Sigma Rescale Neo is a newly authored implementation of range mapping, power shaping and blending for sigma tensors, plus host adapters and metadata interoperability.

Inspiration and context:

- [ClownsharkBatwing / RES4LYF](https://github.com/ClownsharkBatwing/RES4LYF) and its [sigma tools](https://github.com/ClownsharkBatwing/RES4LYF/wiki/Sigmas-Nodes).
- lz-55's standalone Forge sigma-rescale workflow, shared in [Forge discussion #1415](https://github.com/Haoming02/sd-webui-forge-classic/discussions/1415).
- [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) and [ComfyUI](https://github.com/Comfy-Org/ComfyUI) provide the host interfaces.

No original RES4LYF or lz-55 source files are bundled. Original implementations were inspected for behavior and compatibility; this is **not a claim of formal clean-room development**. The code here was authored for this project rather than shipping the prior derivative package. The basic arithmetic order and legacy metadata names are deliberately preserved for existing-PNG interoperability. The optional host tests read selected methods from a checkout supplied by the person running them; host source is not vendored.

This is an independent project, not an official port, continuation, endorsement, or substitute for the authors' complete toolsets. The project's license covers this project's code, not the independently licensed tools linked above.

The standard AGPL-3.0-only license text is reproduced from the [SPDX license-list data](https://github.com/spdx/license-list-data/blob/main/text/AGPL-3.0-only.txt).
