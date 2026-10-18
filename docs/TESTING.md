# Testing the preview

Run from this repository using an environment that already has PyTorch:

```sh
python -m unittest discover -s tests -v
```

Core tests need PyTorch. Optional Forge checks additionally use the host's Gradio, NumPy, SciPy and Pillow packages. They do not import the full Forge application or load models. Set `FORGE_ROOT` to an existing checkout to enable them; otherwise those checks are explicitly skipped.

Windows PowerShell example (substitute your paths):

```powershell
$env:FORGE_ROOT = 'C:\path\to\forge'
$env:GRADIO_ANALYTICS_ENABLED = 'False'
& "$env:FORGE_ROOT\venv\Scripts\python.exe" -m unittest discover -s tests -v
```

The host checks read native scheduler construction, Simple/Beta schedules, Euler sampling and PNG parsing from that checkout at runtime. Euler uses a small deterministic CPU denoiser, not a downloaded image model. This tests integration/numerics, not image quality. Exact basic arithmetic is checked in float16, bfloat16, float32 and float64.

Model-free UI preview:

```sh
python tools/ui_preview.py
```

It binds only to localhost port 7877 (`SIGMA_PREVIEW_PORT` overrides it), does not open a browser automatically and never connects to a running Forge server.

Before calling a release production-ready, run real-model txt2img/img2img/Hires smoke tests on the intended host/model combinations and compare saved PNG restoration. Those full-GPU and cross-host image-quality tests are not claimed for v0.1.0.
