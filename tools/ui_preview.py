"""Local model-free UI preview. Does not connect to or alter a Forge instance."""
import os
from pathlib import Path
import sys

os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gradio as gr
from neo_sigma.ui import build_controls

with gr.Blocks(analytics_enabled=False, title='Sigma Rescale Neo — UI preview', theme=gr.themes.Default()) as demo:
    gr.Markdown('## Sigma Rescale Neo\nModel-free control preview. No images are generated here.')
    build_controls()

if __name__ == '__main__':
    demo.launch(server_name='127.0.0.1', server_port=int(os.environ.get('SIGMA_PREVIEW_PORT', '7877')),
                inbrowser=False, show_api=False)
