"""Native Gradio controls; no browser scripts or changes to Forge's layout."""
import gradio as gr

from .settings import DEFAULTS, PASSES, restore


def reader(key):
    return lambda parameters: getattr(restore(parameters), key)


# Shared callback identities also allow Forge's txt2img/img2img transfer filter
# to match corresponding controls in the two Script instances.
READERS = {key: reader(key) for key in DEFAULTS}


def advanced_visibility(parameters):
    return gr.update(visible=restore(parameters).advanced)


def hires_visibility(parameters):
    settings = restore(parameters)
    return gr.update(visible=settings.advanced and settings.passes != "Main only" and settings.separate_hires)


def build_controls():
    controls = {}

    def slider(key, label, lo, hi, info, step=0.01):
        controls[key] = gr.Slider(lo, hi, value=DEFAULTS[key], step=step, label=label, info=info)

    with gr.Accordion("Sigma Rescale Neo", open=False):
        gr.Markdown("Remap the scheduler's sigma range. Start/End are noise levels, not step numbers.")
        controls["enabled"] = gr.Checkbox(value=False, label="Enable Sigma Rescale Neo")
        with gr.Row():
            slider("start", "Start", -10, 10, "Incoming maximum maps here.")
            slider("end", "End", -10, 10, "Incoming minimum maps here.")
        gr.Markdown("Basic mode affects the main pass only and keeps the familiar Start/End behavior.")
        with gr.Accordion("Advanced — optional", open=False):
            controls["advanced"] = gr.Checkbox(value=False, label="Enable Advanced controls")
            gr.Markdown("These reshape the schedule; they are not a skin enhancer or sharpening filter. "
                        "Compare one change at a time with a fixed seed.")
            with gr.Group(visible=False) as advanced_group:
                with gr.Row():
                    slider("strength", "Strength", 0, 1, "1 = full remap; 0 = original scheduler, not a softer preset.")
                    slider("curve", "Curve", 0.1, 10, "1 = linear. Above 1 moves interior sigmas toward End; below 1 toward Start.")
                controls["final_zero"] = gr.Checkbox(value=False, label="Force final sigma to zero",
                    info="Off preserves End. Turning this on changes the final sampling interval.")
                controls["passes"] = gr.Dropdown(choices=list(PASSES), value=DEFAULTS["passes"], label="Apply to")
                controls["separate_hires"] = gr.Checkbox(value=False, label="Use separate Hires settings")
                with gr.Group(visible=False) as hires_group:
                    with gr.Row():
                        slider("hires_start", "Hires Start", -10, 10, "Maximum for the Hires pass.")
                        slider("hires_end", "Hires End", -10, 10, "Minimum for the Hires pass.")
                    with gr.Row():
                        slider("hires_strength", "Hires Strength", 0, 1, "0 leaves the Hires schedule unchanged.")
                        slider("hires_curve", "Hires Curve", 0.1, 10, "1 keeps linear mapping.")
                    controls["hires_final_zero"] = gr.Checkbox(value=False, label="Force Hires final sigma to zero")

    controls["advanced"].change(lambda value: gr.update(visible=value),
                                inputs=controls["advanced"], outputs=advanced_group, queue=False)
    for key in ("advanced", "passes", "separate_hires"):
        controls[key].change(lambda advanced, passes, separate: gr.update(
            visible=advanced and passes != "Main only" and separate),
            inputs=[controls[k] for k in ("advanced", "passes", "separate_hires")], outputs=hires_group, queue=False)
    fields = [(controls[key], READERS[key]) for key in DEFAULTS]
    fields.extend([(advanced_group, advanced_visibility), (hires_group, hires_visibility)])
    return [controls[key] for key in DEFAULTS], fields
