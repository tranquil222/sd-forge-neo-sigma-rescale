"""Forge entry point. All integration is in-memory and removable."""
from modules import scripts, script_callbacks
from modules.sd_samplers_kdiffusion import KDiffusionSampler

from neo_sigma.forge import ScheduleHook, prepare
from neo_sigma.settings import DEFAULTS, PREFIX, Settings
from neo_sigma.ui import READERS, build_controls


HOOK = ScheduleHook(KDiffusionSampler)
script_callbacks.on_script_unloaded(HOOK.close)


class Script(scripts.Script):
    def title(self):
        return PREFIX

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        controls, self.infotext_fields = build_controls()
        self.paste_field_names = list(READERS.values())
        return controls

    def process(self, processing, *values):
        if len(values) > len(DEFAULTS):
            raise ValueError("Sigma Rescale Neo received too many control values")
        settings = Settings(**dict(zip(DEFAULTS, values)))
        prepare(processing, settings)

    def postprocess(self, processing, processed, *values):
        params = processing.extra_generation_params
        if params.get(PREFIX) and not any(params.get(f"{PREFIX} {phase} applied") for phase in ("main", "Hires")):
            print("[Sigma Rescale Neo] No schedule changed. Check pass selection, Strength, and sampler support.")
