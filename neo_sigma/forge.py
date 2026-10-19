"""A removable, request-local adapter around Forge's constructed schedules."""
from functools import wraps
from contextvars import ContextVar
import hashlib
import inspect
import math
import torch

from .settings import PREFIX, Settings
from .transform import transform

REQUEST_ATTR = "_neo_sigma_settings"
OWNER_ATTR = "_neo_sigma_hook_owner"
ALIASES_ATTR = "_neo_sigma_emitted_aliases"


def is_hires(processing):
    return any(bool(getattr(processing, key, False)) for key in
               ("is_hr_pass", "_is_hr_pass", "_in_hr_pass", "in_hr_pass", "is_hires_pass"))


def prepare(processing, settings: Settings):
    # Store before validation: Forge logs some Script.process errors and proceeds.
    # The sampling boundary validates again rather than silently ignoring them.
    setattr(processing, REQUEST_ATTR, settings)
    params = processing.extra_generation_params
    for key, value in getattr(processing, ALIASES_ATTR, {}).items():
        if params.get(key) == value:
            params.pop(key, None)
    setattr(processing, ALIASES_ATTR, {})
    for key in list(params):
        if key == PREFIX or key.startswith(PREFIX + " "):
            del params[key]
    settings.validate()
    if settings.enabled:
        params.update({k: v for k, v in settings.metadata().items() if k.startswith(PREFIX)})
        for phase in ("main", "Hires"):
            params[f"{PREFIX} {phase} applied"] = False


class ScheduleHook:
    """Keep other extensions' wrappers intact, including wrappers above ours."""

    def __init__(self, sampler_class):
        old = getattr(sampler_class, OWNER_ATTR, None)
        if old is not None:
            old.close()
        self.sampler_class = sampler_class
        self.previous = sampler_class.get_sigmas
        self.previous_img2img = getattr(sampler_class, "sample_img2img", None)
        self.img2img_context = ContextVar("neo_sigma_img2img", default=None)
        self.img2img_wrapper = None
        self.active = True

        @wraps(self.previous)
        def get_sigmas(sampler, processing, steps):
            settings = getattr(processing, REQUEST_ATTR, Settings())
            if not self.active or not settings.enabled:
                return self.previous(sampler, processing, steps)
            settings.validate()
            legacy_full = getattr(processing, "_sigma_rescale_full_options", {}) or {}
            if (getattr(processing, "_sigma_rescale_standalone_enabled", False)
                    or legacy_full.get("enabled", False)):
                raise ValueError("Sigma Rescale Neo: disable the other Sigma Rescale / Sigma Rescale Full "
                                 "panel before using Neo, to avoid applying the transform twice.")
            hires = is_hires(processing)
            chosen = settings.for_pass(hires)
            params = processing.extra_generation_params
            phase = "Hires" if hires else "main"
            # A processing object can span a batch: do not attach the previous
            # image's Hires/hash diagnostics to the next main-pass image.
            for reset_phase in (("Hires",) if hires else ("main", "Hires")):
                for suffix in ("hash", "first", "last", "sigma count", "unchanged steps"):
                    params.pop(f"{PREFIX} {reset_phase} {suffix}", None)
                params[f"{PREFIX} {reset_phase} applied"] = False
            incoming = self.previous(sampler, processing, steps)
            # This also marks Hires-only Advanced images as incompatible with
            # an old Simple reader, even though their main pass is untouched.
            if not hires:
                aliases = {k: v for k, v in settings.metadata().items() if not k.startswith(PREFIX)}
                params.update(aliases)
                setattr(processing, ALIASES_ATTR, aliases)
            if chosen is None:
                return incoming
            sample_start = 0
            if 0 < chosen.begin_percent < 100 and chosen.strength > 0:
                sampler_function = getattr(sampler, "func", None)
                if callable(sampler_function):
                    arguments = inspect.signature(sampler_function).parameters
                    if not {"sigmas", "sigma_sched"}.intersection(arguments):
                        raise ValueError("Sigma Rescale Neo: late application needs a sampler accepting an "
                                         "explicit sigma schedule. Endpoint-only/adaptive samplers cannot "
                                         "honor a percentage of steps.")
            if chosen.begin_percent > 0:
                context = self.img2img_context.get()
                if context is not None and context[0] is sampler and context[1] is processing:
                    from modules.sd_samplers_common import setup_img2img_steps
                    planned_steps, encoded_steps = setup_img2img_steps(processing, context[2])
                    if planned_steps != steps:
                        raise ValueError("Sigma Rescale Neo: img2img step planning changed; cannot locate "
                                         "the late-application boundary reliably.")
                    # Use Python's exact slice normalization, including a negative
                    # start when the host requests an unusually short schedule.
                    sample_start = slice(steps - encoded_steps - 1, None).indices(incoming.numel())[0]
            outgoing = transform(incoming, chosen, sample_start=sample_start)
            params[f"{PREFIX} {phase} unchanged steps"] = math.ceil(
                max(0, incoming.numel() - sample_start - 1) * chosen.begin_percent / 100)
            params[f"{PREFIX} {phase} applied"] = outgoing is not incoming
            params[f"{PREFIX} {phase} sigma count"] = outgoing.numel()
            if outgoing is not incoming:
                digest = hashlib.sha256(outgoing.detach().to(dtype=torch.float64)
                                        .cpu().contiguous().numpy().tobytes()).hexdigest()[:12]
                params[f"{PREFIX} {phase} hash"] = digest
                params[f"{PREFIX} {phase} first"] = outgoing[0].item()
                params[f"{PREFIX} {phase} last"] = outgoing[-1].item()
            return outgoing

        self.wrapper = get_sigmas
        sampler_class.get_sigmas = get_sigmas
        if self.previous_img2img is not None:
            @wraps(self.previous_img2img)
            def sample_img2img(sampler, processing, *args, **kwargs):
                if not self.active or not getattr(processing, REQUEST_ATTR, Settings()).enabled:
                    return self.previous_img2img(sampler, processing, *args, **kwargs)
                bound = inspect.signature(self.previous_img2img).bind(sampler, processing, *args, **kwargs)
                token = self.img2img_context.set((sampler, processing, bound.arguments.get("steps")))
                try:
                    return self.previous_img2img(sampler, processing, *args, **kwargs)
                finally:
                    self.img2img_context.reset(token)

            self.img2img_wrapper = sample_img2img
            sampler_class.sample_img2img = sample_img2img
        setattr(sampler_class, OWNER_ATTR, self)

    def close(self):
        self.active = False
        if self.sampler_class.get_sigmas is self.wrapper:
            self.sampler_class.get_sigmas = self.previous
        if self.img2img_wrapper is not None and self.sampler_class.sample_img2img is self.img2img_wrapper:
            self.sampler_class.sample_img2img = self.previous_img2img
        if getattr(self.sampler_class, OWNER_ATTR, None) is self:
            delattr(self.sampler_class, OWNER_ATTR)
