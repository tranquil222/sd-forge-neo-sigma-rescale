"""A removable, request-local adapter around Forge's constructed schedules."""
from functools import wraps
import hashlib
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
                for suffix in ("hash", "first", "last", "sigma count"):
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
            outgoing = transform(incoming, chosen)
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
        setattr(sampler_class, OWNER_ATTR, self)

    def close(self):
        self.active = False
        if self.sampler_class.get_sigmas is self.wrapper:
            self.sampler_class.get_sigmas = self.previous
        if getattr(self.sampler_class, OWNER_ATTR, None) is self:
            delattr(self.sampler_class, OWNER_ATTR)
