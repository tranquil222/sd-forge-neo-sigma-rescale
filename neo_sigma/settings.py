"""Serializable controls and old/new PNG parameter interoperability."""
from dataclasses import asdict, dataclass
from typing import Mapping

from . import __version__
from .transform import Range

PASSES = ("Main only", "Main + Hires", "Hires only")
PREFIX = "Sigma Rescale Neo"


@dataclass(frozen=True)
class Settings:
    enabled: bool = False
    start: float = 0.9
    end: float = 0.0
    advanced: bool = False
    strength: float = 1.0
    curve: float = 1.0
    final_zero: bool = False
    passes: str = "Main only"
    separate_hires: bool = False
    hires_start: float = 0.9
    hires_end: float = 0.0
    hires_strength: float = 1.0
    hires_curve: float = 1.0
    hires_final_zero: bool = False

    def for_pass(self, hires: bool):
        if not self.enabled:
            return None
        if not self.advanced:
            return None if hires else Range(self.start, self.end).validate()
        if self.passes not in PASSES:
            raise ValueError("Select Main only, Main + Hires, or Hires only")
        if (hires and self.passes == "Main only") or (not hires and self.passes == "Hires only"):
            return None
        keys = ("start", "end", "strength", "curve", "final_zero")
        prefix = "hires_" if hires and self.separate_hires else ""
        return Range(**{k: getattr(self, prefix + k) for k in keys}).validate()

    def validate(self):
        self.for_pass(False)
        self.for_pass(True)
        return self

    def metadata(self):
        if not self.enabled:
            return {}
        fields = {LABELS[key]: value for key, value in asdict(self).items()}
        fields[PREFIX + " version"] = __version__
        # Simple readers must not silently approximate an advanced transform.
        fields["Sigma Rescale"] = not self.advanced
        fields["Sigma Rescale start"] = self.start
        fields["Sigma Rescale end"] = self.end
        return fields


DEFAULTS = asdict(Settings())
LABELS = {key: PREFIX if key == "enabled" else PREFIX + " " + key.replace("_", " ") for key in DEFAULTS}
FULL_LABELS = {
    "enabled": "Sigma Rescale Full", "start": "Sigma Rescale Full start", "end": "Sigma Rescale Full end",
    "strength": "Sigma Rescale Full strength", "curve": "Sigma Rescale Full curve",
    "final_zero": "Sigma Rescale Full final zero", "passes": "Sigma Rescale Full passes",
    "separate_hires": "Sigma Rescale Full separate Hires", "hires_start": "Sigma Rescale Full Hires start",
    "hires_end": "Sigma Rescale Full Hires end", "hires_strength": "Sigma Rescale Full Hires strength",
    "hires_curve": "Sigma Rescale Full Hires curve", "hires_final_zero": "Sigma Rescale Full Hires final zero",
}


def truth(value):
    return value is True or str(value).strip().lower() in {"true", "1", "yes", "on"}


def restore(parameters: Mapping) -> Settings:
    """Reset missing fields; prefer our complete metadata over legacy aliases."""
    if PREFIX in parameters:
        labels, advanced = LABELS, False
    elif truth(parameters.get("Sigma Rescale Full")):
        labels, advanced = FULL_LABELS, True
    else:
        labels = {"enabled": "Sigma Rescale", "start": "Sigma Rescale start", "end": "Sigma Rescale end"}
        advanced = False
    values = dict(DEFAULTS, advanced=advanced)
    for key, label in labels.items():
        if label not in parameters:
            continue
        value = parameters[label]
        default = DEFAULTS[key]
        try:
            values[key] = truth(value) if isinstance(default, bool) else float(value) if isinstance(default, float) else str(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid PNG setting: {label}") from None
    return Settings(**values).validate()
