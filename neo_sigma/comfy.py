"""Optional ComfyUI SIGMAS adapter using the same tensor implementation."""
from .transform import Range, transform


class SigmaRescaleNeo:
    CATEGORY = "sampling/sigmas"
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "apply"
    DESCRIPTION = "Map the input sigma range to Start/End, with optional power shaping and blending."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sigmas": ("SIGMAS",),
            "enabled": ("BOOLEAN", {"default": True}),
            "start": ("FLOAT", {"default": 0.9, "min": -10.0, "max": 10.0, "step": 0.01}),
            "end": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "advanced": ("BOOLEAN", {"default": False}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "curve": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
            "final_zero": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "begin_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
        }}

    def apply(self, sigmas, enabled=True, start=0.9, end=0.0, advanced=False,
              strength=1.0, curve=1.0, final_zero=False, begin_percent=0.0):
        if not enabled:
            return (sigmas,)
        settings = Range(start, end, strength, curve, final_zero, begin_percent) if advanced else Range(start, end)
        return (transform(sigmas, settings),)
