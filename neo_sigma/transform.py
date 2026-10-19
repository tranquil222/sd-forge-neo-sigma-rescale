"""Tensor-only implementation of range mapping, power shaping, and blending.

This module has no Forge/Comfy imports, filesystem writes, or mutable settings.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Range:
    start: float = 0.9
    end: float = 0.0
    strength: float = 1.0
    curve: float = 1.0
    final_zero: bool = False
    begin_percent: float = 0.0

    def validate(self):
        if not all(math.isfinite(float(x)) for x in (self.start, self.end, self.strength, self.curve, self.begin_percent)):
            raise ValueError("Sigma Rescale values must be finite numbers")
        if not 0 <= self.strength <= 1:
            raise ValueError("Strength must be between 0 and 1")
        if not 0.1 <= self.curve <= 10:
            raise ValueError("Curve must be between 0.1 and 10")
        if not 0 <= self.begin_percent <= 100:
            raise ValueError("Begin applying at must be between 0 and 100 percent")
        return self


def transform(schedule: torch.Tensor, settings: Range, sample_start: int = 0) -> torch.Tensor:
    """Preserve tensor shape/device/dtype; never modify the caller's tensor.

    Linear path operation order is specified for existing-PNG compatibility:
    subtract minimum, multiply by target range, divide by source range, add End.
    Empty/constant input, zero Strength and 100% begin are exact no-ops.
    Late application leaves ceil(sampled steps * percent / 100) intervals intact,
    then smoothly ramps the blend to Strength. The full range is mapped before
    any host denoise slicing; sample_start identifies the suffix actually sampled.
    """
    settings.validate()
    if not isinstance(schedule, torch.Tensor) or schedule.ndim != 1 or not schedule.is_floating_point():
        raise ValueError("Expected a one-dimensional floating-point sigma tensor")
    if not torch.isfinite(schedule).all().item():
        raise ValueError("Incoming sigma schedule contains NaN or infinity")
    if schedule.numel() < 2 or settings.strength == 0 or settings.begin_percent == 100:
        return schedule
    if not isinstance(sample_start, int) or not 0 <= sample_start < schedule.numel():
        raise ValueError("Invalid sampled-schedule start index")
    boundary = sample_start + math.ceil((schedule.numel() - sample_start - 1) * settings.begin_percent / 100)
    if boundary >= schedule.numel() - 1:
        return schedule
    low, high = torch.aminmax(schedule)
    width = high - low
    if width.item() == 0:
        return schedule
    limits = schedule.new_tensor([settings.start, settings.end])
    result = schedule.clone().sub_(low)
    if settings.curve == 1:
        result.mul_(limits[0] - limits[1]).div_(width).add_(limits[1])
    else:
        result.div_(width).clamp_(0, 1).pow_(settings.curve)
        result.mul_(limits[0] - limits[1]).add_(limits[1])
    if settings.begin_percent > 0:
        if torch.any(schedule[1:] > schedule[:-1]).item():
            raise ValueError("Late application requires a descending sigma schedule")
        remaining = schedule.numel() - boundary - 1
        # Match existing Full PNG timing semantics, including fp32 ramps for
        # half-precision schedules and an unchanged boundary sigma.
        ramp_type = torch.float64 if schedule.dtype == torch.float64 else torch.float32
        fraction = torch.arange(1, remaining + 1, dtype=ramp_type, device=schedule.device) / remaining
        weight = (fraction.square() * (3 - 2 * fraction) * settings.strength).to(schedule.dtype)
        tail = slice(boundary + 1, None)
        result[tail] = schedule[tail] + weight * (result[tail] - schedule[tail])
        result[:boundary + 1] = schedule[:boundary + 1]
    elif settings.strength != 1:
        result.sub_(schedule).mul_(settings.strength).add_(schedule)
    if settings.final_zero:
        result[-1] = 0
    if not torch.isfinite(result).all().item():
        raise ValueError("Requested sigma transform exceeds the tensor's numeric range")
    if settings.begin_percent > 0 and torch.any(result[1:] > result[:-1]).item():
        raise ValueError("Late application would increase sigma during denoising. Reduce End/Strength, "
                         "adjust the range/curve, or use Force final sigma to zero.")
    return result
