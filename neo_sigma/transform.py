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

    def validate(self):
        if not all(math.isfinite(float(x)) for x in (self.start, self.end, self.strength, self.curve)):
            raise ValueError("Sigma Rescale values must be finite numbers")
        if not 0 <= self.strength <= 1:
            raise ValueError("Strength must be between 0 and 1")
        if not 0.1 <= self.curve <= 10:
            raise ValueError("Curve must be between 0.1 and 10")
        return self


def transform(schedule: torch.Tensor, settings: Range) -> torch.Tensor:
    """Preserve tensor shape/device/dtype; never modify the caller's tensor.

    Linear path operation order is specified for existing-PNG compatibility:
    subtract minimum, multiply by target range, divide by source range, add End.
    Empty/constant input and zero Strength are unchanged, including final-zero.
    """
    settings.validate()
    if not isinstance(schedule, torch.Tensor) or schedule.ndim != 1 or not schedule.is_floating_point():
        raise ValueError("Expected a one-dimensional floating-point sigma tensor")
    if not torch.isfinite(schedule).all().item():
        raise ValueError("Incoming sigma schedule contains NaN or infinity")
    if schedule.numel() < 2 or settings.strength == 0:
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
    if settings.strength != 1:
        result.sub_(schedule).mul_(settings.strength).add_(schedule)
    if settings.final_zero:
        result[-1] = 0
    if not torch.isfinite(result).all().item():
        raise ValueError("Requested sigma transform exceeds the tensor's numeric range")
    return result
