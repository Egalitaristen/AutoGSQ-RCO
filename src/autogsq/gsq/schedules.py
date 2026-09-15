"""Annealing schedules for temperature and logit scale in GSQ."""

from __future__ import annotations

import math
from typing import Tuple


def get_temperature(
    step: int,
    total_steps: int,
    temp_range: Tuple[float, float] = (2.0, 0.05),
    schedule: str = "cosine",
) -> float:
    """Calculate temperature tau for step in total_steps.

    Decays from temp_range[0] to temp_range[1].
    """
    if total_steps <= 1 or step <= 0:
        return float(temp_range[0])
    if step >= total_steps:
        return float(temp_range[1])

    progress = min(max(step / float(total_steps), 0.0), 1.0)
    t_start, t_end = float(temp_range[0]), float(temp_range[1])

    if schedule == "cosine":
        # Cosine decay from t_start to t_end
        cos_val = 0.5 * (1.0 + math.cos(math.pi * progress))
        return t_end + (t_start - t_end) * cos_val
    elif schedule == "exponential":
        # Exponential decay: t_start * (t_end / t_start) ** progress
        ratio = max(t_end / max(t_start, 1e-8), 1e-8)
        return t_start * (ratio**progress)
    elif schedule == "linear":
        return t_start + (t_end - t_start) * progress
    else:
        raise ValueError(f"Unknown schedule type: {schedule}")


def get_logit_scale(
    step: int,
    total_steps: int,
    scale_range: Tuple[float, float] = (100.0, 500.0),
    schedule: str = "linear",
) -> float:
    """Calculate logit scale kappa for step in total_steps.

    Increases from scale_range[0] to scale_range[1].
    """
    if total_steps <= 1 or step <= 0:
        return float(scale_range[0])
    if step >= total_steps:
        return float(scale_range[1])

    progress = min(max(step / float(total_steps), 0.0), 1.0)
    s_start, s_end = float(scale_range[0]), float(scale_range[1])

    if schedule == "cosine":
        # Cosine ramp from s_start to s_end
        cos_val = 0.5 * (1.0 - math.cos(math.pi * progress))
        return s_start + (s_end - s_start) * cos_val
    elif schedule == "linear":
        return s_start + (s_end - s_start) * progress
    elif schedule == "exponential":
        ratio = max(s_end / max(s_start, 1e-8), 1e-8)
        return s_start * (ratio**progress)
    else:
        raise ValueError(f"Unknown schedule type: {schedule}")
