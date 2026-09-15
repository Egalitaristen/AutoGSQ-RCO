"""Lion optimizer (EvoLved Sign Momentum) with per-group settings.

Reference: GSQ trains quantizer logits and scales with Lion using separate
learning rates per parameter group, weight decay on logits only, and a
warmup + decay schedule (applied by the caller setting ``lr`` per step).
"""

from __future__ import annotations

from typing import Iterable, Optional
import torch
from torch.optim import Optimizer


class Lion(Optimizer):
    """Lion optimizer with decoupled weight decay.

    Args:
        params: parameters or param groups (each group may carry its own
            ``lr`` and ``weight_decay``).
        lr: default learning rate.
        betas: (beta1, beta2) momentum coefficients.
        weight_decay: default decoupled weight-decay factor.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 2e-4,
        betas: tuple = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            wd = group.get("weight_decay", 0.0)
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                # Sign update from interpolated momentum.
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1).sign_()
                # Decay the parameters directly (decoupled).
                if wd != 0.0:
                    p.mul_(1.0 - group["lr"] * wd)
                p.add_(update, alpha=-group["lr"])
                # Momentum update.
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)
        return loss
