"""RCOOptimizer wrapping Adam with tangent projection, retraction, and momentum transport."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from autogsq.rco.manifold import (
    budget_normal,
    expected_cost,
    project_gradient,
    retract,
    vector_transport,
)
from autogsq.rco.knapsack import mckp_solve


class RCOOptimizer:
    """Riemannian Constrained Optimizer for discrete budget allocations.

    Maintains logits alpha over K options for N groups on the manifold:
    M = { alpha in R^{N x K} : C(alpha) = target_budget }.
    """

    def __init__(
        self,
        alpha_init: torch.Tensor,
        weights: torch.Tensor,
        costs: torch.Tensor,
        target_budget: float,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        tol: float = 1e-6,
    ) -> None:
        self.device = alpha_init.device
        self.dtype = alpha_init.dtype
        self.weights = weights.to(device=self.device, dtype=self.dtype)
        self.costs = costs.to(device=self.device, dtype=self.dtype)
        self.target_budget = float(target_budget)
        self.lr = float(lr)
        self.beta1, self.beta2 = betas
        self.eps = float(eps)
        self.tol = float(tol)

        # Retract initial logits to be exactly on the manifold
        self.alpha = retract(
            alpha_init.detach().clone(),
            self.weights,
            self.costs,
            self.target_budget,
            tol=self.tol,
        )

        # Adam state
        self.m = torch.zeros_like(self.alpha)
        self.v = torch.zeros_like(self.alpha)
        self.step_count = 0

    def get_current_cost(self) -> float:
        """Return current expected cost C(alpha)."""
        return expected_cost(self.alpha, self.weights, self.costs).item()

    def get_probabilities(self) -> torch.Tensor:
        """Return current softmax probabilities p over options."""
        return F.softmax(self.alpha, dim=-1)

    def step(self, loss_grad: torch.Tensor) -> torch.Tensor:
        """Perform one optimization step: tangent projection -> Adam -> retraction -> momentum transport.

        Args:
            loss_grad: (N, K) Euclidean gradient of loss w.r.t. alpha

        Returns:
            Updated alpha, guaranteed to satisfy C(alpha) == target_budget to tol.
        """
        p = self.get_probabilities()
        n_old = budget_normal(p, self.weights, self.costs)

        # 1. Tangent projection: project Euclidean gradient onto tangent plane
        g_tan = project_gradient(loss_grad, n_old)

        # 2. Adam update using tangent gradient
        self.step_count += 1
        t = self.step_count
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * g_tan
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (g_tan * g_tan)

        m_hat = self.m / (1.0 - self.beta1**t)
        v_hat = self.v / (1.0 - self.beta2**t)

        delta_alpha = -self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        alpha_drifted = self.alpha + delta_alpha

        # 3. Retraction: 1D bisection along cost vector back to manifold
        alpha_new = retract(
            alpha_drifted,
            self.weights,
            self.costs,
            self.target_budget,
            tol=self.tol,
        )

        # 4. Momentum transport: re-project momentum vector onto new tangent plane
        p_new = F.softmax(alpha_new, dim=-1)
        n_new = budget_normal(p_new, self.weights, self.costs)
        self.m = vector_transport(self.m, n_new)

        self.alpha = alpha_new
        return self.alpha

    def current_assignment(self, temperature: float = 1.0) -> List[int]:
        """Gumbel-STE sample + MCKP DP solve to obtain a discrete assignment feasible under budget.

        Returns:
            List of length N containing the chosen option index for each group.
        """
        device = self.alpha.device
        dtype = self.alpha.dtype

        # Sample Gumbel noise
        u = torch.rand(self.alpha.shape, device=device, dtype=dtype)
        eps = 1e-20
        g = -torch.log(-torch.log(u + eps) + eps)

        # Perturbed utilities
        utilities = (self.alpha + g) / max(temperature, 1e-4)

        # Cost matrix: (N, K) where cost[i, k] = weight[i] * cost[k]
        costs_matrix = self.weights.unsqueeze(-1) * self.costs.unsqueeze(0)

        # MCKP DP solve
        chosen = mckp_solve(utilities, costs_matrix, self.target_budget)
        return chosen


def optimize_budget(
    weights: torch.Tensor,
    costs: torch.Tensor,
    target_bpw: float,
    loss_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    group_names: Optional[List[str]] = None,
    num_steps: int = 2000,
    lr: float = 1e-2,
    temperature_start: float = 1.0,
    temperature_end: float = 0.1,
    option_labels: Optional[List[str]] = None,
    alpha_init: Optional[torch.Tensor] = None,
    temperature_schedule: str = "exponential",
    polish: bool = True,
) -> Dict[str, Any]:
    """Execute RCO search loop to find the exact-budget precision allocation.

    Args:
        weights: (N,) tensor of parameter counts per group
        costs: (K,) tensor of bit-width costs per option
        target_bpw: target bits per weight
        loss_fn: callable taking alpha (N, K) and returning (loss_scalar, grad_tensor)
        group_names: optional names for the N groups
        num_steps: total optimization steps
        lr: learning rate for RCOOptimizer
        temperature_start: initial Gumbel temperature
        temperature_end: final Gumbel temperature

    Returns:
        Dictionary with chosen assignments, achieved bpw, and optimization trace.
    """
    N = weights.shape[0]
    K = costs.shape[0]
    total_params = float(weights.sum().item())
    target_budget = target_bpw * total_params

    # Initialize logits uniformly, or continue from a previous phase.
    if alpha_init is None:
        alpha_start = torch.zeros((N, K), dtype=torch.float32, device=weights.device)
    else:
        alpha_start = alpha_init.detach().to(device=weights.device, dtype=torch.float32)
        assert alpha_start.shape == (N, K), f"alpha_init shape {tuple(alpha_start.shape)} != {(N, K)}"
    optimizer = RCOOptimizer(
        alpha_init=alpha_start,
        weights=weights,
        costs=costs,
        target_budget=target_budget,
        lr=lr,
    )

    best_loss = float("inf")
    best_assignment: Optional[List[int]] = None

    for step in range(num_steps):
        # Gumbel temperature schedule. Exponential (reference) spends more
        # steps near-discrete but needs enough total steps to avoid quenching;
        # linear suits short smoke runs.
        progress = step / max(num_steps - 1, 1)
        if temperature_schedule == "exponential":
            temp = max(temperature_end, temperature_start * (temperature_end / max(temperature_start, 1e-8)) ** progress)
        else:
            temp = temperature_start + (temperature_end - temperature_start) * progress

        loss, grad = loss_fn(optimizer.alpha)

        # Take geometric step
        optimizer.step(grad)

        # Evaluate discrete assignment
        discrete_assignment = optimizer.current_assignment(temperature=temp)

        current_loss_val = loss.item()
        if current_loss_val < best_loss:
            best_loss = current_loss_val
            best_assignment = copy.deepcopy(discrete_assignment)

    if best_assignment is None:
        best_assignment = optimizer.current_assignment(temperature=temperature_end)

    # Finalize: harden the final logits at low temperature. The RCO guarantee
    # is an *exact budget*: among feasible assignments we prefer those within
    # tolerance of the target, and only then the lower loss. (Tracked
    # high-temperature samples can otherwise win on noise while missing the
    # budget by far.)
    hard_assignment = optimizer.current_assignment(
        temperature=min(float(temperature_end), 0.05)
    )

    def _hard_alpha(choice: List[int]) -> torch.Tensor:
        a = torch.full_like(optimizer.alpha, -20.0)
        for i, k in enumerate(choice):
            a[i, int(k)] = 0.0
        return a

    # Compute achieved BPW
    costs_arr = costs.cpu().numpy()
    weights_arr = weights.cpu().numpy()

    def _achieved(choice: List[int]) -> float:
        tot = sum(
            float(weights_arr[i]) * float(costs_arr[int(choice[i])]) for i in range(N)
        )
        return tot / total_params

    tol = 0.05
    try:
        best_loss_hard, _ = loss_fn(_hard_alpha(best_assignment))
        hard_loss_hard, _ = loss_fn(_hard_alpha(hard_assignment))
        cands = [
            (best_assignment, float(best_loss_hard.item())),
            (hard_assignment, float(hard_loss_hard.item())),
        ]
        within = [(c, l) for c, l in cands if abs(_achieved(c) - target_bpw) <= tol]
        pool = within if within else cands
        best_assignment, best_loss = min(pool, key=lambda cl: cl[1])
    except Exception:
        pass

    # Greedy discrete polish: coordinate ascent over single-tensor changes,
    # keeping feasibility. Closes the entropy gap between the soft mean (on
    # budget by construction) and its argmax (which can sit far under).
    if polish:
        try:
            budget = target_bpw * total_params

            def _cost(choice: List[int]) -> float:
                return sum(
                    float(weights_arr[i]) * float(costs_arr[int(choice[i])])
                    for i in range(N)
                )

            cur = list(best_assignment)
            cur_loss = float(loss_fn(_hard_alpha(cur))[0].item())
            improved = True
            while improved:
                improved = False
                for i in range(N):
                    for k in range(K):
                        if int(k) == int(cur[i]):
                            continue
                        trial = list(cur)
                        trial[i] = int(k)
                        if _cost(trial) > budget + 1e-9:
                            continue
                        tl = float(loss_fn(_hard_alpha(trial))[0].item())
                        if tl < cur_loss - 1e-12:
                            cur, cur_loss = trial, tl
                            improved = True
                best_assignment, best_loss = list(cur), float(cur_loss)
        except Exception:
            pass

    total_bits = sum(weights_arr[i] * costs_arr[best_assignment[i]] for i in range(N))
    achieved_bpw = total_bits / total_params

    # Build per-group mapping
    per_group_result = {}
    for i in range(N):
        name = group_names[i] if group_names and i < len(group_names) else f"group_{i}"
        chosen_opt = best_assignment[i]
        label = (
            option_labels[chosen_opt]
            if option_labels and chosen_opt < len(option_labels)
            else str(costs_arr[chosen_opt])
        )
        per_group_result[name] = {
            "option_idx": chosen_opt,
            "label": label,
            "bits": float(costs_arr[chosen_opt]),
            "params": int(weights_arr[i]),
        }

    return {
        "target_bpw": target_bpw,
        "achieved_bpw": achieved_bpw,
        "best_loss": best_loss,
        "assignment_indices": best_assignment,
        "per_group": per_group_result,
        "final_alpha": optimizer.alpha.detach().cpu(),
    }
