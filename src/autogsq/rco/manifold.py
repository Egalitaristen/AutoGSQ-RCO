"""Riemannian manifold geometry for budget-constrained optimization.

Implements the budget manifold M = { alpha : C(alpha) = B }:
1. Tangent projection: projects loss gradient onto tangent plane.
2. Retraction: 1D bisection along cost vector restoring C(alpha') = B.
3. Vector transport: re-projects Adam momentum onto new tangent plane.
"""

from __future__ import annotations

from typing import Callable, Optional
import torch
import torch.nn.functional as F


def expected_cost(alpha: torch.Tensor, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Compute expected total cost C(alpha) in double precision for numerical accuracy.

    Args:
        alpha: (N, K) logits over K options for N groups
        w: (N,) group weights (e.g. parameter count)
        c: (K,) per-option costs (e.g. bit-widths)

    Returns:
        Scalar tensor with total cost
    """
    alpha_d = alpha.to(torch.float64)
    w_d = w.to(torch.float64)
    c_d = c.to(torch.float64)

    p = F.softmax(alpha_d, dim=-1)  # (N, K)
    expected_c_per_group = torch.matmul(p, c_d.unsqueeze(-1)).squeeze(-1)  # (N,)
    return torch.sum(w_d * expected_c_per_group)


def budget_normal(p: torch.Tensor, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Compute normal vector to budget manifold: n_{i,k} = w_i * p_{i,k} * (c_k - E_{p_i}[c]).

    Args:
        p: (N, K) softmax probabilities over options
        w: (N,) or (N, 1) group weights
        c: (K,) or (1, K) option costs

    Returns:
        n: (N, K) normal vector to the manifold M
    """
    orig_dtype = p.dtype
    p_d = p.to(torch.float64)
    w_d = w.to(torch.float64)
    c_d = c.to(torch.float64)

    if w_d.dim() == 1:
        w_expanded = w_d.unsqueeze(-1)
    else:
        w_expanded = w_d

    if c_d.dim() == 1:
        c_expanded = c_d.unsqueeze(0)
    else:
        c_expanded = c_d

    # E_{p_i}[c] = sum_k p_{i,k} * c_k
    exp_c = torch.sum(p_d * c_expanded, dim=-1, keepdim=True)  # (N, 1)
    diff = c_expanded - exp_c  # (N, K)
    normal = w_expanded * p_d * diff  # (N, K)
    return normal.to(orig_dtype)


def project_gradient(g: torch.Tensor, n: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Project Euclidean gradient g onto tangent plane orthogonal to normal n.

    g_tan = g - (<g, n> / ||n||^2) * n
    Calculated in float64 for exact orthogonality.
    """
    orig_dtype = g.dtype
    g_d = g.to(torch.float64)
    n_d = n.to(torch.float64)

    dot_prod = torch.sum(g_d * n_d)
    norm_sq = torch.sum(n_d * n_d)

    if norm_sq < eps:
        return g

    proj_scalar = dot_prod / (norm_sq + eps)
    g_tan = g_d - proj_scalar * n_d
    return g_tan.to(orig_dtype)


def retract(
    alpha: torch.Tensor,
    w: torch.Tensor,
    c: torch.Tensor,
    target_budget: float,
    tol: float = 1e-6,
    max_iters: int = 80,
) -> torch.Tensor:
    """Retract alpha back onto manifold M = { alpha : C(alpha) = target_budget }.

    alpha' = alpha + t * c, solved via 1D bisection search for t.
    Uses float64 internally so machine precision is reached even for large budgets.
    """
    orig_dtype = alpha.dtype
    alpha_d = alpha.to(torch.float64)
    c_vec = c.to(device=alpha.device, dtype=torch.float64)
    w_vec = w.to(device=alpha.device, dtype=torch.float64)
    target_b = float(target_budget)

    def cost_at(t_val: float) -> float:
        t_tensor = torch.tensor(t_val, device=alpha.device, dtype=torch.float64)
        alpha_t = alpha_d + t_tensor * c_vec.unsqueeze(0)
        return expected_cost(alpha_t, w_vec, c_vec).item()

    current_cost = cost_at(0.0)
    diff = current_cost - target_b
    if abs(diff) <= tol:
        return alpha.clone()

    # Step 1: Find valid bracket [t_low, t_high]
    step = 0.1
    if diff < 0:
        t_low = 0.0
        t_high = step
        while cost_at(t_high) < target_b and t_high < 100.0:
            t_low = t_high
            t_high *= 2.0
    else:
        t_high = 0.0
        t_low = -step
        while cost_at(t_low) > target_b and t_low > -100.0:
            t_high = t_low
            t_low *= 2.0

    # Step 2: Bisection search
    for _ in range(max_iters):
        t_mid = 0.5 * (t_low + t_high)
        c_mid = cost_at(t_mid)
        if abs(c_mid - target_b) <= tol:
            break
        if c_mid < target_b:
            t_low = t_mid
        else:
            t_high = t_mid

    best_t = 0.5 * (t_low + t_high)
    alpha_retracted = alpha_d + torch.tensor(best_t, device=alpha.device, dtype=torch.float64) * c_vec.unsqueeze(0)
    return alpha_retracted.to(orig_dtype)


def vector_transport(momentum: torch.Tensor, new_n: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Transport momentum vector onto the new tangent plane at the retracted point."""
    return project_gradient(momentum, new_n, eps=eps)
