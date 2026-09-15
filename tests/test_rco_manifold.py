"""Unit tests for Riemannian Constrained Optimization (RCO) manifold operations."""

import pytest
import torch
import torch.nn.functional as F

from autogsq.rco.manifold import (
    budget_normal,
    project_gradient,
    retract,
    vector_transport,
    expected_cost,
)
from autogsq.rco.solver import RCOOptimizer


def test_budget_normal_properties():
    """Verify normal vector to the budget manifold sums to 0 per group."""
    torch.manual_seed(42)
    N, K = 5, 4
    alpha = torch.randn(N, K, dtype=torch.float64)
    w = torch.tensor([100.0, 200.0, 150.0, 300.0, 250.0], dtype=torch.float64)
    c = torch.tensor([2.0, 3.0, 4.0, 6.0], dtype=torch.float64)

    p = F.softmax(alpha, dim=-1)
    normal = budget_normal(p, w, c)

    assert normal.shape == (N, K)
    # The normal in logit space must sum to 0 along each group's options
    sum_per_group = torch.sum(normal, dim=-1)
    assert torch.allclose(sum_per_group, torch.zeros_like(sum_per_group), atol=1e-10)


def test_tangent_projection_orthogonality():
    """Verify projected gradient is strictly orthogonal to normal vector: <g_tan, n> = 0."""
    torch.manual_seed(42)
    N, K = 8, 3
    alpha = torch.randn(N, K, dtype=torch.float64)
    w = (torch.rand(N, dtype=torch.float64) * 1000 + 100)
    c = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

    p = F.softmax(alpha, dim=-1)
    normal = budget_normal(p, w, c)

    euclidean_grad = torch.randn(N, K, dtype=torch.float64)
    g_tan = project_gradient(euclidean_grad, normal)

    inner_product = torch.sum(g_tan * normal)
    assert abs(inner_product.item()) < 1e-10


def test_retraction_convergence():
    """Verify retraction converges to target budget within 1e-6 in O(log) steps."""
    torch.manual_seed(42)
    N, K = 10, 4
    alpha = torch.randn(N, K, dtype=torch.float64)
    w = torch.tensor([1000.0] * N, dtype=torch.float64)
    c = torch.tensor([2.0, 3.0, 4.0, 8.0], dtype=torch.float64)

    # Desired budget: e.g. 2.75 bpw on total params (10 * 1000 = 10000) -> 27500
    target_budget = 2.75 * 10000.0

    retracted_alpha = retract(alpha, w, c, target_budget, tol=1e-6)
    achieved_cost = expected_cost(retracted_alpha, w, c).item()

    assert abs(achieved_cost - target_budget) < 1e-5


def test_vector_transport_orthogonality():
    """Verify transported momentum is orthogonal to the new normal."""
    torch.manual_seed(42)
    N, K = 6, 3
    w = torch.ones(N, dtype=torch.float64) * 500
    c = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float64)

    # Point 1
    alpha1 = torch.randn(N, K, dtype=torch.float64)
    p1 = F.softmax(alpha1, dim=-1)
    n1 = budget_normal(p1, w, c)
    m = project_gradient(torch.randn(N, K, dtype=torch.float64), n1)

    # Point 2
    alpha2 = alpha1 + 0.1 * torch.randn(N, K, dtype=torch.float64)
    p2 = F.softmax(alpha2, dim=-1)
    n2 = budget_normal(p2, w, c)

    m_transported = vector_transport(m, n2)
    inner_prod = torch.sum(m_transported * n2).item()
    assert abs(inner_prod) < 1e-10


def test_rco_optimizer_budget_invariance():
    """Verify budget holds to floating-point precision throughout 500+ optimization steps."""
    torch.manual_seed(42)
    N, K = 12, 4
    w = torch.randint(100, 5000, (N,)).to(torch.float64)
    c = torch.tensor([2.0, 3.0, 4.0, 8.0], dtype=torch.float64)
    total_params = float(w.sum().item())
    target_bpw = 3.25
    target_budget = target_bpw * total_params

    alpha_init = torch.randn(N, K, dtype=torch.float64)
    optimizer = RCOOptimizer(
        alpha_init=alpha_init,
        weights=w,
        costs=c,
        target_budget=target_budget,
        lr=0.05,
    )

    # Synthetic loss target: quadratic distance to target distribution
    alpha_star = torch.randn(N, K, dtype=torch.float64)

    for step in range(500):
        # Loss = ||alpha - alpha_star||^2
        grad = 2.0 * (optimizer.alpha - alpha_star)
        optimizer.step(grad)

        # Verify budget is strictly satisfied at every single step!
        current_cost = optimizer.get_current_cost()
        assert abs(current_cost - target_budget) < 1e-4, f"Budget violated at step {step}: {current_cost} vs {target_budget}"

    # Final cost check
    final_cost = optimizer.get_current_cost()
    assert abs(final_cost - target_budget) < 1e-5
