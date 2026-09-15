"""RCO (Riemannian Constrained Optimization) module."""

from autogsq.rco.manifold import (
    budget_normal,
    project_gradient,
    retract,
    vector_transport,
    expected_cost,
)
from autogsq.rco.knapsack import mckp_solve
from autogsq.rco.solver import RCOOptimizer, optimize_budget

__all__ = [
    "budget_normal",
    "project_gradient",
    "retract",
    "vector_transport",
    "expected_cost",
    "mckp_solve",
    "RCOOptimizer",
    "optimize_budget",
]
