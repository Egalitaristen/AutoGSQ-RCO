"""Unit tests for Multiple-Choice Knapsack Problem (MCKP) solver."""

import itertools
import numpy as np
import pytest
import torch

from autogsq.rco.knapsack import mckp_solve


def brute_force_mckp(utilities: np.ndarray, costs: np.ndarray, budget: float):
    """Exhaustive search over all K^N combinations to find the exact global maximum."""
    N, K = utilities.shape
    best_u = -float("inf")
    best_choice = None

    for choice in itertools.product(range(K), repeat=N):
        total_cost = sum(costs[i, choice[i]] for i in range(N))
        if total_cost <= budget:
            total_u = sum(utilities[i, choice[i]] for i in range(N))
            if total_u > best_u:
                best_u = total_u
                best_choice = list(choice)

    return best_choice, best_u


def test_mckp_matches_brute_force_small():
    """Verify MCKP DP solver finds the optimal assignment matching exhaustive brute force."""
    np.random.seed(42)
    N, K = 4, 3
    utilities = np.random.uniform(1.0, 10.0, size=(N, K))
    costs = np.random.uniform(2.0, 8.0, size=(N, K))

    # Min cost and max cost
    min_c = np.sum(np.min(costs, axis=1))
    max_c = np.sum(np.max(costs, axis=1))
    budget = min_c + 0.5 * (max_c - min_c)  # Intermediate budget

    bf_choice, bf_u = brute_force_mckp(utilities, costs, budget)
    dp_choice = mckp_solve(utilities, costs, budget, max_budget_bins=5000)

    # Calculate DP utility and cost
    dp_cost = sum(costs[i, dp_choice[i]] for i in range(N))
    dp_u = sum(utilities[i, dp_choice[i]] for i in range(N))

    # Feasibility check
    assert dp_cost <= budget + 1e-5, f"DP cost {dp_cost} exceeded budget {budget}"

    # Utility optimality check (within discretization tolerance)
    assert abs(dp_u - bf_u) < 0.1, f"DP utility {dp_u} differs from brute force {bf_u}"
    if abs(dp_u - bf_u) < 1e-4:
        assert dp_choice == bf_choice


def test_mckp_strict_feasibility():
    """Verify that under random instances, MCKP DP solution is strictly feasible."""
    np.random.seed(123)
    for _ in range(10):
        N, K = 8, 4
        utilities = np.random.uniform(0.0, 20.0, size=(N, K))
        costs = np.random.uniform(1.0, 10.0, size=(N, K))

        min_c = np.sum(np.min(costs, axis=1))
        max_c = np.sum(np.max(costs, axis=1))
        budget = min_c + np.random.uniform(0.2, 0.8) * (max_c - min_c)

        chosen = mckp_solve(utilities, costs, budget, max_budget_bins=4000)
        total_cost = sum(costs[i, chosen[i]] for i in range(N))

        assert total_cost <= budget + 1e-4, f"Feasibility violated: {total_cost} > {budget}"


def test_mckp_infeasible_fallback():
    """Verify graceful handling when budget is smaller than sum of minimum options."""
    utilities = np.array([[1.0, 2.0], [3.0, 4.0]])
    costs = np.array([[5.0, 10.0], [5.0, 10.0]])
    budget = 4.0  # Even min cost (5 + 5 = 10) exceeds budget

    chosen = mckp_solve(utilities, costs, budget)
    # Should return all minimum cost choices [0, 0]
    assert chosen == [0, 0]
