"""Multiple-Choice Knapsack Problem (MCKP) solver for discrete feasibility in RCO."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch


def mckp_solve(
    utilities: Union[List[List[float]], np.ndarray, torch.Tensor],
    costs: Union[List[List[float]], np.ndarray, torch.Tensor],
    budget: float,
    max_budget_bins: int = 5000,
) -> List[int]:
    """Exact Multiple-Choice Knapsack Problem (MCKP) DP solver.

    Given N groups, each having K options:
    Pick exactly one option x_i in {0, ..., K-1} per group i to maximize
    sum_i utilities[i][x_i] subject to sum_i costs[i][x_i] <= budget.

    Args:
        utilities: (N, K) utility matrix (e.g. perturbed logits alpha_i + Gumbel)
        costs: (N, K) cost matrix (e.g. w_i * bits_k)
        budget: maximum total cost allowed
        max_budget_bins: discretization resolution for the DP table

    Returns:
        List of length N with the chosen option index for each group.
    """
    if isinstance(utilities, torch.Tensor):
        u_arr = utilities.detach().cpu().to(torch.float64).numpy()
    else:
        u_arr = np.asarray(utilities, dtype=np.float64)

    if isinstance(costs, torch.Tensor):
        c_arr = costs.detach().cpu().to(torch.float64).numpy()
    else:
        c_arr = np.asarray(costs, dtype=np.float64)

    N, K = u_arr.shape
    if N == 0:
        return []

    # Find minimum cost option per group
    min_cost_idx = np.argmin(c_arr, axis=1)
    min_costs = c_arr[np.arange(N), min_cost_idx]
    base_cost = np.sum(min_costs)

    if base_cost > budget:
        # Infeasible: even all-minimum options exceed budget
        # Return all-minimum assignment
        return min_cost_idx.tolist()

    residual_budget = budget - base_cost
    delta_costs = c_arr - min_costs[:, None]  # all >= 0

    # Max delta cost sum across all groups (maximum possible residual budget needed)
    max_delta_sum = np.sum(np.max(delta_costs, axis=1))

    if max_delta_sum <= 0 or residual_budget >= max_delta_sum:
        # Budget is so large that every group can independently pick its max utility option!
        return np.argmax(u_arr, axis=1).tolist()

    # Determine discretization scale factor
    effective_budget = min(residual_budget, max_delta_sum)
    scale = max_budget_bins / max(effective_budget, 1e-8)
    int_budget = int(math.floor(residual_budget * scale))
    int_budget = min(int_budget, max_budget_bins)

    # Discretize residual costs: ceil so we never undercount cost and exceed budget
    int_delta_costs = np.ceil(delta_costs * scale).astype(np.int64)

    # DP Table: dp[b] stores max utility achievable with residual integer budget b
    # Initialize with -infinity
    neg_inf = -1e18
    dp = np.full(int_budget + 1, neg_inf, dtype=np.float64)
    choices = np.full((N, int_budget + 1), -1, dtype=np.int16)

    # Base case for group 0
    for k in range(K):
        c_k = int_delta_costs[0, k]
        if c_k <= int_budget:
            u_k = u_arr[0, k]
            if u_k > dp[c_k]:
                dp[c_k] = u_k
                choices[0, c_k] = k

    # Transition for groups 1 to N-1
    for i in range(1, N):
        next_dp = np.full(int_budget + 1, neg_inf, dtype=np.float64)
        for b in range(int_budget + 1):
            prev_u = dp[b]
            if prev_u <= neg_inf / 2:
                continue
            for k in range(K):
                c_k = int_delta_costs[i, k]
                new_b = b + c_k
                if new_b <= int_budget:
                    new_u = prev_u + u_arr[i, k]
                    if new_u > next_dp[new_b]:
                        next_dp[new_b] = new_u
                        choices[i, new_b] = k
        dp = next_dp

    # Find the best ending budget within int_budget
    best_b = np.argmax(dp)
    if dp[best_b] <= neg_inf / 2:
        # DP did not reach a valid state (e.g. rounding issue), fall back to min cost
        return min_cost_idx.tolist()

    # Backtrack choices from group N-1 down to 0
    chosen = [0] * N
    curr_b = best_b
    for i in range(N - 1, -1, -1):
        k = int(choices[i, curr_b])
        if k < 0:
            k = int(min_cost_idx[i])
        chosen[i] = k
        curr_b -= int(int_delta_costs[i, k])
        curr_b = max(curr_b, 0)

    return chosen
