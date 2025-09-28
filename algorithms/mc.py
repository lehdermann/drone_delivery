from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import numpy as np


def mc_control_epsilon_soft(
    env,
    episodes: int = 10_000,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    first_visit: bool = True,
    seed: int | None = 42,
):
    """On-policy Monte Carlo Control with epsilon-soft policy, first-visit by default.

    Returns:
        policy: np.ndarray [nS]
        Q: np.ndarray [nS, nA]
        returns_avg: list of episode returns for monitoring
    """
    rng = np.random.default_rng(seed)
    nS, nA = env.nS, env.nA
    Q = np.zeros((nS, nA), dtype=np.float64)
    returns_count = np.zeros((nS, nA), dtype=np.int64)

    def epsilon_greedy_action(s: int) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(low=0, high=nA))
        return int(np.argmax(Q[s]))

    returns_avg: List[float] = []

    for _ in range(episodes):
        # Generate episode
        episode: List[Tuple[int, int, float]] = []  # (s, a, r)
        s, _ = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            a = epsilon_greedy_action(s)
            s2, r, done, truncated, _ = env.step(a)
            episode.append((s, a, r))
            s = s2

        # Compute returns G_t backwards
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s_t, a_t, r_t = episode[t]
            G = gamma * G + r_t
            key = (s_t, a_t)
            if first_visit and key in visited:
                continue
            visited.add(key)
            # Incremental mean update for Q: Q <- Q + (1/N) * (G - Q)
            returns_count[s_t, a_t] += 1
            alpha = 1.0 / float(returns_count[s_t, a_t])
            Q[s_t, a_t] += alpha * (G - Q[s_t, a_t])
        returns_avg.append(sum(r for _, _, r in episode))

    policy = np.argmax(Q, axis=1).astype(np.int64)
    return policy, Q, returns_avg


def mc_control_off_policy_is(
    env,
    episodes: int = 10_000,
    gamma: float = 0.99,
    *,
    behavior: str = "epsilon",
    behavior_epsilon: float = 0.2,
    weighted: bool = True,
    seed: int | None = 42,
    # Debug options
    debug_behavior: bool = False,
    debug_behavior_episodes: int = 100,
):
    """Off-policy Monte Carlo Control using Importance Sampling.

    Target policy: deterministic greedy w.r.t. Q.
    Behavior policy b:
      - "uniform": chooses actions uniformly at random
      - "epsilon": epsilon-greedy w.r.t. current Q with epsilon=behavior_epsilon

    Weighted IS (default): incremental update with cumulative weights C[s,a].
    Ordinary IS: uses cumulative product W only (more variance).

    Returns:
        policy: np.ndarray [nS]
        Q: np.ndarray [nS, nA]
        returns_avg: list of episode returns for monitoring
    """
    rng = np.random.default_rng(seed)
    nS, nA = env.nS, env.nA
    Q = np.zeros((nS, nA), dtype=np.float64)
    C = np.zeros((nS, nA), dtype=np.float64)  # cumulative weights for Weighted IS

    def greedy_action(s: int) -> int:
        return int(np.argmax(Q[s]))

    def behavior_sample_action_and_prob(s: int) -> tuple[int, float]:
        """Sample an action from behavior policy and return (a, b_prob(a|s))."""
        if behavior.lower() in ("uniform", "u", "uni"):
            a = int(rng.integers(low=0, high=nA))
            return a, 1.0 / float(nA)
        # epsilon-greedy behavior
        eps = float(max(0.0, min(1.0, behavior_epsilon)))
        g = greedy_action(s)
        if rng.random() < eps:
            a = int(rng.integers(low=0, high=nA))
        else:
            a = g
        # b(a|s)
        if a == g:
            b_prob = eps / float(nA) + (1.0 - eps)
        else:
            b_prob = eps / float(nA)
        return a, float(b_prob)

    returns_avg: List[float] = []
    # Debug counters for behavior policy empirical frequencies (first N episodes)
    dbg_total = 0
    dbg_greedy = 0

    for ep_idx in range(episodes):
        # Generate episode with behavior policy b
        episode: List[Tuple[int, int, float, float]] = []  # (s, a, r, b_prob)
        s, _ = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            a, b_prob = behavior_sample_action_and_prob(s)
            if debug_behavior and ep_idx < int(debug_behavior_episodes):
                g = greedy_action(s)
                dbg_total += 1
                if a == g:
                    dbg_greedy += 1
            s2, r, done, truncated, _ = env.step(a)
            episode.append((s, a, float(r), float(b_prob)))
            s = s2

        # Backward update with IS
        G = 0.0
        W = 1.0
        for t in reversed(range(len(episode))):
            s_t, a_t, r_t, bprob_t = episode[t]
            G = gamma * G + r_t

            if weighted:
                C[s_t, a_t] += W
                # Incremental weighted IS update
                Q[s_t, a_t] += (W / C[s_t, a_t]) * (G - Q[s_t, a_t])
            else:
                # Ordinary IS incremental (treat 1/N via implicit averaging is tricky online).
                # Here we use a diminishing step-size 1/(C+1) where C counts total weight mass.
                # This mirrors the weighted IS form but without the C accumulation in the denominator of the true ordinary IS.
                C[s_t, a_t] += 1.0  # count updates
                alpha = 1.0 / C[s_t, a_t]
                Q[s_t, a_t] += alpha * (W * (G - Q[s_t, a_t]))

            # If action deviates from greedy target policy, break
            if a_t != greedy_action(s_t):
                break

            # Update importance weight W <- W * (pi(a|s)/b(a|s)) ; pi=1 for greedy, 0 otherwise
            if bprob_t <= 0.0:
                break  # safety guard
            W = W / bprob_t

        returns_avg.append(sum(r for (_, _, r, _) in episode))

    # Print debug behavior policy stats
    if debug_behavior and dbg_total > 0:
        frac_greedy = dbg_greedy / float(dbg_total)
        msg = (
            f"[mc_off][debug] Behavior stats over first {min(debug_behavior_episodes, episodes)} episodes: "
            f"greedy_fraction={frac_greedy:.4f}, samples={dbg_total}"
        )
        if behavior.lower().startswith("e"):
            eps = float(max(0.0, min(1.0, behavior_epsilon)))
            expected = (1.0 - eps) + eps / float(nA)
            msg += f"; expected_greedy_fraction≈{expected:.4f} for epsilon={eps:.3f}, nA={nA}"
        elif behavior.lower().startswith("u"):
            expected = 1.0 / float(nA)
            msg += f"; expected_greedy_fraction≈{expected:.4f} for uniform, nA={nA}"
        print(msg)

    policy = np.argmax(Q, axis=1).astype(np.int64)
    return policy, Q, returns_avg
