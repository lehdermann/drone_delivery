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
