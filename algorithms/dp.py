from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np


def value_iteration(env, gamma: float = 0.99, theta: float = 1e-4, max_iterations: int = 10_000, *, store_v_history: bool = False):
    """Value Iteration for finite tabular MDPs with access to env.enumerate_transitions.

    Returns:
        V: np.ndarray of shape [nS]
        policy: np.ndarray of shape [nS] (greedy w.r.t. V)
        stats: dict with keys
            - 'iterations': number of sweeps
            - 'deltas': list of max |V_new - V| per sweep
            - 'Vs' (optional): list of V snapshots per sweep, only if store_v_history=True
    """
    # Guards for tabular + model-based requirement
    if not hasattr(env, "nS") or not hasattr(env, "nA"):
        raise NotImplementedError(
            "value_iteration requires a tabular environment exposing env.nS and env.nA. "
            "For continuous spaces, provide a discretized wrapper that defines these."
        )
    if not hasattr(env, "enumerate_transitions"):
        raise NotImplementedError(
            "value_iteration requires env.enumerate_transitions(s, a) to enumerate the model."
        )

    nS, nA = env.nS, env.nA
    V = np.zeros(nS, dtype=np.float64)
    deltas: List[float] = []
    v_history: List[np.ndarray] = []

    for it in range(max_iterations):
        delta = 0.0
        V_new = V.copy()
        for s in range(nS):
            q_values = []
            for a in range(nA):
                q = 0.0
                for p, s2, r, done in env.enumerate_transitions(s, a):
                    q += p * (r + (0.0 if done else gamma * V[s2]))
                q_values.append(q)
            V_new[s] = np.max(q_values)
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        deltas.append(delta)
        if store_v_history:
            v_history.append(V.copy())
        if delta < theta:
            break

    # Derive greedy policy
    policy = np.zeros(nS, dtype=np.int64)
    for s in range(nS):
        q_values = []
        for a in range(nA):
            q = 0.0
            for p, s2, r, done in env.enumerate_transitions(s, a):
                q += p * (r + (0.0 if done else gamma * V[s2]))
            q_values.append(q)
        policy[s] = int(np.argmax(q_values))

    stats = {"iterations": len(deltas), "deltas": deltas}
    if store_v_history:
        stats["Vs"] = v_history
    return V, policy, stats


def policy_evaluation(env, policy: np.ndarray, gamma: float = 0.99, theta: float = 1e-4, max_iterations: int = 10_000):
    nS = env.nS
    V = np.zeros(nS, dtype=np.float64)
    for _ in range(max_iterations):
        delta = 0.0
        for s in range(nS):
            a = int(policy[s])
            v = 0.0
            for p, s2, r, done in env.enumerate_transitions(s, a):
                v += p * (r + (0.0 if done else gamma * V[s2]))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V


def policy_iteration(env, gamma: float = 0.99, theta: float = 1e-4, max_iterations: int = 10_000, init_policy: Optional[np.ndarray] = None):
    """Policy Iteration (greedy improvement) with iterative policy evaluation for tabular MDPs.

    Returns:
        V: value function
        policy: improved policy
        stats: dict with counts of eval/improve iterations

    Args:
        env: environment exposing enumerate_transitions and nS/nA
        gamma: discount factor
        theta: evaluation convergence threshold
        max_iterations: safety cap on total evaluation+improvement loops
        init_policy: optional initial policy (shape [nS], int in [0, nA-1]). If None, starts with zeros.
    """
    # Guards for tabular + model-based requirement
    if not hasattr(env, "nS") or not hasattr(env, "nA"):
        raise NotImplementedError(
            "policy_iteration requires a tabular environment exposing env.nS and env.nA. "
            "For continuous spaces, provide a discretized wrapper that defines these."
        )
    if not hasattr(env, "enumerate_transitions"):
        raise NotImplementedError(
            "policy_iteration requires env.enumerate_transitions(s, a) to enumerate the model."
        )

    nS, nA = env.nS, env.nA
    if init_policy is None:
        policy = np.zeros(nS, dtype=np.int64)
    else:
        policy = np.array(init_policy, dtype=np.int64).copy()
        assert policy.shape == (nS,), f"init_policy must have shape ({nS},)"
        assert np.all((policy >= 0) & (policy < nA)), "init_policy contains invalid action indices"
    stable = False
    eval_iters = 0
    improve_iters = 0
    while not stable and (eval_iters + improve_iters) < max_iterations:
        # Policy Evaluation
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        eval_iters += 1

        # Policy Improvement
        stable = True
        for s in range(nS):
            old_a = int(policy[s])
            q_values = []
            for a in range(nA):
                q = 0.0
                for p, s2, r, done in env.enumerate_transitions(s, a):
                    q += p * (r + (0.0 if done else gamma * V[s2]))
                q_values.append(q)
            best_a = int(np.argmax(q_values))
            policy[s] = best_a
            if best_a != old_a:
                stable = False
        improve_iters += 1

    stats = {"policy_eval_iters": eval_iters, "policy_improve_iters": improve_iters}
    return V, policy, stats
