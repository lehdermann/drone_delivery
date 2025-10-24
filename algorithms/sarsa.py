from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, List

import numpy as np

try:
    from tqdm import tqdm  # type: ignore
    HAS_TQDM = True
except Exception:  # pragma: no cover - tqdm is optional
    HAS_TQDM = False
    tqdm = None  # type: ignore


@dataclass
class SarsaLambdaResult:
    weights: np.ndarray
    returns: List[float]

    def q_value(self, phi_sa: np.ndarray) -> float:
        return float(self.weights @ phi_sa)


def default_feature_fn(env) -> Callable[[int, int], np.ndarray]:
    """Create a lightweight feature extractor for linear Q(s, a).

    Features (dimension = 1 bias + 4 state + 6 action = 11):
    - 1 bias
    - x_norm, y_norm, battery_norm, has_package
    - one-hot over the action (6)
    """
    nA = env.nA
    width = float(env.width)
    height = float(env.height)
    max_battery = float(env.max_battery)

    def phi(s: int, a: int) -> np.ndarray:
        x, y, b, p = env._decode(s)
        # Simple and stable normalizations
        x_n = x / max(1.0, width - 1.0)
        y_n = y / max(1.0, height - 1.0)
        b_n = b / max(1.0, max_battery)
        p_f = float(p)
        feat = np.zeros(1 + 4 + nA, dtype=np.float64)
        # bias
        feat[0] = 1.0
        # state
        feat[1:5] = np.array([x_n, y_n, b_n, p_f], dtype=np.float64)
        # one-hot action
        feat[5 + int(a)] = 1.0
        return feat

    return phi


def one_hot_feature_fn(env) -> Callable[[int, int], np.ndarray]:
    """Tabular one-hot features for exact representation of Q(s, a).

    Dimension = env.nS * env.nA. Feature for (s, a) is 1 at index (s * nA + a), 0 otherwise.
    """
    nS, nA = env.nS, env.nA

    def phi(s: int, a: int) -> np.ndarray:
        idx = s * nA + int(a)
        feat = np.zeros(nS * nA, dtype=np.float64)
        feat[idx] = 1.0
        return feat

    return phi


def engineered_feature_fn(env) -> Callable[[int, int], np.ndarray]:
    """Engineered features capturing task structure with small dimension.

    Features:
    - bias
    - position normalized: x_n, y_n
    - battery normalized: b_n
    - has_package flag
    - distances (Manhattan) normalized to pickup and dropoff
    - flags: at_charging, at_pickup, at_dropoff
    - simple interactions: b_n * x_n, b_n * y_n
    - one-hot over action (6)
    """
    nA = env.nA
    width = float(env.width)
    height = float(env.height)
    max_battery = float(env.max_battery)
    pickup = env.pickup
    drop = env.dropoff

    def manhattan(ax: int, ay: int, bx: int, by: int) -> float:
        return float(abs(ax - bx) + abs(ay - by))

    max_dist = float((env.width - 1) + (env.height - 1)) if (env.width > 0 and env.height > 0) else 1.0

    def phi(s: int, a: int) -> np.ndarray:
        x, y, b, p = env._decode(s)
        x_n = x / max(1.0, width - 1.0)
        y_n = y / max(1.0, height - 1.0)
        b_n = b / max(1.0, max_battery)
        p_f = float(p)
        d_pick = manhattan(x, y, pickup.x, pickup.y) / max(1.0, max_dist)
        d_drop = manhattan(x, y, drop.x, drop.y) / max(1.0, max_dist)
        at_chg = 1.0 if type(env).Cell(x, y) in env.charging_stations else (1.0 if hasattr(env, "charging_stations") and any(c.x == x and c.y == y for c in env.charging_stations) else 0.0)
        at_pick = 1.0 if (x == pickup.x and y == pickup.y) else 0.0
        at_drop = 1.0 if (x == drop.x and y == drop.y) else 0.0
        bx = b_n * x_n
        by = b_n * y_n

        core = np.array([
            1.0,          # bias
            x_n, y_n,     # position
            b_n, p_f,     # battery, has_package
            d_pick, d_drop,
            at_chg, at_pick, at_drop,
            bx, by,
        ], dtype=np.float64)

        feat = np.zeros(core.shape[0] + nA, dtype=np.float64)
        feat[: core.shape[0]] = core
        feat[core.shape[0] + int(a)] = 1.0
        return feat

    return phi


def sarsa_lambda_linear(
    env,
    episodes: int = 10_000,
    gamma: float = 0.99,
    lam: float = 0.9,
    alpha: float = 0.05,
    epsilon: float = 0.1,
    *,
    feature_fn: Callable[[int, int], np.ndarray] | None = None,
    seed: int | None = 42,
    verbose: bool = False,
    verbose_every: int = 100,
    interrupt_check: callable | None = None,
    # Optional schedules (per-episode): if provided, override epsilon/alpha each episode
    epsilon_schedule: Callable[[int], float] | None = None,
    alpha_schedule: Callable[[int], float] | None = None,
    # Eligibility trace variant
    replacing_traces: bool = False,
    # Optimistic initialization: initial Q estimate to encourage exploration
    optimistic_init: float | None = None,
) -> SarsaLambdaResult:
    """On-policy SARSA(λ) with linear function approximation Q(s, a) = w^T φ(s, a).

    - Accumulating eligibility traces.
    - Epsilon-greedy policy over the approximate Q.
    - Constant step-size alpha (can be scheduled if desired).

    Returns learned weights and per-episode returns history.
    """
    rng = np.random.default_rng(seed)
    nA = env.nA

    # Features
    if feature_fn is None:
        feature_fn = default_feature_fn(env)
    s_ref = env._encode(0, 0, env.max_battery, 1)
    phi_ref = feature_fn(s_ref, 0)
    d = phi_ref.shape[0]

    # Initialize weights (optionally optimistic)
    w = np.zeros(d, dtype=np.float64)
    if optimistic_init is not None:
        try:
            # Detect tabular one-hot features
            is_one_hot = (d == env.nS * env.nA) and (np.isclose(phi_ref.sum(), 1.0)) and (np.isclose(phi_ref.max(), 1.0))
        except Exception:
            is_one_hot = False

        init_val = float(optimistic_init)
        if is_one_hot:
            # For tabular, set each (s,a) weight to optimistic Q
            w.fill(init_val)
        elif phi_ref.shape[0] > 0 and np.isclose(phi_ref[0], 1.0):
            # If there is a bias term, set only bias to the optimistic Q so Q≈init across all (s,a)
            w[0] = init_val
        else:
            # Fallback: scale uniformly by L1 norm of reference features
            denom = float(np.sum(np.abs(phi_ref)))
            if denom <= 1e-12:
                denom = 1.0
            w[:] = init_val / denom
    returns: List[float] = []

    def epsilon_greedy_action(s: int) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(low=0, high=nA))
        # Choose action with highest approximate Q
        q_vals = [float(w @ feature_fn(s, a)) for a in range(nA)]
        return int(np.argmax(q_vals))

    # Progress bar
    if verbose and HAS_TQDM:
        iterator = tqdm(range(episodes), desc="SARSA(λ)", unit="ep")
    else:
        iterator = range(episodes)

    for ep in iterator:
        # Check for interruption
        if interrupt_check and interrupt_check():
            print("\n[SARSA(λ)] Training interrupted by user.")
            break
        # Eligibility trace
        z = np.zeros_like(w)
        s, _ = env.reset()
        # Episode-wise schedules
        eps_ep = float(epsilon_schedule(ep)) if epsilon_schedule else float(epsilon)
        alpha_ep = float(alpha_schedule(ep)) if alpha_schedule else float(alpha)

        def epsilon_greedy_action_ep(s_: int) -> int:
            if rng.random() < eps_ep:
                return int(rng.integers(low=0, high=env.nA))
            q_vals = [float(w @ feature_fn(s_, a_)) for a_ in range(env.nA)]
            return int(np.argmax(q_vals))

        a = epsilon_greedy_action_ep(s)
        done = False
        truncated = False
        G = 0.0

        while not (done or truncated):
            # Mid-episode interruption check
            if interrupt_check and interrupt_check():
                print("\n[SARSA(λ)] Training interrupted by user.")
                done = True
                truncated = True
                break
            s2, r, done, truncated, _ = env.step(a)
            G += float(r)

            # TD target and error
            phi_sa = feature_fn(s, a)
            if done or truncated:
                td_target = float(r)
            else:
                a2 = epsilon_greedy_action_ep(s2)
                phi_s2a2 = feature_fn(s2, a2)
                td_target = float(r) + gamma * float(w @ phi_s2a2)

            delta = td_target - float(w @ phi_sa)

            # Update traces
            if replacing_traces:
                # Replacing: decay then replace with max
                z = gamma * lam * z
                z = np.maximum(z, phi_sa)
            else:
                # Accumulating traces
                z = gamma * lam * z + phi_sa

            # Linear Q gradient: ∇_w Q = φ
            w += alpha_ep * delta * z

            # Advance
            s, a = s2, (epsilon_greedy_action_ep(s2) if not (done or truncated) else a)

        returns.append(G)

        if verbose and HAS_TQDM:
            avg_last = np.mean(returns[-100:]) if returns else 0.0
            iterator.set_postfix({"avg_return": f"{avg_last:.2f}", "last": f"{G:.2f}"})
        elif verbose and (ep + 1) % max(1, verbose_every) == 0:
            avg_last = np.mean(returns[-verbose_every:]) if returns else 0.0
            print(f"[sarsa] ep {ep+1}/{episodes} avg_last={avg_last:.2f} last={G:.2f}")

    if verbose and HAS_TQDM:
        iterator.close()  # type: ignore

    return SarsaLambdaResult(weights=w, returns=returns)
