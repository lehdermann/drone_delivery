#!/usr/bin/env python3
"""Replay a greedy policy computed by Value Iteration with live rendering.

This script:
- Builds the DroneDeliveryEnv with render_mode="human" (matplotlib window).
- Runs Value Iteration to compute V and the greedy policy.
- Replays one episode taking greedy actions a = pi[s], rendering at every step.

Usage:
  python -m examples.replay_vi_policy --width 7 --height 7 --max-battery 12 --charge-rate 2 \
    --wind-slip 0.05 --gamma 0.99 --theta 1e-4 --seed 0 --sleep 0.2

Notes:
- Requires matplotlib (already in requirements.txt).
- Close the matplotlib window to end early if needed.
"""
from __future__ import annotations

import argparse
import time

from envs.drone_delivery import DroneDeliveryEnv
from algorithms.dp import value_iteration


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay VI greedy policy with rendering")
    # Env
    p.add_argument("--width", type=int, default=7)
    p.add_argument("--height", type=int, default=7)
    p.add_argument("--max-battery", dest="max_battery", type=int, default=5)
    p.add_argument("--charge-rate", dest="charge_rate", type=int, default=2)
    p.add_argument("--wind-slip", dest="wind_slip", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    # VI
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--theta", type=float, default=1e-4)
    # Replay
    p.add_argument("--sleep", type=float, default=0.1, help="Seconds to sleep between steps for visualization")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build env with live rendering
    env = DroneDeliveryEnv(
        width=args.width,
        height=args.height,
        max_battery=args.max_battery,
        charge_rate=args.charge_rate,
        wind_slip=args.wind_slip,
        render_mode="human",
        seed=args.seed,
    )

    # Compute VI policy
    V, policy, stats = value_iteration(env, gamma=args.gamma, theta=args.theta)
    print(f"[replay] VI converged in {stats.get('iterations', 0)} sweeps; last delta={stats.get('deltas', [0])[-1] if stats.get('deltas') else 0}")

    # Replay one episode greedily
    s, _ = env.reset(seed=args.seed)
    done = False
    truncated = False
    total_return = 0.0
    steps = 0
    while not (done or truncated):
        a = int(policy[s])
        s, r, done, truncated, info = env.step(a)
        total_return += float(r)
        steps += 1
        time.sleep(max(0.0, args.sleep))

    delivered = bool(info.get("delivered", False))
    print(f"[replay] Done. steps={steps}, return={total_return:.2f}, delivered={delivered}")

    env.close()


if __name__ == "__main__":
    main()
