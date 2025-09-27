#!/usr/bin/env python3
"""Replay a greedy policy computed by On-Policy MC Control with live rendering.

This script:
- Builds the DroneDeliveryEnv with render_mode="human" (matplotlib window).
- Runs On-Policy MC Control to compute Q and the greedy policy.
- Replays one episode taking greedy actions a = pi[s], rendering at every step.

Usage:
  python -m examples.replay_mc_policy --width 7 --height 7 --max-battery 12 \
    --wind-slip 0.05 --gamma 0.99 --epsilon 0.1 --episodes 10000 --seed 0 --sleep 0.2

Notes:
- Requires matplotlib (already in requirements.txt).
- Close the matplotlib window to end early if needed.
"""
from __future__ import annotations

import argparse
import time
import csv
import os
import numpy as np

from envs.drone_delivery import DroneDeliveryEnv
from algorithms.mc import mc_control_epsilon_soft


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay MC greedy policy with rendering")
    # Env
    p.add_argument("--width", type=int, default=7)
    p.add_argument("--height", type=int, default=7)
    p.add_argument("--max-battery", dest="max_battery", type=int, default=12)
    p.add_argument("--charge-rate", dest="charge_rate", type=int, default=2)
    p.add_argument("--wind-slip", dest="wind_slip", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    # MC
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--episodes", type=int, default=20_000)
    p.add_argument("--epsilon", type=float, default=0.1)
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

    # Compute MC policy
    print(f"[replay] Starting MC training for {args.episodes} episodes...")
    policy, Q, returns_avg = mc_control_epsilon_soft(
        env, episodes=args.episodes, gamma=args.gamma, epsilon=args.epsilon, seed=args.seed
    )
    
    # Calculate statistics over the last 100 episodes
    last_100_returns = returns_avg[-100:]
    avg_return = np.mean(last_100_returns)
    var_return = np.var(last_100_returns) if len(last_100_returns) > 0 else 0.0
    std_return = np.std(last_100_returns) if len(last_100_returns) > 0 else 0.0
    print(f"[replay] MC training complete. Stats over last 100 episodes: Avg={avg_return:.2f}, Var={var_return:.2f}, Std={std_return:.2f}")

    # Replay one episode greedily
    # Note: we reset with a different seed for replay to get a representative sample
    # of the policy's performance, rather than replaying a path seen in training.
    replay_seed = args.seed + 1 if args.seed is not None else None
    s, _ = env.reset(seed=replay_seed)
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

    # Save results to CSV
    results_filepath = "mc_experiments.csv"
    run_params = vars(args)
    # We don't need to log the visualization sleep time
    run_params.pop("sleep", None)

    results_data = {
        "avg_return_last_100": avg_return,
        "var_return_last_100": var_return,
        "std_return_last_100": std_return,
        "replay_steps": steps,
        "replay_return": total_return,
        "replay_delivered": delivered,
    }

    # Combine params and results into a single dictionary for the CSV row
    log_data = {**run_params, **results_data}

    file_exists = os.path.isfile(results_filepath)
    with open(results_filepath, mode='a', newline='') as csvfile:
        fieldnames = list(log_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)
    print(f"[replay] Results appended to {results_filepath}")

    env.close()


if __name__ == "__main__":
    main()