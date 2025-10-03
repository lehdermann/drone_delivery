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
import csv
import os
import signal
import numpy as np

import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.drone_delivery import DroneDeliveryEnv
from algorithms.dp import value_iteration


def print_policy_summary(env, policy, V=None):
    """Print a summary of the learned policy."""
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY', 'CHARGE']
    action_counts = {name: 0 for name in action_names}
    
    # Count action distribution
    for s in range(env.nS):
        a = int(policy[s])
        action_counts[action_names[a]] += 1
    
    print("\n" + "="*60)
    print("LEARNED POLICY SUMMARY")
    print("="*60)
    print(f"Total states: {env.nS}")
    print("\nAction distribution across all states:")
    for action, count in action_counts.items():
        pct = 100 * count / env.nS
        print(f"  {action:8s}: {count:4d} states ({pct:5.1f}%)")
    
    # Show policy for initial state
    s0 = env._encode(0, 0, env.max_battery, 1)  # Start: (0,0), full battery, has package
    a0 = int(policy[s0])
    print(f"\nInitial state policy:")
    print(f"  State: pos=(0,0), battery={env.max_battery}, has_package=True")
    print(f"  Action: {action_names[a0]}")
    
    if V is not None:
        print(f"  State value V(s): {V[s0]:.2f}")
    
    print("="*60 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay VI greedy policy with rendering")
    # Env
    p.add_argument("--width", type=int, default=7)
    p.add_argument("--height", type=int, default=7)
    p.add_argument("--max-battery", dest="max_battery", type=int, default=5)
    p.add_argument("--charge-rate", dest="charge_rate", type=int, default=2)
    p.add_argument("--wind-slip", dest="wind_slip", type=float, default=0.1)
    p.add_argument("--obstacles", type=str, default="default", 
                   help="Obstacles: 'none', 'default', or comma-separated x,y pairs like '2,3;3,3;4,3'")
    p.add_argument("--seed", type=int, default=0)
    # VI
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--theta", type=float, default=1e-4)
    # Evaluation (post-policy, no rendering)
    p.add_argument("--eval-episodes", type=int, default=0, help="Greedy evaluation episodes after VI (no rendering)")
    # Replay
    p.add_argument("--sleep", type=float, default=0.1, help="Seconds to sleep between steps for visualization")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Setup signal handler for graceful shutdown
    interrupted = False
    
    def signal_handler(sig, frame):
        nonlocal interrupted
        print("\n[replay] Interrupted by user (Ctrl+C). Cleaning up...")
        interrupted = True
    
    signal.signal(signal.SIGINT, signal_handler)

    # Parse obstacles
    obstacles = None
    if args.obstacles == "default":
        # Default obstacle configuration for 7x7 grid
        obstacles = [
            (3, 1), (3, 2), (3, 3),  # vertical wall
            (1, 4), (2, 4),           # horizontal wall
            (5, 3), (5, 4),           # another wall
        ]
    elif args.obstacles != "none":
        # Parse custom obstacles from string like "2,3;3,3;4,3"
        try:
            obstacles = []
            for pair in args.obstacles.split(";"):
                x, y = map(int, pair.split(","))
                obstacles.append((x, y))
        except Exception:
            print(f"[warning] Could not parse obstacles '{args.obstacles}', using none")
            obstacles = None

    # Add charging stations at strategic locations (including pickup)
    charging_stations = [
        (0, 0),  # At pickup/store
        (3, 6),  # Mid-bottom
        (6, 3),  # Mid-right
    ]

    # Build env with live rendering
    env = DroneDeliveryEnv(
        width=args.width,
        height=args.height,
        max_battery=args.max_battery,
        charge_rate=args.charge_rate,
        wind_slip=args.wind_slip,
        obstacles=obstacles,
        charging_stations=charging_stations,
        render_mode="human",
        seed=args.seed,
    )

    # Compute VI policy
    V, policy, stats = value_iteration(env, gamma=args.gamma, theta=args.theta)
    vi_iterations = stats.get('iterations', 0)
    vi_last_delta = stats.get('deltas', [0])[-1] if stats.get('deltas') else 0
    print(f"[replay] VI converged in {vi_iterations} sweeps; last delta={vi_last_delta}")
    
    # Print policy summary
    print_policy_summary(env, policy, V)

    # Optional greedy evaluation without rendering
    eval_metrics = {}
    if int(args.eval_episodes) > 0:
        eval_env = DroneDeliveryEnv(
            width=args.width,
            height=args.height,
            max_battery=args.max_battery,
            charge_rate=args.charge_rate,
            wind_slip=args.wind_slip,
            render_mode=None,
            seed=args.seed,
        )
        ep_returns = []
        ep_steps = []
        ep_collisions = []
        ep_success = []
        for i in range(int(args.eval_episodes)):
            s, _ = eval_env.reset(seed=(args.seed + 2000 + i) if args.seed is not None else None)
            done = False
            truncated = False
            total_r = 0.0
            steps = 0
            collisions = 0
            delivered = False
            while not (done or truncated):
                a = int(policy[s])
                s, r, done, truncated, info = eval_env.step(a)
                total_r += float(r)
                steps += 1
                if bool(info.get("collision", False)):
                    collisions += 1
                if bool(info.get("delivered", False)):
                    delivered = True
            ep_returns.append(total_r)
            ep_steps.append(steps)
            ep_collisions.append(collisions)
            ep_success.append(1 if delivered else 0)
        eval_metrics = {
            "eval_episodes": int(args.eval_episodes),
            "eval_success_rate": float(np.mean(ep_success)) if ep_success else 0.0,
            "eval_avg_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "eval_avg_steps": float(np.mean(ep_steps)) if ep_steps else 0.0,
            "eval_avg_collisions": float(np.mean(ep_collisions)) if ep_collisions else 0.0,
        }
        print(
            "[replay][eval] N={} success_rate={:.3f} avg_return={:.2f} avg_steps={:.1f} avg_collisions={:.2f}".format(
                eval_metrics["eval_episodes"],
                eval_metrics["eval_success_rate"],
                eval_metrics["eval_avg_return"],
                eval_metrics["eval_avg_steps"],
                eval_metrics["eval_avg_collisions"],
            )
        )
        eval_env.close()

    # Replay one episode greedily
    if not interrupted:
        s, _ = env.reset(seed=args.seed)
        done = False
        truncated = False
        total_return = 0.0
        steps = 0
        
        # Track trajectory
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY', 'CHARGE']
        trajectory = []
        
        print("\n" + "="*60)
        print("TRAJECTORY USING OPTIMAL POLICY")
        print("="*60)
        
        while not (done or truncated) and not interrupted:
            # Decode current state
            x, y, battery, has_pkg = env._decode(s)
            a = int(policy[s])
            
            # Store trajectory info
            trajectory.append({
                'step': steps,
                'pos': (x, y),
                'battery': battery,
                'has_package': has_pkg,
                'action': action_names[a]
            })
            
            # Print step info (only first 10 and last 10 steps for brevity)
            if steps < 10 or steps >= 190:
                pkg_status = "📦" if has_pkg else "✓"
                print(f"Step {steps:3d}: pos=({x},{y}) bat={battery:2d} {pkg_status} → {action_names[a]:6s}", end="")
            elif steps == 10:
                print("  ... (trajectory continues) ...")
            
            s, r, done, truncated, info = env.step(a)
            total_return += float(r)
            steps += 1
            
            # Print reward for last step
            if steps <= 10 or steps >= 190:
                print(f" → reward={r:6.1f}")
            
            time.sleep(max(0.0, args.sleep))

        delivered = bool(info.get("delivered", False))
        
        # Final state
        x, y, battery, has_pkg = env._decode(s)
        print(f"Step {steps:3d}: pos=({x},{y}) bat={battery:2d} {'✓' if delivered else '✗'} → TERMINAL")
        print("="*60)
        print(f"RESULT: {'SUCCESS ✓' if delivered else 'FAILED ✗'}")
        print(f"Total steps: {steps}, Total return: {total_return:.2f}")
        print("="*60 + "\n")
    else:
        print("[replay] Replay skipped due to interruption.")
        steps = 0
        total_return = 0.0
        delivered = False
        trajectory = []

    # Save results to CSV
    results_filepath = "vi_experiments.csv"
    run_params = vars(args)
    # We don't need to log the visualization sleep time
    run_params.pop("sleep", None)

    results_data = {
        "vi_iterations": vi_iterations,
        "vi_last_delta": vi_last_delta,
        "replay_steps": steps,
        "replay_return": total_return,
        "replay_delivered": delivered,
    }

    # Combine params and results into a single dictionary for the CSV row
    log_data = {**run_params, **results_data, **eval_metrics}

    file_exists = os.path.isfile(results_filepath)
    with open(results_filepath, mode='a', newline='') as csvfile:
        fieldnames = list(log_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)
    print(f"[replay] Results appended to {results_filepath}")

    try:
        env.close()
    except Exception as e:
        print(f"[replay] Warning: Error closing environment: {e}")
    
    if interrupted:
        print("[replay] Exiting due to interruption.")
        exit(130)  # Standard exit code for SIGINT


if __name__ == "__main__":
    main()
