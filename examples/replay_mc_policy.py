#!/usr/bin/env python3
"""Replay a greedy policy computed by MC Control (On-Policy or Off-Policy IS) with live rendering.

This script:
- Builds the DroneDeliveryEnv with render_mode="human" (matplotlib window).
- Runs MC Control to compute Q and the greedy policy (on-policy epsilon-soft by default, or off-policy with importance sampling when enabled).
- Replays one episode taking greedy actions a = pi[s], rendering at every step.

Usage:
  # On-policy epsilon-soft
  python -m examples.replay_mc_policy --width 7 --height 7 --max-battery 12 \
    --wind-slip 0.05 --gamma 0.99 --epsilon 0.1 --episodes 10000 --seed 0 --sleep 0.2

  # Off-policy with Importance Sampling (weighted), epsilon-greedy behavior
  python -m examples.replay_mc_policy --width 7 --height 7 --max-battery 12 \
    --wind-slip 0.05 --gamma 0.99 --episodes 10000 --seed 0 --sleep 0.2 \
    --off-policy --behavior epsilon --behavior-epsilon 0.2 --off-weighted

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
from algorithms.mc import mc_control_epsilon_soft, mc_control_off_policy_is


def print_policy_summary(env, policy, Q=None):
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
    
    if Q is not None:
        print(f"  Q-values: {Q[s0]}")
        print(f"  Best Q-value: {Q[s0, a0]:.2f}")
    
    print("="*60 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay MC greedy policy with rendering")
    # Env
    p.add_argument("--width", type=int, default=7)
    p.add_argument("--height", type=int, default=7)
    p.add_argument("--max-battery", dest="max_battery", type=int, default=12)
    p.add_argument("--charge-rate", dest="charge_rate", type=int, default=2)
    p.add_argument("--wind-slip", dest="wind_slip", type=float, default=0.05)
    p.add_argument("--obstacles", type=str, default="default", 
                   help="Obstacles: 'none', 'default', or comma-separated x,y pairs like '2,3;3,3;4,3'")
    p.add_argument("--seed", type=int, default=0)
    # MC
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--episodes", type=int, default=20_000)
    p.add_argument("--epsilon", type=float, default=0.1)
    # Off-policy options
    p.add_argument("--off-policy", action="store_true", help="Use Off-Policy MC with Importance Sampling")
    p.add_argument("--behavior", type=str, default="epsilon", help="Behavior policy for off-policy: epsilon or uniform")
    p.add_argument("--behavior-epsilon", dest="behavior_epsilon", type=float, default=0.2, help="Epsilon for behavior when --behavior=epsilon")
    p.add_argument("--off-weighted", dest="off_weighted", action="store_true", help="Use weighted IS (default) for off-policy")
    p.add_argument("--off-ordinary", dest="off_weighted", action="store_false", help="Use ordinary IS for off-policy")
    p.set_defaults(off_weighted=True)
    # Evaluation and debug
    p.add_argument("--eval-episodes", type=int, default=0, help="Greedy evaluation episodes after training (no rendering)")
    p.add_argument("--debug-behavior", action="store_true", help="Print behavior policy empirical stats (off-policy only)")
    p.add_argument(
        "--debug-behavior-episodes",
        dest="debug_behavior_episodes",
        type=int,
        default=100,
        help="Number of episodes to sample behavior stats from (off-policy debug)",
    )
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

    # Compute MC policy (on-policy by default; off-policy when enabled)
    print(f"[replay] Starting MC training for {args.episodes} episodes... (off_policy={args.off_policy})")
    
    # Create interrupt check function
    def check_interrupt():
        return interrupted
    
    if args.off_policy:
        policy, Q, returns_avg = mc_control_off_policy_is(
            env,
            episodes=args.episodes,
            gamma=args.gamma,
            behavior=args.behavior,
            behavior_epsilon=float(args.behavior_epsilon),
            weighted=bool(args.off_weighted),
            seed=args.seed,
            verbose=True,
            verbose_every=max(1, args.episodes // 50),  # Show ~50 progress updates
            interrupt_check=check_interrupt,
            debug_behavior=bool(args.debug_behavior),
            debug_behavior_episodes=int(args.debug_behavior_episodes),
        )
    else:
        policy, Q, returns_avg = mc_control_epsilon_soft(
            env, 
            episodes=args.episodes, 
            gamma=args.gamma, 
            epsilon=args.epsilon, 
            seed=args.seed,
            verbose=True,
            verbose_every=max(1, args.episodes // 50),  # Show ~50 progress updates
            interrupt_check=check_interrupt,
        )
    
    # Calculate statistics over the last 100 episodes
    last_100_returns = returns_avg[-100:]
    avg_return = np.mean(last_100_returns)
    var_return = np.var(last_100_returns) if len(last_100_returns) > 0 else 0.0
    std_return = np.std(last_100_returns) if len(last_100_returns) > 0 else 0.0
    print(f"[replay] MC training complete. Stats over last 100 episodes: Avg={avg_return:.2f}, Var={var_return:.2f}, Std={std_return:.2f}")
    
    # Print policy summary
    print_policy_summary(env, policy, Q)

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
            s, _ = eval_env.reset(seed=(args.seed + 1000 + i) if args.seed is not None else None)
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
        # Note: we reset with a different seed for replay to get a representative sample
        # of the policy's performance, rather than replaying a path seen in training.
        replay_seed = args.seed + 1 if args.seed is not None else None
        s, _ = env.reset(seed=replay_seed)
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