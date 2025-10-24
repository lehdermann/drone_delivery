#!/usr/bin/env python3
"""Replay a greedy policy computed by SARSA(λ) with linear function approximation.

This script:
- Builds the DroneDeliveryEnv (render_mode optional).
- Trains SARSA(λ) with linear approximation of Q(s,a) = w^T φ(s,a).
- Optionally evaluates greedily without rendering.
- Replays one greedy episode with optional rendering.
- Appends results to sarsa_experiments.csv.

Usage (headless fast):
  python -m examples.replay_sarsa_policy --no-render --episodes 2000 --epsilon 0.1 --alpha 0.05 --lam 0.9
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
from algorithms.sarsa import (
    sarsa_lambda_linear,
    default_feature_fn,
    one_hot_feature_fn,
    engineered_feature_fn,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay SARSA(λ) greedy policy with rendering")
    # Env
    p.add_argument("--width", type=int, default=7)
    p.add_argument("--height", type=int, default=7)
    p.add_argument("--max-battery", dest="max_battery", type=int, default=12)
    p.add_argument("--charge-rate", dest="charge_rate", type=int, default=2)
    p.add_argument("--wind-slip", dest="wind_slip", type=float, default=0.05)
    p.add_argument("--obstacles", type=str, default="default",
                   help="Obstacles: 'none', 'default', or comma-separated x,y pairs like '2,3;3,3;4,3'")
    p.add_argument("--seed", type=int, default=0)
    # SARSA(λ)
    p.add_argument("--episodes", type=int, default=10_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument(
        "--features",
        type=str,
        default="basic",
        choices=["basic", "one_hot", "engineered"],
        help="Feature set for linear FA: basic (default), one_hot (tabular), engineered",
    )
    # Schedules (linear decay)
    p.add_argument(
        "--epsilon-final",
        dest="epsilon_final",
        type=float,
        default=None,
        help="If set, linearly decay epsilon from --epsilon to this value across episodes",
    )
    p.add_argument(
        "--alpha-final",
        dest="alpha_final",
        type=float,
        default=None,
        help="If set, linearly decay alpha from --alpha to this value across episodes",
    )
    # Traces variant
    p.add_argument(
        "--replacing-traces",
        dest="replacing_traces",
        action="store_true",
        help="Use replacing traces instead of accumulating",
    )
    # Optimistic initialization
    p.add_argument(
        "--optimistic-init",
        dest="optimistic_init",
        type=float,
        default=None,
        help="If set, initialize Q optimistically to this value to encourage exploration",
    )
    # Evaluation and replay
    p.add_argument("--eval-episodes", type=int, default=0,
                   help="Greedy evaluation episodes after training (no rendering)")
    p.add_argument("--sleep", type=float, default=0.1,
                   help="Seconds to sleep between steps for visualization")
    p.add_argument("--no-render", dest="no_render", action="store_true",
                   help="Disable rendering (headless mode for faster execution)")
    return p.parse_args()


def greedy_action_from_weights(env: DroneDeliveryEnv, w: np.ndarray, phi) -> callable:
    def act(s: int) -> int:
        q_vals = [float(w @ phi(s, a)) for a in range(env.nA)]
        return int(np.argmax(q_vals))
    return act


def print_policy_summary(env, act_greedy, w=None, phi=None):
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY', 'CHARGE']
    action_counts = {name: 0 for name in action_names}
    for s in range(env.nS):
        a = int(act_greedy(s))
        action_counts[action_names[a]] += 1

    print("\n" + "="*60)
    print("LEARNED POLICY SUMMARY (SARSA λ linear)")
    print("="*60)
    print(f"Total states: {env.nS}")
    print("\nAction distribution across all states:")
    for action, count in action_counts.items():
        pct = 100 * count / env.nS
        print(f"  {action:8s}: {count:4d} states ({pct:5.1f}%)")

    s0 = env._encode(0, 0, env.max_battery, 1)
    a0 = int(act_greedy(s0))
    print(f"\nInitial state policy:")
    print(f"  State: pos=(0,0), battery={env.max_battery}, has_package=True")
    print(f"  Action: {action_names[a0]}")
    if w is not None and phi is not None:
        q = [float(w @ phi(s0, a)) for a in range(env.nA)]
        print(f"  Q-approx: {np.array(q)} (best={q[a0]:.2f})")
    print("="*60 + "\n")


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
        obstacles = [
            (3, 1), (3, 2), (3, 3),
            (1, 4), (2, 4),
            (5, 3), (5, 4),
        ]
    elif args.obstacles != "none":
        try:
            obstacles = []
            for pair in args.obstacles.split(";"):
                x, y = map(int, pair.split(","))
                obstacles.append((x, y))
        except Exception:
            print(f"[warning] Could not parse obstacles '{args.obstacles}', using none")
            obstacles = None

    charging_stations = [
        (0, 0),
        (3, 6),
        (6, 3),
    ]

    render_mode = None if args.no_render else "human"
    env = DroneDeliveryEnv(
        width=args.width,
        height=args.height,
        max_battery=args.max_battery,
        charge_rate=args.charge_rate,
        wind_slip=args.wind_slip,
        obstacles=obstacles,
        charging_stations=charging_stations,
        render_mode=render_mode,
        seed=args.seed,
    )

    # Train SARSA(λ)
    # Select feature function
    if args.features == "one_hot":
        phi = one_hot_feature_fn(env)
    elif args.features == "engineered":
        phi = engineered_feature_fn(env)
    else:
        phi = default_feature_fn(env)
    print(f"[replay] Starting SARSA(λ) training for {args.episodes} episodes...")
    # Create interrupt check function
    def check_interrupt():
        return interrupted
    # Optional linear schedules
    eps0 = float(args.epsilon)
    eps1 = float(args.epsilon_final) if args.epsilon_final is not None else None
    a0 = float(args.alpha)
    a1 = float(args.alpha_final) if args.alpha_final is not None else None

    def make_linear_schedule(v0: float, v1: float):
        def sched(t: int) -> float:
            if args.episodes <= 1:
                return v1
            frac = min(1.0, max(0.0, t / (args.episodes - 1)))
            return (1.0 - frac) * v0 + frac * v1
        return sched

    epsilon_schedule = make_linear_schedule(eps0, eps1) if eps1 is not None else None
    alpha_schedule = make_linear_schedule(a0, a1) if a1 is not None else None

    result = sarsa_lambda_linear(
        env,
        episodes=args.episodes,
        gamma=args.gamma,
        lam=args.lam,
        alpha=args.alpha,
        epsilon=args.epsilon,
        feature_fn=phi,
        seed=args.seed,
        verbose=True,
        verbose_every=max(1, args.episodes // 50),
        interrupt_check=check_interrupt,
        epsilon_schedule=epsilon_schedule,
        alpha_schedule=alpha_schedule,
        replacing_traces=bool(args.replacing_traces),
        optimistic_init=args.optimistic_init,
    )
    w = result.weights

    # Stats over last 100 ep
    last_100 = result.returns[-100:]
    avg_return = float(np.mean(last_100)) if last_100 else (float(np.mean(result.returns)) if result.returns else 0.0)
    var_return = float(np.var(last_100)) if last_100 else 0.0
    std_return = float(np.std(last_100)) if last_100 else 0.0
    print(f"[replay] SARSA(λ) training complete. Stats over last 100 episodes: Avg={avg_return:.2f}, Var={var_return:.2f}, Std={std_return:.2f}")

    # Greedy policy from weights
    act_greedy = greedy_action_from_weights(env, w, phi)
    print_policy_summary(env, act_greedy, w=w, phi=phi)

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
            s, _ = eval_env.reset(seed=(args.seed + 3000 + i) if args.seed is not None else None)
            done = False
            truncated = False
            total_r = 0.0
            steps = 0
            collisions = 0
            delivered = False
            while not (done or truncated):
                a = int(act_greedy(s))
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
        s, _ = env.reset(seed=args.seed + 1 if args.seed is not None else None)
        done = False
        truncated = False
        total_return = 0.0
        steps = 0

        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY', 'CHARGE']
        if not args.no_render:
            print("\n" + "="*60)
            print("TRAJECTORY USING GREEDY POLICY (SARSA λ)")
            print("="*60)

        while not (done or truncated) and not interrupted:
            x, y, battery, has_pkg = env._decode(s)
            a = int(act_greedy(s))
            if not args.no_render:
                if steps < 10 or steps >= 190:
                    pkg_status = "📦" if has_pkg else "✓"
                    print(f"Step {steps:3d}: pos=({x},{y}) bat={battery:2d} {pkg_status} → {action_names[a]:6s}", end="")
                elif steps == 10:
                    print("  ... (trajectory continues) ...")

            s, r, done, truncated, info = env.step(a)
            total_return += float(r)
            steps += 1
            if not args.no_render and (steps <= 10 or steps >= 190):
                print(f" → reward={r:6.1f}")
            if not args.no_render:
                time.sleep(max(0.0, args.sleep))

        delivered = bool(info.get("delivered", False))
        if not args.no_render:
            x, y, battery, has_pkg = env._decode(s)
            print(f"Step {steps:3d}: pos=({x},{y}) bat={battery:2d} {'✓' if delivered else '✗'} → TERMINAL")
            print("="*60)
            print(f"RESULT: {'SUCCESS ✓' if delivered else 'FAILED ✗'}")
            print(f"Total steps: {steps}, Total return: {total_return:.2f}")
            print("="*60 + "\n")
        else:
            print(f"[replay] {'SUCCESS' if delivered else 'FAILED'}: steps={steps}, return={total_return:.2f}")
    else:
        print("[replay] Replay skipped due to interruption.")
        steps = 0
        total_return = 0.0
        delivered = False

    # Append results to CSV
    results_filepath = "sarsa_experiments.csv"
    run_params = vars(args)
    run_params.pop("sleep", None)
    results_data = {
        "avg_return_last_100": avg_return,
        "var_return_last_100": var_return,
        "std_return_last_100": std_return,
        "replay_steps": steps,
        "replay_return": total_return,
        "replay_delivered": delivered,
    }
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
        exit(130)


if __name__ == "__main__":
    main()
