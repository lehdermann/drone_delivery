import os
import sys
import argparse
import time
from typing import Optional

from stable_baselines3 import PPO, A2C, DQN

# Ensure project root is on sys.path so 'envs' and 'wrappers' are importable when running as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.drone_delivery import DroneDeliveryEnv
from wrappers.sb3_wrappers import OneHotObservationWrapper, DecodedObservationWrapper


ALGOS = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
}


def make_env(
    width: int,
    height: int,
    max_battery: int,
    wind_slip: float,
    one_hot: bool,
    obstacles,
    charging_stations,
    charge_rate: int,
) -> DroneDeliveryEnv:
    env = DroneDeliveryEnv(
        width=width,
        height=height,
        max_battery=max_battery,
        charge_rate=charge_rate,
        wind_slip=wind_slip,
        obstacles=obstacles,
        charging_stations=charging_stations,
        render_mode=None,  # will be set by caller if human rendering desired
    )
    env = OneHotObservationWrapper(env) if one_hot else DecodedObservationWrapper(env)
    return env


def default_model_path(algo: str) -> str:
    # Prefer best model if available
    best = {
        "ppo": "models/ppo_best/best_model.zip",
        "a2c": "models/a2c_best/best_model.zip",
        "dqn": "models/dqn_best/best_model.zip",
    }
    fallback = {
        "ppo": "models/ppo_drone_delivery.zip",
        "a2c": "models/a2c_drone_delivery.zip",
        "dqn": "models/dqn_drone_delivery.zip",
    }
    return best[algo] if os.path.exists(best[algo]) else fallback[algo]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay a trained SB3 policy on DroneDeliveryEnv")
    p.add_argument("--algo", choices=list(ALGOS.keys()), default="ppo")
    p.add_argument("--model-path", type=str, default=None, help="Path to the saved SB3 model .zip")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    # Env params
    p.add_argument("--width", type=int, default=7)
    p.add_argument("--height", type=int, default=7)
    p.add_argument("--max-battery", type=int, default=30)
    p.add_argument("--charge-rate", dest="charge_rate", type=int, default=2)
    p.add_argument("--wind-slip", type=float, default=0.05)
    p.add_argument(
        "--obstacles",
        type=str,
        default="default",
        help="Obstacles: 'none', 'default', or comma-separated x,y pairs like '2,3;3,3;4,3'",
    )
    p.add_argument(
        "--charging-stations",
        dest="charging_stations",
        type=str,
        default="default",
        help="Charging stations: 'none', 'default', or comma-separated x,y pairs like '0,0;3,6;6,3'",
    )
    p.add_argument("--one-hot", action="store_true", help="Use OneHotObservationWrapper (default)")
    p.add_argument("--decoded", action="store_true", help="Use DecodedObservationWrapper (4D features)")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic policy for replay")
    p.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between steps (for visualization)")
    p.add_argument("--no-render", dest="no_render", action="store_true", help="Disable rendering (headless)")
    p.add_argument("--to-csv", dest="to_csv", type=str, default=None, help="Append per-episode results to this CSV file")
    args = p.parse_args()
    if args.model_path is None:
        args.model_path = default_model_path(args.algo)
    # default to one-hot unless explicitly decoded
    one_hot = True
    if args.decoded:
        one_hot = False
    elif args.one_hot:
        one_hot = True
    args.one_hot = one_hot
    return args


def main():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}. Train and save a model first.")

    ModelCls = ALGOS[args.algo]
    model = ModelCls.load(args.model_path, device="cpu")

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

    # Parse charging stations
    charging_stations = None
    if args.charging_stations == "default":
        charging_stations = [
            (0, 0),  # pickup
            (3, 6),
            (6, 3),
        ]
    elif args.charging_stations == "none":
        charging_stations = None  # env will default to [pickup]
    else:
        try:
            charging_stations = []
            for pair in args.charging_stations.split(";"):
                x, y = map(int, pair.split(","))
                charging_stations.append((x, y))
        except Exception:
            print(f"[warning] Could not parse charging_stations '{args.charging_stations}', using default [pickup]")
            charging_stations = None

    # Build initial env
    env = make_env(
        args.width,
        args.height,
        args.max_battery,
        args.wind_slip,
        args.one_hot,
        obstacles,
        charging_stations,
        args.charge_rate,
    )
    # Set render mode if needed
    if not args.no_render:
        try:
            env.unwrapped.render_mode = "human"
        except Exception:
            pass

    # Auto-fix observation shape mismatch for OneHotObservationWrapper by inferring max_battery from model
    try:
        expected_shape = getattr(model.observation_space, 'shape', None)
        if args.one_hot and expected_shape is not None and len(expected_shape) == 1:
            exp_dim = int(expected_shape[0])
            # One-hot dimension should equal nS = width * height * (max_battery + 1) * 2
            # Use current env width/height to infer a compatible max_battery
            width, height = getattr(env, 'width', 7), getattr(env, 'height', 7)
            denom = int(width) * int(height) * 2
            if denom > 0:
                # Compute candidate max_battery
                cand = exp_dim / float(denom) - 1.0
                cand_int = int(round(cand))
                if cand_int >= 0 and abs(cand - cand_int) < 1e-6:
                    # If mismatch with requested max_battery, rebuild env
                    current_nS = int(width) * int(height) * (int(args.max_battery) + 1) * 2
                    if current_nS != exp_dim:
                        try:
                            env.close()
                        except Exception:
                            pass
                        env = make_env(
                            args.width,
                            args.height,
                            cand_int,
                            args.wind_slip,
                            args.one_hot,
                            obstacles,
                            charging_stations,
                            args.charge_rate,
                        )
                        print(f"[replay] Adjusted env to match model: max_battery={cand_int} (expected one-hot dim={exp_dim})")
    except Exception as e:
        print(f"[replay] Warning: could not auto-adjust env shape ({e}). Proceeding with requested settings.")

    import csv
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        truncated = False
        ep_return = 0.0
        steps = 0
        delivered = False
        while not (done or truncated):
            # Render current frame
            if not args.no_render:
                try:
                    env.render()
                except Exception:
                    pass

            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_return += float(reward)
            steps += 1
            if float(reward) >= 50.0:
                delivered = True
            if args.sleep > 0:
                time.sleep(args.sleep)
        print(f"[replay] episode {ep+1}/{args.episodes} return={ep_return:.2f} steps={steps}")

        # CSV logging per-episode
        if args.to_csv:
            row = {
                "algo": args.algo,
                "wind_slip": args.wind_slip,
                "max_battery": args.max_battery,
                "width": args.width,
                "height": args.height,
                "obstacles": args.obstacles,
                "charging_stations": args.charging_stations,
                "observation": ("one_hot" if args.one_hot else "decoded"),
                "model_path": args.model_path,
                "seed": args.seed + ep,
                "replay_return": ep_return,
                "replay_steps": steps,
                "replay_delivered": delivered,
            }
            write_header = not os.path.exists(args.to_csv)
            try:
                with open(args.to_csv, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if write_header:
                        w.writeheader()
                    w.writerow(row)
            except Exception as e:
                print(f"[replay] Warning: failed to write CSV '{args.to_csv}': {e}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
