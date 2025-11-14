import os
import sys
import time
from typing import Optional

from stable_baselines3 import A2C, DQN, PPO

# Ensure project root is on sys.path for imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.drone_delivery import DroneDeliveryEnv
from wrappers.sb3_wrappers import OneHotObservationWrapper

ALGOS = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
}


def default_model_path(algo: str) -> str:
    """Returns the default path for a given algorithm's model file."""
    best = {
        "ppo": "models/ppo_best/best_model.zip",
        "a2c": "models/a2c_best/best_model.zip",
        "dqn": "models/dqn_best/best_model.zip",
    }
    return best.get(algo, f"models/{algo}_drone_delivery.zip")


def replay_notebook_policy(
    algo: str,
    model_path: Optional[str] = None,
    render_ax=None,
    seed: int = 42,
    deterministic: bool = True,
    sleep: float = 0.1,
):
    """
    Replays a trained SB3 policy in a notebook, using default DroneDeliveryEnv parameters.

    :param algo: The algorithm name ('ppo', 'a2c', 'dqn').
    :param model_path: Optional path to the saved model .zip file. If None, uses default path.
    :param render_ax: Matplotlib Axes object for rendering. If None, no rendering.
    :param seed: Seed for the environment.
    :param deterministic: Whether to use deterministic actions.
    :param sleep: Time to sleep between steps for animation.
    """
    if model_path is None:
        model_path = default_model_path(algo)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Please train the model first.")

    ModelCls = ALGOS[algo]
    model = ModelCls.load(model_path, device="cpu")

    # Create environment with default parameters from drone_delivery.py
    env = DroneDeliveryEnv()
    env = OneHotObservationWrapper(env)

    obs, info = env.reset(seed=seed)
    done, truncated = False, False
    ep_return, steps = 0.0, 0

    fig = None
    if render_ax:
        from IPython import display
        from matplotlib import pyplot as plt

        fig = render_ax.figure
        plt.close(fig)  # Prevent the initial static plot
        d_handle = display.display(fig, display_id=True)

    while not (done or truncated):
        if render_ax and fig and d_handle:
            env.render(ax=render_ax)
            d_handle.update(fig)
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(int(action))
        ep_return += float(reward)
        steps += 1

        if sleep > 0 and render_ax:
            time.sleep(sleep)

    if render_ax:
        if fig and d_handle:
            env.render(ax=render_ax)  # Final render
            d_handle.update(fig)
        print("Replay finished.")
        print(f"  Total Steps: {steps}")
        print(f"  Total Return: {ep_return:.2f}")
        print(f"  Delivered: {'Yes' if info.get('delivered') else 'No'}")

    try:
        env.close()
    except Exception:
        pass

    return ep_return, steps, info