import os
import sys
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# Ensure project root is on sys.path so 'envs' and 'wrappers' are importable when running as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.drone_delivery import DroneDeliveryEnv
from wrappers.sb3_wrappers import OneHotObservationWrapper

def make_env_fn():
    def _f():
        env = DroneDeliveryEnv()
        env = OneHotObservationWrapper(env)
        return env
    return _f

def train_dqn(
    total_timesteps: int = 1_000_000,
    best_model_save_path: str = "models/dqn_best",
    buffer_size: int = 100_000,
    learning_starts: int = 10_000,
    target_update_interval: int = 1_000,
    train_freq: int = 4,
    exploration_fraction: float = 0.2,
    seed: int = 42,
    verbose: int = 0,
):
    """
    Train a DQN model on the DroneDelivery environment.

    :param total_timesteps: The total number of samples (env steps) to train on.
    :param save_path: Path to save the final model.
    :param best_model_save_path: Path to save the best model during training.
    :param buffer_size: Size of the replay buffer.
    :param learning_starts: How many steps of the model to collect transitions for before learning starts.
    :param target_update_interval: Update frequency of the target network.
    :param train_freq: Update the model every ``train_freq`` steps.
    :param exploration_fraction: Fraction of entire training period over which the exploration rate is reduced.
    :param seed: Random seed.
    :param verbose: Verbosity level.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("tb_logs", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_env = Monitor(make_env_fn()())
    model = DQN(
        "MlpPolicy",
        train_env,
        device=device,
        verbose=verbose,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        target_update_interval=target_update_interval,
        train_freq=(train_freq, "step"),
        exploration_fraction=exploration_fraction,
        tensorboard_log="tb_logs",
        seed=seed,
    )
    eval_env = Monitor(make_env_fn()())
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=20,
        eval_freq=10_000,
        deterministic=True,
        best_model_save_path=best_model_save_path,
        log_path=os.path.join(best_model_save_path, "eval_logs"),
        verbose=verbose,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="DQN",
        progress_bar=True,
    )
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    return model, mean_reward, std_reward

if __name__ == "__main__":
    train_dqn()
