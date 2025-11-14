import os
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
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

def train_ppo(
    total_timesteps: int = 3_000_000,
    n_envs: int = 8,
    best_model_save_path: str = "models/ppo_best",
    n_steps: int = 256,
    batch_size: int = 2048,
    gae_lambda: float = 0.95,
    gamma: float = 0.99,
    ent_coef: float = 0.01,
    learning_rate: float = 2.5e-4,
    n_epochs: int = 10,
    clip_range: float = 0.2,
    seed: int = 42,
    verbose: int = 0,
):
    """
    Train a PPO model on the DroneDelivery environment.

    :param total_timesteps: The total number of samples (env steps) to train on.
    :param n_envs: The number of parallel environments to use.
    :param best_model_save_path: Path to save the best model during training.
    :param n_steps: The number of steps to run for each environment per update.
    :param batch_size: The size of the batch for optimization.
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
    :param gamma: The discount factor.
    :param ent_coef: Entropy coefficient for the loss calculation.
    :param learning_rate: The learning rate.
    :param n_epochs: The number of epochs when optimizing the surrogate loss.
    :param clip_range: Clipping parameter for PPO.
    :param seed: Random seed.
    :param verbose: Verbosity level.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("tb_logs", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vec_env = make_vec_env(make_env_fn(), n_envs=n_envs, seed=seed)
    model = PPO(
        "MlpPolicy",
        vec_env,
        device=device,
        verbose=verbose,
        n_steps=n_steps,
        batch_size=batch_size,
        gae_lambda=gae_lambda,
        gamma=gamma,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        clip_range=clip_range,
        policy_kwargs=dict(net_arch=[128, 128]),
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
        tb_log_name="PPO",
        progress_bar=True,
    )
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    return model, mean_reward, std_reward

if __name__ == "__main__":
    train_ppo()
