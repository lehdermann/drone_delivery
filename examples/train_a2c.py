import os
import sys

# Ensure project root is on sys.path so 'envs' and 'wrappers' are importable when running as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


import torch
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from envs.drone_delivery import DroneDeliveryEnv
from wrappers.sb3_wrappers import OneHotObservationWrapper
from stable_baselines3.common.callbacks import EvalCallback

def make_env_fn():
    def _f():
        env = DroneDeliveryEnv()
        env = OneHotObservationWrapper(env)
        return env
    return _f

def train_a2c(
    total_timesteps: int = 2_000_000,
    n_envs: int = 8,
    best_model_save_path: str = "models/a2c_best",
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ent_coef: float = 0.01,
    learning_rate: float = 3e-4,
    n_steps: int = 20,
    seed: int = 42,
    verbose: int = 0,
):
    """
    Train an A2C model on the DroneDelivery environment.

    :param total_timesteps: The total number of samples (env steps) to train on.
    :param n_envs: The number of parallel environments to use.
    :param save_path: Path to save the final model.
    :param best_model_save_path: Path to save the best model during training.
    :param gamma: The discount factor.
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
    :param ent_coef: Entropy coefficient for the loss calculation.
    :param learning_rate: The learning rate.
    :param n_steps: The number of steps to run for each environment per update.
    :param seed: Random seed.
    :param verbose: Verbosity level.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("tb_logs", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vec_env = make_vec_env(make_env_fn(), n_envs=n_envs, seed=seed)
    model = A2C(
        "MlpPolicy",
        vec_env,
        device=device,
        verbose=verbose,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        n_steps=n_steps,
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
        tb_log_name="A2C",
        progress_bar=True,
    )
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    return model, mean_reward, std_reward

if __name__ == "__main__":
    train_a2c()
