import os
import sys
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Ensure project root is on sys.path so 'envs' and 'wrappers' are importable when running as a script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.drone_delivery import DroneDeliveryEnv
from wrappers.sb3_wrappers import OneHotObservationWrapper
from stable_baselines3.common.callbacks import EvalCallback

def make_env_fn():
    def _f():
        env = DroneDeliveryEnv(max_battery=30, wind_slip=0.05)
        env = OneHotObservationWrapper(env)
        return env
    return _f

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    vec_env = make_vec_env(make_env_fn(), n_envs=8, seed=42)
    model = A2C(
        "MlpPolicy",
        vec_env,
        device="cpu",
        verbose=1,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        learning_rate=3e-4,
        n_steps=20,
        tensorboard_log="tb_logs",
        seed=42,
    )
    model.save("models/a2c_drone_delivery")
    eval_env = make_env_fn()()
    eval_env = Monitor(eval_env)
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=20,
        eval_freq=10_000,
        deterministic=True,
        best_model_save_path="models/a2c_best",
        log_path="models/a2c_eval",
    )
    model.learn(total_timesteps=2_000_000, callback=eval_callback)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"eval_mean_reward={mean_reward:.3f}, std={std_reward:.3f}")
