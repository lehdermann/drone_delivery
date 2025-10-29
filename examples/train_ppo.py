import os
import sys
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
        env = DroneDeliveryEnv(max_battery=30, wind_slip=0.05)
        env = OneHotObservationWrapper(env)
        return env
    return _f

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("tb_logs", exist_ok=True)
    vec_env = make_vec_env(make_env_fn(), n_envs=8, seed=42)
    model = PPO(
        "MlpPolicy",
        vec_env,
        device="cpu",
        verbose=1,
        n_steps=256,
        batch_size=2048,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.01,
        learning_rate=2.5e-4,
        n_epochs=10,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log="tb_logs",
        seed=42,
    )
    eval_env = make_env_fn()()
    eval_env = Monitor(eval_env)
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=20,
        eval_freq=10_000,
        deterministic=True,
        best_model_save_path="models/ppo_best",
        log_path="models/ppo_eval",
    )
    model.learn(total_timesteps=3_000_000, callback=eval_callback)
    model.save("models/ppo_drone_delivery")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"eval_mean_reward={mean_reward:.3f}, std={std_reward:.3f}")
