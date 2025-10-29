import os
import sys
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

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("tb_logs", exist_ok=True)
    env = DroneDeliveryEnv(max_battery=30, wind_slip=0.05)
    env = OneHotObservationWrapper(env)
    env = Monitor(env)
    model = DQN(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        buffer_size=100_000,
        learning_starts=10_000,
        target_update_interval=1_000,
        train_freq=(4, "step"),
        exploration_fraction=0.2,
        tensorboard_log="tb_logs",
        seed=42,
    )
    eval_env = DroneDeliveryEnv(max_battery=30, wind_slip=0.05)
    eval_env = OneHotObservationWrapper(eval_env)
    eval_env = Monitor(eval_env)
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=20,
        eval_freq=10_000,
        deterministic=True,
        best_model_save_path="models/dqn_best",
        log_path="models/dqn_eval",
    )
    model.learn(total_timesteps=1_000_000, callback=eval_callback)
    model.save("models/dqn_drone_delivery")
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"eval_mean_reward={mean_reward:.3f}, std={std_reward:.3f}")
