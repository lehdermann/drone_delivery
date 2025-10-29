# Monitoring Training with TensorBoard

This guide shows how to use TensorBoard to monitor Stable-Baselines3 (SB3) training runs (PPO, A2C, DQN) in this project.

## 1) Prerequisites
- Install in your active virtual environment:
  ```bash
  pip install tensorboard
  ```
- SB3 training scripts already enable logging via `tensorboard_log="tb_logs"`.
  - Files will be created under `./tb_logs/` when you run `examples/train_*.py`.

## 2) Start training (creates logs)
Run any of the SB3 trainers (they log to `./tb_logs` automatically):
```bash
# PPO (~3M steps)
python examples/train_ppo.py

# A2C (~2M steps)
python examples/train_a2c.py

# DQN (~1M steps)
python examples/train_dqn.py
```

Optionally, name a run for easier comparison (SB3 supports `tb_log_name` in `learn`):
```python
model.learn(total_timesteps=3_000_000, callback=eval_callback, tb_log_name="ppo_bat30_ws005")
```

## 3) Launch TensorBoard
From the project root, run one of the following:
```bash
# Recommended (works well inside WSL/venv)
python -m tensorboard.main --logdir ./tb_logs --port 6006 --bind_all

# Or, if the tensorboard binary is available in your venv
~/.venv/bin/tensorboard --logdir ./tb_logs --port 6006 --bind_all
```
Then open your browser at:
- http://localhost:6006

Notes for WSL:
- If using VS Code Remote/WSL, the above URL usually works directly.
- If the page is blank, try a hard refresh or a different browser.

## 4) What to look at (common SB3 scalars)
Typical useful tags in the Scalars tab:
- rollout/ep_rew_mean — moving average of episode returns
- rollout/ep_len_mean — moving average of episode lengths
- train/approx_kl — PPO/A2C approximate KL (lower is typically more stable)
- train/clip_fraction — PPO clipping fraction
- train/entropy_loss — policy entropy term
- train/value_loss — critic/value loss
- time/fps — training speed (steps per second)

Tips:
- Use the Smoothing slider to reduce noise.
- Use the Runs panel to compare PPO, A2C, DQN side by side.

## 5) Organizing multiple runs
Each call to `learn()` creates a new run directory inside `tb_logs/`.
- To keep runs tidy, set `tb_log_name` in `learn()`:
  ```python
  model.learn(total_timesteps=1_000_000, tb_log_name="dqn_bat30_ws005")
  ```
- Alternatively, move or rename subfolders under `tb_logs/` after training finishes.

## 6) Troubleshooting
- "command not found: tensorboard": run via the module form:
  ```bash
  python -m tensorboard.main --logdir ./tb_logs
  ```
- No data appears:
  - Confirm logs exist: `ls -R tb_logs/` should show `events.out.tfevents...` files
  - Ensure you ran one of the `examples/train_*.py` scripts
  - Make sure `tensorboard_log="tb_logs"` is set in the script
- Port already in use:
  - Pick another port: `--port 6007`, then open http://localhost:6007
- Slow or frozen UI:
  - Reduce smoothing; filter visible tags; or close other heavy tabs

## 7) Limitations and notes
- EvalCallback metrics are saved under `models/<algo>_eval/` (CSV-like logs), not TensorBoard.
- Replay scripts (`examples/replay_*`) do not write TensorBoard logs; they can write CSVs for plotting via `create_plot.py`.
- All examples run on CPU by default (no GPU required).

## 8) Quick reference
- Start TB: `python -m tensorboard.main --logdir ./tb_logs --port 6006 --bind_all`
- Open UI: `http://localhost:6006`
- Logs directory: `./tb_logs/`
- Trainers: `examples/train_ppo.py`, `train_a2c.py`, `train_dqn.py`
