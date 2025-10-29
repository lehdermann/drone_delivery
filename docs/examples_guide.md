# Drone Delivery Examples Guide

This guide explains how to use the example scripts to visualize and evaluate learned policies for the Drone Delivery MDP.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Value Iteration Examples](#value-iteration-examples)
- [Monte Carlo Examples](#monte-carlo-examples)
- [Stable-Baselines3 (PPO/A2C/DQN)](#stable-baselines3-ppoacddqn)
- [Understanding the Output](#understanding-the-output)
- [Running Experiments](#running-experiments)
- [Advanced Usage](#advanced-usage)

---

## Overview

The `examples/` directory contains two main scripts:

1. **`replay_vi_policy.py`** - Trains using Value Iteration (DP) and visualizes the optimal policy
2. **`replay_mc_policy.py`** - Trains using Monte Carlo Control and visualizes the learned policy
3. **`replay_sb3_policy.py`** - Replays SB3-trained policies (PPO, A2C, DQN) with visualization

Both scripts provide:
- Live visualization with matplotlib
- Trajectory tracking and analysis
- Policy summary statistics
- CSV logging of results

---

## Quick Start

### Basic Value Iteration Demo

```bash
cd /Users/familia/src/rl_gym/drone_delivery

python examples/replay_vi_policy.py \
  --max-battery 30 \
  --obstacles default \
  --sleep 0.5
```

### Basic Monte Carlo Demo

```bash
python examples/replay_mc_policy.py \
  --max-battery 30 \
  --episodes 5000 \
  --obstacles default \
  --sleep 0.5
```

### Basic SB3 Replay (PPO)

```bash
python examples/replay_sb3_policy.py \
  --algo ppo \
  --episodes 5 \
  --deterministic \
  --max-battery 30 \
  --wind-slip 0.05 \
  --obstacles default --charging-stations default
```

---

## SARSA(λ) Examples

### Command Structure

```bash
python examples/replay_sarsa_policy.py [OPTIONS]
```

### Common Options

All VI options plus:

| Option | Default | Description |
|--------|---------|-------------|
| `--episodes` | 10000 | Number of training episodes |
| `--epsilon` | 0.1 | Initial exploration rate |
| `--epsilon-final` | None | Final epsilon for linear decay (if set) |
| `--alpha` | 0.05 | Initial step size |
| `--alpha-final` | None | Final alpha for linear decay (if set) |
| `--lam` | 0.9 | Trace decay parameter λ |
| `--replacing-traces` | False | Use replacing instead of accumulating traces |
| `--features` | basic | Feature set: `basic`, `one_hot`, or `engineered` |
| `--optimistic-init` | None | Optimistic initialization value for Q |

#### Tile Coding Options

| Option | Default | Description |
|--------|---------|-------------|
| `--features` | tile | Use tile coding features |
| `--tile-tilings` | 8 | Number of overlapping tilings |
| `--tile-bins-x` | 8 | Bins along x |
| `--tile-bins-y` | 8 | Bins along y |
| `--tile-bins-b` | 8 | Bins along battery |

Tips:
- Start with `tile-tilings=8` and `bins=6`–`8` per dimension.
- Reduce `alpha` roughly proportional to `1/num_tilings` and use decay.
- `λ` in `0.85–0.95` is often stable; `replacing-traces` helps.

### Example Scenarios

#### 1. One-Hot (Tabular) with Decays, Replacing Traces, and Optimistic Init

```bash
python examples/replay_sarsa_policy.py \
  --no-render \
  --episodes 10000 \
  --features one_hot \
  --gamma 0.995 \
  --epsilon 0.3 --epsilon-final 0.05 \
  --alpha 0.05 --alpha-final 0.01 \
  --lam 0.95 --replacing-traces \
  --optimistic-init 10.0 \
  --max-battery 30 \
  --obstacles default \
  --eval-episodes 200
```

#### 3. Tile Coding (stable and efficient)

```bash
python examples/replay_sarsa_policy.py \
  --no-render \
  --episodes 20000 \
  --features tile \
  --tile-tilings 8 --tile-bins-x 6 --tile-bins-y 6 --tile-bins-b 6 \
  --gamma 0.995 \
  --epsilon 0.3 --epsilon-final 0.02 \
  --alpha 0.007 --alpha-final 0.002 \
  --lam 0.90 --replacing-traces \
  --optimistic-init 5.0 \
  --max-battery 30 \
  --obstacles default \
  --eval-episodes 200
```

#### 2. Engineered Features (Faster Learning) with Decays and Replacing Traces

```bash
python examples/replay_sarsa_policy.py \
  --no-render \
  --episodes 15000 \
  --features engineered \
  --gamma 0.995 \
  --epsilon 0.2 --epsilon-final 0.05 \
  --alpha 0.02 --alpha-final 0.005 \
  --lam 0.95 --replacing-traces \
  --optimistic-init 5.0 \
  --max-battery 30 \
  --obstacles default \
  --eval-episodes 200
```


## Value Iteration Examples

### Command Structure

```bash
python examples/replay_vi_policy.py [OPTIONS]
```

### Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--width` | 7 | Grid width |
| `--height` | 7 | Grid height |
| `--max-battery` | 5 | Maximum battery capacity |
| `--charge-rate` | 2 | Battery units recharged per charge action |
| `--wind-slip` | 0.1 | Probability of wind deviation (0.0-0.5) |
| `--obstacles` | default | Obstacle configuration: 'none', 'default', or custom |
| `--gamma` | 0.99 | Discount factor |
| `--theta` | 1e-4 | Convergence threshold |
| `--seed` | 0 | Random seed |
| `--sleep` | 0.1 | Seconds between visualization steps |
| `--eval-episodes` | 0 | Number of evaluation episodes (no rendering) |

### Example Scenarios

#### 1. No Wind, Easy Navigation

```bash
python examples/replay_vi_policy.py \
  --wind-slip 0.0 \
  --max-battery 20 \
  --obstacles none \
  --sleep 0.3
```

#### 2. High Wind, Challenging Environment

```bash
python examples/replay_vi_policy.py \
  --wind-slip 0.2 \
  --max-battery 30 \
  --obstacles default \
  --sleep 0.5
```

#### 3. Custom Obstacles

```bash
python examples/replay_vi_policy.py \
  --obstacles "2,2;2,3;2,4;4,2;4,3;4,4" \
  --max-battery 25 \
  --sleep 0.4
```

#### 4. Fast Evaluation (No Visualization)

```bash
python examples/replay_vi_policy.py \
  --max-battery 30 \
  --obstacles default \
  --sleep 0 \
  --eval-episodes 100
```

---

## Monte Carlo Examples

### Command Structure

```bash
python examples/replay_mc_policy.py [OPTIONS]
```

### Common Options

All VI options plus:

| Option | Default | Description |
|--------|---------|-------------|
| `--episodes` | 20000 | Number of training episodes |
| `--epsilon` | 0.1 | Exploration rate (on-policy) |
| `--off-policy` | False | Use off-policy MC with importance sampling |
| `--behavior` | epsilon | Behavior policy: 'epsilon' or 'uniform' |
| `--behavior-epsilon` | 0.2 | Epsilon for behavior policy (off-policy) |
| `--off-weighted` | True | Use weighted IS (vs ordinary IS) |
| `--debug-behavior` | False | Print behavior policy statistics |

### Example Scenarios

#### 1. On-Policy MC (Epsilon-Soft)

```bash
python examples/replay_mc_policy.py \
  --episodes 5000 \
  --epsilon 0.1 \
  --max-battery 30 \
  --obstacles default \
  --sleep 0.5
```

#### 2. Off-Policy MC with Weighted Importance Sampling

```bash
python examples/replay_mc_policy.py \
  --episodes 5000 \
  --max-battery 30 \
  --obstacles default \
  --off-policy \
  --behavior epsilon \
  --behavior-epsilon 0.2 \
  --off-weighted \
  --sleep 0.5
```

#### 3. Quick Training Test (Fewer Episodes)

```bash
python examples/replay_mc_policy.py \
  --episodes 1000 \
  --max-battery 30 \
  --obstacles default \
  --sleep 0.3
```

#### 4. Extensive Training with Evaluation

```bash
python examples/replay_mc_policy.py \
  --episodes 10000 \
  --epsilon 0.05 \
  --max-battery 30 \
  --obstacles default \
  --eval-episodes 100 \
  --sleep 0.5
```

---

## Stable-Baselines3 (PPO/A2C/DQN)

### Training

Treinamento padrão (One-Hot, CPU, TensorBoard habilitado):

```bash
# PPO (~3M steps)
python examples/train_ppo.py

# A2C (~2M steps)
python examples/train_a2c.py

# DQN (~1M steps)
python examples/train_dqn.py
```

TensorBoard:

```bash
python -m tensorboard.main --logdir ./tb_logs
```

### Replay with Obstacles and Charging Stations

```bash
python examples/replay_sb3_policy.py \
  --algo ppo \
  --episodes 10 \
  --deterministic \
  --width 7 --height 7 \
  --max-battery 30 --charge-rate 2 --wind-slip 0.05 \
  --obstacles "2,3;3,3;4,3" \
  --charging-stations "0,0;3,6;6,3"
```

Headless + per-episode CSV:

```bash
python examples/replay_sb3_policy.py \
  --algo dqn \
  --episodes 100 \
  --no-render \
  --to-csv sb3_experiments.csv \
  --max-battery 30 --wind-slip 0.05 \
  --obstacles default --charging-stations default
```

Note: The script automatically adjusts `max_battery` in one-hot mode if the model expects a different one-hot dimension (e.g., trained with another battery size).

---

## Understanding the Output

### 1. Policy Summary

```
============================================================
LEARNED POLICY SUMMARY
============================================================
Total states: 2352
Action distribution across all states:
  UP      :  450 states ( 19.1%)
  DOWN    :  520 states ( 22.1%)
  LEFT    :  180 states (  7.7%)
  RIGHT   :  890 states ( 37.8%)
  STAY    :   12 states (  0.5%)
  CHARGE  :  300 states ( 12.8%)

Initial state policy:
  State: pos=(0,0), battery=30, has_package=True
  Action: RIGHT
  Q-values: [-45.2 -52.1 -48.3 32.5 -50.0 -15.2]
  Best Q-value: 32.50
============================================================
```

**Interpretation:**
- Shows how many states use each action
- Initial state action indicates the first move from start
- Q-values show expected returns for each action

### 2. Trajectory Visualization

```
============================================================
TRAJECTORY USING OPTIMAL POLICY
============================================================
Step   0: pos=(0,0) bat=30 📦 → RIGHT  → reward=  -1.0
Step   1: pos=(1,0) bat=29 📦 → RIGHT  → reward=  -1.0
Step   2: pos=(2,0) bat=28 📦 → RIGHT  → reward=  -1.0
...
Step  16: pos=(6,6) bat=14 ✓ → TERMINAL
============================================================
RESULT: SUCCESS ✓
Total steps: 16, Total return: 34.00
============================================================
```

**Symbols:**
- 📦 = Carrying package
- ✓ = Package delivered
- ✗ = Failed delivery

### 3. Evaluation Metrics

```
[replay][eval] N=100 success_rate=0.950 avg_return=32.45 avg_steps=16.2 avg_collisions=0.15
```

**Metrics:**
- `success_rate`: Fraction of successful deliveries
- `avg_return`: Average total reward
- `avg_steps`: Average number of steps
- `avg_collisions`: Average collisions per episode

### 4. Visual Elements

The matplotlib window shows:
- **🏪 S** = Store (pickup location)
- **🏠 H** = House (delivery location)
- **⬛ X** = Obstacles
- **⭐ ★** = Charging stations
- **🚁** = Drone (red triangle)
- **📦** = Package (brown square, when carried)

**Legend:**
- Blue background = Store
- Green background = House
- Dark gray = Obstacles
- Orange background = Charging stations

---

## Running Experiments

### Automated Experiment Suite

Run comprehensive experiments with:

```bash
cd /Users/familia/src/rl_gym/drone_delivery
./run_experiments.sh
```

This script:
1. Runs 10 VI experiments with random parameters
2. Runs MC on-policy and off-policy (reduced sampling)
3. Runs SARSA(λ) (one_hot and tile)
4. Runs SB3 replays for PPO/A2C/DQN, saving to `sb3_experiments.csv`

**Output:**
- `vi_experiments.csv` - VI results
- `mc_experiments.csv` - MC results
- `sarsa_experiments.csv` - SARSA results
- `sb3_experiments.csv` - SB3 replay results
- `run_experiments.out` - Full console log

### Customizing the Experiment Script

Edit `run_experiments.sh` to modify:

```bash
# Line 51: Number of MC episodes
mc_episodes=5000  # Increase for better convergence

# Lines 48-50: Parameter grids
wind_slips=(0.00 0.05 0.10 0.15)  # Add more wind levels
seeds=(0 1 2 3 4)                  # More random seeds
epsilons=(0.05 0.10 0.15)         # Different exploration rates
```

---

## Advanced Usage

### 1. Interrupting Training (Ctrl+C)

Both scripts handle interruption gracefully:
- Saves partial results to CSV
- Closes visualization properly
- Shows statistics up to interruption point

```bash
# Start training
python examples/replay_mc_policy.py --episodes 10000

# Press Ctrl+C to stop
# Output:
# [replay] Interrupted by user (Ctrl+C). Cleaning up...
# [MC] Training interrupted by user.
# [replay] Results appended to mc_experiments.csv
```

### 2. Progress Monitoring (tqdm)

MC training shows a progress bar (requires `tqdm`):

```bash
pip install tqdm

python examples/replay_mc_policy.py --episodes 5000
# MC Training:  40%|████████▍         | 2000/5000 [02:15<03:22, avg_return=28.45, last_return=32.10]
```

### 3. Headless Mode (No Visualization)

For faster execution without GUI:

```bash
# Set sleep to 0 and use eval-episodes for metrics
python examples/replay_vi_policy.py \
  --sleep 0 \
  --eval-episodes 100
```

### 4. Comparing Algorithms

Run both algorithms with same parameters:

```bash
# VI
python examples/replay_vi_policy.py \
  --max-battery 30 \
  --wind-slip 0.1 \
  --seed 42 \
  --sleep 0

# MC
python examples/replay_mc_policy.py \
  --max-battery 30 \
  --wind-slip 0.1 \
  --episodes 5000 \
  --seed 42 \
  --sleep 0
```

Then compare results and generate the consolidated plot:

```bash
python create_plot.py
open performance_plot.png  # or xdg-open on Linux
```

### 5. Debugging Policies

To understand why a policy fails:

```bash
# Slow visualization with trajectory logging
python examples/replay_vi_policy.py \
  --max-battery 15 \
  --obstacles default \
  --sleep 1.0  # 1 second per step
```

Watch the drone's decisions step-by-step and check the trajectory output.

---

## Troubleshooting

### Issue: "success_rate=0.000"

**Cause:** Insufficient battery or too few MC episodes

**Solution:**
```bash
# Increase battery
--max-battery 30

# Increase MC episodes
--episodes 5000
```

### Issue: Drone gets stuck charging

**Cause:** MC hasn't learned optimal policy yet

**Solution:**
```bash
# More episodes
--episodes 10000

# Lower exploration
--epsilon 0.05
```

### Issue: "ModuleNotFoundError: No module named 'envs'"

**Cause:** Running from wrong directory

**Solution:**
```bash
cd /Users/familia/src/rl_gym/drone_delivery
python examples/replay_vi_policy.py ...
```

### Issue: No matplotlib window appears

**Cause:** Backend issue or headless environment

**Solution:**
```bash
# Check matplotlib backend
python -c "import matplotlib; print(matplotlib.get_backend())"

# Use eval-episodes instead
--eval-episodes 100 --sleep 0
```

---

## Tips for Best Results

1. **Battery Sizing:** Use `max_battery ≥ 2 × grid_distance`
   - For 7×7 grid: `--max-battery 30` is safe

2. **MC Convergence:** Use at least 5000 episodes
   - More episodes = better policy
   - Monitor avg_return in progress bar

3. **Wind Effects:** Higher wind requires more battery
   - `wind_slip=0.0`: Deterministic
   - `wind_slip=0.1`: Realistic
   - `wind_slip=0.2`: Challenging

4. **Visualization Speed:**
   - `--sleep 0.3`: Good for watching
   - `--sleep 0.5`: Easier to follow
   - `--sleep 1.0`: Step-by-step debugging

5. **Evaluation:** Always use `--eval-episodes 100` for reliable metrics

---

## Example Workflow

### Complete Analysis Workflow

```bash
cd /Users/familia/src/rl_gym/drone_delivery

# 1. Train with VI (fast, optimal)
python examples/replay_vi_policy.py \
  --max-battery 30 \
  --obstacles default \
  --eval-episodes 100 \
  --sleep 0.5

# 2. Train with MC (slower, model-free)
python examples/replay_mc_policy.py \
  --max-battery 30 \
  --episodes 5000 \
  --obstacles default \
  --eval-episodes 100 \
  --sleep 0.5

# 3. Compare results
head -n 2 vi_experiments.csv
head -n 2 mc_experiments.csv

# 4. Run full experiment suite
./run_experiments.sh

# 5. Analyze results
python -c "
import pandas as pd
vi = pd.read_csv('vi_experiments.csv')
mc = pd.read_csv('mc_experiments.csv')
print('VI avg return:', vi['replay_return'].mean())
print('MC avg return:', mc['replay_return'].mean())
"
```


