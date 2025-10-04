#!/bin/bash

# --- Log file setup ---
# Redirect all output (stdout and stderr) to a log file while also printing to the console.
# The 'tee' command handles the duplication.
LOG_FILE="run_experiments.out"
exec &> >(tee -a "$LOG_FILE")


# This script runs a series of experiments for both Value Iteration (VI)
# and Monte Carlo (MC) algorithms, logging the results to their respective
# CSV files (vi_experiments.csv and mc_experiments.csv).

# Usage:
#   ./run_experiments.sh [MC_EPISODES]
#
# Arguments:
#   MC_EPISODES: Number of episodes for Monte Carlo training (default: 50)
#
# Examples:
#   ./run_experiments.sh          # Use default (50 episodes)
#   ./run_experiments.sh 5000     # Use 5000 episodes

# Parse command line arguments
MC_EPISODES=${1:-50}  # Default to 50 if not provided

echo "========================================================================"
echo "Starting new experiment suite at $(date)"
echo "MC Episodes: ${MC_EPISODES}"
echo "========================================================================"

echo "Starting experiment suite..."

# --- Value Iteration Experiments (10 runs) ---
echo ""
echo "--- Running Value Iteration (VI) Experiments ---"
for i in {0..9}
do
    # Vary wind_slip between 0.0 and 0.2, and max_battery between 20 and 40
    # Note: Grid 7x7 with obstacles requires ~12-16 steps minimum
    # With wind and detours, need 20-40 battery for reliable completion
    wind_slip=$(echo "scale=2; $RANDOM/32767 * 0.2" | bc)
    max_battery=$((RANDOM % 21 + 20))  # Random between 20 and 40

    echo "[VI Run ${i}/9] wind_slip=${wind_slip}, max_battery=${max_battery}, seed=${i}"
    python examples/replay_vi_policy.py \
        --wind-slip "$wind_slip" \
        --max-battery "$max_battery" \
        --seed "$i" \
        --obstacles default \
        --no-render # Disable rendering for faster execution
done

# --- Monte Carlo Experiments: On-Policy and Off-Policy (grid, reproducible) ---
echo ""
echo "--- Running Monte Carlo (MC) Experiments: On-Policy ---"

# Grids (reproducible): adjust as needed
# Reduce for faster experiments: fewer wind levels and seeds
wind_slips=(0.00 0.10)  # Reduced from 3 to 2 levels (saves 33% time)
seeds=(0 1)             # Reduced from 3 to 2 seeds (saves 33% time)
epsilons=(0.10)
# mc_episodes is now set from command line argument (default: 50)
# Total MC runs: 2 wind × 2 seeds × (1 on + 2 off) = 12 runs (was 27)

for ws in "${wind_slips[@]}"; do
  for seed in "${seeds[@]}"; do
    for eps in "${epsilons[@]}"; do
      echo "[MC On-Policy] wind_slip=${ws}, epsilon=${eps}, episodes=${MC_EPISODES}, seed=${seed}"
      python examples/replay_mc_policy.py --wind-slip "$ws" --epsilon "$eps" --episodes "$MC_EPISODES" --seed "$seed" --max-battery 30 --obstacles default --no-render --eval-episodes 100
    done
  done
done

echo ""
echo "--- Running Monte Carlo (MC) Experiments: Off-Policy (Weighted IS, epsilon behavior) ---"

behavior_epsilons=(0.20)  # Reduced from 2 to 1 (saves 50% off-policy time)
debug_behavior_episodes=200

for ws in "${wind_slips[@]}"; do
  for seed in "${seeds[@]}"; do
    for beps in "${behavior_epsilons[@]}"; do
      echo "[MC Off-Policy] wind_slip=${ws}, behavior=epsilon, behavior_epsilon=${beps}, episodes=${MC_EPISODES}, seed=${seed}"
      python examples/replay_mc_policy.py --wind-slip "$ws" --episodes "$MC_EPISODES" --seed "$seed" --max-battery 30 --obstacles default --no-render --off-policy --behavior epsilon --behavior-epsilon "$beps" --off-weighted --eval-episodes 100 --debug-behavior --debug-behavior-episodes "$debug_behavior_episodes"
    done
  done
done

echo ""
echo "Experiment suite finished. Results are in vi_experiments.csv and mc_experiments.csv."
echo "Full console output logged to ${LOG_FILE}."
