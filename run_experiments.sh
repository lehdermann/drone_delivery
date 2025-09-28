#!/bin/bash

# --- Log file setup ---
# Redirect all output (stdout and stderr) to a log file while also printing to the console.
# The 'tee' command handles the duplication.
LOG_FILE="run_experiments.out"
exec &> >(tee -a "$LOG_FILE")


# This script runs a series of experiments for both Value Iteration (VI)
# and Monte Carlo (MC) algorithms, logging the results to their respective
# CSV files (vi_experiments.csv and mc_experiments.csv).

# To run this script, make it executable first:
# chmod +x run_experiments.sh
# Then execute it:
# ./run_experiments.sh

echo "========================================================================"
echo "Starting new experiment suite at $(date)"
echo "========================================================================"

echo "Starting experiment suite..."

# --- Value Iteration Experiments (10 runs) ---
echo ""
echo "--- Running Value Iteration (VI) Experiments ---"
for i in {0..9}
do
    # Vary wind_slip between 0.0 and 0.4, and max_battery between 5 and 15
    wind_slip=$(echo "scale=2; $RANDOM/32767 * 0.4" | bc)
    max_battery=$((RANDOM % 11 + 5))

    echo "[VI Run ${i}/9] wind_slip=${wind_slip}, max_battery=${max_battery}, seed=${i}"
    python -m examples.replay_vi_policy \
        --wind-slip "$wind_slip" \
        --max-battery "$max_battery" \
        --seed "$i" \
        --sleep 0 # Disable sleep for faster execution
done

# --- Monte Carlo Experiments: On-Policy and Off-Policy (grid, reproducible) ---
echo ""
echo "--- Running Monte Carlo (MC) Experiments: On-Policy ---"

# Grids (reproducible): adjust as needed
wind_slips=(0.00 0.05 0.10)
seeds=(0 1 2)
epsilons=(0.10)
mc_episodes=5000

for ws in "${wind_slips[@]}"; do
  for seed in "${seeds[@]}"; do
    for eps in "${epsilons[@]}"; do
      echo "[MC On-Policy] wind_slip=${ws}, epsilon=${eps}, episodes=${mc_episodes}, seed=${seed}"
      python -m examples.replay_mc_policy \
        --wind-slip "$ws" \
        --epsilon "$eps" \
        --episodes "$mc_episodes" \
        --seed "$seed" \
        --sleep 0 \
        --eval-episodes 100
    done
  done
done

echo ""
echo "--- Running Monte Carlo (MC) Experiments: Off-Policy (Weighted IS, epsilon behavior) ---"

behavior_epsilons=(0.10 0.20)
debug_behavior_episodes=200

for ws in "${wind_slips[@]}"; do
  for seed in "${seeds[@]}"; do
    for beps in "${behavior_epsilons[@]}"; do
      echo "[MC Off-Policy] wind_slip=${ws}, behavior=epsilon, behavior_epsilon=${beps}, episodes=${mc_episodes}, seed=${seed}"
      python -m examples.replay_mc_policy \
        --wind-slip "$ws" \
        --episodes "$mc_episodes" \
        --seed "$seed" \
        --sleep 0 \
        --off-policy --behavior epsilon --behavior-epsilon "$beps" --off-weighted \
        --eval-episodes 100 \
        --debug-behavior --debug-behavior-episodes "$debug_behavior_episodes"
    done
  done
done

echo ""
echo "Experiment suite finished. Results are in vi_experiments.csv and mc_experiments.csv."
echo "Full console output logged to ${LOG_FILE}."