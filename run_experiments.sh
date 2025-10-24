#!/bin/bash

# --- Log file setup ---
# Redirect all output (stdout and stderr) to a log file while also printing to the console.
# The 'tee' command handles the duplication.
LOG_FILE="run_experiments.out"
exec &> >(tee -a "$LOG_FILE")


# This script runs a series of experiments for Value Iteration (VI),
# Monte Carlo (MC), and SARSA(λ), logging the results to their respective
# CSV files (vi_experiments.csv, mc_experiments.csv, sarsa_experiments.csv).

# Usage:
#   ./run_experiments.sh [MC_EPISODES] [SARSA_EPISODES]
#
# Arguments:
#   MC_EPISODES: Number of episodes for Monte Carlo training (default: 50)
#   SARSA_EPISODES: Number of episodes for SARSA training (default: 5000)
#
# Examples:
#   ./run_experiments.sh                   # Use defaults (MC:50, SARSA:5000)
#   ./run_experiments.sh 5000 10000        # MC:5000 episodes, SARSA:10000 episodes

# Parse command line arguments
MC_EPISODES=${1:-50}       # Default to 50 if not provided
SARSA_EPISODES=${2:-5000}  # Default to 5000 if not provided

echo "========================================================================"
echo "Starting new experiment suite at $(date)"
echo "MC Episodes: ${MC_EPISODES}"
echo "SARSA Episodes: ${SARSA_EPISODES}"
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

# --- SARSA(λ) Experiments (grid, reproducible) ---
echo ""
echo "--- Running SARSA(λ) Experiments ---"

# Use a small grid to keep runtime reasonable; tuned based on prior validation
sarsa_wind_slips=(0.00 0.10)
sarsa_seeds=(0 1)

for ws in "${sarsa_wind_slips[@]}"; do
  for seed in "${sarsa_seeds[@]}"; do
    echo "[SARSA] wind_slip=${ws}, episodes=${SARSA_EPISODES}, seed=${seed}, features=one_hot"
    python examples/replay_sarsa_policy.py \
      --no-render \
      --episodes "${SARSA_EPISODES}" \
      --features one_hot \
      --gamma 0.995 \
      --epsilon 0.3 --epsilon-final 0.05 \
      --alpha 0.05 --alpha-final 0.01 \
      --lam 0.95 --replacing-traces \
      --optimistic-init 10.0 \
      --wind-slip "$ws" \
      --max-battery 30 \
      --obstacles default \
      --eval-episodes 100 \
      --seed "$seed"
  done
done

# --- SARSA(λ) Experiments: Tile Coding (grid, reproducible) ---
echo ""
echo "--- Running SARSA(λ) Experiments (Tile Coding) ---"

tile_wind_slips=(0.00 0.10)
tile_seeds=(0 1)

for ws in "${tile_wind_slips[@]}"; do
  for seed in "${tile_seeds[@]}"; do
    echo "[SARSA Tile] wind_slip=${ws}, episodes=${SARSA_EPISODES}, seed=${seed}, features=tile"
    python examples/replay_sarsa_policy.py \
      --no-render \
      --episodes "${SARSA_EPISODES}" \
      --features tile \
      --tile-tilings 8 --tile-bins-x 6 --tile-bins-y 6 --tile-bins-b 6 \
      --gamma 0.995 \
      --epsilon 0.3 --epsilon-final 0.02 \
      --alpha 0.007 --alpha-final 0.002 \
      --lam 0.90 --replacing-traces \
      --optimistic-init 5.0 \
      --wind-slip "$ws" \
      --max-battery 30 \
      --obstacles default \
      --eval-episodes 100 \
      --seed "$seed"
  done
done

echo ""
echo "Experiment suite finished. Results are in vi_experiments.csv, mc_experiments.csv, and sarsa_experiments.csv."
echo "Full console output logged to ${LOG_FILE}."
