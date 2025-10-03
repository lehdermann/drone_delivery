
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Load the data with error handling
try:
    print("Loading mc_experiments.csv...")
    mc_df = pd.read_csv('mc_experiments.csv', on_bad_lines='skip')
    print(f"Loaded {len(mc_df)} MC experiments")
except Exception as e:
    print(f"Error loading mc_experiments.csv: {e}")
    print("Trying with different encoding...")
    try:
        mc_df = pd.read_csv('mc_experiments.csv', on_bad_lines='skip', encoding='utf-8', quoting=1)
        print(f"Loaded {len(mc_df)} MC experiments with alternative settings")
    except Exception as e2:
        print(f"Failed to load MC data: {e2}")
        sys.exit(1)

try:
    print("Loading vi_experiments.csv...")
    vi_df = pd.read_csv('vi_experiments.csv', on_bad_lines='skip')
    print(f"Loaded {len(vi_df)} VI experiments")
except Exception as e:
    print(f"Error loading vi_experiments.csv: {e}")
    sys.exit(1)

# Check and print column names
print("\nMC columns:", mc_df.columns.tolist())
print("VI columns:", vi_df.columns.tolist())

# Convert boolean columns from string to actual boolean
if 'replay_delivered' in mc_df.columns:
    mc_df['replay_delivered'] = mc_df['replay_delivered'].map({'True': True, 'False': False, True: True, False: False})
if 'replay_delivered' in vi_df.columns:
    vi_df['replay_delivered'] = vi_df['replay_delivered'].map({'True': True, 'False': False, True: True, False: False})

# Convert wind_slip to numeric
mc_df['wind_slip'] = pd.to_numeric(mc_df['wind_slip'], errors='coerce')
vi_df['wind_slip'] = pd.to_numeric(vi_df['wind_slip'], errors='coerce')

# Drop rows with NaN values
mc_df = mc_df.dropna(subset=['wind_slip', 'replay_delivered'])
vi_df = vi_df.dropna(subset=['wind_slip', 'replay_delivered'])

print(f"\nAfter cleaning: {len(mc_df)} MC experiments, {len(vi_df)} VI experiments")

# Process Monte Carlo data
mc_grouped = mc_df.groupby('wind_slip').agg(
    avg_iterations=('episodes', 'mean'),
    success_rate=('replay_delivered', 'mean')
).reset_index()

# Process Value Iteration data
vi_grouped = vi_df.groupby('wind_slip').agg(
    avg_iterations=('vi_iterations', 'mean'),
    success_rate=('replay_delivered', 'mean')
).reset_index()

print("\nMC grouped data:")
print(mc_grouped)
print("\nVI grouped data:")
print(vi_grouped)

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot iterations for both algorithms on the primary y-axis
ax1.plot(mc_grouped['wind_slip'], mc_grouped['avg_iterations'], 'o-', label='MC Iterations', color='tab:blue')
ax1.plot(vi_grouped['wind_slip'], vi_grouped['avg_iterations'], 's-', label='VI Iterations', color='tab:cyan')
ax1.set_xlabel('Probabilidade de Slip (p_slip)')
ax1.set_ylabel('Iterações para Convergência', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

# Create a secondary y-axis for success rate
ax2 = ax1.twinx()
ax2.plot(mc_grouped['wind_slip'], mc_grouped['success_rate'], 'o--', label='MC Success Rate', color='tab:red')
ax2.plot(vi_grouped['wind_slip'], vi_grouped['success_rate'], 's--', label='VI Success Rate', color='tab:orange')
ax2.set_ylabel('Taxa de Sucesso', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# Add title and legend
plt.title('Performance do Algoritmo vs. Dificuldade do Ambiente')
fig.tight_layout()
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

# Save the plot
plt.savefig('performance_plot.png')

print("Plot saved as performance_plot.png")
