
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

# Try load SARSA data (optional)
try:
    print("Loading sarsa_experiments.csv...")
    sarsa_df = pd.read_csv('sarsa_experiments.csv', on_bad_lines='skip')
    print(f"Loaded {len(sarsa_df)} SARSA experiments")
except Exception as e:
    print(f"Error loading sarsa_experiments.csv: {e}")
    sarsa_df = pd.DataFrame()

# Check and print column names
print("\nMC columns:", mc_df.columns.tolist())
print("VI columns:", vi_df.columns.tolist())
if not sarsa_df.empty:
    print("SARSA columns:", sarsa_df.columns.tolist())

# Convert boolean columns from string to actual boolean
if 'replay_delivered' in mc_df.columns:
    # A more robust way to convert string 'True'/'False' to boolean, handling case-insensitivity
    mc_df['replay_delivered'] = mc_df['replay_delivered'].apply(lambda x: str(x).lower() == 'true' if isinstance(x, str) else bool(x))

if 'replay_delivered' in vi_df.columns:
    # Apply the same robust conversion for the VI dataframe
    vi_df['replay_delivered'] = vi_df['replay_delivered'].apply(lambda x: str(x).lower() == 'true' if isinstance(x, str) else bool(x))

if not sarsa_df.empty and 'replay_delivered' in sarsa_df.columns:
    sarsa_df['replay_delivered'] = sarsa_df['replay_delivered'].apply(lambda x: str(x).lower() == 'true' if isinstance(x, str) else bool(x))

# Convert wind_slip to numeric
mc_df['wind_slip'] = pd.to_numeric(mc_df['wind_slip'], errors='coerce')
vi_df['wind_slip'] = pd.to_numeric(vi_df['wind_slip'], errors='coerce')
if not sarsa_df.empty and 'wind_slip' in sarsa_df.columns:
    sarsa_df['wind_slip'] = pd.to_numeric(sarsa_df['wind_slip'], errors='coerce')

# Identify MC on-policy vs off-policy
# Off-policy has 'behavior' column or can be identified by presence of off_policy flag
if 'off_policy' in mc_df.columns:
    mc_on = mc_df[mc_df['off_policy'] == False].copy()
    mc_off = mc_df[mc_df['off_policy'] == True].copy()
else:
    # Fallback: assume first half is on-policy, second half is off-policy
    # Better: check for specific columns that only exist in off-policy
    mc_on = mc_df.copy()
    mc_off = pd.DataFrame()

print(f"\nSplit MC data: {len(mc_on)} on-policy, {len(mc_off)} off-policy")

# Drop rows with NaN values
mc_on = mc_on.dropna(subset=['wind_slip', 'replay_delivered'])
if len(mc_off) > 0:
    mc_off = mc_off.dropna(subset=['wind_slip', 'replay_delivered'])
vi_df = vi_df.dropna(subset=['wind_slip', 'replay_delivered'])
if not sarsa_df.empty:
    sarsa_df = sarsa_df.dropna(subset=['wind_slip', 'replay_delivered'])

print(f"After cleaning: {len(mc_on)} MC on-policy, {len(mc_off)} MC off-policy, {len(vi_df)} VI experiments")

# Process data by algorithm type
mc_on_grouped = mc_on.groupby('wind_slip').agg(
    success_rate=('replay_delivered', 'mean'),
    avg_return=('replay_return', 'mean'),
    avg_steps=('replay_steps', 'mean')
).reset_index()

if len(mc_off) > 0:
    mc_off_grouped = mc_off.groupby('wind_slip').agg(
        success_rate=('replay_delivered', 'mean'),
        avg_return=('replay_return', 'mean'),
        avg_steps=('replay_steps', 'mean')
    ).reset_index()
else:
    mc_off_grouped = pd.DataFrame()

vi_grouped = vi_df.groupby('wind_slip').agg(
    success_rate=('replay_delivered', 'mean'),
    avg_return=('replay_return', 'mean'),
    avg_steps=('replay_steps', 'mean'),
    avg_vi_iters=('vi_iterations', 'mean')
).reset_index()

sarsa_grouped = pd.DataFrame()
if not sarsa_df.empty:
    sarsa_grouped = sarsa_df.groupby('wind_slip').agg(
        success_rate=('replay_delivered', 'mean'),
        avg_return=('replay_return', 'mean'),
        avg_steps=('replay_steps', 'mean')
    ).reset_index()

print("\nMC On-Policy grouped:")
print(mc_on_grouped)
if len(mc_off_grouped) > 0:
    print("\nMC Off-Policy grouped:")
    print(mc_off_grouped)
print("\nVI grouped:")
print(vi_grouped)
if len(sarsa_grouped) > 0:
    print("\nSARSA grouped:")
    print(sarsa_grouped)

# Create subplots for comprehensive comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Algorithm Comparison: VI vs MC On-Policy vs MC Off-Policy vs SARSA', fontsize=16, fontweight='bold')

# Plot 1: Success Rate vs Wind Slip
ax = axes[0, 0]
ax.plot(vi_grouped['wind_slip'], vi_grouped['success_rate'], 's-', label='VI', color='tab:blue', linewidth=2, markersize=8)
ax.plot(mc_on_grouped['wind_slip'], mc_on_grouped['success_rate'], 'o-', label='MC On-Policy', color='tab:orange', linewidth=2, markersize=8)
if len(mc_off_grouped) > 0:
    ax.plot(mc_off_grouped['wind_slip'], mc_off_grouped['success_rate'], '^-', label='MC Off-Policy', color='tab:green', linewidth=2, markersize=8)
if len(sarsa_grouped) > 0:
    ax.plot(sarsa_grouped['wind_slip'], sarsa_grouped['success_rate'], 'x-', label='SARSA', color='tab:red', linewidth=2, markersize=8)
ax.set_xlabel('Wind Slip Probability', fontsize=11)
ax.set_ylabel('Success Rate', fontsize=11)
ax.set_title('Success Rate vs Wind Slip', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])

# Plot 2: Average Return vs Wind Slip
ax = axes[0, 1]
ax.plot(vi_grouped['wind_slip'], vi_grouped['avg_return'], 's-', label='VI', color='tab:blue', linewidth=2, markersize=8)
ax.plot(mc_on_grouped['wind_slip'], mc_on_grouped['avg_return'], 'o-', label='MC On-Policy', color='tab:orange', linewidth=2, markersize=8)
if len(mc_off_grouped) > 0:
    ax.plot(mc_off_grouped['wind_slip'], mc_off_grouped['avg_return'], '^-', label='MC Off-Policy', color='tab:green', linewidth=2, markersize=8)
if len(sarsa_grouped) > 0:
    ax.plot(sarsa_grouped['wind_slip'], sarsa_grouped['avg_return'], 'x-', label='SARSA', color='tab:red', linewidth=2, markersize=8)
ax.set_xlabel('Wind Slip Probability', fontsize=11)
ax.set_ylabel('Average Return', fontsize=11)
ax.set_title('Average Return vs Wind Slip', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Average Steps vs Wind Slip
ax = axes[1, 0]
ax.plot(vi_grouped['wind_slip'], vi_grouped['avg_steps'], 's-', label='VI', color='tab:blue', linewidth=2, markersize=8)
ax.plot(mc_on_grouped['wind_slip'], mc_on_grouped['avg_steps'], 'o-', label='MC On-Policy', color='tab:orange', linewidth=2, markersize=8)
if len(mc_off_grouped) > 0:
    ax.plot(mc_off_grouped['wind_slip'], mc_off_grouped['avg_steps'], '^-', label='MC Off-Policy', color='tab:green', linewidth=2, markersize=8)
if len(sarsa_grouped) > 0:
    ax.plot(sarsa_grouped['wind_slip'], sarsa_grouped['avg_steps'], 'x-', label='SARSA', color='tab:red', linewidth=2, markersize=8)
ax.set_xlabel('Wind Slip Probability', fontsize=11)
ax.set_ylabel('Average Steps', fontsize=11)
ax.set_title('Average Steps to Delivery vs Wind Slip', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: VI Iterations (only for VI)
ax = axes[1, 1]
ax.plot(vi_grouped['wind_slip'], vi_grouped['avg_vi_iters'], 's-', label='VI Iterations', color='tab:purple', linewidth=2, markersize=8)
ax.set_xlabel('Wind Slip Probability', fontsize=11)
ax.set_ylabel('VI Iterations to Convergence', fontsize=11)
ax.set_title('VI Convergence Speed', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.5, 0.95, 'Note: MC uses fixed episodes, not convergence-based', 
        transform=ax.transAxes, ha='center', va='top', fontsize=9, style='italic', 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save the plot
plt.savefig('performance_plot.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as performance_plot.png")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print("\nValue Iteration:")
print(vi_grouped.to_string(index=False))
print("\nMC On-Policy:")
print(mc_on_grouped.to_string(index=False))
if len(mc_off_grouped) > 0:
    print("\nMC Off-Policy:")
    print(mc_off_grouped.to_string(index=False))
if len(sarsa_grouped) > 0:
    print("\nSARSA:")
    print(sarsa_grouped.to_string(index=False))
