import matplotlib.pyplot as plt

# Your Official 50-Problem Unseen Dataset Data
n_values = [5, 10, 20, 50]
accuracies = [60.0, 62.0, 70.0, 78.0]
wall_clock_times = [449.63, 906.61, 721.30, 1777.20]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Accuracy (Left Y-Axis)
color = 'tab:blue'
ax1.set_xlabel('Number of Samples (N)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', color=color, fontsize=12, fontweight='bold')
ax1.plot(n_values, accuracies, marker='o', color=color, linewidth=2, markersize=8, label="Accuracy")
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(40, 100)
ax1.grid(True, linestyle='--', alpha=0.6)

# Create a second Y-axis for Time (Right Y-Axis)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Total Wall-Clock Time (Seconds)', color=color, fontsize=12, fontweight='bold')
ax2.plot(n_values, wall_clock_times, marker='s', color=color, linewidth=2, markersize=8, linestyle='dashed', label="Time (s)")
ax2.tick_params(axis='y', labelcolor=color)

# Highlight the Optimal N
plt.axvline(x=20, color='green', linestyle=':', linewidth=2, label="Optimal N (Mac)")
plt.annotate('Optimal Balance\n(High Acc, Fast Early Stopping)', xy=(20, 721.30), xytext=(22, 1200),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

# Title and Layout
plt.title('Best-of-N Pareto Curve: Accuracy vs. Wall-Clock Time (50 Unseen Problems)', fontsize=13, fontweight='bold')
fig.tight_layout()

# Save the plot
plt.savefig("pareto_curve_2.png", dpi=300)
print("Pareto curve saved successfully as 'final_pareto_curve.png'!")