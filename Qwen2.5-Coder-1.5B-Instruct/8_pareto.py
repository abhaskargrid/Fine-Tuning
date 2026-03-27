import matplotlib.pyplot as plt

# Your NEW Unseen Dataset Data
n_values = [5, 10, 20, 50]
accuracies = [65.0, 65.0, 75.0, 80.0]
wall_clock_times = [148.40, 340.35, 339.38, 724.74]

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Accuracy (Left Y-Axis)
color = 'tab:blue'
ax1.set_xlabel('Number of Samples (N)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', color=color, fontsize=12, fontweight='bold')
ax1.plot(n_values, accuracies, marker='o', color=color, linewidth=2, markersize=8, label="Accuracy")
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(40, 100) # Adjusted Y-axis for the new data range
ax1.grid(True, linestyle='--', alpha=0.6)

# Create a second Y-axis for Time (Right Y-Axis)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Total Wall-Clock Time (Seconds)', color=color, fontsize=12, fontweight='bold')
ax2.plot(n_values, wall_clock_times, marker='s', color=color, linewidth=2, markersize=8, linestyle='dashed', label="Time (s)")
ax2.tick_params(axis='y', labelcolor=color)

# Highlight the Optimal N
plt.axvline(x=20, color='green', linestyle=':', linewidth=2, label="Optimal N (Mac)")
plt.annotate('Optimal Balance\n(Max Acc, Efficient Time)', xy=(20, 241.00), xytext=(22, 400),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

# Title and Layout
plt.title('Best-of-N Pareto Curve (Unseen Data): Accuracy vs. Wall-Clock Time', fontsize=14, fontweight='bold')
fig.tight_layout()

# Save the plot
plt.savefig("pareto_curve_unseen.png", dpi=300)
print("Pareto curve saved successfully as 'pareto_curve_unseen.png'!")