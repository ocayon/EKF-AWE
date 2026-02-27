"""
Plot variations in v_app, alpha, and g_k across depower settings (u_dp).

Creates a 1-row, 3-column plot showing:
- Column 1: Apparent wind speed (v_app) vs u_dp
- Column 2: Angle of attack (alpha) vs u_dp
- Column 3: Steering gain coefficients (g_k_uni and g_k_dyn) vs u_dp
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style


set_plot_style()

# Data from varying-up dataset
# Based on statistics from circle_batch_analysis_varying_up.csv
u_dp = np.array([0.18, 0.250, 0.275, 0.300, 0.325, 0.350, 0.375, 0.400, 0.420])

# Apparent wind speed (m/s)
v_app = np.array([28.56, 44.09, 41.94, 38.22, 35.13, 31.84, 28.69, 26.02, 24.32])

# Angle of attack (degrees)
alpha = np.array([22.25, 4.61, 2.90, 2.64, 2.01, 1.52, 1.14, 0.83, 0.64])

# Steering gain - uniform model
g_k_uni = np.array([7.998, 10.064, 9.594, 8.767, 7.135, 6.399, 5.970, 5.537, 5.375])

# Steering gain - dynamic model
g_k_dyn = np.array([12.319, 11.953, 11.043, 9.606, 8.440, 7.653, 6.737, 6.045, 5.702])

# Create figure with 1 row, 3 columns
fig, axes = plt.subplots(1, 3, figsize=(9, 3))


msize = 3
linewidth = 1

# Column 1: Apparent wind speed
axes[0].plot(
    u_dp,
    v_app,
    "o-",
    color="black",
    markersize=3,
    linewidth=1,
    label="_nolegend_",
)
axes[0].plot(
    u_dp[-1],
    v_app[-1],
    "+",
    color="black",
    markersize=2 * msize,
    label=r"2019 $u_\mathrm{dp}$$=0.18$",
)
axes[0].plot(
    u_dp[0],
    v_app[0],
    "x",
    color="black",
    markersize=2 * msize,
    label=r"2025 $u_\mathrm{dp}$$=0.42$",
)
axes[0].set_xlabel(r"$u_\mathrm{dp}$ (-)")
axes[0].set_ylabel(r"$v_\mathrm{a}$ ($\mathrm{ms^{-1}}$)")
axes[0].legend(loc="best", frameon=True)
# axes[0].grid(True, alpha=0.3)


# Column 2: Angle of attack
axes[1].plot(u_dp, alpha, "o-", color="black", markersize=msize, linewidth=linewidth)
axes[1].plot(u_dp[0], alpha[0], "x", color="black", markersize=2 * msize)
axes[1].plot(u_dp[-1], alpha[-1], "+", color="black", markersize=2 * msize)
axes[1].set_xlabel(r"$u_\mathrm{dp}$ (-)")
axes[1].set_ylabel(r"$\alpha$ ($^\circ$)")
# axes[1].grid(True, alpha=0.3)

# Column 3: Steering gain (both models)
axes[2].plot(
    u_dp,
    g_k_uni,
    "o-",
    color="black",
    label=r"Sim. uniform",
    markersize=msize,
    linewidth=linewidth,
)
axes[2].plot(
    u_dp,
    g_k_dyn,
    "s--",
    color="C1",
    label=r"Sim. dynamic",
    markersize=msize,
    linewidth=linewidth,
)
axes[2].plot(u_dp[0], g_k_uni[0], "+", color="black", markersize=msize * 2)
axes[2].plot(u_dp[0], g_k_dyn[0], "+", color="C1", markersize=msize * 2)
axes[2].plot(u_dp[-1], g_k_uni[-1], "x", color="black", markersize=msize * 2)
axes[2].plot(u_dp[-1], g_k_dyn[-1], "x", color="C1", markersize=msize * 2)
axes[2].set_xlabel(r"$u_\mathrm{dp}$ (-)")
axes[2].set_ylabel(r"$g_\mathrm{k}$ (-)")
axes[2].legend(loc="best", frameon=True)
# axes[2].grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()

# Save figure
output_path = Path("results/plots_paper/udp_variations.pdf")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, bbox_inches="tight")
print(f"Saved {output_path}")

# plt.show()
