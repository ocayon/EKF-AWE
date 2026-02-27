"""
Create a 2-column plot of apparent wind speed (v_app) and steering gain (g_k) vs depower setting (u_dp).

Data are pulled from the circle batch CSVs (varying-up, 2019, 2025) using the same
statistics extraction approach as _load_plot_udp_variations.py, but only the
v_app and g_k panels are rendered here.
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from awes_ekf.plotting.color_palette import set_plot_style


def extract_statistics_arrays(
    circle_df: pd.DataFrame,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Extract u_dp, v_app (uni/dyn), alpha (uni/dyn), g_k (uni/dyn) arrays."""

    up_values = sorted(circle_df["up"].dropna().unique())

    u_dp = []
    v_app_uni = []
    v_app_dyn = []
    alpha_uni = []
    alpha_dyn = []
    g_k_uni_arr = []
    g_k_dyn_arr = []

    for up in up_values:
        df_up = circle_df[circle_df["up"] == up]

        # Uniform values (mean of columns)
        u_dp.append(up)
        v_app_uni.append(df_up["v_app"].mean())
        alpha_uni.append(df_up["aoa"].mean())

        # Calculate uniform g_k
        x_uniform = df_up["us"] * df_up["v_app"]
        y_uniform = df_up["yaw_rate"]
        finite_uniform = np.isfinite(x_uniform) & np.isfinite(y_uniform)

        if finite_uniform.sum() > 1:
            slope_uniform, _, _, _, _ = linregress(
                x_uniform[finite_uniform], y_uniform[finite_uniform]
            )
            g_k_uni_arr.append(slope_uniform)
        else:
            g_k_uni_arr.append(np.nan)

        # Dynamic v_app: average of va_3 to va_10
        dyn_va = []
        dyn_aoa = []
        for n in range(3, 11):
            va_col = f"va{n}"
            aoa_col = f"aoa{n}"
            if va_col in df_up.columns:
                dyn_va.extend(df_up[va_col].values)
            if aoa_col in df_up.columns:
                dyn_aoa.extend(df_up[aoa_col].values)

        if dyn_va:
            dyn_va_arr = np.array(dyn_va)
            finite_va = np.isfinite(dyn_va_arr)
            v_app_dyn.append(
                dyn_va_arr[finite_va].mean() if finite_va.sum() > 0 else np.nan
            )
        else:
            v_app_dyn.append(np.nan)

        if dyn_aoa:
            dyn_aoa_arr = np.array(dyn_aoa)
            finite_aoa = np.isfinite(dyn_aoa_arr)
            alpha_dyn.append(
                dyn_aoa_arr[finite_aoa].mean() if finite_aoa.sum() > 0 else np.nan
            )
        else:
            alpha_dyn.append(np.nan)

        # Calculate dynamic g_k
        dyn_x = []
        dyn_y = []
        for n in range(3, 11):
            usva_col = f"usva_{n}"
            yaw_col = f"yaw_rate_{n}"
            if usva_col in df_up.columns and yaw_col in df_up.columns:
                dyn_x.extend(df_up[usva_col].values)
                dyn_y.extend(df_up[yaw_col].values)

        if dyn_x and dyn_y:
            dyn_x = np.array(dyn_x)
            dyn_y = np.array(dyn_y)
            finite_dyn = np.isfinite(dyn_x) & np.isfinite(dyn_y)

            if finite_dyn.sum() > 1:
                slope_dyn, _, _, _, _ = linregress(dyn_x[finite_dyn], dyn_y[finite_dyn])
                g_k_dyn_arr.append(slope_dyn)
            else:
                g_k_dyn_arr.append(np.nan)
        else:
            g_k_dyn_arr.append(np.nan)

    return (
        np.array(u_dp),
        np.array(v_app_uni),
        np.array(v_app_dyn),
        np.array(alpha_uni),
        np.array(alpha_dyn),
        np.array(g_k_uni_arr),
        np.array(g_k_dyn_arr),
    )


def plot_udp_variations_2col(
    u_dp: np.ndarray,
    v_app_uni: np.ndarray,
    v_app_dyn: np.ndarray,
    g_k_uni: np.ndarray,
    g_k_dyn: np.ndarray,
    output_path: Path,
) -> None:
    """Plot 2 columns: v_app (uni/dyn) and g_k (uni/dyn)."""

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.6))

    msize = 3
    linewidth = 1

    # skipping the first entry
    # u_dp = u_dp[1:]
    # v_app_uni = v_app_uni[1:]
    # v_app_dyn = v_app_dyn[1:]
    # g_k_uni = g_k_uni[1:]
    # g_k_dyn = g_k_dyn[1:]

    # Column 1: Apparent wind speed (uniform and dynamic)
    axes[0].plot(
        u_dp,
        v_app_uni,
        "o-",
        color="black",
        markersize=msize,
        linewidth=linewidth,
        label=r"Uniform",
    )
    axes[0].plot(
        u_dp,
        v_app_dyn,
        "s--",
        color="C1",
        markersize=msize,
        linewidth=linewidth,
        label=r"Dynamic",
    )
    axes[0].set_xlabel(r"$u_\mathrm{dp}$ (-)")
    axes[0].set_ylabel(r"$v_\mathrm{a}$ ($\mathrm{ms^{-1}}$)")
    axes[0].legend(loc="best", frameon=True)

    # Column 2: Steering gain (both models)
    axes[1].plot(
        u_dp,
        g_k_uni,
        "o-",
        color="black",
        label=r"Sim. uniform",
        markersize=msize,
        linewidth=linewidth,
    )
    axes[1].plot(
        u_dp,
        g_k_dyn,
        "s--",
        color="C1",
        label=r"Sim. dynamic",
        markersize=msize,
        linewidth=linewidth,
    )
    axes[1].set_xlabel(r"$u_\mathrm{dp}$ (-)")
    axes[1].set_ylabel(r"$g_\mathrm{k}$ (-)")
    # axes[1].legend(loc="best", frameon=True)

    axes[0].set_ylim(20, 45)
    axes[1].set_ylim(5, 14)

    # Manual font size controls (similar to _plot_gk_2col_AWEC.py)
    for ax in axes:
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        ax.title.set_size(18)

    legend = axes[0].get_legend()
    for text in legend.get_texts():
        text.set_fontsize(18)

    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=15)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")


def main() -> None:
    set_plot_style()

    base_path = Path(__file__).resolve().parents[1] / "data"

    # Prefer the "complete" varying-up CSV if available, else fall back.
    varying_candidates = [
        base_path / "circle_batch_analysis_varying_up_complete.csv",
        base_path / "circle_batch_analysis_varying_up.csv",
    ]

    circle_data_varying = None
    for candidate in varying_candidates:
        if candidate.is_file():
            circle_data_varying = pd.read_csv(candidate)
            print(f"Loaded varying-up circle batch data from {candidate}")
            break
    if circle_data_varying is None:
        print("No varying-up circle batch CSV found; nothing to plot.")
        return

    circle_csv_path_2019 = base_path / "circle_batch_analysis_2019.csv"
    circle_csv_path_2025 = base_path / "circle_batch_analysis_2025.csv"

    circle_data_2019 = None
    circle_data_2025 = None

    if circle_csv_path_2019.is_file():
        circle_data_2019 = pd.read_csv(circle_csv_path_2019)
        # Filter 2019 data: only include points where up * v_app < 10 and yaw_rate < 120
        circle_data_2019 = circle_data_2019[
            circle_data_2019["us"] * circle_data_2019["v_app"] < 10
        ]
        circle_data_2019 = circle_data_2019[circle_data_2019["yaw_rate"] < 120]
        print(f"Loaded 2019 circle batch data from {circle_csv_path_2019}")
    else:
        print(f"2019 circle batch CSV not found: {circle_csv_path_2019}")

    if circle_csv_path_2025.is_file():
        circle_data_2025 = pd.read_csv(circle_csv_path_2025)
        print(f"Loaded 2025 circle batch data from {circle_csv_path_2025}")
    else:
        print(f"2025 circle batch CSV not found: {circle_csv_path_2025}")

    # Extract arrays from varying-up data
    (
        u_dp,
        v_app_uni,
        v_app_dyn,
        _alpha_uni,
        _alpha_dyn,
        g_k_uni,
        g_k_dyn,
    ) = extract_statistics_arrays(circle_data_varying)

    # Append 2019 data if available
    if circle_data_2019 is not None:
        (
            u_dp_2019,
            v_app_uni_2019,
            v_app_dyn_2019,
            _alpha_uni_2019,
            _alpha_dyn_2019,
            g_k_uni_2019,
            g_k_dyn_2019,
        ) = extract_statistics_arrays(circle_data_2019)
        u_dp = np.append(u_dp, u_dp_2019)
        v_app_uni = np.append(v_app_uni, v_app_uni_2019)
        v_app_dyn = np.append(v_app_dyn, v_app_dyn_2019)
        g_k_uni = np.append(g_k_uni, g_k_uni_2019)
        g_k_dyn = np.append(g_k_dyn, g_k_dyn_2019)

    # Append 2025 data if available
    if circle_data_2025 is not None:
        (
            u_dp_2025,
            v_app_uni_2025,
            v_app_dyn_2025,
            _alpha_uni_2025,
            _alpha_dyn_2025,
            g_k_uni_2025,
            g_k_dyn_2025,
        ) = extract_statistics_arrays(circle_data_2025)
        u_dp = np.append(u_dp, u_dp_2025)
        v_app_uni = np.append(v_app_uni, v_app_uni_2025)
        v_app_dyn = np.append(v_app_dyn, v_app_dyn_2025)
        g_k_uni = np.append(g_k_uni, g_k_uni_2025)
        g_k_dyn = np.append(g_k_dyn, g_k_dyn_2025)

    # Sort by u_dp to create continuous lines
    sort_indices = np.argsort(u_dp)
    u_dp = u_dp[sort_indices]
    v_app_uni = v_app_uni[sort_indices]
    v_app_dyn = v_app_dyn[sort_indices]
    g_k_uni = g_k_uni[sort_indices]
    g_k_dyn = g_k_dyn[sort_indices]

    output_path = Path("results/plots_paper/udp_variations_AWEC.pdf")
    plot_udp_variations_2col(
        u_dp,
        v_app_uni,
        v_app_dyn,
        g_k_uni,
        g_k_dyn,
        output_path,
    )


if __name__ == "__main__":
    main()
