"""
Plot variations in v_app, alpha, and g_k across depower settings (u_dp).

Creates a 1-row, 3-column plot showing:
- Column 1: Apparent wind speed (v_app) vs u_dp
- Column 2: Angle of attack (alpha) vs u_dp
- Column 3: Steering gain coefficients (g_k_uni and g_k_dyn) vs u_dp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
from pathlib import Path
from scipy.stats import linregress
from typing import Optional, List, Tuple, Dict
from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from awes_ekf.load_data.read_data import read_results
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style


def extract_statistics_arrays(
    circle_df: pd.DataFrame,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Extract arrays of u_dp, v_app (uni/dyn), alpha (uni/dyn), g_k (uni/dyn) from circle batch data.

    Parameters
    ----------
    circle_df : pd.DataFrame
        Circle batch data with columns: up, us, v_app, aoa, yaw_rate, turn_radius, usva_N, yaw_rate_N, va_N, aoa_N

    Returns
    -------
    tuple
        (u_dp, v_app_uni, v_app_dyn, alpha_uni, alpha_dyn, g_k_uni, g_k_dyn) as numpy arrays
    """
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


def plot_udp_variations(
    u_dp: np.ndarray,
    v_app_uni: np.ndarray,
    v_app_dyn: np.ndarray,
    alpha_uni: np.ndarray,
    alpha_dyn: np.ndarray,
    g_k_uni: np.ndarray,
    g_k_dyn: np.ndarray,
    output_path: str = "results/plots_paper/udp_variations.pdf",
) -> None:
    """
    Create a 1-row, 3-column plot of v_app, alpha, and g_k vs u_dp (both uniform and dynamic).

    Parameters
    ----------
    u_dp : np.ndarray
        Depower setting values (includes varying-up, 2019, and 2025 data combined)
    v_app_uni : np.ndarray
        Uniform apparent wind speed values (m/s)
    v_app_dyn : np.ndarray
        Dynamic apparent wind speed values (m/s)
    alpha_uni : np.ndarray
        Uniform angle of attack values (degrees)
    alpha_dyn : np.ndarray
        Dynamic angle of attack values (degrees)
    g_k_uni : np.ndarray
        Uniform steering gain coefficients
    g_k_dyn : np.ndarray
        Dynamic steering gain coefficients
    output_path : str, optional
        Path to save the output PDF
    """
    # Create figure with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.2))

    msize = 3
    linewidth = 1

    # Column 1: Apparent wind speed (uniform and dynamic)
    axes[0].plot(
        u_dp,
        v_app_uni,
        "o-",
        color="black",
        markersize=3,
        linewidth=1,
        label=r"Uniform",
    )
    axes[0].plot(
        u_dp,
        v_app_dyn,
        "s--",
        color="C1",
        markersize=3,
        linewidth=1,
        label=r"Dynamic",
    )
    # axes[0].plot(
    #     u_dp[-1],
    #     v_app_uni[-1],
    #     "+",
    #     color="black",
    #     markersize=2 * msize,
    # )
    # axes[0].plot(
    #     u_dp[-1],
    #     v_app_dyn[-1],
    #     "+",
    #     color="C1",
    #     markersize=2 * msize,
    # )
    # axes[0].plot(
    #     u_dp[0],
    #     v_app_uni[0],
    #     "x",
    #     color="black",
    #     markersize=2 * msize,
    # )
    # axes[0].plot(
    #     u_dp[0],
    #     v_app_dyn[0],
    #     "x",
    #     color="C1",
    #     markersize=2 * msize,
    # )
    axes[0].set_xlabel(r"$u_\mathrm{dp}$ (-)")
    axes[0].set_ylabel(r"$v_\mathrm{a}$ ($\mathrm{ms^{-1}}$)")
    axes[0].legend(loc="best", frameon=True)

    # Column 2: Angle of attack (uniform and dynamic)
    axes[1].plot(
        u_dp,
        alpha_uni,
        "o-",
        color="black",
        markersize=msize,
        linewidth=linewidth,
        label=r"Sim. uniform",
    )
    axes[1].plot(
        u_dp,
        alpha_dyn,
        "s--",
        color="C1",
        markersize=msize,
        linewidth=linewidth,
        label=r"Sim. dynamic",
    )
    # axes[1].plot(u_dp[0], alpha_uni[0], "x", color="black", markersize=2 * msize)
    # axes[1].plot(u_dp[0], alpha_dyn[0], "x", color="C1", markersize=2 * msize)
    # axes[1].plot(u_dp[-1], alpha_uni[-1], "+", color="black", markersize=2 * msize)
    # axes[1].plot(u_dp[-1], alpha_dyn[-1], "+", color="C1", markersize=2 * msize)
    axes[1].set_xlabel(r"$u_\mathrm{dp}$ (-)")
    axes[1].set_ylabel(r"$\alpha$ ($^\circ$)")
    # axes[1].legend(loc="best", frameon=True)

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
    # axes[2].plot(u_dp[0], g_k_uni[0], "x", color="black", markersize=msize * 2)
    # axes[2].plot(u_dp[0], g_k_dyn[0], "x", color="C1", markersize=msize * 2)
    # axes[2].plot(u_dp[-1], g_k_uni[-1], "+", color="black", markersize=msize * 2)
    # axes[2].plot(u_dp[-1], g_k_dyn[-1], "+", color="C1", markersize=msize * 2)
    axes[2].set_xlabel(r"$u_\mathrm{dp}$ (-)")
    axes[2].set_ylabel(r"$g_\mathrm{k}$ (-)")
    # axes[2].legend(loc="best", frameon=True)

    # x and y limits
    axes[0].set_ylim(15, 45)
    axes[1].set_ylim(0, 30)
    axes[2].set_ylim(4, 14)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path_obj, bbox_inches="tight")
    print(f"Saved {output_path_obj}")
    plt.close()


def print_statistics(circle_df: pd.DataFrame):
    """Print statistics for each up category in the circle batch data."""

    # Load all three datasets
    base_path = Path(__file__).resolve().parents[1] / "data"

    datasets = {
        "Varying up": circle_df,
        "2019": None,
        "2025": None,
    }

    csv_2019 = base_path / "circle_batch_analysis_2019.csv"
    csv_2025 = base_path / "circle_batch_analysis_2025.csv"

    if csv_2019.is_file():
        datasets["2019"] = pd.read_csv(csv_2019)

    if csv_2025.is_file():
        datasets["2025"] = pd.read_csv(csv_2025)

    # Print statistics for each dataset
    for dataset_name, df in datasets.items():
        if df is None or df.empty:
            print(f"\nNo data available for {dataset_name}")
            continue

        up_values = sorted(df["up"].dropna().unique())

        print("\n" + "=" * 80)
        print(f"STATISTICS BY DEPOWER (up) CATEGORY - {dataset_name.upper()}")
        print("=" * 80)

        for up in up_values:
            df_up = df[df["up"] == up]

            print(f"\nup = {up:.2f}")
            print("-" * 40)

            # Uniform g_k and R²
            x_uniform = df_up["us"] * df_up["v_app"]
            y_uniform = df_up["yaw_rate"]
            finite_uniform = np.isfinite(x_uniform) & np.isfinite(y_uniform)

            if finite_uniform.sum() > 1:
                slope_uniform, _, r_uniform, _, _ = linregress(
                    x_uniform[finite_uniform], y_uniform[finite_uniform]
                )
                r2_uniform = r_uniform**2
                print(f"  Uniform:  g_k = {slope_uniform:.3f}, R² = {r2_uniform:.3f}")
            else:
                print(f"  Uniform:  insufficient data")

            # Dynamic g_k and R²
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
                    slope_dyn, _, r_dyn, _, _ = linregress(
                        dyn_x[finite_dyn], dyn_y[finite_dyn]
                    )
                    r2_dyn = r_dyn**2
                    print(f"  Dynamic:  g_k = {slope_dyn:.3f}, R² = {r2_dyn:.3f}")
                else:
                    print(f"  Dynamic:  insufficient data")
            else:
                print(f"  Dynamic:  no data available")

            # Averaged values
            avg_v_app = df_up["v_app"].mean()
            avg_aoa = df_up["aoa"].mean()
            avg_turn_radius = df_up["turn_radius"].mean()

            print(f"  Avg v_app:        {avg_v_app:.2f}")
            print(f"  Avg AoA:          {avg_aoa:.2f}°")
            print(f"  Avg turn_radius:  {avg_turn_radius:.2f} m")

        print("\n" + "=" * 80 + "\n")


def prepare_dataset(
    *,
    year: str,
    month: str,
    day: str,
    kite_model: str,
    addition: str,
    time_range: Tuple[float, float],
    downsample_fraction: float = 0.5,
):
    """Load, filter, and downsample data for a given flight."""

    results, flight_data, _ = read_results(
        year, month, day, kite_model, addition=addition
    )

    time_mask = (results["time"] >= time_range[0]) & (results["time"] <= time_range[1])

    print(
        f"Total time range for {year}-{month}-{day}{addition}: {results['time'].min()} s to {results['time'].max()} s"
    )
    print(
        f"Using masked window: {time_range[0]} s to {time_range[1]} s for {kite_model}"
    )
    print(
        f"Loaded results file: results/{kite_model}/{kite_model}_{year}-{month}-{day}{addition}.h5"
    )
    print(f"Results columns: {list(results.columns)}")
    print(f"Flight data columns: {list(flight_data.columns)}")

    results = results.loc[time_mask].reset_index(drop=True)
    flight_data = flight_data.loc[time_mask].reset_index(drop=True)

    # Add yaw rate and delayed steering
    flight_data["kite_yaw_rate"] = flight_data["kite_yaw_rate_1"]
    flight_data["kcu_actual_steering_delay"] = np.roll(
        flight_data["kcu_actual_steering"], int(8)
    )

    # Downsample the data

    if year == "2019":
        downsample_fraction = 0.1
    else:
        downsample_fraction = 1

    downsampled_data = flight_data.sample(frac=downsample_fraction, random_state=42)
    downsampled_results = results.loc[downsampled_data.index]
    downsampled_data = downsampled_data[downsampled_data["powered"] == "powered"]
    downsampled_results = downsampled_results.loc[downsampled_data.index]

    # Calculate steering normalization
    max_abs_steering = flight_data["kcu_actual_steering"].abs().max()
    if max_abs_steering == 0:
        max_abs_steering = 1.0
    max_abs_us = flight_data["us"].abs().max()
    if max_abs_us == 0:
        max_abs_us = 1.0
    steering_norm = max_abs_steering / max_abs_us

    return downsampled_data, downsampled_results, steering_norm


def main():
    """Main execution function."""
    # Set plot style
    set_plot_style()

    # Load circle batch data for both years
    circle_data_2019 = None
    circle_data_2025 = None
    circle_data_varying = None

    circle_csv_path_2019 = (
        Path(__file__).resolve().parents[1] / "data" / "circle_batch_analysis_2019.csv"
    )
    circle_csv_path_2025 = (
        Path(__file__).resolve().parents[1] / "data" / "circle_batch_analysis_2025.csv"
    )
    circle_csv_path_varying = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "circle_batch_analysis_varying_up_complete.csv"
    )

    if circle_csv_path_2019.is_file():
        circle_data_2019 = pd.read_csv(circle_csv_path_2019)
        ## Filter 2019 data: only include points where up * v_app < 12
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

    if circle_csv_path_varying.is_file():
        circle_data_varying = pd.read_csv(circle_csv_path_varying)
        print(f"Loaded varying-up circle batch data from {circle_csv_path_varying}")
        # Print statistics for varying-up data
        print_statistics(circle_data_varying)
    else:
        print(f"Varying-up circle batch CSV not found: {circle_csv_path_varying}")

    colors = get_color_list()

    # dataset_configs = [
    #     {
    #         "title": "2019-10-08",
    #         "year": "2019",
    #         "month": "10",
    #         "day": "08",
    #         "kite_model": "v3",
    #         "addition": "_t26",
    #         "time_range": (1800.0, 9986.2),
    #     },
    #     {
    #         "title": "2025-10-09",
    #         "year": "2025",
    #         "month": "10",
    #         "day": "09",
    #         "kite_model": "v3",
    #         "addition": "",
    #         "time_range": (300.0, 1080.0),
    #     },
    # ]

    # prepared_datasets = []
    # for cfg in dataset_configs:
    #     cfg_no_title = {k: v for k, v in cfg.items() if k != "title"}
    #     ds_data, ds_results, steering_norm = prepare_dataset(**cfg_no_title)
    #     prepared_datasets.append(
    #         {
    #             "data": ds_data,
    #             "results": ds_results,
    #             "steering_norm": steering_norm,
    #             "title": cfg["title"],
    #             "year": cfg["year"],
    #         }
    #     )

    # # Build three-column figure
    # fig, axes = plt.subplots(1, 3, figsize=(11, 3.6667), sharey=True)

    # x_limits = []
    # for idx, dataset in enumerate(prepared_datasets):
    #     exclude_quadrant = dataset["title"] == "2025-10-09"
    #     # Show region labels only in column 2 (2025) and hide in column 1 (2019)
    #     hide_regions = dataset["title"] != "2025-10-09"

    #     # Select appropriate circle data based on year
    #     circle_data = (
    #         circle_data_2019 if dataset["year"] == "2019" else circle_data_2025
    #     )

    # plot_yaw_rate_vs_steering(
    #     dataset["data"],
    #     dataset["results"],
    #     colors,
    #     output_filename=None,
    #     bucket_type="regions",
    #     circle_df=circle_data,
    #     circle_ups=None,
    #     circle_colors=None,
    #     steering_norm=dataset["steering_norm"],
    #     ax=axes[idx],
    #     show_legend=True,
    #     exclude_quadrant_filter=exclude_quadrant,
    #     hide_region_labels=hide_regions,
    #     year=dataset["year"],
    # )
    # axes[idx].set_title(f"{dataset['title']} exp. \& sim.")
    # axes[idx].set_ylim(-120, 120)
    # x_limits.append(axes[idx].get_xlim())

    # Third column: simulation-only grouped by up
    # plot_simulation_by_up(
    #     circle_data_varying,
    #     colors,
    #     axes[2],
    #     show_legend=True,
    # )
    # axes[2].set_title("Varying $u_\\mathrm{dp}$ sim.")
    # axes[2].set_ylim(-120, 120)
    # x_limits.append(axes[2].get_xlim())

    # # Remove y-label from second and third columns
    # axes[1].set_ylabel("")
    # axes[2].set_ylabel("")

    # # Synchronize x-axis limits across both panels
    # if x_limits:
    #     xmin = min(lim[0] for lim in x_limits)
    #     xmax = max(lim[1] for lim in x_limits)
    #     xmin = 0
    #     xmax = 16
    #     ymin = 0
    #     ymax = 140
    #     for ax in axes:
    #         ax.set_xlim(xmin, xmax)
    #         ax.set_ylim(ymin, ymax)

    # fig.tight_layout()

    # output_path = Path("./results/plots_paper") / "yaw.pdf"
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # fig.savefig(output_path)
    # print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

    # Load circle batch data for varying-up, 2019, and 2025
    set_plot_style()
    base_path = Path(__file__).resolve().parents[1] / "data"
    circle_csv_path_varying = (
        base_path / "circle_batch_analysis_varying_up_complete.csv"
    )
    circle_csv_path_2019 = base_path / "circle_batch_analysis_2019.csv"
    circle_csv_path_2025 = base_path / "circle_batch_analysis_2025.csv"

    circle_data_varying = None
    circle_data_2019 = None
    circle_data_2025 = None

    if circle_csv_path_varying.is_file():
        circle_data_varying = pd.read_csv(circle_csv_path_varying)
        print(f"Loaded varying-up circle batch data from {circle_csv_path_varying}")
    else:
        print(f"Varying-up circle batch CSV not found: {circle_csv_path_varying}")

    if circle_csv_path_2019.is_file():
        circle_data_2019 = pd.read_csv(circle_csv_path_2019)
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
    if circle_data_varying is not None:
        u_dp, v_app_uni, v_app_dyn, alpha_uni, alpha_dyn, g_k_uni, g_k_dyn = (
            extract_statistics_arrays(circle_data_varying)
        )

        # Append 2019 data if available
        if circle_data_2019 is not None:
            (
                u_dp_2019,
                v_app_uni_2019,
                v_app_dyn_2019,
                alpha_uni_2019,
                alpha_dyn_2019,
                g_k_uni_2019,
                g_k_dyn_2019,
            ) = extract_statistics_arrays(circle_data_2019)
            u_dp = np.append(u_dp, u_dp_2019)
            v_app_uni = np.append(v_app_uni, v_app_uni_2019)
            v_app_dyn = np.append(v_app_dyn, v_app_dyn_2019)
            alpha_uni = np.append(alpha_uni, alpha_uni_2019)
            alpha_dyn = np.append(alpha_dyn, alpha_dyn_2019)
            g_k_uni = np.append(g_k_uni, g_k_uni_2019)
            g_k_dyn = np.append(g_k_dyn, g_k_dyn_2019)

        # Append 2025 data if available
        if circle_data_2025 is not None:
            (
                u_dp_2025,
                v_app_uni_2025,
                v_app_dyn_2025,
                alpha_uni_2025,
                alpha_dyn_2025,
                g_k_uni_2025,
                g_k_dyn_2025,
            ) = extract_statistics_arrays(circle_data_2025)
            u_dp = np.append(u_dp, u_dp_2025)
            v_app_uni = np.append(v_app_uni, v_app_uni_2025)
            v_app_dyn = np.append(v_app_dyn, v_app_dyn_2025)
            alpha_uni = np.append(alpha_uni, alpha_uni_2025)
            alpha_dyn = np.append(alpha_dyn, alpha_dyn_2025)
            g_k_uni = np.append(g_k_uni, g_k_uni_2025)
            g_k_dyn = np.append(g_k_dyn, g_k_dyn_2025)

        # Sort by u_dp to create continuous lines
        sort_indices = np.argsort(u_dp)
        u_dp = u_dp[sort_indices]
        v_app_uni = v_app_uni[sort_indices]
        v_app_dyn = v_app_dyn[sort_indices]
        alpha_uni = alpha_uni[sort_indices]
        alpha_dyn = alpha_dyn[sort_indices]
        g_k_uni = g_k_uni[sort_indices]
        g_k_dyn = g_k_dyn[sort_indices]

        # Create and save the plot with combined data
        plot_udp_variations(
            u_dp,
            v_app_uni,
            v_app_dyn,
            alpha_uni,
            alpha_dyn,
            g_k_uni,
            g_k_dyn,
            output_path="results/plots_paper/udp_variations_complete.pdf",
        )
