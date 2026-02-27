"""
Generalized plotting functions for yaw rate analysis.
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

from awes_ekf.load_data.read_data import read_results
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style


def convert_2019_depower_to_2025_updata(
    x19_depower,
    x19_pow=22.68,
    x19_dep=0.02,
    ld_0=1.098,
    delta_d=0.08,
    delta_ld_max=4.8,
    ld_2025_offset=0.2,
    ld_2025_scale=5.0,
):
    """
    Convert 2019 kcu_actual_depower (depower angle in degrees) to 2025-equivalent up_data.

    This conversion assumes both systems measure the same physical tape deployment (ld),
    using the paper's physics as the bridge:

    1. Map 2019 depower angle → paper's normalized up_paper ∈ [0,1]
    2. Calculate physical tape deployment: ld = ld_0 + δd·Δld,max·(1 - up_paper)
    3. Convert to 2025 notation: up_data_2025 = (ld - 0.2) / 5

    From paper (Table 2, Equations 1-2):
    - ld_0 = 1.098 m (baseline at fully powered)
    - δd = 0.08 (8% of max tape used, from 2019 campaign)
    - Δld,max = 4.8 m (maximum tape capacity)

    2025 affine relation: ld = 0.2 + 5·up_data_2025

    Verification:
    - At powered (up_paper=1): ld=1.098m → up_data_2025=0.1796
    - At depowered (up_paper=0): ld=1.482m → up_data_2025=0.2564
    """
    x19_depower = np.asarray(x19_depower)

    # Step 1: Normalize 2019 depower to paper's up_paper ∈ [0, 1]
    # up_paper = 1 at fully powered, 0 at fully depowered
    up_paper_2019 = np.clip((x19_depower - x19_dep) / (x19_pow - x19_dep), 0, 1)

    # Step 2: Calculate physical tape deployment using paper's physics
    # ld = ld_0 + δd·Δld,max·(1 - up_paper)
    ld_2019 = ld_0 + delta_d * delta_ld_max * (1.0 - up_paper_2019)

    # Step 3: Convert to 2025 notation
    # From ld = 0.2 + 5·up_data_2025, solve for up_data_2025
    up_data_2025 = (ld_2019 - ld_2025_offset) / ld_2025_scale

    return up_data_2025


def plot_yaw_rate_vs_steering(
    downsampled_data,
    downsampled_results,
    colors,
    output_filename: Optional[str],
    bucket_type: str = "regions",
    bucket_variable: Optional[str] = None,
    bucket_ranges: Optional[List[Tuple[float, float, str, any]]] = None,
    circle_df: Optional[object] = None,
    circle_ups: Optional[List] = None,
    circle_colors: Optional[List] = None,
    steering_norm: float = 1.0,
    ax: Optional[Axes] = None,
    show_legend: bool = True,
    exclude_quadrant_filter: bool = False,
    y_exclude_threshold_deg: float = 90.0,
    hide_region_labels: bool = False,
    year: str = "2019",
):
    """
    Create a generalized yaw rate vs steering*windspeed plot.

    Parameters
    ----------
    downsampled_data : DataFrame
        Flight data with steering inputs and yaw rate
    downsampled_results : DataFrame
        EKF results with apparent windspeed
    colors : list
        Color palette for plotting
    output_filename : str, optional
        Path to save the output PDF when creating a standalone figure
    bucket_type : str, optional
        Type of bucketing: 'regions' (straight/left/right) or 'continuous' (variable-based buckets)
    bucket_variable : str, optional
        Column name in downsampled_data to bucket on (e.g., 'tether_length', 'kcu_actual_depower')
        Required if bucket_type='continuous'
    bucket_ranges : list of tuples, optional
        List of (low, high, label, color) for continuous buckets
        Required if bucket_type='continuous'
    circle_df : DataFrame, optional
        Batch circle data to overlay
    circle_ups : list, optional
        List of up values from circle data
    circle_colors : list, optional
        Colors for circle data points
    steering_norm : float, optional
        Normalization factor for steering (max_abs_steering / max_abs_us)
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    show_legend : bool, optional
        Whether to draw the legend on the provided axis.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    # Prepare x variables
    x_no_delay_kcu = -downsampled_data["kcu_actual_steering"] / 100
    x_no_delay_us = -downsampled_data["us"]

    x_kcu = x_no_delay_kcu * downsampled_results["kite_apparent_windspeed"]
    x_us = x_no_delay_us * downsampled_results["kite_apparent_windspeed"]
    y = downsampled_data["kite_yaw_rate"]
    y_deg = y * 180 / np.pi

    # Quadrant filter (optional) for 2025 data: mismatched sign/high magnitude points
    mask_excluded = np.zeros_like(y_deg, dtype=bool)
    if exclude_quadrant_filter:
        mask_excluded = ((y_deg > y_exclude_threshold_deg) & (x_kcu < 0)) | (
            (y_deg < -y_exclude_threshold_deg) & (x_kcu > 0)
        )
    mask_include = ~mask_excluded

    # Finite mask for regression (exclude circle batch overlays by construction)
    finite_mask = np.isfinite(x_kcu) & np.isfinite(y_deg) & mask_include

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        created_fig = True
    else:
        fig = ax.figure

    if bucket_type == "regions":
        # Define regions based on steering thresholds
        upper_threshold = 0.08
        lower_threshold = -0.06
        lower_threshold_us = lower_threshold / (steering_norm / 100.0)
        upper_threshold_us = upper_threshold / (steering_norm / 100.0)

        mask_straight_kcu = mask_include & x_no_delay_kcu.between(
            lower_threshold, upper_threshold
        )
        mask_left_kcu = mask_include & (x_no_delay_kcu < lower_threshold)
        mask_right_kcu = mask_include & (x_no_delay_kcu > upper_threshold)

        mask_straight_us = mask_include & x_no_delay_us.between(
            lower_threshold_us, upper_threshold_us
        )
        mask_left_us = mask_include & (x_no_delay_us < lower_threshold_us)
        mask_right_us = mask_include & (x_no_delay_us > upper_threshold_us)

        # Plot regions for kcu panel
        label_straight = None if hide_region_labels else "Experimental straight"
        label_left = None if hide_region_labels else "Left Turn"
        label_left = None
        label_right = None if hide_region_labels else "Experimental right turn"

        ax.scatter(
            x_kcu[mask_straight_kcu],
            y_deg[mask_straight_kcu],
            color=colors[1],
            alpha=0.4,
            marker=".",
            label=label_straight,
        )
        ax.scatter(
            x_kcu[mask_left_kcu],
            y_deg[mask_left_kcu],
            color=colors[2],
            marker=".",
            alpha=0.4,
            label=label_left,
        )
        ax.scatter(
            x_kcu[mask_right_kcu],
            y_deg[mask_right_kcu],
            color=colors[3],
            marker=".",
            alpha=0.4,
            label=label_right,
        )

    elif bucket_type == "continuous":
        if bucket_variable is None or bucket_ranges is None:
            raise ValueError(
                "bucket_variable and bucket_ranges required for continuous bucketing"
            )

        bucket_data = downsampled_data[bucket_variable]

        for low, high, label, color in bucket_ranges:
            # Create mask for this bucket
            if high >= 600.0:  # Special case for last bucket
                mask_bucket = (bucket_data >= low) & (bucket_data <= high)
            else:
                mask_bucket = (bucket_data >= low) & (bucket_data < high)

            # Process kcu panel
            x_vals_kcu = x_kcu[mask_bucket & mask_include]
            y_vals_kcu = y_deg[mask_bucket & mask_include]
            finite_kcu = np.isfinite(x_vals_kcu) & np.isfinite(y_vals_kcu)
            x_vals_kcu = x_vals_kcu[finite_kcu]
            y_vals_kcu = y_vals_kcu[finite_kcu]
            label_kcu = label

            if len(x_vals_kcu) > 1:
                slope_kcu, intercept_kcu, r_kcu, _, _ = linregress(
                    x_vals_kcu, y_vals_kcu
                )
                label_kcu = (
                    f"{label} ($g_\\textrm{{k}}={slope_kcu:.3f}$, R$^2$={r_kcu**2:.2f})"
                )
                x_line_kcu = np.linspace(x_vals_kcu.min(), x_vals_kcu.max(), 100)
                y_line_kcu = slope_kcu * x_line_kcu + intercept_kcu
                ax.plot(
                    x_line_kcu,
                    y_line_kcu,
                    color=color,
                    linestyle="--",
                    alpha=0.9,
                    label="_nolegend_",
                )

            ax.scatter(
                x_vals_kcu,
                y_vals_kcu,
                color=color,
                alpha=0.4,
                marker=".",
                label=label_kcu,
            )

    # Add circle batch data if provided (uniform and dynamic simulations)
    if circle_df is not None and not circle_df.empty:
        # Plot uniform simulation (hollow circles)
        x_uniform = circle_df["us"] * circle_df["v_app"]
        y_uniform = circle_df["yaw_rate"]
        # Filter for finite values
        finite_uniform = np.isfinite(x_uniform) & np.isfinite(y_uniform)
        ax.scatter(
            x_uniform[finite_uniform],
            y_uniform[finite_uniform],
            s=60,
            alpha=0.8,
            color="black",
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=1.5,
            label="Uniform simulation",
        )

        # Plot dynamic simulation (dots from usva_N and yaw_rate_N)
        for n in range(3, 11):  # usva_3 to usva_10
            usva_col = f"usva_{n}"
            yaw_col = f"yaw_rate_{n}"
            if usva_col in circle_df.columns and yaw_col in circle_df.columns:
                x_dyn = circle_df[usva_col]
                y_dyn = circle_df[yaw_col]
                finite_dyn = np.isfinite(x_dyn) & np.isfinite(y_dyn)
                ax.scatter(
                    x_dyn[finite_dyn],
                    y_dyn[finite_dyn],
                    s=20,
                    alpha=0.8,
                    color="black",
                    marker=".",
                    label="Dynamic simulation" if n == 3 else "",  # Only label once
                )

        # Fit through all simulated data (uniform + dynamic)
        x_sim_all = []
        y_sim_all = []

        # Add uniform points
        x_sim_all.extend(x_uniform[finite_uniform].values)
        y_sim_all.extend(y_uniform[finite_uniform].values)

        # Add dynamic points
        for n in range(3, 11):  # usva_3 to usva_10
            usva_col = f"usva_{n}"
            yaw_col = f"yaw_rate_{n}"
            if usva_col in circle_df.columns and yaw_col in circle_df.columns:
                x_dyn = circle_df[usva_col]
                y_dyn = circle_df[yaw_col]
                finite_dyn = np.isfinite(x_dyn) & np.isfinite(y_dyn)
                x_sim_all.extend(x_dyn[finite_dyn].values)
                y_sim_all.extend(y_dyn[finite_dyn].values)

        # Perform linear regression on all simulated data
        if len(x_sim_all) > 1:
            x_sim_all = np.array(x_sim_all)
            y_sim_all = np.array(y_sim_all)
            sim_slope, sim_intercept, sim_r, _, _ = linregress(x_sim_all, y_sim_all)
            sim_r2 = sim_r**2
            x_sim_line = np.linspace(x_sim_all.min(), x_sim_all.max(), 200)
            y_sim_line = sim_slope * x_sim_line + sim_intercept
            ax.plot(
                x_sim_line,
                y_sim_line,
                color="gray",
                linestyle="-",
                linewidth=2.0,
                alpha=0.7,
                label=f"Sim fit $g_\\textrm{{k}}={sim_slope:.3f}$, R$^2$={sim_r2:.2f}",
            )

    # Plot excluded quadrant-mismatch points (still visible, not used in fit)
    if mask_excluded.any():
        ax.scatter(
            x_kcu[mask_excluded],
            y_deg[mask_excluded],
            color="0.7",
            alpha=0.5,
            marker=".",
            label="_nolegend_",
        )

    # Global linear fit on experimental points (not circle markers)
    x_fit = x_kcu[finite_mask]
    y_fit = y_deg[finite_mask]
    if len(x_fit) > 1:
        slope, intercept, r, _, _ = linregress(x_fit, y_fit)
        r2 = r**2
        x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Fit $g_\\textrm{{k}}={slope:.3f}$, R$^2$={r2:.2f}",
        )

    # Set labels
    ax.set_xlabel(r"$u_\mathrm{dp}v_\mathrm{a}$ (-)")
    ax.set_ylabel(r"$\dot{\psi}\;(^\circ\,\mathrm{s^{-1}})$")
    if show_legend:
        ax.legend(frameon=True)

    # Save figure only when we created it here
    if output_filename and created_fig:
        fig.tight_layout()
        fig.savefig(output_filename)
        print(f"Saved {output_filename}")

    return fig


def plot_simulation_by_up(
    circle_df: pd.DataFrame,
    colors,
    ax: Axes,
    show_legend: bool = True,
):
    """Plot uniform and dynamic simulations grouped by depower ``up`` value."""

    if circle_df is None or circle_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return ax.figure

    up_values = sorted(circle_df["up"].dropna().unique())
    color_cycle = cycle(colors)

    handles = []
    labels = []

    for up in up_values:
        color = next(color_cycle)
        df_up = circle_df[circle_df["up"] == up]

        # Uniform simulation for this up
        x_uniform = df_up["us"] * df_up["v_app"]
        y_uniform = df_up["yaw_rate"]
        finite_uniform = np.isfinite(x_uniform) & np.isfinite(y_uniform)
        scatter_uniform = ax.scatter(
            x_uniform[finite_uniform],
            y_uniform[finite_uniform],
            s=60,
            alpha=0.8,
            color=color,
            marker="o",
            facecolors="none",
            edgecolors=color,
            linewidths=1.2,
            label="_nolegend_",
        )

        # Dynamic simulations across harmonics for this up
        dyn_x = []
        dyn_y = []
        for n in range(3, 11):
            usva_col = f"usva_{n}"
            yaw_col = f"yaw_rate_{n}"
            if usva_col in df_up.columns and yaw_col in df_up.columns:
                dyn_x.append(df_up[usva_col])
                dyn_y.append(df_up[yaw_col])

        scatter_dyn = None
        if dyn_x and dyn_y:
            dyn_x = pd.concat(dyn_x, ignore_index=True)
            dyn_y = pd.concat(dyn_y, ignore_index=True)
            finite_dyn = np.isfinite(dyn_x) & np.isfinite(dyn_y)
            scatter_dyn = ax.scatter(
                dyn_x[finite_dyn],
                dyn_y[finite_dyn],
                s=20,
                alpha=0.9,
                color=color,
                marker=".",
                label="_nolegend_",
            )
            x_all = np.concatenate([x_uniform[finite_uniform], dyn_x[finite_dyn]])
            y_all = np.concatenate([y_uniform[finite_uniform], dyn_y[finite_dyn]])
        else:
            x_all = x_uniform[finite_uniform].to_numpy()
            y_all = y_uniform[finite_uniform].to_numpy()

        # Linear fit per up group (uniform + dynamic)
        if len(x_all) > 1:
            slope, intercept, _, _, _ = linregress(x_all, y_all)
            x_line = np.linspace(x_all.min(), x_all.max(), 200)
            y_line = slope * x_line + intercept
            ax.plot(
                x_line,
                y_line,
                color=color,
                linestyle="-",
                linewidth=1.5,
                alpha=0.9,
                label="_nolegend_",
            )
            label = f"$u_\\textrm{{dp}}={up:.2f}$, $g_\\textrm{{k}}={slope:.1f}$"
        else:
            label = f"$u_\\textrm{{dp}}={up:.2f}$"

        handle = (
            (scatter_uniform,)
            if scatter_dyn is None
            else (scatter_uniform, scatter_dyn)
        )
        handles.append(handle)
        labels.append(label)

    ax.set_xlabel(r"$u_\mathrm{dp}v_\mathrm{a}$ (-)")
    if show_legend:
        ax.legend(
            handles=handles,
            labels=labels,
            frameon=True,
            handler_map={tuple: HandlerTuple(ndivide=None)},
        )

    return ax.figure


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
        / "circle_batch_analysis_varying_up.csv"
    )

    if circle_csv_path_2019.is_file():
        circle_data_2019 = pd.read_csv(circle_csv_path_2019)
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
    else:
        print(f"Varying-up circle batch CSV not found: {circle_csv_path_varying}")

    colors = get_color_list()

    dataset_configs = [
        {
            "title": "2019-10-08",
            "year": "2019",
            "month": "10",
            "day": "08",
            "kite_model": "v3",
            "addition": "_t26",
            "time_range": (1800.0, 9986.2),
        },
        {
            "title": "2025-10-09",
            "year": "2025",
            "month": "10",
            "day": "09",
            "kite_model": "v3",
            "addition": "",
            "time_range": (300.0, 1080.0),
        },
    ]

    prepared_datasets = []
    for cfg in dataset_configs:
        cfg_no_title = {k: v for k, v in cfg.items() if k != "title"}
        ds_data, ds_results, steering_norm = prepare_dataset(**cfg_no_title)
        prepared_datasets.append(
            {
                "data": ds_data,
                "results": ds_results,
                "steering_norm": steering_norm,
                "title": cfg["title"],
                "year": cfg["year"],
            }
        )

    # Build three-column figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

    x_limits = []
    for idx, dataset in enumerate(prepared_datasets):
        exclude_quadrant = dataset["title"] == "2025-10-09"
        hide_regions = exclude_quadrant  # simplify legend on col 2

        # Select appropriate circle data based on year
        circle_data = (
            circle_data_2019 if dataset["year"] == "2019" else circle_data_2025
        )

        plot_yaw_rate_vs_steering(
            dataset["data"],
            dataset["results"],
            colors,
            output_filename=None,
            bucket_type="regions",
            circle_df=circle_data,
            circle_ups=None,
            circle_colors=None,
            steering_norm=dataset["steering_norm"],
            ax=axes[idx],
            show_legend=True,
            exclude_quadrant_filter=exclude_quadrant,
            hide_region_labels=hide_regions,
            year=dataset["year"],
        )
        axes[idx].set_title(dataset["title"])
        axes[idx].set_ylim(-120, 120)
        x_limits.append(axes[idx].get_xlim())

    # Third column: simulation-only grouped by up
    plot_simulation_by_up(
        circle_data_varying,
        colors,
        axes[2],
        show_legend=True,
    )
    axes[2].set_title("Sim varying up")
    axes[2].set_ylim(-120, 120)
    x_limits.append(axes[2].get_xlim())

    # Remove y-label from second and third columns
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    # Synchronize x-axis limits across both panels
    if x_limits:
        xmin = min(lim[0] for lim in x_limits)
        xmax = max(lim[1] for lim in x_limits)
        xmin = 0
        xmax = 10
        ymin = 0
        ymax = 140
        for ax in axes:
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    fig.tight_layout()

    output_path = Path("./results/plots_paper") / "yaw.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
