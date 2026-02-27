"""
Two-column comparison plots of 2019 vs 2025 flight statistics.
Produces statistics_all.pdf and statistics_right_turn.pdf.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from awes_ekf.load_data.read_data import read_results
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style
from awes_ekf.setup.settings import SimulationConfig
from awes_ekf.setup.kite import PointMassEKF
from awes_ekf.setup.kcu import KCU


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

    Note: If observed 2025 range differs from [0.18, 0.26], it may indicate:
    - Different δd value used in 2025 operations
    - Different ld_0 baseline configuration
    - Different physical measurements (not the same tape)

    Parameters:
    -----------
    x19_depower : array-like
        2019 depower angle in degrees
    x19_pow : float
        2019 depower angle at fully powered state (degrees)
    x19_dep : float
        2019 depower angle at fully depowered state (degrees)
    ld_0 : float
        Baseline tape deployment at fully powered (meters)
    delta_d : float
        Fraction of max tape capacity used (dimensionless)
    delta_ld_max : float
        Maximum tape capacity (meters)
    ld_2025_offset : float
        2025 affine relation offset (meters)
    ld_2025_scale : float
        2025 affine relation scale (meters)

    Returns:
    --------
    up_data_2025 : array-like
        2025-equivalent up_data values
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


def load_and_process_data(
    year, month, day, kite_model, addition, time_range, downsample_frac
):
    """Load and process flight data for a given year."""
    results, flight_data, config_data = read_results(
        year, month, day, kite_model, addition=addition
    )

    # Time-based filtering
    time_mask = (results["time"] >= time_range[0]) & (results["time"] <= time_range[1])
    results = results.loc[time_mask].reset_index(drop=True)
    flight_data = flight_data.loc[time_mask].reset_index(drop=True)

    # Create system components
    simConfig = SimulationConfig(**config_data["simulation_parameters"])
    kite = PointMassEKF(simConfig, **config_data["kite"])
    kcu = KCU(**config_data["kcu"])

    # Prepare yaw rate
    flight_data["kite_yaw_rate"] = flight_data["kite_yaw_rate_1"]
    flight_data["kcu_actual_steering_delay"] = np.roll(
        flight_data["kcu_actual_steering"], int(8)
    )

    # Downsample and filter to powered mode
    downsampled_data = flight_data.sample(frac=downsample_frac, random_state=42)
    downsampled_results = results.loc[downsampled_data.index]
    downsampled_data = downsampled_data[downsampled_data["powered"] == "powered"]
    downsampled_results = downsampled_results.loc[downsampled_data.index]

    downsampled_sorted = downsampled_data.sort_values("time")
    downsampled_results_sorted = downsampled_results.loc[downsampled_sorted.index]

    return (
        downsampled_data,
        downsampled_results,
        downsampled_sorted,
        downsampled_results_sorted,
    )


def get_multi_row_signals(
    downsampled_data,
    downsampled_results_sorted,
    downsampled_sorted,
):
    """Define the multi-row signal list."""
    signals = [
        (
            "tether_length",
            (
                downsampled_sorted["tether_length"]
                if "tether_length" in downsampled_sorted.columns
                else None
            ),
            r"$\mathrm{tether\_length}\;(\mathrm{m})$",
            r"\mathrm{m}",
        ),
        (
            "kite_elevation",
            (
                np.degrees(downsampled_sorted["kite_elevation"])
                if "kite_elevation" in downsampled_sorted.columns
                else None
            ),
            r"$\mathrm{kite\_elevation}\;(^\circ)$",
            r"^\circ",
        ),
        (
            "wind_speed_horizontal",
            (
                downsampled_results_sorted["wind_speed_horizontal"]
                if "wind_speed_horizontal" in downsampled_results_sorted.columns
                else None
            ),
            r"$\mathrm{wind\_speed\_horizontal}\;(\mathrm{m\,s^{-1}})$",
            r"\mathrm{m\,s^{-1}}",
        ),
        (
            "wing_angle_of_attack",
            (
                downsampled_results_sorted["wing_angle_of_attack"]
                if "wing_angle_of_attack" in downsampled_results_sorted.columns
                else None
            ),
            r"$\mathrm{wing\_angle\_of\_attack}\;(^\circ)$",
            r"^\circ",
        ),
        (
            "kite_apparent_windspeed",
            (
                downsampled_results_sorted["kite_apparent_windspeed"]
                if "kite_apparent_windspeed" in downsampled_results_sorted.columns
                else None
            ),
            r"$\mathrm{kite\_apparent\_windspeed}\;(\mathrm{m\,s^{-1}})$",
            r"\mathrm{m\,s^{-1}}",
        ),
        (
            "radius_turn",
            (
                downsampled_results_sorted["radius_turn"]
                if "radius_turn" in downsampled_results_sorted.columns
                else None
            ),
            r"$\mathrm{radius\_turn}\;(\mathrm{m})$",
            r"\mathrm{m}",
        ),
        (
            "kcu_actual_depower",
            (
                downsampled_sorted["kcu_actual_depower"]
                if "kcu_actual_depower" in downsampled_sorted.columns
                else None
            ),
            r"$\mathrm{kcu\_actual\_depower}$",
            r"",
        ),
        (
            "kcu_actual_steering",
            (
                downsampled_sorted["kcu_actual_steering"]
                if "kcu_actual_steering" in downsampled_sorted.columns
                else None
            ),
            r"$\mathrm{kcu\_actual\_steering}\;(\%)$",
            r"\%",
        ),
    ]

    return [entry for entry in signals if entry[1] is not None]


def plot_multi_row_comparison(
    data_2019,
    results_2019,
    sorted_2019,
    results_sorted_2019,
    data_2025,
    results_2025,
    sorted_2025,
    results_sorted_2025,
    colors,
    output_filename,
):
    """Create 2-column multi-row comparison plot (all data)."""

    # Get signal definitions
    signals_2019 = get_multi_row_signals(data_2019, results_sorted_2019, sorted_2019)
    signals_2025 = get_multi_row_signals(data_2025, results_sorted_2025, sorted_2025)

    # Use signals that exist in both years
    signal_names_2019 = {s[0] for s in signals_2019}
    signal_names_2025 = {s[0] for s in signals_2025}
    common_names = signal_names_2019 & signal_names_2025

    signals_2019 = [s for s in signals_2019 if s[0] in common_names]
    signals_2025 = [s for s in signals_2025 if s[0] in common_names]

    n_rows = len(signals_2019)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3 * n_rows), sharex=False)

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (sig_2019, sig_2025) in enumerate(zip(signals_2019, signals_2025)):
        name_19, series_19, label_19, unit_19 = sig_2019
        name_25, series_25, label_25, unit_25 = sig_2025

        # 2019 column (left)
        ax_left = axes[row_idx, 0]
        mean_val = float(series_19.mean())
        min_val = float(series_19.min())
        max_val = float(series_19.max())

        ax_left.plot(
            sorted_2019["time"],
            series_19,
            color=colors[0],
            marker=".",
            linestyle="None",
            alpha=0.6,
            label=label_19,
        )
        ax_left.axhline(
            mean_val,
            color=colors[1],
            linestyle="--",
            label=rf"$\mathrm{{mean}} = {mean_val:.3f}\,{unit_19}$",
        )
        ax_left.axhline(
            min_val,
            color=colors[2],
            linestyle=":",
            label=rf"$\mathrm{{min}} = {min_val:.3f}\,{unit_19}$",
        )
        ax_left.axhline(
            max_val,
            color=colors[3],
            linestyle=":",
            label=rf"$\mathrm{{max}} = {max_val:.3f}\,{unit_19}$",
        )
        ax_left.set_ylabel(label_19)
        ax_left.legend(frameon=True, loc="upper left", framealpha=1.0, fontsize=8)
        ax_left.set_title("2019" if row_idx == 0 else "")

        # 2025 column (right)
        ax_right = axes[row_idx, 1]
        mean_val = float(series_25.mean())
        min_val = float(series_25.min())
        max_val = float(series_25.max())

        ax_right.plot(
            sorted_2025["time"],
            series_25,
            color=colors[0],
            marker=".",
            linestyle="None",
            alpha=0.6,
            label=label_25,
        )
        ax_right.axhline(
            mean_val,
            color=colors[1],
            linestyle="--",
            label=rf"$\mathrm{{mean}} = {mean_val:.3f}\,{unit_25}$",
        )
        ax_right.axhline(
            min_val,
            color=colors[2],
            linestyle=":",
            label=rf"$\mathrm{{min}} = {min_val:.3f}\,{unit_25}$",
        )
        ax_right.axhline(
            max_val,
            color=colors[3],
            linestyle=":",
            label=rf"$\mathrm{{max}} = {max_val:.3f}\,{unit_25}$",
        )
        ax_right.set_ylabel(label_25)
        ax_right.legend(frameon=True, loc="upper left", framealpha=1.0, fontsize=8)
        ax_right.set_title("2025" if row_idx == 0 else "")

    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(output_filename, dpi=150)
    print(f"Saved {output_filename}")
    plt.close(fig)


def plot_turn_comparison(
    data_2019,
    results_2019,
    sorted_2019,
    results_sorted_2019,
    data_2025,
    results_2025,
    sorted_2025,
    results_sorted_2025,
    colors,
    output_filename,
):
    """Create 2-column multi-row comparison plot (right turns only)."""

    # Define steering thresholds
    upper_threshold_19 = 0.08 * 100  # Convert to % scale for 2019 data
    upper_threshold_25 = 0.08 * 100

    # Get masks for right turns
    x_full_kcu_sorted_19 = -sorted_2019["kcu_actual_steering_delay"] / 100
    mask_right_19 = x_full_kcu_sorted_19 > upper_threshold_19 / 100

    x_full_kcu_sorted_25 = -sorted_2025["kcu_actual_steering_delay"] / 100
    mask_right_25 = x_full_kcu_sorted_25 > upper_threshold_25 / 100

    # Filter signals to right-turn periods
    signals_2019 = get_multi_row_signals(data_2019, results_sorted_2019, sorted_2019)
    signals_2025 = get_multi_row_signals(data_2025, results_sorted_2025, sorted_2025)

    signals_2019_turn = [
        (name, series.loc[mask_right_19], label, unit)
        for (name, series, label, unit) in signals_2019
    ]
    signals_2025_turn = [
        (name, series.loc[mask_right_25], label, unit)
        for (name, series, label, unit) in signals_2025
    ]

    # Filter out empty series
    signals_2019_turn = [
        s for s in signals_2019_turn if s[1] is not None and not s[1].empty
    ]
    signals_2025_turn = [
        s for s in signals_2025_turn if s[1] is not None and not s[1].empty
    ]

    # Use signals that exist in both
    signal_names_19 = {s[0] for s in signals_2019_turn}
    signal_names_25 = {s[0] for s in signals_2025_turn}
    common_names = signal_names_19 & signal_names_25

    signals_2019_turn = [s for s in signals_2019_turn if s[0] in common_names]
    signals_2025_turn = [s for s in signals_2025_turn if s[0] in common_names]

    n_rows = len(signals_2019_turn)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3 * n_rows), sharex=False)

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    time_turn_2019 = sorted_2019.loc[mask_right_19, "time"]
    time_turn_2025 = sorted_2025.loc[mask_right_25, "time"]

    for row_idx, (sig_2019, sig_2025) in enumerate(
        zip(signals_2019_turn, signals_2025_turn)
    ):
        name_19, series_19, label_19, unit_19 = sig_2019
        name_25, series_25, label_25, unit_25 = sig_2025

        # 2019 column (left)
        ax_left = axes[row_idx, 0]
        if len(series_19) > 0:
            mean_val = float(series_19.mean())
            min_val = float(series_19.min())
            max_val = float(series_19.max())

            ax_left.plot(
                time_turn_2019,
                series_19,
                color=colors[0],
                marker=".",
                linestyle="None",
                alpha=0.6,
                label=label_19,
            )
            ax_left.axhline(
                mean_val,
                color=colors[1],
                linestyle="--",
                label=rf"$\mathrm{{mean}} = {mean_val:.3f}\,{unit_19}$",
            )
            ax_left.axhline(
                min_val,
                color=colors[2],
                linestyle=":",
                label=rf"$\mathrm{{min}} = {min_val:.3f}\,{unit_19}$",
            )
            ax_left.axhline(
                max_val,
                color=colors[3],
                linestyle=":",
                label=rf"$\mathrm{{max}} = {max_val:.3f}\,{unit_19}$",
            )
        ax_left.set_ylabel(label_19)
        ax_left.legend(frameon=True, loc="upper left", framealpha=1.0, fontsize=8)
        ax_left.set_title("2019 (Right Turns)" if row_idx == 0 else "")

        # 2025 column (right)
        ax_right = axes[row_idx, 1]
        if len(series_25) > 0:
            mean_val = float(series_25.mean())
            min_val = float(series_25.min())
            max_val = float(series_25.max())

            ax_right.plot(
                time_turn_2025,
                series_25,
                color=colors[0],
                marker=".",
                linestyle="None",
                alpha=0.6,
                label=label_25,
            )
            ax_right.axhline(
                mean_val,
                color=colors[1],
                linestyle="--",
                label=rf"$\mathrm{{mean}} = {mean_val:.3f}\,{unit_25}$",
            )
            ax_right.axhline(
                min_val,
                color=colors[2],
                linestyle=":",
                label=rf"$\mathrm{{min}} = {min_val:.3f}\,{unit_25}$",
            )
            ax_right.axhline(
                max_val,
                color=colors[3],
                linestyle=":",
                label=rf"$\mathrm{{max}} = {max_val:.3f}\,{unit_25}$",
            )
        ax_right.set_ylabel(label_25)
        ax_right.legend(frameon=True, loc="upper left", framealpha=1.0, fontsize=8)
        ax_right.set_title("2025 (Right Turns)" if row_idx == 0 else "")

    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(output_filename, dpi=150)
    print(f"Saved {output_filename}")
    plt.close(fig)


def main():
    set_plot_style()
    colors = get_color_list()

    # Load 2019 data
    print("Loading 2019 data...")
    data_19, results_19, sorted_19, results_sorted_19 = load_and_process_data(
        year="2019",
        month="10",
        day="08",
        kite_model="v3",
        addition="_t26",
        time_range=(1800.0, 9986.2),
        downsample_frac=0.1,
    )

    # Convert 2019 depower to 2025-equivalent up_data using paper physics
    if "kcu_actual_depower" in sorted_19.columns:
        sorted_19["kcu_actual_depower"] = convert_2019_depower_to_2025_updata(
            sorted_19["kcu_actual_depower"]
        )
        print(
            f"2019 kcu_actual_depower converted to 2025-equivalent (via paper physics): "
            f"min={sorted_19['kcu_actual_depower'].min():.4f}, "
            f"max={sorted_19['kcu_actual_depower'].max():.4f}"
        )

    # Load 2025 data
    print("Loading 2025 data...")
    data_25, results_25, sorted_25, results_sorted_25 = load_and_process_data(
        year="2025",
        month="10",
        day="09",
        kite_model="v3",
        addition="",
        time_range=(400.0, 1000.0),
        # time_range=(0, 2000),
        downsample_frac=1,
    )

    # Create comparison plots
    print("Creating statistics_all.pdf...")
    plot_multi_row_comparison(
        data_19,
        results_19,
        sorted_19,
        results_sorted_19,
        data_25,
        results_25,
        sorted_25,
        results_sorted_25,
        colors,
        "./results/plots_paper/statistics_all.pdf",
    )

    print("Creating statistics_right_turn.pdf...")
    plot_turn_comparison(
        data_19,
        results_19,
        sorted_19,
        results_sorted_19,
        data_25,
        results_25,
        sorted_25,
        results_sorted_25,
        colors,
        "./results/plots_paper/statistics_right_turn.pdf",
    )

    print("Done!")


if __name__ == "__main__":
    main()
