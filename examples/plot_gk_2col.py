"""
Generalized plotting functions for yaw rate analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
from typing import Optional, List, Tuple, Dict
from matplotlib.axes import Axes

from awes_ekf.load_data.read_data import read_results
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style


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

        mask_straight_kcu = x_no_delay_kcu.between(lower_threshold, upper_threshold)
        mask_left_kcu = x_no_delay_kcu < lower_threshold
        mask_right_kcu = x_no_delay_kcu > upper_threshold

        mask_straight_us = x_no_delay_us.between(lower_threshold_us, upper_threshold_us)
        mask_left_us = x_no_delay_us < lower_threshold_us
        mask_right_us = x_no_delay_us > upper_threshold_us

        # Plot regions for kcu panel
        ax.scatter(
            x_kcu[mask_straight_kcu],
            y_deg[mask_straight_kcu],
            color=colors[1],
            alpha=0.4,
            marker=".",
            label="Straight",
        )
        ax.scatter(
            x_kcu[mask_left_kcu],
            y_deg[mask_left_kcu],
            color=colors[2],
            marker=".",
            alpha=0.4,
            label="Left Turn",
        )
        ax.scatter(
            x_kcu[mask_right_kcu],
            y_deg[mask_right_kcu],
            color=colors[3],
            marker=".",
            alpha=0.4,
            label="Right Turn",
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
            x_vals_kcu = x_kcu[mask_bucket]
            y_vals_kcu = y_deg[mask_bucket]
            finite_kcu = np.isfinite(x_vals_kcu) & np.isfinite(y_vals_kcu)
            x_vals_kcu = x_vals_kcu[finite_kcu]
            y_vals_kcu = y_vals_kcu[finite_kcu]
            label_kcu = label

            if len(x_vals_kcu) > 1:
                slope_kcu, intercept_kcu, r_kcu, _, _ = linregress(
                    x_vals_kcu, y_vals_kcu
                )
                label_kcu = f"{label} (k={slope_kcu:.3f}, R$^2$={r_kcu**2:.2f})"
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

    # Add circle batch data if provided
    if circle_df is not None and not circle_df.empty and circle_ups is not None:
        for i, up_val in enumerate(circle_ups):
            rows = circle_df[circle_df["up"] == up_val]
            if len(rows) > 0:
                ax.scatter(
                    rows["us"] * rows["v_app"],
                    rows["yaw_rate_paper"],
                    s=80,
                    alpha=1.0,
                    color=(
                        circle_colors[i] if circle_colors else colors[i % len(colors)]
                    ),
                    marker="x",
                    label="_nolegend_",
                )

    # Set labels
    ax.set_xlabel(
        r"$\mathrm{kcu\_actual\_steering}/100 \cdot v_a\;(\mathrm{m\,s^{-1}})$"
    )
    ax.set_ylabel(r"$\dot{\psi}\;(^\circ\,\mathrm{s^{-1}})$")
    if show_legend:
        ax.legend(frameon=True)

    # Save figure only when we created it here
    if output_filename and created_fig:
        fig.tight_layout()
        fig.savefig(output_filename)
        print(f"Saved {output_filename}")

    return fig


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

    # Load circle batch data if available
    circle_df = None
    circle_ups = []
    circle_colors = []
    circle_csv_path = (
        Path(__file__).resolve().parents[1] / "data" / "circle_batch_analysis.csv"
    )

    if circle_csv_path.is_file():
        circle_df = pd.read_csv(circle_csv_path)
        required_cols = ["us", "v_app", "yaw_rate_paper", "cs", "up"]
        missing_cols = [col for col in required_cols if col not in circle_df.columns]

        if missing_cols:
            raise ValueError(
                f"Missing columns in {circle_csv_path}: {', '.join(missing_cols)}"
            )

        circle_mask = (
            np.isfinite(circle_df["us"])
            & np.isfinite(circle_df["v_app"])
            & np.isfinite(circle_df["yaw_rate_paper"])
            & np.isfinite(circle_df["cs"])
            & np.isfinite(circle_df["up"])
        )
        circle_df = circle_df.loc[circle_mask]
        circle_ups = np.sort(circle_df["up"].unique())
        circle_cmap = plt.get_cmap("tab10")
        circle_colors = [circle_cmap(i % circle_cmap.N) for i in range(len(circle_ups))]
    else:
        print(f"Circle batch CSV not found: {circle_csv_path}")

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
            "time_range": (0.0, 1500.0),
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
            }
        )

    # Build two-column figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for idx, dataset in enumerate(prepared_datasets):
        plot_yaw_rate_vs_steering(
            dataset["data"],
            dataset["results"],
            colors,
            output_filename=None,
            bucket_type="regions",
            circle_df=circle_df,
            circle_ups=circle_ups,
            circle_colors=circle_colors,
            steering_norm=dataset["steering_norm"],
            ax=axes[idx],
            show_legend=(idx == 0),
        )
        axes[idx].set_title(dataset["title"])
        axes[idx].set_ylim(-100, 100)

    fig.tight_layout()

    output_path = Path("./results/plots_paper") / "yaw.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
