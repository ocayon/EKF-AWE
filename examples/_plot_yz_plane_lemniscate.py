"""
Plot kite position in the Y-Z plane for two v3 flights loaded from EKF .h5 files.

Creates a 1x2 scatter figure (2019 and 2025), with marker color mapped to a
selected scalar signal and one shared colorbar.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from awes_ekf.load_data.read_data import read_results
from awes_ekf.plotting.color_palette import set_plot_style


def resolve_color_settings(color_by: str | bool) -> tuple[str, str, float, str]:
    """Map color selector input to column, colorbar label, display scale and unit."""
    if isinstance(color_by, bool):
        key = "tether_force_kite" if color_by else "kite_speed"
    else:
        key = color_by.strip().lower()

    aliases = {
        "kite_speed": "kite_speed",
        "v_k": "kite_speed",
        "vk": "kite_speed",
        "speed": "kite_speed",
        "tether_force_kite": "tether_force_kite",
        "tether_force": "tether_force_kite",
        "force": "tether_force_kite",
        "f_tether": "tether_force_kite",
    }
    labels = {
        "kite_speed": r"$v_\mathrm{k}$ (ms$^{-1}$)",
        "tether_force_kite": r"$F_\mathrm{t,KCU}$ (kN)",
    }
    scales = {
        "kite_speed": 1.0,
        "tether_force_kite": 1e-3,  # N -> kN
    }
    units = {
        "kite_speed": "m/s",
        "tether_force_kite": "kN",
    }

    if key not in aliases:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unsupported color_by='{color_by}'. Use one of: {allowed}, "
            "or pass bool (True=tether_force_kite, False=kite_speed)."
        )

    column = aliases[key]
    return column, labels[column], scales[column], units[column]


def load_and_process_data(
    year: str,
    month: str,
    day: str,
    kite_model: str,
    addition: str,
    time_range: tuple[float, float],
    downsample_frac: float,
) -> pd.DataFrame:
    """Load .h5 results and apply the same filtering pipeline used before."""
    results, flight_data, _ = read_results(
        year, month, day, kite_model, addition=addition
    )

    time_mask = (results["time"] >= time_range[0]) & (results["time"] <= time_range[1])
    results = results.loc[time_mask].reset_index(drop=True)
    flight_data = flight_data.loc[time_mask].reset_index(drop=True)

    # Keep preprocessing steps consistent with prior script behavior.
    if "kite_yaw_rate_1" in flight_data.columns:
        flight_data["kite_yaw_rate"] = flight_data["kite_yaw_rate_1"]
    if "kcu_actual_steering" in flight_data.columns:
        flight_data["kcu_actual_steering_delay"] = np.roll(
            flight_data["kcu_actual_steering"], int(8)
        )

    downsampled_data = flight_data.sample(frac=downsample_frac, random_state=42)
    downsampled_results = results.loc[downsampled_data.index]
    downsampled_data = downsampled_data[downsampled_data["powered"] == "powered"]
    downsampled_results = downsampled_results.loc[downsampled_data.index]

    downsampled_sorted = downsampled_data.sort_values("time")
    downsampled_results_sorted = downsampled_results.loc[downsampled_sorted.index]

    if all(
        col in downsampled_sorted.columns
        for col in ("kite_velocity_x", "kite_velocity_y", "kite_velocity_z")
    ):
        downsampled_sorted["kite_speed"] = np.sqrt(
            downsampled_sorted["kite_velocity_x"] ** 2
            + downsampled_sorted["kite_velocity_y"] ** 2
            + downsampled_sorted["kite_velocity_z"] ** 2
        )
    elif "kite_apparent_windspeed" in downsampled_sorted.columns:
        downsampled_sorted["kite_speed"] = downsampled_sorted["kite_apparent_windspeed"]
    else:
        raise ValueError(
            "Could not compute kite speed: require velocity components "
            "('kite_velocity_x', 'kite_velocity_y', 'kite_velocity_z') "
            "or 'kite_apparent_windspeed'."
        )

    if "tether_force_kite" in downsampled_results_sorted.columns:
        downsampled_sorted["tether_force_kite"] = downsampled_results_sorted[
            "tether_force_kite"
        ].to_numpy(dtype=float)

    if "wind_direction" in downsampled_results_sorted.columns:
        wind_dir = downsampled_results_sorted["wind_direction"].to_numpy(dtype=float)
    elif "ground_wind_direction" in downsampled_sorted.columns:
        wind_dir = downsampled_sorted["ground_wind_direction"].to_numpy(dtype=float)
        # In legacy logs, ground_wind_direction is often stored in degrees.
        if np.nanmax(np.abs(wind_dir)) > 2 * np.pi:
            wind_dir = np.deg2rad(wind_dir)
    else:
        raise ValueError(
            "No wind-direction signal found in results/flight_data "
            "(expected 'wind_direction' or 'ground_wind_direction')."
        )

    # Circular mean avoids bias across angle wrap-around.
    mean_wind_dir = np.arctan2(
        np.nanmean(np.sin(wind_dir)), np.nanmean(np.cos(wind_dir))
    )
    x = downsampled_sorted["kite_position_x"].to_numpy(dtype=float)
    y = downsampled_sorted["kite_position_y"].to_numpy(dtype=float)
    crosswind = -x * np.sin(mean_wind_dir) + y * np.cos(mean_wind_dir)
    downsampled_sorted["kite_position_y_wind"] = crosswind - np.nanmean(crosswind)
    downsampled_sorted["mean_wind_direction_deg"] = (
        np.rad2deg(mean_wind_dir) + 360
    ) % 360

    required_columns = {
        "kite_position_x",
        "kite_position_y",
        "kite_position_y_wind",
        "kite_position_z",
        "kite_speed",
    }
    missing_columns = required_columns - set(downsampled_sorted.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns {sorted(missing_columns)} in "
            f"results/{kite_model}/{kite_model}_{year}-{month}-{day}{addition}.h5"
        )

    return downsampled_sorted


def scatter_yz(
    ax: plt.Axes,
    df: pd.DataFrame,
    label: str,
    marker: str,
    marker_size: float,
    norm: Normalize,
    color_column: str = "kite_speed",
    color_scale: float = 1.0,
    y_column: str = "kite_position_y_wind",
    x_label: str = r"crosswind position $y_\perp$ (m)",
    cmap: str = "viridis",
    alpha: float = 0.5,
) -> plt.PathCollection:
    """Scatter kite Y-Z positions with marker color based on a selected signal."""
    scatter = ax.scatter(
        df[y_column],
        df["kite_position_z"],
        c=df[color_column] * color_scale,
        cmap=cmap,
        norm=norm,
        s=marker_size,
        alpha=alpha,
        marker=marker,
        linewidths=0,
        label=label,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("kite_position_z (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    return scatter


def set_shared_limits(
    ax: plt.Axes,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    y_column: str = "kite_position_y_wind",
) -> None:
    """Apply equal axis limits for the combined dataset."""
    y_all = pd.concat([df_a[y_column], df_b[y_column]], ignore_index=True)
    z_all = pd.concat(
        [df_a["kite_position_z"], df_b["kite_position_z"]], ignore_index=True
    )

    y_min, y_max = y_all.min(), y_all.max()
    z_min, z_max = z_all.min(), z_all.max()

    y_pad = 0.03 * (y_max - y_min) if y_max > y_min else 1.0
    z_pad = 0.03 * (z_max - z_min) if z_max > z_min else 1.0

    x_limits = (y_min - y_pad, y_max + y_pad)
    y_limits = (z_min - z_pad, z_max + z_pad)

    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)


def main(color_by: str | bool = "tether_force_kite") -> None:
    set_plot_style()

    repo_root = Path(__file__).resolve().parents[1]
    color_column, colorbar_label, color_scale, color_unit = resolve_color_settings(
        color_by
    )
    df_2019 = load_and_process_data(
        year="2019",
        month="10",
        day="08",
        kite_model="v3",
        addition="_t26",
        time_range=(2190, 2255),  # (1800.0, 9986.2),
        downsample_frac=1.0,
    )
    df_2025 = load_and_process_data(
        year="2025",
        month="10",
        day="09",
        kite_model="v3",
        addition="",
        time_range=(700, 800),  # (400.0, 1000.0),
        downsample_frac=1.0,
    )

    fig = plt.figure(figsize=(8, 3), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.08], wspace=0.02)
    ax_2019 = fig.add_subplot(gs[0, 0])
    ax_2025 = fig.add_subplot(gs[0, 1], sharey=ax_2019)
    cax = fig.add_subplot(gs[0, 2])

    missing_years = []
    if color_column not in df_2019.columns:
        missing_years.append("2019")
    if color_column not in df_2025.columns:
        missing_years.append("2025")
    if missing_years:
        raise ValueError(
            f"Column '{color_column}' not found for flight(s): {', '.join(missing_years)}"
        )

    print(f"Investigated variable: {color_column} [{color_unit}]")
    for dataset_label, dataset_df in [
        ("2019-10-08", df_2019),
        ("2025-10-09", df_2025),
    ]:
        dataset_values = dataset_df[color_column].to_numpy(dtype=float) * color_scale
        finite_values = dataset_values[np.isfinite(dataset_values)]
        if finite_values.size == 0:
            print(f"{dataset_label}: min=nan, max=nan [{color_unit}]")
            continue
        print(
            f"{dataset_label}: min={float(np.nanmin(finite_values)):.3f}, "
            f"max={float(np.nanmax(finite_values)):.3f} [{color_unit}]"
        )

    color_values = (
        pd.concat(
            [df_2019[color_column], df_2025[color_column]], ignore_index=True
        ).to_numpy(dtype=float)
        * color_scale
    )
    if color_values.size == 0 or np.all(np.isnan(color_values)):
        raise ValueError(f"No valid data available for color column '{color_column}'.")

    color_min = float(np.nanmin(color_values))
    color_max = float(np.nanmax(color_values))
    if np.isclose(color_min, color_max):
        color_min -= 1.0
        color_max += 1.0
    norm = Normalize(vmin=color_min, vmax=color_max)

    # Marker size 10 = 5x larger than previous size 2.
    marker_size = 18
    scatter_yz(
        ax_2019,
        df_2019,
        label="2019",
        marker="o",
        marker_size=marker_size,
        norm=norm,
        color_column=color_column,
        color_scale=color_scale,
        alpha=0.7,  # 0.45,
    )
    scatter_yz(
        ax_2025,
        df_2025,
        label="2025",
        marker="o",
        marker_size=marker_size,
        norm=norm,
        color_column=color_column,
        color_scale=color_scale,
        alpha=0.6,  # 0.45,
    )
    label_2019 = (
        r"\textbf{2019-10-08}"
        if plt.rcParams.get("text.usetex", False)
        else "2019-10-08"
    )
    label_2025 = (
        r"\textbf{2025-10-09}"
        if plt.rcParams.get("text.usetex", False)
        else "2025-10-09"
    )
    ax_2019.text(
        0.02,
        0.98,
        label_2019,
        transform=ax_2019.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        # fontsize=12,
    )
    ax_2025.text(
        0.02,
        0.98,
        label_2025,
        transform=ax_2025.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        # fontsize=12,
    )
    # ax_2019.set_title("2019-10-08")
    # ax_2025.set_title("2025-10-09")
    ax_2019.set_ylabel(r"$z_\mathrm{W}$ (m)")
    ax_2019.set_xlabel(r"$y_\mathrm{W,\perp}$ (m)")
    ax_2025.set_xlabel(r"$y_\mathrm{W,\perp}$ (m)")
    # ax_2019.text(
    #     0.02,
    #     0.98,
    #     f"mean downwind: {df_2019['mean_wind_direction_deg'].iloc[0]:.1f} deg",
    #     transform=ax_2019.transAxes,
    #     ha="left",
    #     va="top",
    #     fontsize=9,
    # )
    # ax_2025.text(
    #     0.02,
    #     0.98,
    #     f"mean downwind: {df_2025['mean_wind_direction_deg'].iloc[0]:.1f} deg",
    #     transform=ax_2025.transAxes,
    #     ha="left",
    #     va="top",
    #     fontsize=9,
    # )

    # Use a clean scalar mappable so colorbar is fully opaque (independent of point alpha).
    sm_color = ScalarMappable(norm=norm, cmap="viridis")
    sm_color.set_array([])
    cbar = fig.colorbar(sm_color, cax=cax)
    cbar.set_label(colorbar_label)  # , pad=6)

    set_shared_limits(ax_2019, df_2019, df_2025, y_column="kite_position_y_wind")
    set_shared_limits(ax_2025, df_2019, df_2025, y_column="kite_position_y_wind")

    # Right panel shares z-axis with left panel, so hide duplicate z-axis labels/ticks.
    ax_2025.set_ylabel("")
    ax_2025.tick_params(axis="y", which="both", left=False, labelleft=False)

    # Match colorbar height to the actually rendered panel height with equal aspect.
    fig.canvas.draw()
    pos_2019 = ax_2019.get_position()
    pos_2025 = ax_2025.get_position()
    pos_cax = cax.get_position()
    y0 = min(pos_2019.y0, pos_2025.y0) - 0.03
    y1 = max(pos_2019.y1, pos_2025.y1) + 0.025
    cax.set_position([pos_cax.x0 + 0.062, y0, pos_cax.width - 0.01, y1 - y0])

    output_path = repo_root / "results/plots_paper/yz_plane_lemniscate.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot kite Y-Z plane with configurable color variable."
    )
    parser.add_argument(
        "--color-by",
        default="tether_force_kite",
        help=(
            "Color variable ('tether_force_kite' or 'kite_speed'; aliases: "
            "'tether_force', 'v_k')."
        ),
    )
    parser.add_argument(
        "--use-tether-force",
        action="store_true",
        help="Boolean shortcut to color by tether_force_kite.",
    )
    args = parser.parse_args()
    selected_color = True if args.use_tether_force else args.color_by
    main(color_by=selected_color)
