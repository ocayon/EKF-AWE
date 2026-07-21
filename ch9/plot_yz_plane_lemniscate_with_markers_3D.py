"""Plot the 2025 flight path in 3D with photogrammetry measurement markers.

The original ``plot_yz_plane_lemniscate.py`` is intentionally left unchanged.
This script covers the complete interval containing all named in-flight
photogrammetry reconstructions:

* powered straight reel-out: frame 7182;
* powered turns: frames 7362, 7721, and 7811;
* powered straight reel-out: frame 17372;
* depowered reel-in: frames 17611 and 17701.

The merged stereo-video frame numbers are mapped to EKF time with a 30 Hz
video rate and the table-consistent synchronized anchor frame
7182 -> EKF t=67.633 s. Both values can be overridden from the command line.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from awes_ekf.load_data.read_data import read_results
from awes_ekf.plotting.color_palette import set_plot_style

DEFAULT_VIDEO_FPS = 30.0
DEFAULT_ANCHOR_FRAME = 7182
DEFAULT_ANCHOR_EKF_TIME_S = 67.63333333333334
DEFAULT_ELEVATION_DEG = 35.0
DEFAULT_AZIMUTH_DEG = -65  # -58.0
DEFAULT_FIGURE_SIZE_IN = (9.2, 5.2)
DEFAULT_COLORBAR_FRACTION = 0.022
DEFAULT_COLORBAR_SHRINK = 0.65
DEFAULT_RETAIN_POINTS = [1, 2]
DEFAULT_TIME_WINDOW_S = 50.0
DEFAULT_X_LIM_M = (100, 280.0)
DEFAULT_Y_LIM_M = (-50, 100)
DEFAULT_Z_LIM_M = (0, 180)
DEFAULT_X_TICKS_M = np.arange(DEFAULT_X_LIM_M[0], DEFAULT_X_LIM_M[1] + 1, 50)
DEFAULT_Y_TICKS_M = np.arange(DEFAULT_Y_LIM_M[0], DEFAULT_Y_LIM_M[1] + 1, 50)
DEFAULT_Z_TICKS_M = np.arange(DEFAULT_Z_LIM_M[0], DEFAULT_Z_LIM_M[1] + 1, 50)


@dataclass(frozen=True)
class Measurement:
    short_label: str
    frame: int
    label: str
    category: str
    marker: str
    color: str


MEASUREMENTS = (
    Measurement(
        "PS_1", 7182, "Powered straight reel-out", "powered_straight", "o", "C0"
    ),
    Measurement("PT_1", 7362, "Powered right-turn reel-out", "powered_turn", "o", "C7"),
    Measurement(
        "PS_2", 7721, "Powered straight reel-out", "powered_straight", "^", "C0"
    ),
    Measurement("PT_2", 7811, "Powered left-turn reel-out", "powered_turn", "^", "C7"),
    Measurement(
        "PS_3", 17372, "Powered straight reel-out", "powered_straight", "s", "C0"
    ),
    Measurement("DR_1", 17611, "Depowered reel-in", "depowered_reelin", "o", "C2"),
    Measurement("DR_2", 17701, "Depowered reel-in", "depowered_reelin", "^", "C2"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the 2025 EKF flight path in wind-aligned 3D coordinates and "
            "mark all named photogrammetry measurement frames."
        )
    )
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument(
        "--anchor-frame",
        type=int,
        default=DEFAULT_ANCHOR_FRAME,
    )
    parser.add_argument(
        "--anchor-ekf-time-s",
        type=float,
        default=DEFAULT_ANCHOR_EKF_TIME_S,
    )
    parser.add_argument(
        "--padding-s",
        type=float,
        default=15.0,
        help="Time included before the first and after the last marker.",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=DEFAULT_TIME_WINDOW_S,
        help="Keep trajectory samples within +/- window-s of each selected marker time.",
    )
    parser.add_argument(
        "--trajectory-step",
        type=int,
        default=3,
        help="Plot every Nth EKF sample in the colored trajectory scatter.",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=DEFAULT_ELEVATION_DEG,
        help="Initial 3D camera elevation in degrees.",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=DEFAULT_AZIMUTH_DEG,
        help="Initial 3D camera azimuth in degrees.",
    )
    parser.add_argument(
        "--output-stem",
        default="yz_plane_lemniscate_with_markers_3d",
    )
    return parser


def frame_to_ekf_time(
    frame: int,
    *,
    video_fps: float,
    anchor_frame: int,
    anchor_ekf_time_s: float,
) -> float:
    return anchor_ekf_time_s + (float(frame) - anchor_frame) / video_fps


def load_interval(
    start_time_s: float,
    end_time_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results, flight_data, _ = read_results(
        "2025",
        "10",
        "09",
        "v3",
        addition="",
    )
    mask = (flight_data["time"] >= start_time_s) & (flight_data["time"] <= end_time_s)
    results = results.loc[mask].reset_index(drop=True)
    flight_data = flight_data.loc[mask].reset_index(drop=True)
    if flight_data.empty:
        raise ValueError(
            f"No 2025 flight data found between {start_time_s:.1f} and "
            f"{end_time_s:.1f} s."
        )
    if "tether_force_kite" in results.columns:
        flight_data["tether_force_kite"] = results["tether_force_kite"].to_numpy(
            dtype=float
        )
    if "tether_force_kite" not in flight_data:
        raise ValueError("Flight data do not contain 'tether_force_kite'.")
    return results, flight_data


def add_wind_aligned_coordinates(
    results: pd.DataFrame,
    flight_data: pd.DataFrame,
) -> float:
    if "wind_direction" in results:
        wind_direction = results["wind_direction"].to_numpy(dtype=float)
    elif "ground_wind_direction" in flight_data:
        wind_direction = flight_data["ground_wind_direction"].to_numpy(dtype=float)
        if np.nanmax(np.abs(wind_direction)) > 2.0 * np.pi:
            wind_direction = np.deg2rad(wind_direction)
    else:
        raise ValueError("No wind-direction signal is available.")

    finite = np.isfinite(wind_direction)
    if not np.any(finite):
        raise ValueError("The wind-direction signal contains no finite values.")
    mean_wind_direction = np.arctan2(
        np.nanmean(np.sin(wind_direction[finite])),
        np.nanmean(np.cos(wind_direction[finite])),
    )

    x_world = flight_data["kite_position_x"].to_numpy(dtype=float)
    y_world = flight_data["kite_position_y"].to_numpy(dtype=float)
    downwind = x_world * np.cos(mean_wind_direction) + y_world * np.sin(
        mean_wind_direction
    )
    crosswind = -x_world * np.sin(mean_wind_direction) + y_world * np.cos(
        mean_wind_direction
    )
    flight_data["kite_position_downwind"] = downwind
    flight_data["kite_position_crosswind"] = crosswind - np.nanmean(crosswind)
    return float(mean_wind_direction)


def nearest_measurement_rows(
    flight_data: pd.DataFrame,
    measurements: tuple[Measurement, ...],
    *,
    video_fps: float,
    anchor_frame: int,
    anchor_ekf_time_s: float,
) -> pd.DataFrame:
    rows = []
    times = flight_data["time"].to_numpy(dtype=float)
    time_min = float(np.nanmin(times))
    time_max = float(np.nanmax(times))
    for measurement in measurements:
        target_time = frame_to_ekf_time(
            measurement.frame,
            video_fps=video_fps,
            anchor_frame=anchor_frame,
            anchor_ekf_time_s=anchor_ekf_time_s,
        )
        if target_time < time_min or target_time > time_max:
            print(
                f"Skipping {measurement.short_label}: target time "
                f"{target_time:.3f} s is outside the available EKF interval "
                f"[{time_min:.3f}, {time_max:.3f}] s."
            )
            continue
        index = int(np.argmin(np.abs(times - target_time)))
        sample = flight_data.iloc[index]
        rows.append(
            {
                "short_label": measurement.short_label,
                "frame": measurement.frame,
                "measurement": measurement.label,
                "category": measurement.category,
                "marker": measurement.marker,
                "color": measurement.color,
                "target_ekf_time_s": target_time,
                "sample_ekf_time_s": float(sample["time"]),
                "time_error_s": float(sample["time"] - target_time),
                "downwind_position_m": float(sample["kite_position_downwind"]),
                "crosswind_position_m": float(sample["kite_position_crosswind"]),
                "height_m": float(sample["kite_position_z"]),
                "flight_power_state": str(sample.get("powered", "")),
                "flight_turn_state": str(sample.get("turn_straight", "")),
                "steering_percent": float(sample.get("kcu_actual_steering", np.nan)),
                "depower_percent": float(sample.get("kcu_actual_depower", np.nan)),
                "reelout_speed_ms": float(sample.get("tether_reelout_speed", np.nan)),
                "tether_force_kcu_n": float(sample.get("tether_force_kite", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def apply_axes_limits_and_ticks(ax: plt.Axes) -> None:
    x_min, x_max = DEFAULT_X_LIM_M
    y_min, y_max = DEFAULT_Y_LIM_M
    z_min, z_max = DEFAULT_Z_LIM_M
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xticks(DEFAULT_X_TICKS_M)
    ax.set_yticks(DEFAULT_Y_TICKS_M)
    ax.set_zticks(DEFAULT_Z_TICKS_M)

    # Keep axis scale in meters consistent across x/y/z.
    x_span = max(float(x_max - x_min), 1.0)
    y_span = max(float(y_max - y_min), 1.0)
    z_span = max(float(z_max - z_min), 1.0)
    ax.set_box_aspect((x_span, y_span, z_span))


def plot_trajectory(
    flight_data: pd.DataFrame,
    marker_data: pd.DataFrame,
    *,
    elevation: float,
    azimuth: float,
    trajectory_step: int,
    output_stem: str,
) -> list[Path]:
    tether_force_kn = flight_data["tether_force_kite"].to_numpy(dtype=float) * 1e-3
    norm = Normalize(
        vmin=float(np.nanmin(tether_force_kn)),
        vmax=float(np.nanmax(tether_force_kn)),
    )
    cmap = "viridis"

    fig = plt.figure(figsize=DEFAULT_FIGURE_SIZE_IN)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        flight_data["kite_position_downwind"],
        flight_data["kite_position_crosswind"],
        flight_data["kite_position_z"],
        color="0.45",
        linewidth=0.75,
        alpha=0.45,
        zorder=1,
    )
    step = max(1, int(trajectory_step))
    plotted = flight_data.iloc[::step]
    ax.scatter(
        plotted["kite_position_downwind"],
        plotted["kite_position_crosswind"],
        plotted["kite_position_z"],
        c=plotted["tether_force_kite"] * 1e-3,
        cmap=cmap,
        norm=norm,
        s=4.0,
        alpha=0.55,
        linewidths=0,
        zorder=2,
    )

    for row in marker_data.itertuples(index=False):
        marker_artist = ax.scatter(
            row.downwind_position_m,
            row.crosswind_position_m,
            row.height_m,
            marker=row.marker,
            color=row.color,
            edgecolor="black",
            linewidth=0.8,
            s=95 / 1.5,
            depthshade=False,
            zorder=200,
        )
        marker_artist.set_sort_zpos(1e9)
        label_text = str(row.short_label).replace("_", "")
        ax.text(
            row.downwind_position_m,
            row.crosswind_position_m,
            row.height_m + 2.5,
            label_text,
            color="black",
            fontsize=8,
            ha="left",
            va="bottom",
            zorder=300,
        )

    ax.set_xlabel(r"$x_{\mathrm{W},\parallel}$ (m)")
    ax.set_ylabel(r"$y_{\mathrm{W},\perp}$ (m)")
    ax.set_zlabel(r"$z_\mathrm{W}$ (m)")
    ax.view_init(elev=elevation, azim=azimuth)
    ax.grid(True, alpha=0.25)
    apply_axes_limits_and_ticks(ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="C0",
            markeredgecolor="black",
            markersize=8 / np.sqrt(1.5),
            label="Powered straight, reel-out",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="C7",
            markeredgecolor="black",
            markersize=8 / np.sqrt(1.5),
            label="Powered turn, reel-out",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="C2",
            markeredgecolor="black",
            markersize=7 / np.sqrt(1.5),
            label="Depowered reel-in",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7.5 / np.sqrt(1.5),
            label=r"$*_{1}$",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7.5 / np.sqrt(1.5),
            label=r"$*_{2}$",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(-0.23, 0.4),
        frameon=True,
    )

    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(
        scalar_mappable,
        ax=ax,
        pad=0.03,
        fraction=DEFAULT_COLORBAR_FRACTION,
        shrink=DEFAULT_COLORBAR_SHRINK,
    )
    colorbar.set_label(r"$F_{\mathrm{t,KCU}}$ (kN)")

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for suffix in ("pdf", "png"):
        output_path = output_dir / f"{output_stem}.{suffix}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        output_paths.append(output_path)
    plt.close(fig)
    return output_paths


def main() -> None:
    args = build_parser().parse_args()
    if args.video_fps <= 0.0:
        raise ValueError("--video-fps must be positive.")
    if args.padding_s < 0.0:
        raise ValueError("--padding-s must be non-negative.")
    if args.window_s < 0.0:
        raise ValueError("--window-s must be non-negative.")

    set_plot_style()
    retain_suffixes = {f"_{n}" for n in DEFAULT_RETAIN_POINTS}
    selected_measurements = tuple(
        measurement
        for measurement in MEASUREMENTS
        if any(measurement.short_label.endswith(suffix) for suffix in retain_suffixes)
    )
    if not selected_measurements:
        raise ValueError("No measurements selected by DEFAULT_RETAIN_POINTS.")

    measurement_times = [
        frame_to_ekf_time(
            measurement.frame,
            video_fps=args.video_fps,
            anchor_frame=args.anchor_frame,
            anchor_ekf_time_s=args.anchor_ekf_time_s,
        )
        for measurement in selected_measurements
    ]
    start_time = min(measurement_times) - args.padding_s
    end_time = max(measurement_times) + args.padding_s
    results, flight_data = load_interval(start_time, end_time)
    mean_wind_direction = add_wind_aligned_coordinates(results, flight_data)

    # Keep only trajectory samples within +/-window-s around selected marker times.
    times = flight_data["time"].to_numpy(dtype=float)
    keep = np.zeros(len(times), dtype=bool)
    for marker_time in measurement_times:
        keep |= np.abs(times - marker_time) <= args.window_s
    flight_data = flight_data.loc[keep].reset_index(drop=True)

    marker_data = nearest_measurement_rows(
        flight_data,
        selected_measurements,
        video_fps=args.video_fps,
        anchor_frame=args.anchor_frame,
        anchor_ekf_time_s=args.anchor_ekf_time_s,
    )
    output_paths = plot_trajectory(
        flight_data,
        marker_data,
        elevation=args.elevation,
        azimuth=args.azimuth,
        trajectory_step=args.trajectory_step,
        output_stem=args.output_stem,
    )

    marker_csv = (
        Path(__file__).resolve().parent / "results" / f"{args.output_stem}_markers.csv"
    )
    marker_data.to_csv(marker_csv, index=False)

    print(
        f"Frame-to-time mapping: frame {args.anchor_frame} -> "
        f"{args.anchor_ekf_time_s:.3f} s at {args.video_fps:.3f} fps"
    )
    print(
        f"Plotted interval: {start_time:.3f} to {end_time:.3f} s; "
        f"mean wind direction={np.degrees(mean_wind_direction) % 360.0:.2f} deg"
    )
    print(
        marker_data[
            [
                "short_label",
                "frame",
                "measurement",
                "target_ekf_time_s",
                "sample_ekf_time_s",
                "flight_power_state",
                "flight_turn_state",
                "steering_percent",
                "depower_percent",
                "reelout_speed_ms",
                "tether_force_kcu_n",
            ]
        ].to_string(index=False)
    )
    for output_path in output_paths:
        print(f"Saved {output_path}")
    print(f"Saved {marker_csv}")


if __name__ == "__main__":
    main()
