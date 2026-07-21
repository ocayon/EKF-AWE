"""Plot the 2025 flight path in 2D (Y-Z plane) with photogrammetry measurement markers.

Same data, interval, and markers as ``plot_yz_plane_lemniscate_with_markers_3D.py``
but projected onto the crosswind / height plane.
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
DEFAULT_FIGURE_SIZE_IN = (5.0, 4.5)
DEFAULT_RETAIN_POINTS = [1, 2]  # keep measurements whose index suffix is in this list


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
            "Plot the 2025 EKF flight path in the 2D Y-Z plane and "
            "mark all named photogrammetry measurement frames."
        )
    )
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--anchor-frame", type=int, default=DEFAULT_ANCHOR_FRAME)
    parser.add_argument(
        "--anchor-ekf-time-s", type=float, default=DEFAULT_ANCHOR_EKF_TIME_S
    )
    parser.add_argument(
        "--padding-s",
        type=float,
        default=15.0,
        help="Time included before the first and after the last marker.",
    )
    parser.add_argument(
        "--trajectory-step",
        type=int,
        default=3,
        help="Plot every Nth EKF sample in the colored trajectory scatter.",
    )
    parser.add_argument(
        "--output-stem",
        default="yz_plane_lemniscate_with_markers_2d",
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
    results, flight_data, _ = read_results("2025", "10", "09", "v3", addition="")
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
    crosswind = -x_world * np.sin(mean_wind_direction) + y_world * np.cos(
        mean_wind_direction
    )
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
                f"{target_time:.3f} s is outside [{time_min:.3f}, {time_max:.3f}] s."
            )
            continue
        index = int(np.argmin(np.abs(times - target_time)))
        sample = flight_data.iloc[index]
        rows.append(
            {
                "short_label": measurement.short_label,
                "frame": measurement.frame,
                "marker": measurement.marker,
                "color": measurement.color,
                "crosswind_position_m": float(sample["kite_position_crosswind"]),
                "height_m": float(sample["kite_position_z"]),
            }
        )
    return pd.DataFrame(rows)


def plot_trajectory_2d(
    flight_data: pd.DataFrame,
    marker_data: pd.DataFrame,
    *,
    trajectory_step: int,
    output_stem: str,
) -> list[Path]:
    tether_force_kn = flight_data["tether_force_kite"].to_numpy(dtype=float) * 1e-3
    norm = Normalize(
        vmin=float(np.nanmin(tether_force_kn)),
        vmax=float(np.nanmax(tether_force_kn)),
    )
    cmap = "viridis"

    fig, ax = plt.subplots(figsize=DEFAULT_FIGURE_SIZE_IN)

    ax.plot(
        flight_data["kite_position_crosswind"],
        flight_data["kite_position_z"],
        color="0.45",
        linewidth=0.5,
        alpha=0.2,
        zorder=1,
    )
    step = max(1, int(trajectory_step))
    plotted = flight_data.iloc[::step]
    ax.scatter(
        plotted["kite_position_crosswind"],
        plotted["kite_position_z"],
        c=plotted["tether_force_kite"] * 1e-3,
        cmap=cmap,
        norm=norm,
        s=8.0,
        alpha=0.7,
        linewidths=0,
        zorder=2,
    )

    marker_size = 95 / 1.5
    for row in marker_data.itertuples(index=False):
        ax.scatter(
            row.crosswind_position_m,
            row.height_m,
            marker=row.marker,
            color=row.color,
            edgecolors="black",
            linewidths=0.5,
            s=marker_size,
            zorder=10,
        )
        label_text = str(row.short_label).replace("_", "")
        ax.annotate(
            label_text,
            (row.crosswind_position_m, row.height_m),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="black",
            zorder=11,
        )

    ax.set_xlabel(r"$y_{\mathrm{W},\perp}$ (m)")
    ax.set_ylabel(r"$z_\mathrm{W}$ (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.25)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="C0",
            markeredgecolor="black",
            markersize=7,
            label="Powered straight, reel-out",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="C7",
            markeredgecolor="black",
            markersize=7,
            label="Powered turn, reel-out",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6.5,
            label=r"$*_{1}$",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6.5,
            label=r"$*_{2}$",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.37),
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02, shrink=0.4)
    cbar.set_label(r"$F_{\mathrm{t,KCU}}$ (kN)")

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

    set_plot_style()
    retain_suffixes = {f"_{n}" for n in DEFAULT_RETAIN_POINTS}
    reel_out_measurements = tuple(
        m
        for m in MEASUREMENTS
        if m.category != "depowered_reelin"
        and any(m.short_label.endswith(suffix) for suffix in retain_suffixes)
    )

    if not reel_out_measurements:
        raise ValueError("No reel-out measurements selected by DEFAULT_RETAIN_POINTS.")
    measurement_times = [
        frame_to_ekf_time(
            m.frame,
            video_fps=args.video_fps,
            anchor_frame=args.anchor_frame,
            anchor_ekf_time_s=args.anchor_ekf_time_s,
        )
        for m in reel_out_measurements
    ]
    start_time = min(measurement_times) - args.padding_s
    end_time = max(measurement_times) + args.padding_s

    results, flight_data = load_interval(start_time, end_time)
    mean_wind_direction = add_wind_aligned_coordinates(results, flight_data)

    # Keep only samples within 10 s of any reel-out measurement.
    window_s = 10.0
    times = flight_data["time"].to_numpy(dtype=float)
    keep = np.zeros(len(times), dtype=bool)
    for t in measurement_times:
        keep |= np.abs(times - t) <= window_s
    flight_data = flight_data.loc[keep].reset_index(drop=True)

    marker_data = nearest_measurement_rows(
        flight_data,
        reel_out_measurements,
        video_fps=args.video_fps,
        anchor_frame=args.anchor_frame,
        anchor_ekf_time_s=args.anchor_ekf_time_s,
    )
    output_paths = plot_trajectory_2d(
        flight_data,
        marker_data,
        trajectory_step=args.trajectory_step,
        output_stem=args.output_stem,
    )

    print(
        f"Plotted interval: {start_time:.3f} to {end_time:.3f} s; "
        f"mean wind direction={np.degrees(mean_wind_direction) % 360.0:.2f} deg"
    )
    for output_path in output_paths:
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
