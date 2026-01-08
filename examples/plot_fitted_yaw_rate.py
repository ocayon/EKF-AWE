"""
Generate three yaw-rate fit plots:
1) Recreate the Fig. 16-style fit using the standard yaw rate via plot_identify_turn_dynamics.
2) Repeat the fit with yaw rate redefined as the time derivative of a wind-perpendicular heading.
3) Repeat the fit with yaw rate from a spherical/tangent-plane heading (Julia yaw_sphere-aligned).

Usage examples:
    python examples/plot_fitted_yaw_rate.py --hdf5 results/v3/v3_2019-10-08.h5
    python examples/plot_fitted_yaw_rate.py --year 2019 --month 10 --day 08 --kite-model v3 --addition _va
Optional:
    --wind-dir-column est_upwind_direction  # pick a wind direction column if multiple exist
    --heading-column heading                # reuse an existing heading column instead of deriving it
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from awes_ekf.load_data.read_data import read_results, read_results_from_hdf5
from awes_ekf.plotting.plot_aerodynamics import plot_identify_turn_dynamics
from awes_ekf.utils import project_onto_plane


def select_first(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[pd.Series]:
    for name in candidates:
        if name and name in df.columns:
            return df[name]
    return None


def to_radians(series: pd.Series) -> np.ndarray:
    arr = np.asarray(series)
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size == 0:
        return arr
    return np.deg2rad(arr) if np.nanmax(finite) > 2 * np.pi else arr


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm


def compute_wind_perp_heading_rate(
    flight_data: pd.DataFrame,
    results: pd.DataFrame,
    wind_dir_col: Optional[str],
    heading_col: Optional[str],
    smooth_window: int = 10,
    use_mean_wind_dir: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute yaw_rate_perp_wind = d/dt(heading_wind_plane).
    heading_wind_plane is the angle of the kite velocity projected onto the plane
    perpendicular to the wind vector. The derivative is smoothed with a moving average.
    """
    time = np.asarray(flight_data["time"])

    # 1) If a heading column is provided, use it directly.
    if heading_col:
        heading_series = select_first(flight_data, [heading_col])
        if heading_series is None:
            heading_series = select_first(results, [heading_col])
        if heading_series is not None:
            heading = np.unwrap(to_radians(heading_series))
            yaw_rate = np.gradient(heading, time)
            if smooth_window > 1:
                yaw_rate = np.convolve(
                    yaw_rate, np.ones(smooth_window) / smooth_window, mode="same"
                )
            return heading, yaw_rate

    # 2) Derive heading from velocity projected onto the wind-perpendicular plane.
    wind_dir_series = select_first(
        flight_data,
        [
            wind_dir_col,
            "est_upwind_direction",
            "ground_upwind_direction",
            "wind_direction",
        ],
    )
    if wind_dir_series is None:
        wind_dir_series = select_first(
            results, [wind_dir_col, "wind_direction", "est_upwind_direction"]
        )
    if wind_dir_series is None:
        raise ValueError("No wind direction column found; provide --wind-dir-column.")
    wind_dir = to_radians(wind_dir_series)
    if use_mean_wind_dir:
        wind_dir = np.full_like(wind_dir, np.nanmean(wind_dir))

    res_has_vel = all(
        col in results.columns
        for col in ["kite_velocity_x", "kite_velocity_y", "kite_velocity_z"]
    )
    if res_has_vel:
        vx = np.asarray(results["kite_velocity_x"])
        vy = np.asarray(results["kite_velocity_y"])
        vz = np.asarray(results["kite_velocity_z"])
    else:
        vx = select_first(flight_data, ["kite_velocity_x", "kite_0_vx"])
        vy = select_first(flight_data, ["kite_velocity_y", "kite_0_vy"])
        vz = select_first(flight_data, ["kite_velocity_z", "kite_0_vz"])
        if vx is None or vy is None or vz is None:
            raise ValueError("No velocity components found to derive heading.")
        vx, vy, vz = np.asarray(vx), np.asarray(vy), np.asarray(vz)

    # Treat wind_dir as an upwind direction; flip sign to get wind-velocity direction.
    wind_heading = np.column_stack(
        [-np.cos(wind_dir), -np.sin(wind_dir), np.zeros_like(wind_dir)]
    )
    heading_angles = np.full_like(wind_dir, np.nan, dtype=float)

    for i, (vx_i, vy_i, vz_i, w_hat) in enumerate(zip(vx, vy, vz, wind_heading)):
        v_vec = np.array([vx_i, vy_i, vz_i], dtype=float)
        w_hat = _unit(np.array(w_hat, dtype=float))

        # Build an orthonormal basis in the wind-perpendicular plane using "up" as reference.
        e_up = np.array([0.0, 0.0, 1.0])
        e1 = project_onto_plane(e_up, w_hat)  # zero angle along projected up
        if np.linalg.norm(e1) < 1e-6:
            e1 = project_onto_plane(np.array([0.0, 1.0, 0.0]), w_hat)
        e1 = _unit(e1)
        e2 = _unit(np.cross(w_hat, e1))  # crosswind direction in the plane

        v_proj = project_onto_plane(v_vec, w_hat)
        v_proj_norm = np.linalg.norm(v_proj)
        if v_proj_norm < 1e-8:
            continue
        heading_angles[i] = np.arctan2(np.dot(v_proj, e2), np.dot(v_proj, e1))

    heading_unwrapped = np.full_like(heading_angles, np.nan, dtype=float)
    yaw_rate = np.full_like(time, np.nan, dtype=float)

    finite_idx = np.where(np.isfinite(heading_angles))[0]
    if finite_idx.size:
        segments = np.split(finite_idx, np.where(np.diff(finite_idx) > 1)[0] + 1)
        for seg in segments:
            if seg.size == 0:
                continue

            seg_unwrapped = np.unwrap(heading_angles[seg])
            heading_unwrapped[seg] = seg_unwrapped

            if seg.size > 1:
                seg_yaw_rate = np.gradient(seg_unwrapped, time[seg])
                if smooth_window > 1 and seg_yaw_rate.size > 1:
                    window = min(smooth_window, seg_yaw_rate.size)
                    kernel = np.ones(window) / window
                    seg_yaw_rate = np.convolve(seg_yaw_rate, kernel, mode="same")
                yaw_rate[seg] = seg_yaw_rate

    return heading_unwrapped, yaw_rate


def compute_heading_spherical(
    flight_data: pd.DataFrame,
    results: pd.DataFrame,
    smooth_window: int = 10,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute spherical/tangent-plane heading (matches Julia yaw_sphere) and its derivative.
    Heading is the direction of motion projected onto the sphere tangent plane, expressed
    in the local frame built from position:
      radial = p/||p||; up_y_raw = (-p_y, p_x, 0); up_x = radial x up_y; up_y = radial x up_x.
      heading = atan2(heading_vec_y, heading_vec_x) where heading_vec = R_up.T @ tangential_velocity.
    """
    time = np.asarray(flight_data["time"])

    # Pull position/velocity (prefer results, fall back to flight_data)
    px = select_first(results, ["kite_position_x"])
    py = select_first(results, ["kite_position_y"])
    pz = select_first(results, ["kite_position_z"])
    vx = select_first(results, ["kite_velocity_x"])
    vy = select_first(results, ["kite_velocity_y"])
    vz = select_first(results, ["kite_velocity_z"])

    if px is None or py is None or pz is None:
        px = select_first(flight_data, ["kite_position_x", "kite_pos_east"])
        py = select_first(flight_data, ["kite_position_y", "kite_pos_north"])
        pz = select_first(flight_data, ["kite_position_z", "kite_height"])
    if vx is None or vy is None or vz is None:
        vx = select_first(flight_data, ["kite_velocity_x", "kite_0_vx"])
        vy = select_first(flight_data, ["kite_velocity_y", "kite_0_vy"])
        vz = select_first(flight_data, ["kite_velocity_z", "kite_0_vz"])

    if any(s is None for s in (px, py, pz, vx, vy, vz)):
        raise ValueError("Missing position or velocity columns for spherical heading.")

    pos = np.column_stack([np.asarray(px), np.asarray(py), np.asarray(pz)])
    vel = np.column_stack([np.asarray(vx), np.asarray(vy), np.asarray(vz)])

    n = pos.shape[0]
    yaw = np.full(n, np.nan, dtype=float)
    last = 0.0
    have_last = False

    for k in range(n):
        p = pos[k]
        v = vel[k]
        p_norm = np.linalg.norm(p)
        v_norm = np.linalg.norm(v)
        if p_norm <= eps or v_norm <= eps:
            if have_last:
                yaw[k] = last
            continue

        radial = p / p_norm
        tang_vel = v - np.dot(v, radial) * radial
        tv_norm = np.linalg.norm(tang_vel)
        if tv_norm <= eps:
            if have_last:
                yaw[k] = last
            continue
        tang_vel /= tv_norm

        up_z = radial
        up_y_raw = np.array([-p[1], p[0], 0.0], dtype=float)
        uy_norm = np.linalg.norm(up_y_raw)
        if uy_norm <= eps:
            if have_last:
                yaw[k] = last
            continue
        up_y = up_y_raw / uy_norm

        up_x = np.cross(up_z, up_y)
        ux_norm = np.linalg.norm(up_x)
        if ux_norm <= eps:
            if have_last:
                yaw[k] = last
            continue
        up_x /= ux_norm
        up_y = np.cross(up_z, up_x)  # re-orthonormalize

        R_up = np.column_stack([up_x, up_y, up_z])
        heading_vec = R_up.T @ tang_vel
        yaw_k = np.arctan2(heading_vec[1], heading_vec[0])

        yaw[k] = yaw_k
        last = yaw_k
        have_last = True

    heading_unwrapped = np.unwrap(yaw)
    yaw_rate = np.gradient(heading_unwrapped, time)
    if smooth_window > 1:
        yaw_rate = np.convolve(
            yaw_rate, np.ones(smooth_window) / smooth_window, mode="same"
        )

    return heading_unwrapped, yaw_rate


# def find_default_hdf5() -> Optional[Path]:
#     candidates = sorted(Path(".").rglob("*.h5"))
#     if not candidates:
#         return None
#     if len(candidates) == 1:
#         return candidates[0]
#     # Prefer files under results/ if multiple exist
#     results_candidates = [c for c in candidates if "results" in c.parts]
#     if len(results_candidates) == 1:
#         return results_candidates[0]
#     return candidates[0]


def load_data(args):
    path_to_h5 = (
        "/home/jellepoland/ownCloud/phd/code/EKF-AWE/results/v3/v3_2019-10-08.h5"
    )
    path_to_h5 = (
        "/home/jellepoland/ownCloud/phd/code/EKF-AWE/results/v3/v3_2025-10-09.h5"
    )
    return read_results_from_hdf5(path_to_h5)

    # # Priority: explicit --hdf5, then explicit year/month/day/model, else auto-detect a lone HDF5.
    # if args.hdf5:
    #     return read_results_from_hdf5(args.hdf5)
    # if all([args.year, args.month, args.day, args.kite_model]):
    #     return read_results(
    #         args.year,
    #         args.month,
    #         args.day,
    #         args.kite_model,
    #         addition=args.addition or "",
    #     )

    # auto = find_default_hdf5()
    # if auto:
    #     print(f"Using detected HDF5: {auto}")
    #     return read_results_from_hdf5(auto)

    # raise SystemExit(
    #     "No HDF5 results file found. Run your pipeline first (e.g., examples/run_analysis.py "
    #     "with process_v3_data.py) to generate a results/*.h5 file."
    # )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot fitted yaw rate with two yaw-rate definitions."
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        help="Path to a results HDF5 file (optional if there is only one available).",
    )
    parser.add_argument("--year", type=str, help="Year of the dataset (e.g., 2019).")
    parser.add_argument("--month", type=str, help="Month of the dataset (e.g., 10).")
    parser.add_argument("--day", type=str, help="Day of the dataset (e.g., 08).")
    parser.add_argument("--kite-model", type=str, help="Kite model name (e.g., v3).")
    parser.add_argument(
        "--addition", type=str, default="", help="Optional filename suffix, e.g., _va."
    )
    parser.add_argument(
        "--wind-dir-column",
        type=str,
        default=None,
        help="Column name for wind direction (rad or deg).",
    )
    parser.add_argument(
        "--heading-column",
        type=str,
        default=None,
        help="Optional heading column to differentiate directly.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Moving-average window for yaw rate smoothing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results, flight_data, config_data = load_data(args)

    # 1) Standard Fig. 16-style plot.
    plot_identify_turn_dynamics(
        results.copy(),
        flight_data.copy(),
        config_data,
        title="Python (& paper?) Yaw Rate Definition",
    )

    # 2) Same plot with yaw_rate redefined as d/dt of wind-perpendicular heading.
    fd_perp = flight_data.copy()
    _, yaw_rate_perp = compute_wind_perp_heading_rate(
        fd_perp,
        results.copy(),
        wind_dir_col=args.wind_dir_column,
        heading_col=args.heading_column,
        smooth_window=args.smooth_window,
    )
    fd_perp["kite_yaw_rate"] = yaw_rate_perp
    plot_identify_turn_dynamics(
        results.copy(),
        fd_perp,
        config_data,
        title="Wind-Perpendicular Yaw Rate Definition",
    )

    # 3) Wind-perpendicular heading rate with mean (frozen) wind direction
    fd_perp_mean = flight_data.copy()
    _, yaw_rate_perp_mean = compute_wind_perp_heading_rate(
        fd_perp_mean,
        results.copy(),
        wind_dir_col=args.wind_dir_column,
        heading_col=args.heading_column,
        smooth_window=args.smooth_window,
        use_mean_wind_dir=True,
    )
    fd_perp_mean["kite_yaw_rate"] = yaw_rate_perp_mean
    plot_identify_turn_dynamics(
        results.copy(),
        fd_perp_mean,
        config_data,
        title="Wind-Perpendicular Averaged Wind-Direction",
    )

    # 3) Spherical/tangent-plane heading rate
    fd_sph = flight_data.copy()
    _, yaw_rate_sph = compute_heading_spherical(
        fd_sph,
        results.copy(),
        smooth_window=args.smooth_window,
    )
    fd_sph["kite_yaw_rate"] = yaw_rate_sph
    plot_identify_turn_dynamics(
        results.copy(),
        fd_sph,
        config_data,
        title="Spherical/Tangent-Plane Yaw Rate Definition",
    )

    plt.show()


if __name__ == "__main__":
    main()
