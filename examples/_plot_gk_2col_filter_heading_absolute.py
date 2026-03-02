"""
Two-column yaw-rate plot (2019 vs 2025), colored by heading bins.

Behavior:
- 1 row, 3 columns (2019, 2025, simulation)
- Experimental data only
- Shared heading bins across both years
- Two linear fit lines per panel (upward/downward), fitted only for u_s < -0.05
- 2D fits are computed/printed only for u_s < -0.05
- Uncertainty-band plotting is currently disabled (kept commented out)
- 2D fit results are printed (g_k, M, R^2), not plotted
- Absolute x-axis (0 to positive), combining left and right turns
- Absolute yaw-rate axis in rad/s (0 to positive)
- Filtering retained: time-window + powered-only + downsampling
- Scatter plotting only for turn cases: u_s < -0.05 (left) or u_s > 0.05 (right)
- Heading coloring and wheel use full 360-degree flight direction
"""

from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import linregress

from awes_ekf.load_data.read_data import read_results
from awes_ekf.plotting.color_palette import set_plot_style

RIGHT_TURN_HATCH = "///"


@dataclass(frozen=True)
class FlightConfig:
    title: str
    year: str
    month: str
    day: str
    kite_model: str
    addition: str
    time_range: tuple[float, float]
    downsample_frac: float
    apply_quadrant_filter: bool = False
    y_exclude_threshold_deg: float = 90.0


def configure_typography(fig_width: float, fig_height: float) -> dict[str, float]:
    """Scale text sizes from figure size while keeping the active font family."""
    base_size = float(np.clip(0.75 * min(fig_width, fig_height) + 5.0, 7.0, 12.5))
    sizes = {
        "base": base_size,
        "title": base_size + 1.2,
        "label": base_size + 0.7,
        "tick": base_size - 0.2,
        "legend": base_size - 0.3,
    }

    plt.rcParams.update(
        {
            "font.size": sizes["base"],
            "axes.titlesize": sizes["title"],
            "axes.labelsize": sizes["label"],
            "xtick.labelsize": sizes["tick"],
            "ytick.labelsize": sizes["tick"],
            "legend.fontsize": sizes["legend"],
            "legend.title_fontsize": sizes["legend"],
            "figure.titlesize": sizes["title"],
        }
    )
    return sizes


def load_simulation_varying_dataset() -> pd.DataFrame:
    """Load varying-up simulation CSV used for the third panel."""
    base = Path(__file__).resolve().parents[1] / "data"
    candidates = [
        base / "circle_batch_analysis_varying_up_manually_reduced.csv",
        # base / "circle_batch_analysis_varying_up.csv",
    ]

    for path in candidates:
        if path.is_file():
            df = pd.read_csv(path)
            print(f"Loaded simulation data from {path}")
            return df

    print("Simulation CSV not found for third panel (varying-up).")
    return pd.DataFrame()


def load_filtered_dataset(cfg: FlightConfig) -> pd.DataFrame:
    """Load one flight and keep only filtered data needed for plotting."""
    results, flight_data, _ = read_results(
        cfg.year, cfg.month, cfg.day, cfg.kite_model, addition=cfg.addition
    )

    time_mask = (results["time"] >= cfg.time_range[0]) & (
        results["time"] <= cfg.time_range[1]
    )
    results = results.loc[time_mask].reset_index(drop=True)
    flight_data = flight_data.loc[time_mask].reset_index(drop=True)

    if "powered" in flight_data.columns:
        powered_mask = flight_data["powered"] == "powered"
        results = results.loc[powered_mask].reset_index(drop=True)
        flight_data = flight_data.loc[powered_mask].reset_index(drop=True)

    if cfg.downsample_frac < 1.0:
        sampled = flight_data.sample(frac=cfg.downsample_frac, random_state=42)
        results = results.loc[sampled.index]
        flight_data = sampled

    yaw_rate_col = (
        "kite_yaw_rate_1"
        if "kite_yaw_rate_1" in flight_data.columns
        else "kite_yaw_rate"
    )
    if yaw_rate_col not in flight_data.columns:
        raise ValueError(f"Missing yaw-rate column in {cfg.title}")

    required = ["kcu_actual_steering", "kite_heading", "kite_elevation"]
    for col in required:
        if col not in flight_data.columns:
            raise ValueError(f"Missing column '{col}' in {cfg.title}")
    if "kite_apparent_windspeed" not in results.columns:
        raise ValueError(
            f"Missing column 'kite_apparent_windspeed' in results for {cfg.title}"
        )

    va = results["kite_apparent_windspeed"].to_numpy(dtype=float)
    steering = flight_data["kcu_actual_steering"].to_numpy(dtype=float)
    us_signed = steering / 100.0
    x_signed = -us_signed * va
    x = np.abs(x_signed)
    y_signed = flight_data[yaw_rate_col].to_numpy(dtype=float)
    y = np.abs(y_signed)
    heading_raw = flight_data["kite_heading"].to_numpy(dtype=float)
    # Support both heading encodings:
    # - radians in [0, 2pi]
    # - degrees in [0, 360]
    if np.nanmax(np.abs(heading_raw)) > (2.0 * np.pi + 0.5):
        heading_nav = np.deg2rad(np.mod(heading_raw, 360.0))
    else:
        heading_nav = np.mod(heading_raw, 2.0 * np.pi)

    # Convert navigation bearing (0=north, clockwise positive) to mathematical
    # angle (0=right/east, counter-clockwise positive).
    heading = np.mod((np.pi / 2.0) - heading_nav, 2.0 * np.pi)

    # Eq. 8.41 uses orientation angles wrt gravity (Table 8.1): psi_k, theta_k.
    # Use control-pod yaw angle as psi_k when available; fallback to heading only
    # for datasets where yaw is unavailable.
    if "kite_yaw_0" in flight_data.columns:
        psi_k = flight_data["kite_yaw_0"].to_numpy(dtype=float)
        psi_source = "flight_data.kite_yaw_0"
    else:
        psi_k = heading_nav
        psi_source = "flight_data.kite_heading (fallback)"

    if "kite_elevation" in results.columns:
        theta_k = results["kite_elevation"].to_numpy(dtype=float)
        theta_source = "results.kite_elevation"
    else:
        elevation_raw = flight_data["kite_elevation"].to_numpy(dtype=float)
        if np.nanmax(np.abs(elevation_raw)) > (np.pi + 0.5):
            theta_k = np.deg2rad(elevation_raw)
        else:
            theta_k = elevation_raw
        theta_source = "flight_data.kite_elevation (fallback)"

    print(f"{cfg.title}: using psi_k from {psi_source}, theta_k from {theta_source}.")

    # Eq. 8.41 second regressor:
    # x2 = cos(theta_k) * sin(psi_k) / va
    va_safe = np.where(np.abs(va) > 1e-6, va, np.nan)
    x2_signed = np.cos(theta_k) * np.sin(psi_k) / va_safe

    finite = (
        np.isfinite(x)
        & np.isfinite(y_signed)
        & np.isfinite(heading)
        & np.isfinite(x2_signed)
        & np.isfinite(theta_k)
        & np.isfinite(psi_k)
        & np.isfinite(us_signed)
    )

    if cfg.apply_quadrant_filter:
        # Legacy incorrect-data filter used for 2025:
        # remove high-|yaw-rate| points whose sign conflicts with steering*va sign.
        y_exclude_threshold_rad = np.deg2rad(cfg.y_exclude_threshold_deg)
        mismatch_mask = ((y_signed > y_exclude_threshold_rad) & (x_signed < 0.0)) | (
            (y_signed < -y_exclude_threshold_rad) & (x_signed > 0.0)
        )
        before_n = int(np.count_nonzero(finite))
        finite &= ~mismatch_mask
        after_n = int(np.count_nonzero(finite))
        print(
            f"{cfg.title}: removed {before_n - after_n} quadrant-mismatch points "
            f"(threshold={cfg.y_exclude_threshold_deg:.1f} deg)."
        )

    return pd.DataFrame(
        {
            "x": np.asarray(x[finite]),
            "y": np.asarray(y[finite]),
            "heading": np.asarray(heading[finite]),
            "us_signed": np.asarray(us_signed[finite]),
            "us_was_negative": np.asarray(us_signed[finite] < 0.0),
            "x1_signed": np.asarray(x_signed[finite]),
            "x2_signed": np.asarray(x2_signed[finite]),
            "y_signed": np.asarray(y_signed[finite]),
        }
    )


def fit_two_dimensional_model(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray
) -> tuple[float, float, float, np.ndarray]:
    """Fit y = g_k*x1 + M*x2 using linear least squares."""
    design = np.column_stack([x1, x2])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    gk = float(coeffs[0])
    m = float(coeffs[1])
    y_hat = design @ coeffs

    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    return gk, m, r2, y_hat


def binned_mean_curve(
    x: np.ndarray, y: np.ndarray, n_bins: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Return bin-averaged y(x) to display a smooth model line."""
    if len(x) < 2:
        return x, y

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min <= 1e-12:
        return x, y

    bins = np.linspace(x_min, x_max, n_bins + 1)
    bin_idx = np.digitize(x, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    x_line = []
    y_line = []
    for i in range(n_bins):
        mask = bin_idx == i
        if np.any(mask):
            x_line.append(float(np.mean(x[mask])))
            y_line.append(float(np.mean(y[mask])))

    return np.asarray(x_line), np.asarray(y_line)


def plot_heading_binned_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    heading_bin_centers_deg: np.ndarray,
    cmap,
) -> list[tuple[str, str, float, str]]:
    """Scatter data and overlay 2 linear fits + 2D-based uncertainty bands."""
    fit_legend_entries: list[tuple[str, str, float, str]] = []
    if df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fit_legend_entries

    heading_math = df["heading"].to_numpy(dtype=float)
    # Convert back to navigation heading for 360deg binning:
    # 0deg = North, clockwise positive.
    heading_nav = np.mod((np.pi / 2.0) - heading_math, 2.0 * np.pi)
    heading_nav_deg = np.rad2deg(heading_nav)

    # Deterministic nearest-center assignment on circular distance.
    circ_dist = np.abs(
        ((heading_nav_deg[:, None] - heading_bin_centers_deg[None, :] + 180.0) % 360.0)
        - 180.0
    )
    bin_idx = np.argmin(circ_dist, axis=1)

    # Keep mirrored orientation only for the up/down split in linear fits.
    heading_orient = np.arctan2(np.sin(heading_math), np.abs(np.cos(heading_math)))
    us_signed = df["us_signed"].to_numpy(dtype=float)
    plot_left_mask = us_signed < -0.05
    plot_right_mask = us_signed > 0.05

    for i in range(len(heading_bin_centers_deg)):
        mask_bin = bin_idx == i
        if np.any(mask_bin):
            mask_negative_us = mask_bin & plot_left_mask
            mask_positive_us = mask_bin & plot_right_mask

            if np.any(mask_positive_us):
                ax.scatter(
                    df.loc[mask_positive_us, "x"],
                    df.loc[mask_positive_us, "y"],
                    s=40,
                    alpha=0.55,
                    color=cmap(i),
                    marker="o",
                    hatch=RIGHT_TURN_HATCH,
                    edgecolors="black",
                    linewidths=0.6,
                )

            if np.any(mask_negative_us):
                ax.scatter(
                    df.loc[mask_negative_us, "x"],
                    df.loc[mask_negative_us, "y"],
                    s=40,
                    alpha=0.55,
                    color=cmap(i),
                    marker="o",
                    edgecolors="none",
                    linewidths=0,
                )

    # Plotted lines: simple linear fits split by upward/downward flight.
    x_plot = df["x"].to_numpy(dtype=float)
    y_plot = df["y"].to_numpy(dtype=float)
    fit_us_mask_main = us_signed < -0.05
    fit_us_mask_other = us_signed > 0.05

    def fit_linear(mask: np.ndarray):
        if np.count_nonzero(mask) < 2:
            return None
        x_fit = x_plot[mask]
        y_fit = y_plot[mask]
        if np.ptp(x_fit) <= 1e-12:
            return None
        slope, intercept, r, _, _ = linregress(x_fit, y_fit)
        return slope, intercept, float(r**2), x_fit

    linear_specs = [
        (
            heading_orient >= 0.0,
            "-",
            "Plotted fit (upward, left turn)",
            "upward",
        ),
        (
            heading_orient < 0.0,
            "--",
            "Plotted fit (downward, left turn)",
            "downward",
        ),
    ]
    for direction_mask, linestyle, direction_label, direction_name in linear_specs:
        fit_main = fit_linear(direction_mask & fit_us_mask_main)
        if fit_main is None:
            continue
        slope, intercept, r2_main, x_fit_main = fit_main
        x_line = np.linspace(np.min(x_fit_main), np.max(x_fit_main), 200)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line, y_line, color="black", linestyle=linestyle, linewidth=2.6, zorder=4
        )

        fit_other = fit_linear(direction_mask & fit_us_mask_other)
        gk_label = (
            rf"$g_{{\mathrm{{k,left}}}}={slope:.2f}$, "
            + rf"$g_{{\mathrm{{k,right}}}}=\mathrm{{n/a}}$"
        )
        if fit_other is not None:
            slope_other, _, _, _ = fit_other
            gk_deviation = abs(slope - slope_other)
            gk_label = (
                rf"$g_{{\mathrm{{k,left}}}}={slope:.2f}$, "
                + rf"($g_{{\mathrm{{k,right}}}}={slope_other:.2f}$)"
                # + rf"$\Delta g_{{\mathrm{{k}}}}={gk_deviation:.2f}$"
            )
            print(
                f"{title} linear fit ({direction_name}): "
                f"g_k(u_s<-0.05)={slope:.3f}, "
                f"g_k(u_s>0.05)={slope_other:.3f}, "
                f"deviation={gk_deviation:.3f}"
            )

        fit_legend_entries.append(
            (
                "line",
                linestyle,
                2.6,
                direction_label
                + "\n"
                + rf"$R^2_{{\mathrm{{left}}}}={r2_main:.2f}, $"
                + gk_label,
            )
        )

    # Disabled on request:
    # extra full linear fits by steering sign (left/right proxy).

    # 2D model from Eq. 8.41:
    # y = g_k*x1 + M*x2, with x1 = va*delta and x2 = cos(theta_k)*sin(psi_k)/va.
    x1_model = df["x1_signed"].to_numpy(dtype=float)
    x2_model = df["x2_signed"].to_numpy(dtype=float)
    y_model = df["y_signed"].to_numpy(dtype=float)

    def fit_2d_curve(mask: np.ndarray):
        if np.count_nonzero(mask) < 3:
            return None
        x1_fit = x1_model[mask]
        x2_fit = x2_model[mask]
        y_fit = y_model[mask]
        design = np.column_stack([x1_fit, x2_fit])
        if np.linalg.matrix_rank(design) < 2:
            return None

        gk_2d, m_2d, r2_2d, y_hat_signed = fit_two_dimensional_model(
            x1_fit, x2_fit, y_fit
        )
        x_curve, y_curve = binned_mean_curve(
            np.abs(x1_fit), np.abs(y_hat_signed), n_bins=40
        )
        if len(x_curve) <= 1:
            return None
        order = np.argsort(x_curve)
        return (
            gk_2d,
            m_2d,
            r2_2d,
            x_curve[order],
            y_curve[order],
            int(np.count_nonzero(mask)),
        )

    # Hidden 2D fits used for uncertainty and printed diagnostics.
    up_mask = heading_orient >= 0.0
    down_mask = heading_orient < 0.0
    us_fit_mask = us_signed < -0.05

    fit_up_pos = fit_2d_curve(up_mask & us_fit_mask)
    fit_down_pos = fit_2d_curve(down_mask & us_fit_mask)

    def print_fit_result(fit, direction: str, us_group: str) -> None:
        if fit is None:
            print(f"{title} 2D fit ({direction}, {us_group}): insufficient data")
            return
        gk_fit, m_fit, r2_fit, _, _, n_fit = fit
        print(
            f"{title} 2D fit ({direction}, {us_group}): "
            f"n={n_fit}, g_k={gk_fit:.3f}, M={m_fit:.3f}, R^2={r2_fit:.3f}"
        )

    # Focus 2D-fit reporting on u_s < -0.05 only (edge case).
    print_fit_result(fit_up_pos, "upward", "u_s<-0.05")
    print_fit_result(fit_down_pos, "downward", "u_s<-0.05")

    def add_band(fit_a, fit_b, label: str) -> None:
        if fit_a is None or fit_b is None:
            return
        _, _, _, x_a, y_a, _ = fit_a
        _, _, _, x_b, y_b, _ = fit_b
        x_min = max(float(np.min(x_a)), float(np.min(x_b)))
        x_max = min(float(np.max(x_a)), float(np.max(x_b)))
        if x_max <= x_min + 1e-9:
            return

        x_common = np.linspace(x_min, x_max, 250)
        y_a_i = np.interp(x_common, x_a, y_a)
        y_b_i = np.interp(x_common, x_b, y_b)
        y_low = np.minimum(y_a_i, y_b_i)
        y_high = np.maximum(y_a_i, y_b_i)
        ax.fill_between(
            x_common,
            y_low,
            y_high,
            color="0.5",
            alpha=0.80,
            linewidth=0.0,
            zorder=0,
        )
        fit_legend_entries.append(("band", "", 0.0, label))

    # Uncertainty-band plotting intentionally disabled.

    ax.set_title(title)
    ax.set_xlabel(r"$|u_\mathrm{s}v_\mathrm{a}|$ ($\mathrm{m\,s^{-1}}$)")
    ax.grid(True, alpha=0.25)

    return fit_legend_entries


def plot_simulation_panel(
    ax: plt.Axes, sim_df: pd.DataFrame, title: str
) -> list[tuple[float, float, float, object]]:
    """Plot varying-up simulation data in absolute coordinates (rad/s)."""
    sim_fit_entries: list[tuple[float, float, float, object]] = []
    ax.set_title(title)
    ax.set_xlabel(r"$|u_\mathrm{s}v_\mathrm{a}|$ ($\mathrm{m\,s^{-1}}$)")
    ax.grid(True, alpha=0.25)

    if sim_df.empty:
        ax.text(
            0.5,
            0.5,
            "No simulation data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return sim_fit_entries

    required = {"up", "us", "v_app", "yaw_rate"}
    if not required.issubset(sim_df.columns):
        missing = sorted(required - set(sim_df.columns))
        ax.text(
            0.5,
            0.5,
            f"Missing sim columns:\n{', '.join(missing)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return sim_fit_entries

    up_values = sorted(sim_df["up"].dropna().unique())
    color_cycle = cycle(plt.get_cmap("tab10").colors)

    for up in up_values:
        color = next(color_cycle)
        df_up = sim_df[sim_df["up"] == up]
        fit_x_parts = []
        fit_y_parts = []

        # Uniform simulation points (dot marker).
        x_uniform = np.abs(
            df_up["us"].to_numpy(dtype=float) * df_up["v_app"].to_numpy(dtype=float)
        )
        y_uniform = np.abs(np.deg2rad(df_up["yaw_rate"].to_numpy(dtype=float)))
        finite_uniform = np.isfinite(x_uniform) & np.isfinite(y_uniform)
        if np.any(finite_uniform):
            x_uniform_plot = x_uniform[finite_uniform]
            y_uniform_plot = y_uniform[finite_uniform]
            ax.scatter(
                x_uniform_plot,
                y_uniform_plot,
                s=22,
                alpha=0.85,
                marker=".",
                color=color,
            )
            fit_x_parts.append(x_uniform_plot)
            fit_y_parts.append(y_uniform_plot)

        # Transient harmonics points (open-circle marker).
        transient_x_all = []
        transient_y_all = []
        for n in range(3, 11):
            usva_col = f"usva_{n}"
            yaw_col = f"yaw_rate_{n}"
            if usva_col not in df_up.columns or yaw_col not in df_up.columns:
                continue
            x_dyn = np.abs(df_up[usva_col].to_numpy(dtype=float))
            y_dyn = np.abs(np.deg2rad(df_up[yaw_col].to_numpy(dtype=float)))
            finite_dyn = np.isfinite(x_dyn) & np.isfinite(y_dyn)
            if np.any(finite_dyn):
                transient_x_all.append(x_dyn[finite_dyn])
                transient_y_all.append(y_dyn[finite_dyn])

        if transient_x_all:
            x_transient = np.concatenate(transient_x_all)
            y_transient = np.concatenate(transient_y_all)
            ax.scatter(
                x_transient,
                y_transient,
                s=36,
                alpha=0.90,
                marker="o",
                facecolors="none",
                edgecolors=[color],
                linewidths=1.1,
            )
            fit_x_parts.append(x_transient)
            fit_y_parts.append(y_transient)

        gk_fit = np.nan
        r2_fit = np.nan
        if fit_x_parts:
            x_fit = np.concatenate(fit_x_parts)
            y_fit = np.concatenate(fit_y_parts)
            finite_fit = np.isfinite(x_fit) & np.isfinite(y_fit)
            x_fit = x_fit[finite_fit]
            y_fit = y_fit[finite_fit]
            if x_fit.size >= 2 and np.ptp(x_fit) > 1e-12:
                slope, _, r_value, _, _ = linregress(x_fit, y_fit)
                gk_fit = float(slope)
                r2_fit = float(r_value**2)
                print(
                    f"{title} fit (u_dp={float(up):.2f}): "
                    f"n={x_fit.size}, g_k={gk_fit:.3f}, R^2={r2_fit:.3f}"
                )
            else:
                print(
                    f"{title} fit (u_dp={float(up):.2f}): insufficient spread for fit"
                )

        sim_fit_entries.append((float(up), gk_fit, r2_fit, color))

    return sim_fit_entries


def get_simulation_abs_limits(sim_df: pd.DataFrame) -> tuple[float, float]:
    """Return absolute x/y maxima for simulation panel in rad/s."""
    if sim_df.empty:
        return 0.0, 0.0

    x_vals = []
    y_vals = []

    if {"us", "v_app", "yaw_rate"}.issubset(sim_df.columns):
        x_uniform = np.abs(
            sim_df["us"].to_numpy(dtype=float) * sim_df["v_app"].to_numpy(dtype=float)
        )
        y_uniform = np.abs(np.deg2rad(sim_df["yaw_rate"].to_numpy(dtype=float)))
        finite_uniform = np.isfinite(x_uniform) & np.isfinite(y_uniform)
        if np.any(finite_uniform):
            x_vals.append(x_uniform[finite_uniform])
            y_vals.append(y_uniform[finite_uniform])

    for n in range(3, 11):
        usva_col = f"usva_{n}"
        yaw_col = f"yaw_rate_{n}"
        if usva_col not in sim_df.columns or yaw_col not in sim_df.columns:
            continue
        x_dyn = np.abs(sim_df[usva_col].to_numpy(dtype=float))
        y_dyn = np.abs(np.deg2rad(sim_df[yaw_col].to_numpy(dtype=float)))
        finite_dyn = np.isfinite(x_dyn) & np.isfinite(y_dyn)
        if np.any(finite_dyn):
            x_vals.append(x_dyn[finite_dyn])
            y_vals.append(y_dyn[finite_dyn])

    if not x_vals or not y_vals:
        return 0.0, 0.0

    return float(np.max(np.concatenate(x_vals))), float(np.max(np.concatenate(y_vals)))


def add_heading_color_wheel(
    fig: plt.Figure,
    anchor_ax: plt.Axes,
    heading_bin_centers_deg: np.ndarray,
    heading_labels: list[str],
    heading_colors,
    tick_fontsize: float,
    title_fontsize: float,
    right_side_hatch: str = RIGHT_TURN_HATCH,
) -> None:
    """Add a circular heading legend for full 360deg navigation headings."""
    # Place a compact wheel inside the bottom-right of the chosen subplot.
    bbox = anchor_ax.get_position()
    wheel_w = 0.42 * bbox.width
    wheel_h = 0.42 * bbox.height
    wheel_x = bbox.x0 + 0.65 * bbox.width
    wheel_y = bbox.y0 + 0.15 * bbox.height
    wheel_ax = fig.add_axes([wheel_x, wheel_y, wheel_w, wheel_h], projection="polar")
    n_bins = len(heading_colors)
    theta_centers = np.deg2rad(heading_bin_centers_deg)
    step = 2.0 * np.pi / n_bins
    theta_edges = theta_centers - 0.5 * step

    wheel_bars = wheel_ax.bar(
        theta_edges,
        np.ones(n_bins),
        width=step,
        bottom=0.0,
        align="edge",
        color=heading_colors,
        edgecolor="white",
        linewidth=0.8,
    )
    right_side_mask = (heading_bin_centers_deg > 0.0) & (
        heading_bin_centers_deg < 180.0
    )
    for bar, is_right_side in zip(wheel_bars, right_side_mask):
        if not bool(is_right_side):
            continue
        bar.set_hatch(right_side_hatch)
        bar.set_edgecolor("black")
        bar.set_linewidth(0.9)

    wheel_ax.set_theta_zero_location("N")
    wheel_ax.set_theta_direction(-1)
    wheel_ax.set_ylim(0.0, 1.0)
    wheel_ax.set_yticks([])
    wheel_ax.grid(False)

    wheel_ax.set_xticks(theta_centers)
    wheel_ax.set_xticklabels(heading_labels, fontsize=tick_fontsize)
    wheel_ax.tick_params(pad=3)
    wheel_ax.set_title("Flight direction", pad=3, fontsize=title_fontsize)


def main() -> None:
    set_plot_style()
    fig_width, fig_height = 12.4, 4.0
    font_sizes = configure_typography(fig_width, fig_height)

    datasets_cfg = [
        FlightConfig(
            title="2019-10-08",
            year="2019",
            month="10",
            day="08",
            kite_model="v3",
            addition="_t26",
            time_range=(2190, 2255),  # (1800.0, 9986.2),
            downsample_frac=1.0,
            apply_quadrant_filter=False,
        ),
        FlightConfig(
            title="2025-10-09",
            year="2025",
            month="10",
            day="09",
            kite_model="v3",
            addition="",
            time_range=(700, 800),  # (400.0, 1080.0),
            downsample_frac=1.0,
            apply_quadrant_filter=True,
            y_exclude_threshold_deg=10.0,
        ),
    ]

    datasets = [load_filtered_dataset(cfg) for cfg in datasets_cfg]
    sim_varying_df = load_simulation_varying_dataset()

    heading_bin_centers_deg = np.arange(0.0, 360.0, 45.0)
    heading_labels = [
        "0$^{\\circ}$\nN",
        "45$^{\\circ}$\nNE",
        "90$^{\\circ}$\nE",
        "135$^{\\circ}$\nSE",
        "180$^{\\circ}$\nS",
        "225$^{\\circ}$\nSW",
        "270$^{\\circ}$\nW",
        "315$^{\\circ}$\nNW",
    ]
    # Symmetric left/right coloring:
    # NE=NW, E=W, SE=SW (with distinct N and S).
    heading_bin_colors = [
        "#1f77b4",  # N
        "#2ca02c",  # NE
        "#ff7f0e",  # E
        "#d62728",  # SE
        "#9467bd",  # S
        "#d62728",  # SW (same as SE)
        "#ff7f0e",  # W  (same as E)
        "#2ca02c",  # NW (same as NE)
    ]

    cmap = ListedColormap(heading_bin_colors, name="heading_bins")

    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=True)
    fig.subplots_adjust(bottom=0.3, wspace=0.14)

    fit_legend_entries_per_ax: list[list[tuple[str, str, float, str]]] = []
    for ax, cfg, df in zip(axes[:2], datasets_cfg, datasets):
        fit_legend_entries_per_ax.append(
            plot_heading_binned_panel(ax, df, cfg.title, heading_bin_centers_deg, cmap)
        )
    sim_fit_entries = plot_simulation_panel(
        axes[2], sim_varying_df, title="Varying $u_\\mathrm{dp}$ sim."
    )

    axes[0].set_ylabel(r"$|\dot{\psi}|$ ($\mathrm{rad\,s^{-1}}$)")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")

    # Absolute x-axis for all panels.
    all_x = np.concatenate([df["x"].to_numpy() for df in datasets if not df.empty])
    sim_x_abs, sim_y_abs = get_simulation_abs_limits(sim_varying_df)
    x_abs = np.nanmax(np.abs(all_x)) if all_x.size > 0 else 1.0
    x_lim = max(1.0, 1.05 * max(x_abs, sim_x_abs))
    for ax in axes:
        ax.set_xlim(0.0, x_lim)

    all_y = np.concatenate([df["y"].to_numpy() for df in datasets if not df.empty])
    y_abs = np.nanmax(np.abs(all_y)) if all_y.size > 0 else 1.0
    y_lim = max(0.25, 1.05 * max(y_abs, sim_y_abs))
    for ax in axes:
        ax.set_ylim(0.0, y_lim)

    add_heading_color_wheel(
        fig,
        anchor_ax=axes[1],
        heading_bin_centers_deg=heading_bin_centers_deg,
        heading_labels=heading_labels,
        heading_colors=heading_bin_colors,
        tick_fontsize=font_sizes["tick"],
        title_fontsize=font_sizes["label"],
        right_side_hatch=RIGHT_TURN_HATCH,
    )

    # Small in-plot marker legend (only for left subplot).
    marker_handles = [
        Patch(
            facecolor="0.55",
            edgecolor="none",
            label="Left turn ($u_\\mathrm{s}< -0.05$)",
        ),
        Patch(
            facecolor="0.55",
            edgecolor="black",
            linewidth=0.9,
            hatch=RIGHT_TURN_HATCH,
            label="Right turn ($u_\\mathrm{s}>0.05$)",
        ),
    ]
    marker_legend = axes[1].legend(
        handles=marker_handles,
        loc="upper left",
        fontsize=font_sizes["legend"],
        frameon=True,
        handlelength=1.4,
        borderpad=0.3,
        labelspacing=0.3,
    )
    axes[1].add_artist(marker_legend)

    # In-plot marker-style legend for simulation panel (only for right subplot).
    sim_marker_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=5.8,
            markerfacecolor="none",
            markeredgecolor="0.2",
            markeredgewidth=1.0,
            label="Transient",
        ),
        Line2D(
            [0],
            [0],
            marker=".",
            linestyle="None",
            markersize=8.0,
            color="0.2",
            label="Uniform",
        ),
    ]
    sim_marker_legend = axes[2].legend(
        handles=sim_marker_handles,
        loc="upper left",
        fontsize=font_sizes["legend"],
        frameon=True,
        handlelength=1.2,
        borderpad=0.3,
        labelspacing=0.3,
    )
    axes[2].add_artist(sim_marker_legend)

    # One external legend per subplot (below each axis), without marker entries.
    for ax, panel_entries in zip(axes[:2], fit_legend_entries_per_ax):
        legend_handles = []
        seen_labels = {h.get_label() for h in legend_handles}
        for kind, linestyle, linewidth, label in panel_entries:
            if label in seen_labels:
                continue
            seen_labels.add(label)
            if kind == "line":
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color="black",
                        linestyle=linestyle,
                        linewidth=linewidth,
                        label=label,
                    )
                )
            elif kind == "band":
                legend_handles.append(
                    Patch(facecolor="0.5", edgecolor="none", alpha=0.40, label=label)
                )
        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=1,
            fontsize=font_sizes["legend"],
            frameon=False,
            handlelength=1,
            columnspacing=1.2,
        )

    sim_legend_handles = []
    for up, gk_fit, r2_fit, color in sim_fit_entries:
        if np.isfinite(gk_fit) and np.isfinite(r2_fit):
            label = (
                rf"$u_{{\mathrm{{dp}}}}={up:.2f},\ "
                rf"g_{{\mathrm{{k}}}}={gk_fit:.2f},\ R^2={r2_fit:.2f}$"
            )
        else:
            label = (
                rf"$u_{{\mathrm{{dp}}}}={up:.2f},\ "
                rf"g_{{\mathrm{{k}}}}=\mathrm{{n/a}},\ R^2=\mathrm{{n/a}}$"
            )
        sim_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2.2,
                label=label,
            )
        )
    if sim_legend_handles:
        axes[2].legend(
            handles=sim_legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=1,
            fontsize=font_sizes["legend"],
            frameon=False,
            handlelength=1.0,
            columnspacing=1.2,
        )

    y_lim = 2
    x_lim = 8
    for ax in axes:
        ax.set_xlim(0.5, x_lim)
        ax.set_ylim(0.0, y_lim)

    fig.tight_layout()
    output_path = Path("./results/plots_paper") / "gk_2col_filter_heading_absolute.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
