"""
Two-column yaw-rate plot (2019 vs 2025), colored by heading.

Behavior:
- 1 row, 2 columns (2019, 2025)
- Experimental data only
- Shared continuous heading colormap across both years
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
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import linregress

from awes_ekf.load_data.read_data import read_results
from awes_ekf.plotting.color_palette import set_plot_style


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


@dataclass(frozen=True)
class OverlayFitSummary:
    up: float
    transient_gk: float
    transient_r2: float
    dynamic_gk: float
    dynamic_r2: float


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
    """Load varying-up simulation CSV used for overlays."""
    base = Path(__file__).resolve().parents[1] / "data"
    candidates = [
        # base / "circle_batch_analysis_varying_up_manually_reduced.csv",
        base
        / "circles_batch_analysis_2025_3March.csv",
    ]

    for path in candidates:
        if path.is_file():
            df = pd.read_csv(path)
            print(f"Loaded simulation data from {path}")
            return df

    print("Simulation CSV not found for overlay plotting (varying-up).")
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


def map_heading_to_viridis_mirrored(
    heading_deg: np.ndarray, heading_norm: Normalize
) -> np.ndarray:
    """Map headings to [0,1] using 315->180 (forward) and 180->45 (reversed)."""
    vmin = float(heading_norm.vmin)
    vmax = float(heading_norm.vmax)
    vmid = 0.5 * (vmin + vmax)
    span = vmax - vmid
    if span <= 0.0:
        return np.full_like(heading_deg, np.nan, dtype=float)

    values = np.full_like(heading_deg, np.nan, dtype=float)

    upper_mask = (heading_deg >= vmid) & (heading_deg <= vmax)
    if np.any(upper_mask):
        values[upper_mask] = (vmax - heading_deg[upper_mask]) / span

    lower_mask = (heading_deg >= vmin) & (heading_deg < vmid)
    if np.any(lower_mask):
        values[lower_mask] = (heading_deg[lower_mask] - vmin) / span

    return np.clip(values, 0.0, 1.0)


def format_compact_latex_legend(
    title: str,
    metric_1_tex: str,
    metric_1_value_tex: str,
    metric_2_tex: str,
    metric_2_value_tex: str,
    title_width_cm: float = 2.7,
    metric_width_cm: float = 1.0,
    value_1_width_cm: float = 1.5,
    value_2_width_cm: float = 1.0,
) -> str:
    """Create a compact one-line LaTeX legend label with aligned fields."""
    return (
        rf"\makebox[{title_width_cm:.2f}cm][l]{{{title}}}"
        rf"\makebox[{metric_width_cm:.2f}cm][l]{{$ {metric_1_tex} $}}"
        rf"$=$"
        rf"\makebox[{value_1_width_cm:.2f}cm][l]{{$ {metric_1_value_tex} $}}"
        rf"$,\ $"
        rf"\makebox[{metric_width_cm:.2f}cm][l]{{$ {metric_2_tex} $}}"
        rf"$=$"
        rf"\makebox[{value_2_width_cm:.2f}cm][l]{{$ {metric_2_value_tex} $}}"
    )


def plot_heading_binned_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    heading_norm: Normalize,
    cmap,
) -> list[tuple[str, str, float, str]]:
    """Scatter data and overlay 2 linear fits + 2D-based uncertainty bands."""
    fit_legend_entries: list[tuple[str, str, float, str]] = []
    if df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fit_legend_entries

    heading_math = df["heading"].to_numpy(dtype=float)
    # Convert back to navigation heading for 360deg coloring:
    # 0deg = North, clockwise positive.
    heading_nav = np.mod((np.pi / 2.0) - heading_math, 2.0 * np.pi)
    heading_nav_deg = np.mod(np.rad2deg(heading_nav), 360.0)

    # Keep mirrored orientation only for the up/down split in linear fits.
    heading_orient = np.arctan2(np.sin(heading_math), np.abs(np.cos(heading_math)))
    us_signed = df["us_signed"].to_numpy(dtype=float)
    plot_left_mask = us_signed < -0.05
    heading_color_vals = map_heading_to_viridis_mirrored(heading_nav_deg, heading_norm)
    scatter_mask = plot_left_mask & np.isfinite(heading_color_vals)

    if np.any(scatter_mask):
        ax.scatter(
            df.loc[scatter_mask, "x"],
            df.loc[scatter_mask, "y"],
            c=heading_color_vals[scatter_mask],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=30,
            alpha=0.55,
            marker="o",
            edgecolors="none",
            linewidths=0,
            zorder=1,
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
            "Exp. fit upward",
            "upward",
        ),
        (
            heading_orient < 0.0,
            "--",
            "Exp. fit downward",
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
            x_line, y_line, color="black", linestyle=linestyle, linewidth=1.5, zorder=2
        )

        fit_other = fit_linear(direction_mask & fit_us_mask_other)
        gk_lr_label = rf"{slope:.2f}-\mathrm{{n/a}}"
        r2_lr_label = r"\mathrm{n/a}"
        if fit_other is not None:
            slope_other, _, r2_other, _ = fit_other
            gk_deviation = abs(slope - slope_other)
            slope_low = min(slope, slope_other)
            slope_high = max(slope, slope_other)
            gk_lr_label = rf"{slope_low:.2f}-{slope_high:.2f}"
            r2_lr_label = rf"{0.5 * (r2_main + r2_other):.2f}"
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
                1.5,
                format_compact_latex_legend(
                    title=direction_label,
                    metric_1_tex=r"g_\mathrm{k,l-r}",
                    metric_1_value_tex=gk_lr_label,
                    metric_2_tex=r"\bar{R}_\mathrm{l-r}^{2}",
                    metric_2_value_tex=r2_lr_label,
                ),
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


def _fit_line(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float, np.ndarray, np.ndarray, int] | None:
    """Fit a linear line y = g_k*x + b and return slope/R²/line points."""
    finite = np.isfinite(x) & np.isfinite(y)
    x_fit = np.asarray(x[finite], dtype=float)
    y_fit = np.asarray(y[finite], dtype=float)
    if x_fit.size < 2 or np.ptp(x_fit) <= 1e-12:
        return None
    slope, intercept, r_value, _, _ = linregress(x_fit, y_fit)
    x_line = np.linspace(float(np.min(x_fit)), float(np.max(x_fit)), 200)
    y_line = slope * x_line + intercept
    return float(slope), float(r_value**2), x_line, y_line, int(x_fit.size)


def overlay_simulation_case(
    ax: plt.Axes, sim_df: pd.DataFrame, up_value: float
) -> tuple[list[tuple[str, str, float, str]], OverlayFitSummary, float, float]:
    """Overlay one simulation up-case on an experimental panel in black."""
    legend_entries: list[tuple[str, str, float, str]] = []
    summary = OverlayFitSummary(
        up=up_value,
        transient_gk=np.nan,
        transient_r2=np.nan,
        dynamic_gk=np.nan,
        dynamic_r2=np.nan,
    )
    if sim_df.empty:
        return legend_entries, summary, 0.0, 0.0

    required = {"up", "us", "v_app", "yaw_rate"}
    if not required.issubset(sim_df.columns):
        return legend_entries, summary, 0.0, 0.0

    up_arr = sim_df["up"].to_numpy(dtype=float)
    mask = np.isfinite(up_arr) & np.isclose(up_arr, up_value, atol=1e-9)
    if not np.any(mask):
        print(f"Simulation overlay: no rows for u_dp={up_value:.2f}")
        return legend_entries, summary, 0.0, 0.0

    df_up = sim_df.loc[mask]

    # Transient points: base simulation trajectory.
    x_transient = np.abs(
        df_up["us"].to_numpy(dtype=float) * df_up["v_app"].to_numpy(dtype=float)
    )
    y_transient = np.abs(np.deg2rad(df_up["yaw_rate"].to_numpy(dtype=float)))
    finite_transient = np.isfinite(x_transient) & np.isfinite(y_transient)
    x_transient = x_transient[finite_transient]
    y_transient = y_transient[finite_transient]
    if x_transient.size > 0:
        ax.scatter(
            x_transient,
            y_transient,
            s=40,
            alpha=1,
            marker="X",
            facecolors="white",
            edgecolors="red",
            linewidths=1.0,
            zorder=4,
        )

    # Dynamic points: harmonic cases.
    dyn_x_all = []
    dyn_y_all = []
    for n in range(3, 11):
        usva_col = f"usva_{n}"
        yaw_col = f"yaw_rate_{n}"
        if usva_col not in df_up.columns or yaw_col not in df_up.columns:
            continue
        x_dyn = np.abs(df_up[usva_col].to_numpy(dtype=float))
        y_dyn = np.abs(np.deg2rad(df_up[yaw_col].to_numpy(dtype=float)))
        finite_dyn = np.isfinite(x_dyn) & np.isfinite(y_dyn)
        if np.any(finite_dyn):
            dyn_x_all.append(x_dyn[finite_dyn])
            dyn_y_all.append(y_dyn[finite_dyn])

    if dyn_x_all:
        x_dynamic = np.concatenate(dyn_x_all)
        y_dynamic = np.concatenate(dyn_y_all)
        ax.scatter(
            x_dynamic,
            y_dynamic,
            s=25,
            alpha=0.9,
            marker="+",
            color="red",
            zorder=3,
        )
    else:
        x_dynamic = np.asarray([], dtype=float)
        y_dynamic = np.asarray([], dtype=float)

    x_max = 0.0
    y_max = 0.0
    if x_transient.size > 0:
        x_max = max(x_max, float(np.max(x_transient)))
        y_max = max(y_max, float(np.max(y_transient)))
    if x_dynamic.size > 0:
        x_max = max(x_max, float(np.max(x_dynamic)))
        y_max = max(y_max, float(np.max(y_dynamic)))

    fit_transient = _fit_line(x_transient, y_transient)
    if fit_transient is None:
        label_transient = (
            rf"Transient sim ($u_{{\mathrm{{dp}}}}={up_value:.2f}$)"
            + "\n"
            + rf"$R^2=\mathrm{{n/a}},\ g_{{\mathrm{{k}}}}=\mathrm{{n/a}}$"
        )
    else:
        gk_t, r2_t, x_line_t, y_line_t, n_t = fit_transient
        summary = OverlayFitSummary(
            up=summary.up,
            transient_gk=gk_t,
            transient_r2=r2_t,
            dynamic_gk=summary.dynamic_gk,
            dynamic_r2=summary.dynamic_r2,
        )
        print(
            f"Overlay transient fit (u_dp={up_value:.2f}): "
            f"n={n_t}, g_k={gk_t:.3f}, R^2={r2_t:.3f}"
        )
        label_transient = (
            rf"Transient sim ($u_{{\mathrm{{dp}}}}={up_value:.2f}$)"
            + "\n"
            + rf"$R^2={r2_t:.2f},\ g_{{\mathrm{{k}}}}={gk_t:.2f}$"
        )
    fit_dynamic = _fit_line(x_dynamic, y_dynamic)
    if fit_dynamic is None:
        label_dynamic = (
            rf"Dynamic sim ($u_{{\mathrm{{dp}}}}={up_value:.2f}$)"
            + "\n"
            + rf"$R^2=\mathrm{{n/a}},\ g_{{\mathrm{{k}}}}=\mathrm{{n/a}}$"
        )
    else:
        gk_d, r2_d, x_line_d, y_line_d, n_d = fit_dynamic
        summary = OverlayFitSummary(
            up=summary.up,
            transient_gk=summary.transient_gk,
            transient_r2=summary.transient_r2,
            dynamic_gk=gk_d,
            dynamic_r2=r2_d,
        )
        print(
            f"Overlay dynamic fit (u_dp={up_value:.2f}): "
            f"n={n_d}, g_k={gk_d:.3f}, R^2={r2_d:.3f}"
        )
        label_dynamic = (
            rf"Dynamic sim ($u_{{\mathrm{{dp}}}}={up_value:.2f}$)"
            + "\n"
            + rf"$R^2={r2_d:.2f},\ g_{{\mathrm{{k}}}}={gk_d:.2f}$"
        )
    return legend_entries, summary, x_max, y_max


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
    heading_tick_deg: np.ndarray,
    heading_tick_labels: list[str],
    heading_norm: Normalize,
    cmap,
    tick_fontsize: float,
) -> None:
    """Add a circular heading legend for full 360deg navigation headings."""
    bbox = anchor_ax.get_position()
    wheel_w = 0.3 * bbox.width
    wheel_h = 0.3 * bbox.height
    wheel_x = bbox.x0 - 0.05 * bbox.width
    wheel_y = bbox.y0 + 0.28 * bbox.height
    wheel_ax = fig.add_axes([wheel_x, wheel_y, wheel_w, wheel_h], projection="polar")
    n_segments = int(heading_norm.vmax - heading_norm.vmin)
    heading_vals_deg = np.linspace(
        heading_norm.vmin, heading_norm.vmax, n_segments, endpoint=False
    )
    heading_color_vals = map_heading_to_viridis_mirrored(heading_vals_deg, heading_norm)
    theta_edges = np.deg2rad(heading_vals_deg)
    step_deg = (heading_norm.vmax - heading_norm.vmin) / n_segments
    step = np.deg2rad(step_deg)

    wheel_ax.bar(
        theta_edges,
        np.ones(n_segments),
        width=step,
        bottom=0.0,
        align="edge",
        color=cmap(heading_color_vals),
        edgecolor="white",
        linewidth=0.0,
    )

    wheel_ax.set_theta_zero_location("N")
    wheel_ax.set_theta_direction(-1)
    wheel_ax.set_ylim(0.0, 1.0)
    wheel_ax.set_yticks([])
    wheel_ax.grid(False)

    wheel_ax.set_xticks(np.deg2rad(heading_tick_deg))
    wheel_ax.set_xticklabels(heading_tick_labels, fontsize=tick_fontsize)
    wheel_ax.tick_params(pad=3)


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
            time_range=(2190, 2255),
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
            time_range=(700, 800),
            downsample_frac=1.0,
            apply_quadrant_filter=True,
            y_exclude_threshold_deg=10.0,
        ),
    ]

    datasets = [load_filtered_dataset(cfg) for cfg in datasets_cfg]
    sim_varying_df = load_simulation_varying_dataset()

    heading_tick_deg = np.arange(45.0, 315.0 + 1e-9, 45.0)
    heading_tick_labels = [rf"{int(v):d}$^{{\circ}}$" for v in heading_tick_deg]
    cmap = plt.get_cmap("viridis")
    heading_norm = Normalize(vmin=45.0, vmax=315.0)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    fig.subplots_adjust(bottom=0.3, wspace=0.14)

    fit_legend_entries_per_ax: list[list[tuple[str, str, float, str]]] = []
    for ax, cfg, df in zip(axes, datasets_cfg, datasets):
        fit_legend_entries_per_ax.append(
            plot_heading_binned_panel(ax, df, cfg.title, heading_norm, cmap)
        )

    overlay_up_values = [0.25, 0.42]
    sim_overlay_summaries: list[OverlayFitSummary] = []
    sim_case_limits: list[tuple[float, float]] = []
    for ax, up_value in zip(axes, overlay_up_values):
        _overlay_entries, overlay_summary, sim_x_max, sim_y_max = (
            overlay_simulation_case(ax, sim_varying_df, up_value)
        )
        sim_overlay_summaries.append(overlay_summary)
        sim_case_limits.append((sim_x_max, sim_y_max))

    for ax, cfg, overlay_summary in zip(axes, datasets_cfg, sim_overlay_summaries):
        ax.set_title(rf"{cfg.title} ($u_{{\mathrm{{dp}}}}={overlay_summary.up:.2f}$)")

    axes[0].set_ylabel(r"$|\dot{\psi}|$ ($\mathrm{rad\,s^{-1}}$)")
    axes[1].set_ylabel("")

    # Absolute x-axis for all panels.
    all_x = np.concatenate([df["x"].to_numpy() for df in datasets if not df.empty])
    sim_x_abs = max((limits[0] for limits in sim_case_limits), default=0.0)
    sim_y_abs = max((limits[1] for limits in sim_case_limits), default=0.0)
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
        anchor_ax=axes[0],
        heading_tick_deg=heading_tick_deg,
        heading_tick_labels=heading_tick_labels,
        heading_norm=heading_norm,
        cmap=cmap,
        tick_fontsize=font_sizes["tick"],
    )

    for ax_idx, (ax, panel_entries, overlay_summary) in enumerate(
        zip(axes, fit_legend_entries_per_ax, sim_overlay_summaries)
    ):
        transient_gk_label = (
            f"{overlay_summary.transient_gk:.2f}"
            if np.isfinite(overlay_summary.transient_gk)
            else r"\mathrm{n/a}"
        )
        transient_r2_label = (
            f"{overlay_summary.transient_r2:.2f}"
            if np.isfinite(overlay_summary.transient_r2)
            else r"\mathrm{n/a}"
        )
        transient_label = format_compact_latex_legend(
            title="Sim. uniform",
            metric_1_tex=r"g_\mathrm{k}",
            metric_1_value_tex=transient_gk_label,
            metric_2_tex=r"R^{2}",
            metric_2_value_tex=transient_r2_label,
        )

        dynamic_gk_label = (
            f"{overlay_summary.dynamic_gk:.2f}"
            if np.isfinite(overlay_summary.dynamic_gk)
            else r"\mathrm{n/a}"
        )
        dynamic_r2_label = (
            f"{overlay_summary.dynamic_r2:.2f}"
            if np.isfinite(overlay_summary.dynamic_r2)
            else r"\mathrm{n/a}"
        )
        dynamic_label = format_compact_latex_legend(
            title="Sim. dynamic",
            metric_1_tex=r"g_\mathrm{k}",
            metric_1_value_tex=dynamic_gk_label,
            metric_2_tex=r"R^{2}",
            metric_2_value_tex=dynamic_r2_label,
        )

        transient_marker_size = float(np.sqrt(34.0))
        dynamic_marker_size = float(np.sqrt(18.0))
        sim_marker_handles = [
            Line2D(
                [0],
                [0],
                marker="X",
                linestyle="None",
                markersize=transient_marker_size,
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=1.0,
                label=transient_label,
            ),
            Line2D(
                [0],
                [0],
                marker="+",
                linestyle="None",
                markersize=dynamic_marker_size,
                color="red",
                label=dynamic_label,
            ),
        ]

        fit_handles = []
        seen_fit_labels = set()
        for kind, linestyle, linewidth, label in panel_entries:
            if label in seen_fit_labels:
                continue
            seen_fit_labels.add(label)
            if kind == "line":
                fit_handles.append(
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
                fit_handles.append(
                    Patch(facecolor="0.5", edgecolor="none", alpha=0.40, label=label)
                )

        remaining_fit_handles = list(fit_handles)
        remaining_sim_handles = list(sim_marker_handles)

        def pop_first_with_prefix(handles: list, prefix: str):
            for idx, handle in enumerate(handles):
                if handle.get_label().startswith(prefix):
                    return handles.pop(idx)
            return None

        combined_handles = []
        for prefix in ("Upward", "Downward"):
            handle = pop_first_with_prefix(remaining_fit_handles, prefix)
            if handle is not None:
                combined_handles.append(handle)
        for prefix in ("Dynamic", "Transient"):
            handle = pop_first_with_prefix(remaining_sim_handles, prefix)
            if handle is not None:
                combined_handles.append(handle)

        combined_handles.extend(remaining_fit_handles)
        combined_handles.extend(remaining_sim_handles)

        combined_legend = ax.legend(
            handles=combined_handles,
            loc="upper left",
            fontsize=font_sizes["legend"],
            frameon=True,
            handlelength=1.4,
            borderpad=0.3,
            labelspacing=0.3,
        )
        if ax_idx == 0:
            ax.add_artist(combined_legend)

    y_lim = 2.25
    x_lim = 8
    for ax in axes:
        ax.set_xlim(0.5, x_lim)
        ax.set_ylim(0.0, y_lim)

    fig.tight_layout()
    output_path = (
        Path("./results/plots_paper") / "gk_2col_filter_heading_absolute_trial.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
