"""
Two-row turn-rate plot (yaw and course rate, 2019 vs 2025), colored by heading.

Behavior:
- 2 rows, 2 columns (yaw/course-rate rows, 2019/2025 columns)
- Experimental data only
- Shared continuous heading colormap across both years
- One linear fit line per panel (2019 and 2025), constrained through (0, 0)
- Plot and fit data use independent filters (configured in SETTINGS)
- 2025 yaw-rate panel drops quadrant-II/IV points by signed x/y angle
- 2025 yaw-rate panel drops high-y/low-x outliers
- Signed x/y axes so left and right turns are shown together
- Heading coloring and wheel use full 360-degree flight direction
- 2019 steering uses an EKF-derived constant offset correction:
    fit y = a(us*va) + b(va), set u_s0 = -b/a, then use u_s - u_s0
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
from utils import (
    PLOT_FILTER_DEFAULT,
    FIT_FILTER_DEFAULT,
    PLOT_FILTER_2019,
    FIT_FILTER_2019,
    PLOT_FILTER_2025,
    FIT_FILTER_2025,
    save_plot_data_to_csv,
)


@dataclass(frozen=True)
class FlightConfig:
    title: str
    year: str
    month: str
    day: str
    kite_model: str
    addition: str
    plot_filter: dict[str, float | bool | tuple[float, float] | None]
    fit_filter: dict[str, float | bool | tuple[float, float] | None]


@dataclass(frozen=True)
class OverlayFitSummary:
    transient_gk: float
    transient_r2: float
    dynamic_gk: float
    dynamic_r2: float


@dataclass(frozen=True)
class RateRowConfig:
    key: str
    value_col: str
    signed_col: str
    ylabel: str
    log_label: str


def configure_typography(fig_width: float, fig_height: float) -> dict[str, float]:
    """Apply one font size to all plot text."""
    font_size = float(np.clip(0.75 * min(fig_width, fig_height) + 5.0, 7.0, 12.5))
    font_size = 12
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "legend.title_fontsize": font_size,
            "figure.titlesize": font_size,
        }
    )
    return {
        "base": font_size,
        "title": font_size,
        "label": font_size,
        "tick": font_size,
        "legend": font_size,
    }


def read_simulation_csv(path: Path) -> pd.DataFrame:
    """Read simulation CSV while guarding against malformed row widths."""
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    if not lines:
        return pd.DataFrame()

    expected_fields = lines[0].count(",") + 1
    mismatch_rows = sum(
        1
        for line in lines[1:]
        if line.strip() and (line.count(",") + 1) != expected_fields
    )
    if mismatch_rows:
        print(
            f"{path.name}: {mismatch_rows} rows do not match header width; "
            "ignoring trailing fields."
        )

    # Keep only header-defined columns so malformed trailing values cannot
    # shift expected columns (e.g. 'up', 'us', 'v_app', 'course_rate').
    return pd.read_csv(path, index_col=False, usecols=range(expected_fields))


def to_rate_rad_s(rate: np.ndarray) -> np.ndarray:
    """Convert angular-rate series to rad/s when values appear to be in deg/s."""
    arr = np.asarray(rate, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    if float(np.max(np.abs(finite))) > (2.0 * np.pi + 0.5):
        return np.deg2rad(arr)
    return arr


def estimate_steering_offset_us0(
    us_signed: np.ndarray,
    va: np.ndarray,
    y_signed: np.ndarray,
    dataset_title: str,
) -> float:
    """Estimate constant steering offset via y = a(us*va) + b(va), u_s0 = -b/a."""
    x1 = np.asarray(us_signed, dtype=float) * np.asarray(va, dtype=float)
    x2 = np.asarray(va, dtype=float)
    y = np.asarray(y_signed, dtype=float)

    finite = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(y)
    x1 = x1[finite]
    x2 = x2[finite]
    y = y[finite]

    if x1.size < 3:
        print(f"{dataset_title}: steering-offset fit skipped (insufficient points).")
        return 0.0

    design = np.column_stack([x1, x2])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    a = float(coeffs[0])
    b = float(coeffs[1])
    if abs(a) <= 1e-10:
        print(
            f"{dataset_title}: steering-offset fit skipped "
            "(near-zero steering slope)."
        )
        return 0.0

    y_hat = design @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    us0 = -b / a
    print(
        f"{dataset_title}: steering offset fit a={a:.5f}, b={b:.5f}, "
        f"u_s0={us0:.5f}, R^2={r2:.3f}, n={x1.size}"
    )
    return float(us0)


COURSE_RATE_COLUMNS = (
    "kite_course_rate_1",
    "kite_course_rate_0",
    "kite_course_rate",
    "course_rate",
)
YAW_RATE_COLUMNS = (
    "kite_yaw_rate_1",
    "kite_yaw_rate_0",
    "kite_yaw_rate",
    "yaw_rate",
)
YAW_ANGLE_COLUMNS = ("kite_yaw_0", "kite_yaw_1", "kite_yaw", "yaw")


def first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate column present in a dataframe."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def time_seconds(
    results: pd.DataFrame, flight_data: pd.DataFrame, cfg: FlightConfig, purpose: str
) -> tuple[np.ndarray, str]:
    """Return the time vector used to derive an angular rate."""
    if "time" in results.columns:
        return results["time"].to_numpy(dtype=float), "results.time"
    if "time" in flight_data.columns:
        return flight_data["time"].to_numpy(dtype=float), "flight_data.time"
    raise ValueError(f"Missing time column to derive {purpose} in {cfg.title}")


def angle_to_rad(angle: np.ndarray) -> np.ndarray:
    """Convert wrapped angle samples to radians."""
    arr = np.asarray(angle, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    if float(np.max(np.abs(finite))) > (2.0 * np.pi + 0.5):
        return np.deg2rad(np.mod(arr, 360.0))
    return np.mod(arr, 2.0 * np.pi)


def derive_rate_from_angle(angle: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Derive angular rate from wrapped angle samples."""
    return np.gradient(np.unwrap(angle_to_rad(angle)), time_s)


def load_course_rate(
    cfg: FlightConfig, results: pd.DataFrame, flight_data: pd.DataFrame
) -> tuple[np.ndarray, str]:
    """Load course rate in rad/s, deriving from kite_course when needed."""
    course_rate_col = first_existing_column(flight_data, COURSE_RATE_COLUMNS)
    if course_rate_col is not None:
        return (
            to_rate_rad_s(flight_data[course_rate_col].to_numpy(dtype=float)),
            f"flight_data.{course_rate_col}",
        )

    if "kite_course" not in flight_data.columns:
        raise ValueError(f"Missing course-rate source columns in {cfg.title}")

    time_s, time_source = time_seconds(results, flight_data, cfg, "course rate")
    return (
        derive_rate_from_angle(
            flight_data["kite_course"].to_numpy(dtype=float), time_s
        ),
        f"flight_data.kite_course using {time_source}",
    )


def load_yaw_rate(
    cfg: FlightConfig, results: pd.DataFrame, flight_data: pd.DataFrame
) -> tuple[np.ndarray, str]:
    """Load yaw rate in rad/s, deriving from yaw angle when needed."""
    yaw_rate_col = first_existing_column(flight_data, YAW_RATE_COLUMNS)
    if yaw_rate_col is not None:
        return (
            to_rate_rad_s(flight_data[yaw_rate_col].to_numpy(dtype=float)),
            f"flight_data.{yaw_rate_col}",
        )

    yaw_angle_col = first_existing_column(flight_data, YAW_ANGLE_COLUMNS)
    if yaw_angle_col is None:
        raise ValueError(f"Missing yaw-rate source columns in {cfg.title}")

    time_s, time_source = time_seconds(results, flight_data, cfg, "yaw rate")
    return (
        derive_rate_from_angle(
            flight_data[yaw_angle_col].to_numpy(dtype=float), time_s
        ),
        f"flight_data.{yaw_angle_col} using {time_source}",
    )


def select_rate_columns(df: pd.DataFrame, rate_cfg: RateRowConfig) -> pd.DataFrame:
    """Return a plotting dataframe with y columns set for one rate row."""
    selected = df.copy()
    selected["y"] = selected[rate_cfg.value_col]
    selected["y_signed"] = selected[rate_cfg.signed_col]
    return selected


def remove_xy_angle_quadrants(
    df: pd.DataFrame,
    quadrants: tuple[int, ...],
    panel_label: str,
) -> pd.DataFrame:
    """Drop points whose signed x/y angle falls in selected Cartesian quadrants."""
    if df.empty:
        return df

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    angle_deg = np.mod(np.rad2deg(np.arctan2(y, x)), 360.0)

    remove_mask = np.zeros(len(df), dtype=bool)
    if 1 in quadrants:
        remove_mask |= (angle_deg > 0.0) & (angle_deg < 90.0)
    if 2 in quadrants:
        remove_mask |= (angle_deg > 90.0) & (angle_deg < 180.0)
    if 3 in quadrants:
        remove_mask |= (angle_deg > 180.0) & (angle_deg < 270.0)
    if 4 in quadrants:
        remove_mask |= (angle_deg > 270.0) & (angle_deg < 360.0)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(angle_deg)
    remove_mask &= finite
    removed = int(np.count_nonzero(remove_mask))
    if removed:
        quadrant_label = ", ".join(f"Q{q}" for q in quadrants)
        print(
            f"{panel_label}: removed {removed} points in {quadrant_label} "
            "using atan2(y, x)."
        )
    return df.loc[~remove_mask].reset_index(drop=True)


def remove_xy_threshold_region(
    df: pd.DataFrame,
    *,
    x_max: float,
    y_min: float,
    panel_label: str,
) -> pd.DataFrame:
    """Drop points above a y threshold and below an x threshold."""
    if df.empty:
        return df

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    remove_mask = np.isfinite(x) & np.isfinite(y) & (x < x_max) & (y > y_min)
    removed = int(np.count_nonzero(remove_mask))
    if removed:
        print(
            f"{panel_label}: removed {removed} points with "
            f"y>{y_min:.2f} rad/s and x<{x_max:.2f} m/s."
        )
    return df.loc[~remove_mask].reset_index(drop=True)


def select_panel_dataset(
    df: pd.DataFrame,
    cfg: FlightConfig,
    rate_cfg: RateRowConfig,
    angle_quadrant_filters: dict[tuple[str, str], tuple[int, ...]],
    xy_threshold_filters: dict[tuple[str, str], tuple[float, float]],
) -> pd.DataFrame:
    """Select the rate columns and apply any panel-specific angle filters."""
    selected = select_rate_columns(df, rate_cfg)
    quadrants = angle_quadrant_filters.get((rate_cfg.key, cfg.year))
    panel_label = f"{cfg.title} {rate_cfg.log_label}"
    if quadrants is not None:
        selected = remove_xy_angle_quadrants(
            selected,
            quadrants=quadrants,
            panel_label=panel_label,
        )

    threshold = xy_threshold_filters.get((rate_cfg.key, cfg.year))
    if threshold is not None:
        x_max, y_min = threshold
        selected = remove_xy_threshold_region(
            selected,
            x_max=x_max,
            y_min=y_min,
            panel_label=panel_label,
        )
    return selected


def set_matching_y_limits_and_ticks(
    axes: np.ndarray,
    y_min: float,
    y_max: float,
    tick_step: float = 1.0,
) -> None:
    """Apply identical y limits and ticks to all panels."""
    tick_min = np.ceil(y_min / tick_step) * tick_step
    tick_max = np.floor(y_max / tick_step) * tick_step
    ticks = np.arange(tick_min, tick_max + 0.5 * tick_step, tick_step)
    for ax in axes.flat:
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(ticks)


def load_simulation_dataset(candidates: list[Path], dataset_label: str) -> pd.DataFrame:
    """Load and combine one simulation dataset from a list of candidate CSV paths."""
    frames: list[pd.DataFrame] = []
    for path in candidates:
        if not path.is_file():
            continue
        df = read_simulation_csv(path)
        if df.empty:
            print(f"Simulation CSV is empty ({dataset_label}): {path}")
            continue
        print(f"Loaded simulation data ({dataset_label}) from {path} ({len(df)} rows)")
        frames.append(df)

    if not frames:
        print(f"Simulation CSV not found for overlay plotting ({dataset_label}).")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(
        f"Combined simulation overlays ({dataset_label}) from {len(frames)} file(s): "
        f"{len(combined)} rows total."
    )
    return combined


def load_filtered_dataset(
    cfg: FlightConfig,
    filter_cfg: dict[str, float | bool | tuple[float, float] | None],
) -> pd.DataFrame:
    """Load one flight and keep only filtered data needed for plotting."""
    results, flight_data, _ = read_results(
        cfg.year, cfg.month, cfg.day, cfg.kite_model, addition=cfg.addition
    )

    time_range = filter_cfg.get("time_range")
    if time_range is not None:
        t0, t1 = time_range
        time_mask = (results["time"] >= t0) & (results["time"] <= t1)
        results = results.loc[time_mask].reset_index(drop=True)
        flight_data = flight_data.loc[time_mask].reset_index(drop=True)

    powered_only = bool(filter_cfg.get("powered_only", True))
    if powered_only and "powered" in flight_data.columns:
        powered_mask = flight_data["powered"] == "powered"
        results = results.loc[powered_mask].reset_index(drop=True)
        flight_data = flight_data.loc[powered_mask].reset_index(drop=True)

    downsample_frac = float(filter_cfg.get("downsample_frac", 1.0))
    if downsample_frac < 1.0:
        sampled = flight_data.sample(frac=downsample_frac, random_state=42)
        results = results.loc[sampled.index]
        flight_data = sampled

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
    us_signed_raw = steering / 100.0
    if "kcu_actual_depower" in flight_data.columns:
        udp_signed = flight_data["kcu_actual_depower"].to_numpy(dtype=float) / 100.0
    else:
        udp_signed = np.full_like(us_signed_raw, np.nan, dtype=float)
    course_rate_signed, course_rate_source = load_course_rate(cfg, results, flight_data)
    yaw_rate_signed, yaw_rate_source = load_yaw_rate(cfg, results, flight_data)
    print(f"{cfg.title}: using course rate from {course_rate_source}.")
    print(f"{cfg.title}: using yaw rate from {yaw_rate_source}.")

    apply_steering_offset = bool(
        filter_cfg.get("apply_steering_offset", cfg.year == "2019")
    )
    if apply_steering_offset:
        us0 = estimate_steering_offset_us0(
            us_signed_raw, va, course_rate_signed, cfg.title
        )
    else:
        us0 = 0.0

    us_signed = us_signed_raw - us0
    x_signed = -us_signed * va
    x = x_signed
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
        & np.isfinite(course_rate_signed)
        & np.isfinite(yaw_rate_signed)
        & np.isfinite(heading)
        & np.isfinite(x2_signed)
        & np.isfinite(theta_k)
        & np.isfinite(psi_k)
        & np.isfinite(us_signed)
    )

    apply_quadrant_filter = bool(filter_cfg.get("apply_quadrant_filter", False))
    if apply_quadrant_filter:
        # Legacy incorrect-data filter used for 2025:
        # remove high-|course-rate| points whose sign conflicts with steering*va sign.
        y_exclude_threshold_deg = float(filter_cfg.get("y_exclude_threshold_deg", 90.0))
        y_exclude_threshold_rad = np.deg2rad(y_exclude_threshold_deg)
        mismatch_mask = (
            (course_rate_signed > y_exclude_threshold_rad) & (x_signed < 0.0)
        ) | ((course_rate_signed < -y_exclude_threshold_rad) & (x_signed > 0.0))
        before_n = int(np.count_nonzero(finite))
        finite &= ~mismatch_mask
        after_n = int(np.count_nonzero(finite))
        print(
            f"{cfg.title}: removed {before_n - after_n} quadrant-mismatch points "
            f"(threshold={y_exclude_threshold_deg:.1f} deg)."
        )

    df = pd.DataFrame(
        {
            "x": np.asarray(x[finite]),
            "y": np.asarray(course_rate_signed[finite]),
            "heading": np.asarray(heading[finite]),
            "us_signed": np.asarray(us_signed[finite]),
            "us_signed_raw": np.asarray(us_signed_raw[finite]),
            "us0": np.full(int(np.count_nonzero(finite)), us0, dtype=float),
            "udp_signed": np.asarray(udp_signed[finite]),
            "us_was_negative": np.asarray(us_signed[finite] < 0.0),
            "x1_signed": np.asarray(x_signed[finite]),
            "x2_signed": np.asarray(x2_signed[finite]),
            "y_signed": np.asarray(course_rate_signed[finite]),
            "course_rate": np.asarray(course_rate_signed[finite]),
            "course_rate_signed": np.asarray(course_rate_signed[finite]),
            "yaw_rate": np.asarray(yaw_rate_signed[finite]),
            "yaw_rate_signed": np.asarray(yaw_rate_signed[finite]),
        }
    )

    us_min = filter_cfg.get("us_min")
    us_max = filter_cfg.get("us_max")
    if us_min is not None:
        df = df.loc[df["us_signed"] >= float(us_min)]
    if us_max is not None:
        df = df.loc[df["us_signed"] <= float(us_max)]

    udp_min = filter_cfg.get("udp_min")
    udp_max = filter_cfg.get("udp_max")
    if udp_min is not None and "udp_signed" in df.columns:
        df = df.loc[df["udp_signed"] >= float(udp_min)]
    if udp_max is not None and "udp_signed" in df.columns:
        df = df.loc[df["udp_signed"] <= float(udp_max)]

    return df.reset_index(drop=True)


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
    metric_rows = (
        rf"\makebox[{metric_width_cm:.2f}cm][l]{{$ {metric_1_tex} $}}"
        rf"$=$"
        rf"\makebox[{value_1_width_cm:.2f}cm][l]{{$ {metric_1_value_tex} $}}"
        rf"$,\ $"
        f"\n"
        rf"\makebox[{metric_width_cm:.2f}cm][l]{{$ {metric_2_tex} $}}"
        rf"$=$"
        rf"\makebox[{value_2_width_cm:.2f}cm][l]{{$ {metric_2_value_tex} $}}"
    )
    if not title:
        return metric_rows
    return rf"\makebox[{title_width_cm:.2f}cm][l]{{{title}}}" + "\n" + metric_rows


def plot_heading_binned_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    fit_df: pd.DataFrame,
    title: str,
    heading_norm: Normalize,
    cmap,
    rate_label: str,
) -> list[tuple[str, str, float, str]]:
    """Scatter data and overlay one origin-constrained linear fit."""
    fit_legend_entries: list[tuple[str, str, float, str]] = []
    log_title = f"{title} {rate_label}"

    def add_panel_label() -> None:
        label_text = (
            rf"\textbf{{{title}}}" if plt.rcParams.get("text.usetex", False) else title
        )
        ax.text(
            0.02,
            0.98,
            label_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            # fontsize=8,
        )

    if plot_df.empty:
        add_panel_label()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fit_legend_entries

    heading_math = plot_df["heading"].to_numpy(dtype=float)
    # Convert back to navigation heading for 360deg coloring:
    # 0deg = North, clockwise positive.
    heading_nav = np.mod((np.pi / 2.0) - heading_math, 2.0 * np.pi)
    heading_nav_deg = np.mod(np.rad2deg(heading_nav), 360.0)

    heading_color_vals = map_heading_to_viridis_mirrored(heading_nav_deg, heading_norm)
    scatter_mask = np.isfinite(heading_color_vals)

    if np.any(scatter_mask):
        ax.scatter(
            plot_df.loc[scatter_mask, "x"],
            plot_df.loc[scatter_mask, "y"],
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

    x_fit = fit_df["x"].to_numpy(dtype=float) if not fit_df.empty else np.asarray([])
    y_fit = fit_df["y"].to_numpy(dtype=float) if not fit_df.empty else np.asarray([])
    finite_fit = np.isfinite(x_fit) & np.isfinite(y_fit)
    x_fit = x_fit[finite_fit]
    y_fit = y_fit[finite_fit]

    if x_fit.size >= 2 and float(np.sum(x_fit**2)) > 1e-12:
        slope = float(np.sum(x_fit * y_fit) / np.sum(x_fit**2))
        y_hat = slope * x_fit
        ss_res = float(np.sum((y_fit - y_hat) ** 2))
        ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
        x_line = np.linspace(float(np.min(x_fit)), float(np.max(x_fit)), 200)
        y_line = slope * x_line
        ax.plot(x_line, y_line, color="black", linestyle="-", linewidth=1.8, zorder=2)
        print(
            f"{log_title} origin-fit: n={x_fit.size}, " f"g_k={slope:.3f}, R^2={r2:.3f}"
        )
        fit_legend_entries.append(
            (
                "line",
                "-",
                1.8,
                format_compact_latex_legend(
                    title="",
                    metric_1_tex=r"g_\mathrm{k}",
                    metric_1_value_tex=rf"{slope:.2f}",
                    metric_2_tex=r"R^{2}",
                    metric_2_value_tex=(
                        rf"{r2:.2f}" if np.isfinite(r2) else r"\mathrm{n/a}"
                    ),
                ),
            )
        )
    else:
        print(f"{log_title} origin-fit: insufficient data")

    add_panel_label()
    if str(title).startswith("2025"):
        ax.set_xlabel(r"$v_\mathrm{a}u_\mathrm{s}$ " r"($\mathrm{m\,s^{-1}}$)")
    else:
        ax.set_xlabel(
            r"$v_\mathrm{a}(u_\mathrm{s}-u_{\mathrm{s},0})$ " r"($\mathrm{m\,s^{-1}}$)"
        )
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
    ax: plt.Axes, sim_df: pd.DataFrame, sim_label: str, rate_key: str
) -> tuple[list[tuple[str, str, float, str]], OverlayFitSummary, float, float]:
    """Overlay one full simulation dataset on an experimental panel in red."""
    legend_entries: list[tuple[str, str, float, str]] = []
    summary = OverlayFitSummary(
        transient_gk=np.nan,
        transient_r2=np.nan,
        dynamic_gk=np.nan,
        dynamic_r2=np.nan,
    )
    if sim_df.empty:
        return legend_entries, summary, 0.0, 0.0

    required = {"us", "v_app"}
    if not required.issubset(sim_df.columns):
        return legend_entries, summary, 0.0, 0.0

    transient_rate_col = f"{rate_key}_rate"
    if transient_rate_col not in sim_df.columns:
        print(f"Simulation overlay: missing {transient_rate_col} for {sim_label}")
        return legend_entries, summary, 0.0, 0.0

    df_sim = sim_df.copy()
    if df_sim.empty:
        print(f"Simulation overlay: no rows for {sim_label}")
        return legend_entries, summary, 0.0, 0.0

    # Transient points: base simulation trajectory.
    x_transient = df_sim["us"].to_numpy(dtype=float) * df_sim["v_app"].to_numpy(
        dtype=float
    )
    y_transient = to_rate_rad_s(df_sim[transient_rate_col].to_numpy(dtype=float))
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
        rate_col = f"{rate_key}_rate_{n}"
        if usva_col not in df_sim.columns or rate_col not in df_sim.columns:
            continue
        x_dyn = df_sim[usva_col].to_numpy(dtype=float)
        y_dyn = to_rate_rad_s(df_sim[rate_col].to_numpy(dtype=float))
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
        x_max = max(x_max, float(np.nanmax(np.abs(x_transient))))
        y_max = max(y_max, float(np.nanmax(np.abs(y_transient))))
    if x_dynamic.size > 0:
        x_max = max(x_max, float(np.nanmax(np.abs(x_dynamic))))
        y_max = max(y_max, float(np.nanmax(np.abs(y_dynamic))))

    fit_transient = _fit_line(x_transient, y_transient)
    if fit_transient is None:
        label_transient = (
            rf"Transient sim ({sim_label})"
            + "\n"
            + rf"$R^2=\mathrm{{n/a}},\ g_{{\mathrm{{k}}}}=\mathrm{{n/a}}$"
        )
    else:
        gk_t, r2_t, x_line_t, y_line_t, n_t = fit_transient
        summary = OverlayFitSummary(
            transient_gk=gk_t,
            transient_r2=r2_t,
            dynamic_gk=summary.dynamic_gk,
            dynamic_r2=summary.dynamic_r2,
        )
        print(
            f"Overlay transient fit ({sim_label}, {rate_key}): "
            f"n={n_t}, g_k={gk_t:.3f}, R^2={r2_t:.3f}"
        )
        label_transient = (
            rf"Transient sim ({sim_label})"
            + "\n"
            + rf"$R^2={r2_t:.2f},\ g_{{\mathrm{{k}}}}={gk_t:.2f}$"
        )
    fit_dynamic = _fit_line(x_dynamic, y_dynamic)
    if fit_dynamic is None:
        label_dynamic = (
            rf"Dynamic sim ({sim_label})"
            + "\n"
            + rf"$R^2=\mathrm{{n/a}},\ g_{{\mathrm{{k}}}}=\mathrm{{n/a}}$"
        )
    else:
        gk_d, r2_d, x_line_d, y_line_d, n_d = fit_dynamic
        summary = OverlayFitSummary(
            transient_gk=summary.transient_gk,
            transient_r2=summary.transient_r2,
            dynamic_gk=gk_d,
            dynamic_r2=r2_d,
        )
        print(
            f"Overlay dynamic fit ({sim_label}, {rate_key}): "
            f"n={n_d}, g_k={gk_d:.3f}, R^2={r2_d:.3f}"
        )
        label_dynamic = (
            rf"Dynamic sim ({sim_label})"
            + "\n"
            + rf"$R^2={r2_d:.2f},\ g_{{\mathrm{{k}}}}={gk_d:.2f}$"
        )
    return legend_entries, summary, x_max, y_max


def get_simulation_abs_limits(
    sim_df: pd.DataFrame, rate_key: str
) -> tuple[float, float]:
    """Return absolute x/y maxima for simulation panel in rad/s."""
    if sim_df.empty:
        return 0.0, 0.0

    x_vals = []
    y_vals = []

    transient_rate_col = f"{rate_key}_rate"
    if transient_rate_col in sim_df.columns and {"us", "v_app"}.issubset(
        sim_df.columns
    ):
        x_uniform = np.abs(
            sim_df["us"].to_numpy(dtype=float) * sim_df["v_app"].to_numpy(dtype=float)
        )
        y_uniform = np.abs(
            to_rate_rad_s(sim_df[transient_rate_col].to_numpy(dtype=float))
        )
        finite_uniform = np.isfinite(x_uniform) & np.isfinite(y_uniform)
        if np.any(finite_uniform):
            x_vals.append(x_uniform[finite_uniform])
            y_vals.append(y_uniform[finite_uniform])

    for n in range(3, 11):
        usva_col = f"usva_{n}"
        rate_col = f"{rate_key}_rate_{n}"
        if usva_col not in sim_df.columns or rate_col not in sim_df.columns:
            continue
        x_dyn = np.abs(sim_df[usva_col].to_numpy(dtype=float))
        y_dyn = np.abs(to_rate_rad_s(sim_df[rate_col].to_numpy(dtype=float)))
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
    wheel_w = 0.24 * bbox.width
    wheel_h = 0.24 * bbox.height
    wheel_x = bbox.x0 + 0.05 * bbox.width
    wheel_y = bbox.y0 + 0.6 * bbox.height
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
    wheel_ax.set_xticklabels(heading_tick_labels, fontsize=10)
    wheel_ax.tick_params(pad=3)


def main() -> None:
    set_plot_style()
    fig_width, fig_height = 8.8, 6.6
    font_sizes = configure_typography(fig_width, fig_height)

    datasets_cfg = [
        FlightConfig(
            title="2019-10-08",
            year="2019",
            month="10",
            day="08",
            kite_model="v3",
            addition="_t26",
            plot_filter=PLOT_FILTER_2019,
            fit_filter=FIT_FILTER_2019,
        ),
        FlightConfig(
            title="2025-10-09",
            year="2025",
            month="10",
            day="09",
            kite_model="v3",
            addition="",
            plot_filter=PLOT_FILTER_2025,
            fit_filter=FIT_FILTER_2025,
        ),
    ]
    rate_rows = [
        RateRowConfig(
            key="yaw",
            value_col="yaw_rate",
            signed_col="yaw_rate_signed",
            ylabel=r"$\dot{\psi}$ ($\mathrm{rad\,s^{-1}}$)",
            log_label="yaw rate",
        ),
        RateRowConfig(
            key="course",
            value_col="course_rate",
            signed_col="course_rate_signed",
            ylabel=r"$\dot{\chi}$ ($\mathrm{rad\,s^{-1}}$)",
            log_label="course rate",
        ),
    ]
    angle_quadrant_filters = {
        ("yaw", "2025"): (2, 4),
    }
    xy_threshold_filters = {
        ("yaw", "2025"): (1.5, 1.0),  # x < 1.5 m/s and y > 1.0 rad/s
    }

    plot_datasets = [
        load_filtered_dataset(cfg, cfg.plot_filter) for cfg in datasets_cfg
    ]
    fit_datasets = [load_filtered_dataset(cfg, cfg.fit_filter) for cfg in datasets_cfg]
    panel_plot_datasets: list[list[pd.DataFrame]] = []
    panel_fit_datasets: list[list[pd.DataFrame]] = []
    for rate_cfg in rate_rows:
        row_plot_datasets: list[pd.DataFrame] = []
        row_fit_datasets: list[pd.DataFrame] = []
        for cfg, plot_df_raw, fit_df_raw in zip(
            datasets_cfg, plot_datasets, fit_datasets
        ):
            row_plot_datasets.append(
                select_panel_dataset(
                    plot_df_raw,
                    cfg,
                    rate_cfg,
                    angle_quadrant_filters,
                    xy_threshold_filters,
                )
            )
            row_fit_datasets.append(
                select_panel_dataset(
                    fit_df_raw,
                    cfg,
                    rate_cfg,
                    angle_quadrant_filters,
                    xy_threshold_filters,
                )
            )
        panel_plot_datasets.append(row_plot_datasets)
        panel_fit_datasets.append(row_fit_datasets)

    processed_data_dir = Path(__file__).resolve().parent.parent / "processed_data"
    ch9_processed_data_dir = Path(__file__).resolve().parent / "processed_data"
    fit_records: list[dict] = []
    for row_idx, rate_cfg in enumerate(rate_rows):
        for col_idx, cfg in enumerate(datasets_cfg):
            plot_df = panel_plot_datasets[row_idx][col_idx]
            fit_df = panel_fit_datasets[row_idx][col_idx]
            if plot_df.empty or fit_df.empty:
                continue
            x_f = fit_df["x"].to_numpy(dtype=float)
            y_f = fit_df["y"].to_numpy(dtype=float)
            finite = np.isfinite(x_f) & np.isfinite(y_f)
            x_f, y_f = x_f[finite], y_f[finite]
            if x_f.size >= 2 and float(np.sum(x_f**2)) > 1e-12:
                slope = float(np.sum(x_f * y_f) / np.sum(x_f**2))
                y_hat = slope * x_f
                ss_tot = float(np.sum((y_f - np.mean(y_f)) ** 2))
                ss_res = float(np.sum((y_f - y_hat) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
                save_plot_data_to_csv(
                    plot_df,
                    cfg.year,
                    slope,
                    r2,
                    processed_data_dir,
                    addendum=f"_{rate_cfg.key}_rate",
                )
                fit_records.append(
                    {
                        "year": cfg.year,
                        "rate": rate_cfg.key,
                        "gk": slope,
                        "R2": r2,
                    }
                )

    if fit_records:
        ch9_processed_data_dir.mkdir(parents=True, exist_ok=True)
        fitted_rates_path = ch9_processed_data_dir / "fitted_rates.csv"
        pd.DataFrame(fit_records).to_csv(fitted_rates_path, index=False)
        print(f"Saved fitted rates → {fitted_rates_path}")

    base = Path(__file__).resolve().parents[2] / "data"
    # Edit these candidate lists to choose which simulation CSVs are used per panel.
    candidates_2019 = [
        # base / "circles_batch_analysis_2019_5March.csv",
        # base / "circles_batch_analysis_2019.csv",
        # base / "circles_batch_analysis.csv",
    ]
    candidates_2025 = [
        # base / "circles_batch_analysis_2025_5March.csv",
        # base / "circles_batch_analysis_2025_3March.csv",
        # base / "circles_batch_analysis_2025_3March_.csv",
    ]
    sim_datasets = [
        load_simulation_dataset(candidates_2019, dataset_label="2019"),
        load_simulation_dataset(candidates_2025, dataset_label="2025"),
    ]

    heading_tick_deg = np.arange(45.0, 315.0 + 1e-9, 45.0)
    heading_tick_labels = [rf"{int(v):d}$^{{\circ}}$" for v in heading_tick_deg]
    cmap = plt.get_cmap("viridis")
    heading_norm = Normalize(vmin=45.0, vmax=315.0)

    fig, axes = plt.subplots(
        2, 2, figsize=(fig_width, fig_height), sharex=True, sharey=True
    )
    fig.subplots_adjust(
        left=0.09, right=0.98, bottom=0.08, top=0.98, hspace=0.12, wspace=0.08
    )

    fit_legend_entries_per_ax: list[list[list[tuple[str, str, float, str]]]] = []
    for row_idx, rate_cfg in enumerate(rate_rows):
        row_entries: list[list[tuple[str, str, float, str]]] = []
        for col_idx, cfg in enumerate(datasets_cfg):
            plot_df = panel_plot_datasets[row_idx][col_idx]
            fit_df = panel_fit_datasets[row_idx][col_idx]
            row_entries.append(
                plot_heading_binned_panel(
                    axes[row_idx, col_idx],
                    plot_df,
                    fit_df,
                    cfg.title,
                    heading_norm,
                    cmap,
                    rate_cfg.log_label,
                )
            )
        fit_legend_entries_per_ax.append(row_entries)

    sim_case_limits: list[list[tuple[float, float]]] = []
    sim_labels = ["2019", "2025"]
    for row_idx, rate_cfg in enumerate(rate_rows):
        row_limits: list[tuple[float, float]] = []
        for col_idx, (sim_df, sim_label) in enumerate(zip(sim_datasets, sim_labels)):
            _overlay_entries, _overlay_summary, sim_x_max, sim_y_max = (
                overlay_simulation_case(
                    axes[row_idx, col_idx], sim_df, sim_label, rate_cfg.key
                )
            )
            row_limits.append((sim_x_max, sim_y_max))
        sim_case_limits.append(row_limits)

    for row_idx, rate_cfg in enumerate(rate_rows):
        axes[row_idx, 0].set_ylabel(rate_cfg.ylabel)
        axes[row_idx, 1].set_ylabel("")

    # Symmetric signed axes for all panels.
    all_x_arrays = [df["x"].to_numpy() for df in plot_datasets if not df.empty] + [
        df["x"].to_numpy() for df in fit_datasets if not df.empty
    ]
    all_x = np.concatenate(all_x_arrays) if all_x_arrays else np.asarray([])
    sim_x_abs = max(
        (limits[0] for row_limits in sim_case_limits for limits in row_limits),
        default=0.0,
    )
    x_abs = np.nanmax(np.abs(all_x)) if all_x.size > 0 else 1.0
    x_lim = max(1.0, 1.05 * max(x_abs, sim_x_abs))
    for ax in axes.flat:
        ax.set_xlim(-x_lim, x_lim)

    set_matching_y_limits_and_ticks(axes, y_min=-2.3, y_max=2.3)

    for ax in axes[0, :]:
        ax.set_xlabel("")

    for row_idx in range(len(rate_rows)):
        add_heading_color_wheel(
            fig,
            anchor_ax=axes[row_idx, 1],
            heading_tick_deg=heading_tick_deg,
            heading_tick_labels=heading_tick_labels,
            heading_norm=heading_norm,
            cmap=cmap,
            tick_fontsize=font_sizes["tick"],
        )

    for row_idx in range(len(rate_rows)):
        for col_idx, (ax, panel_entries) in enumerate(
            zip(axes[row_idx, :], fit_legend_entries_per_ax[row_idx])
        ):
            legend_handles = []
            if not panel_plot_datasets[row_idx][col_idx].empty:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="None",
                        marker="o",
                        markersize=6,
                        markerfacecolor="0.55",
                        markeredgecolor="none",
                        alpha=0.55,
                        label="Experimental data",
                    )
                )

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
                        Patch(
                            facecolor="0.5",
                            edgecolor="none",
                            alpha=0.40,
                            label=label,
                        )
                    )

            # Force visible vertical separation between legend entries by inserting
            # explicit spacer rows instead of relying only on labelspacing.
            spacer_label = (
                r"$\vphantom{\int}$" if plt.rcParams.get("text.usetex", False) else " "
            )
            spaced_handles = list(legend_handles)
            for idx, handle in enumerate(fit_handles):
                spaced_handles.append(handle)
                if idx < len(fit_handles) - 1:
                    spaced_handles.append(
                        Line2D(
                            [0],
                            [0],
                            linestyle="None",
                            marker="",
                            linewidth=0.0,
                            label=spacer_label,
                        )
                    )

            if spaced_handles:
                ax.legend(
                    handles=spaced_handles,
                    loc="lower right",
                    fontsize=font_sizes["legend"],
                    frameon=True,
                    handlelength=1.4,
                    borderpad=0.3,
                    labelspacing=0.2,
                    handletextpad=0.6,
                    handleheight=1.2,
                )

    output_path = (
        Path(__file__).resolve().parent
        / "results"
        / "2x2_fitted_turn_rate_2019_and_2025.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=500)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
