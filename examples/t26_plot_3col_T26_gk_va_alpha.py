"""
Plot 1x3 panels versus depower (u_dp).

Panels:
- g_k vs u_dp
- v_a vs u_dp
- turn radius vs u_dp

For g_k and v_a:
- Four bands are shown: (2019, up/down) and (2025, up/down).
- Up/down splitting follows heading orientation (mirrored left/right).
- VW8 varying-us points are overlaid.

For turn radius:
- Experimental bands are shown for (2019/2025, up/down).
- Two simulation lines are shown for u_s = 0.10 and u_s = 0.15.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerPatch
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
    downsample_frac: float = 1.0
    apply_quadrant_filter: bool = False
    y_exclude_threshold_deg: float = 10.0


@dataclass(frozen=True)
class Band:
    year: str
    direction: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float


FIXED_BAND_X_RANGES = {
    "2019": (0.15, 0.4),
    "2025": (0.4, 0.45),
}
TURN_RADIUS_US_MIN = 0.05


class TwoLayerBandHandler(HandlerPatch):
    """Legend handler that matches two-layer band drawing in the axes."""

    def __init__(self, color: str, **kwargs):
        self.color = color
        super().__init__(**kwargs)

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        fill = mpatches.Rectangle(
            (-xdescent, -ydescent),
            width,
            height,
            facecolor=to_rgba(self.color, alpha=0.12),
            edgecolor="none",
            transform=trans,
        )
        hatch = mpatches.Rectangle(
            (-xdescent, -ydescent),
            width,
            height,
            facecolor="none",
            edgecolor=to_rgba(self.color, alpha=1.0),
            hatch="///",
            linewidth=1.0,
            transform=trans,
        )
        return [fill, hatch]


def _read_csv_with_header_width(path: Path) -> pd.DataFrame:
    """Read CSV and keep only header-defined columns."""
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
            f"{path.name}: {mismatch_rows} rows exceed header width; "
            "ignoring trailing fields."
        )

    return pd.read_csv(path, index_col=False, usecols=range(expected_fields))


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
    """Convert 2019 depower angle to 2025-equivalent u_dp."""
    x19_depower = np.asarray(x19_depower)
    up_paper_2019 = np.clip((x19_depower - x19_dep) / (x19_pow - x19_dep), 0, 1)
    ld_2019 = ld_0 + delta_d * delta_ld_max * (1.0 - up_paper_2019)
    up_data_2025 = (ld_2019 - ld_2025_offset) / ld_2025_scale
    return up_data_2025


def _heading_to_orientation(heading_raw: np.ndarray) -> np.ndarray:
    """Return mirrored heading orientation in radians, positive=upward."""
    if np.nanmax(np.abs(heading_raw)) > (2.0 * np.pi + 0.5):
        heading_nav = np.deg2rad(np.mod(heading_raw, 360.0))
    else:
        heading_nav = np.mod(heading_raw, 2.0 * np.pi)

    heading_math = np.mod((np.pi / 2.0) - heading_nav, 2.0 * np.pi)
    return np.arctan2(np.sin(heading_math), np.abs(np.cos(heading_math)))


def _first_available_alpha(results: pd.DataFrame) -> np.ndarray:
    """Pick the best available alpha signal from EKF results."""
    candidates = [
        "wing_angle_of_attack",
        "wing_angle_of_attack_bridle",
        "alpha",
    ]
    for col in candidates:
        if col in results.columns:
            return results[col].to_numpy(dtype=float)
    raise ValueError(f"No alpha column found. Tried: {candidates}")


def _fit_gk_slope(
    df: pd.DataFrame, mask: np.ndarray
) -> tuple[float, float, int] | None:
    """Fit yaw_rate_rad = g_k * x_gk + b on a masked subset."""
    x = df.loc[mask, "x_gk"].to_numpy(dtype=float)
    y = df.loc[mask, "yaw_rate_rad"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2 or np.ptp(x) <= 1e-12:
        return None
    slope, _, r_value, _, _ = linregress(x, y)
    return float(slope), float(r_value**2), int(x.size)


def load_experimental_dataset(cfg: FlightConfig) -> pd.DataFrame:
    """Load and preprocess one flight window from the EKF .h5 output."""
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
    if "kcu_actual_depower" not in flight_data.columns:
        raise ValueError(f"Missing depower column in {cfg.title}")
    if "kcu_actual_steering" not in flight_data.columns:
        raise ValueError(f"Missing steering column in {cfg.title}")
    if "kite_heading" not in flight_data.columns:
        raise ValueError(f"Missing kite_heading column in {cfg.title}")
    if "kite_apparent_windspeed" not in results.columns:
        raise ValueError(f"Missing kite_apparent_windspeed in results for {cfg.title}")
    if "radius_turn" not in results.columns:
        raise ValueError(f"Missing radius_turn in results for {cfg.title}")

    u_dp = flight_data["kcu_actual_depower"].to_numpy(dtype=float)
    if cfg.year == "2019":
        u_dp = convert_2019_depower_to_2025_updata(u_dp)
    elif np.nanmax(np.abs(u_dp)) > 1.5:
        # Some datasets store depower as percent; convert to fraction-like u_dp.
        u_dp = u_dp / 100.0
        print(f"{cfg.title}: scaled kcu_actual_depower by 1/100 for u_dp axis.")

    steering = flight_data["kcu_actual_steering"].to_numpy(dtype=float)
    us_signed = steering / 100.0
    v_a = results["kite_apparent_windspeed"].to_numpy(dtype=float)
    # Use magnitude for direct comparison to positive simulation turn-radius curves.
    turn_radius = np.abs(results["radius_turn"].to_numpy(dtype=float))
    turn_radius_valid = np.abs(us_signed) > TURN_RADIUS_US_MIN
    alpha = _first_available_alpha(results)
    yaw_rate_rad = flight_data[yaw_rate_col].to_numpy(dtype=float)
    x_gk = -us_signed * v_a
    heading_orient = _heading_to_orientation(
        flight_data["kite_heading"].to_numpy(dtype=float)
    )
    direction_upward = heading_orient >= 0.0

    finite = (
        np.isfinite(u_dp)
        & np.isfinite(us_signed)
        & np.isfinite(v_a)
        & np.isfinite(turn_radius)
        & np.isfinite(alpha)
        & np.isfinite(yaw_rate_rad)
        & np.isfinite(x_gk)
        & np.isfinite(heading_orient)
    )

    if cfg.apply_quadrant_filter:
        y_threshold_rad = np.deg2rad(cfg.y_exclude_threshold_deg)
        mismatch = ((yaw_rate_rad > y_threshold_rad) & (x_gk < 0.0)) | (
            (yaw_rate_rad < -y_threshold_rad) & (x_gk > 0.0)
        )
        before = int(np.count_nonzero(finite))
        finite &= ~mismatch
        after = int(np.count_nonzero(finite))
        print(
            f"{cfg.title}: removed {before - after} quadrant-mismatch points "
            f"(threshold={cfg.y_exclude_threshold_deg:.1f} deg)."
        )

    df = pd.DataFrame(
        {
            "u_dp": u_dp[finite],
            "us_signed": us_signed[finite],
            "v_a": v_a[finite],
            "turn_radius": turn_radius[finite],
            "turn_radius_valid": turn_radius_valid[finite],
            "alpha": alpha[finite],
            "yaw_rate_rad": yaw_rate_rad[finite],
            "x_gk": x_gk[finite],
            "heading_orient": heading_orient[finite],
            "direction": np.where(direction_upward[finite], "upward", "downward"),
        }
    ).reset_index(drop=True)

    # Point-wise gain proxy for scatter visualization in panel 1.
    turn_mask = np.abs(df["us_signed"].to_numpy(dtype=float)) > TURN_RADIUS_US_MIN
    x_nonzero = np.abs(df["x_gk"].to_numpy(dtype=float)) > 1e-6
    gk_inst = np.full(len(df), np.nan, dtype=float)
    valid_inst = turn_mask & x_nonzero
    gk_inst[valid_inst] = df.loc[valid_inst, "yaw_rate_rad"].to_numpy(
        dtype=float
    ) / df.loc[valid_inst, "x_gk"].to_numpy(dtype=float)
    df["g_k_inst"] = gk_inst

    return df


def compute_bands(df: pd.DataFrame, year: str) -> dict[str, list[Band]]:
    """Compute up/down bands for g_k, v_a, alpha, and turn radius."""
    bands: dict[str, list[Band]] = {
        "g_k": [],
        "v_a": [],
        "alpha": [],
        "turn_radius": [],
    }
    directions = ("upward", "downward")
    x_min_fixed, x_max_fixed = FIXED_BAND_X_RANGES.get(
        year,
        (
            float(np.nanmin(df["u_dp"].to_numpy(dtype=float))),
            float(np.nanmax(df["u_dp"].to_numpy(dtype=float))),
        ),
    )

    for direction in directions:
        dir_mask = df["direction"].to_numpy() == direction
        if not np.any(dir_mask):
            continue

        for metric in ("v_a", "alpha", "turn_radius"):
            if metric == "turn_radius":
                valid_turn = df["turn_radius_valid"].to_numpy(dtype=bool)
                y = df.loc[dir_mask & valid_turn, metric].to_numpy(dtype=float)
                y = y[np.isfinite(y)]
                if y.size == 0:
                    continue
                y_min, y_max = np.nanpercentile(y, [5, 95])
                y_min = float(y_min)
                y_max = float(y_max)
            else:
                y = df.loc[dir_mask, metric].to_numpy(dtype=float)
                y_min = float(np.nanmin(y))
                y_max = float(np.nanmax(y))
            if abs(y_max - y_min) < 1e-9:
                pad = max(0.05, 0.02 * abs(y_min))
                y_min -= pad
                y_max += pad
            bands[metric].append(
                Band(
                    year=year,
                    direction=direction,
                    x_min=x_min_fixed,
                    x_max=x_max_fixed,
                    y_min=y_min,
                    y_max=y_max,
                )
            )

        # g_k band from left/right turn fits in this direction.
        left_mask = dir_mask & (
            df["us_signed"].to_numpy(dtype=float) < -TURN_RADIUS_US_MIN
        )
        right_mask = dir_mask & (
            df["us_signed"].to_numpy(dtype=float) > TURN_RADIUS_US_MIN
        )

        left_fit = _fit_gk_slope(df, left_mask)
        right_fit = _fit_gk_slope(df, right_mask)

        slopes = []
        if left_fit is not None:
            slopes.append(left_fit[0])
        if right_fit is not None:
            slopes.append(right_fit[0])

        if slopes:
            y_min = float(np.min(slopes))
            y_max = float(np.max(slopes))
            if abs(y_max - y_min) < 1e-9:
                pad = max(0.05, 0.02 * abs(y_min))
                y_min -= pad
                y_max += pad

            bands["g_k"].append(
                Band(
                    year=year,
                    direction=direction,
                    x_min=x_min_fixed,
                    x_max=x_max_fixed,
                    y_min=y_min,
                    y_max=y_max,
                )
            )

            left_str = (
                f"{left_fit[0]:.3f} (R2={left_fit[1]:.2f}, n={left_fit[2]})"
                if left_fit is not None
                else "n/a"
            )
            right_str = (
                f"{right_fit[0]:.3f} (R2={right_fit[1]:.2f}, n={right_fit[2]})"
                if right_fit is not None
                else "n/a"
            )
            print(
                f"{year} {direction} g_k from left/right: "
                f"left={left_str}, right={right_str}, band=[{y_min:.3f}, {y_max:.3f}]"
            )

    return bands


def load_vw8_varying_us_points(
    csv_path: Path,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    """Load VW8 means and per-level uncertainty as (x=u_dp, y=metric) arrays."""
    if not csv_path.is_file():
        print(f"VW8 CSV not found: {csv_path}")
        return {}, {}

    df = _read_csv_with_header_width(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {csv_path}")

    # User requested: interpret CSV column 2 as u_dp and keep it positive.
    # Then place points only on the defined discrete levels.
    u_dp_raw = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    u_dp_pos = np.abs(u_dp_raw)
    u_dp_levels = np.array(
        [0.22, 0.28, 0.30, 0.32, 0.34, 0.38, 0.40, 0.42], dtype=float
    )
    u_dp = u_dp_levels[
        np.argmin(
            np.abs(u_dp_pos[:, None] - u_dp_levels[None, :]),
            axis=1,
        )
    ]

    required = ["us", "v_app", "yaw_rate", "aoa"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required VW8 columns in {csv_path}: {missing}")

    # Match _plot_gk_2col_filter_heading_absolute_trial.py conventions:
    # x = |u_s * v_app| and y = |deg2rad(yaw_rate)|, then g_k = y/x per point.
    us = pd.to_numeric(df["us"], errors="coerce").to_numpy(dtype=float)
    v_app = pd.to_numeric(df["v_app"], errors="coerce").to_numpy(dtype=float)
    yaw_rate = pd.to_numeric(df["yaw_rate"], errors="coerce").to_numpy(dtype=float)
    x_sim = np.abs(us * v_app)
    y_sim = np.abs(np.deg2rad(yaw_rate))
    g_k = np.full_like(x_sim, np.nan, dtype=float)
    nonzero = np.abs(x_sim) > 1e-9
    g_k[nonzero] = y_sim[nonzero] / x_sim[nonzero]

    # Keep v_app exactly as in CSV (no absolute transform).
    v_a = v_app
    alpha = pd.to_numeric(df["aoa"], errors="coerce").to_numpy(dtype=float)

    points_raw = {"g_k": g_k, "v_a": v_a, "alpha": alpha}
    points: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    uncertainty: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for metric, y_vals in points_raw.items():
        finite = np.isfinite(u_dp) & np.isfinite(y_vals)
        if not np.any(finite):
            points[metric] = (np.array([], dtype=float), np.array([], dtype=float))
            uncertainty[metric] = (
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )
            print(f"Loaded VW8 overlay for {metric}: n=0")
            continue
        grouped_stats = (
            pd.DataFrame({"u_dp": u_dp[finite], "y": y_vals[finite]})
            .groupby("u_dp", as_index=False)["y"]
            .agg(y_mean="mean", y_std="std")
            .sort_values("u_dp")
        )
        x_vals = grouped_stats["u_dp"].to_numpy(dtype=float)
        y_mean = grouped_stats["y_mean"].to_numpy(dtype=float)
        y_std = grouped_stats["y_std"].fillna(0.0).to_numpy(dtype=float)
        points[metric] = (
            x_vals,
            y_mean,
        )
        uncertainty[metric] = (
            x_vals,
            y_mean - y_std,
            y_mean + y_std,
        )
        print(
            f"Loaded VW8 overlay for {metric}: "
            f"n_raw={int(np.count_nonzero(finite))}, n_avg={len(grouped_stats)}"
        )

    unique_levels = sorted({float(v) for v in u_dp[np.isfinite(u_dp)]})
    print(f"VW8 u_dp levels from CSV column 2: {unique_levels}")
    return points, uncertainty


def load_vw8_transient_average_points(
    csv_path: Path,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
]:
    """Load transient-harmonic means and per-level uncertainty (n=4..9)."""
    if not csv_path.is_file():
        print(f"VW8 CSV not found: {csv_path}")
        return {}, {}

    df = _read_csv_with_header_width(csv_path)
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {csv_path}")

    u_dp_raw = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    u_dp_pos = np.abs(u_dp_raw)
    u_dp_levels = np.array(
        [0.22, 0.28, 0.30, 0.32, 0.34, 0.38, 0.40, 0.42], dtype=float
    )
    u_dp = u_dp_levels[
        np.argmin(
            np.abs(u_dp_pos[:, None] - u_dp_levels[None, :]),
            axis=1,
        )
    ]

    gk_series = []
    va_series = []
    aoa_series = []
    for n in range(4, 10):
        usva_col = f"usva_{n}"
        yaw_col = f"yaw_rate_{n}"
        va_col = f"va{n}"
        aoa_col = f"aoa{n}"
        if not {usva_col, yaw_col, va_col, aoa_col}.issubset(df.columns):
            continue

        usva_n = np.abs(
            pd.to_numeric(df[usva_col], errors="coerce").to_numpy(dtype=float)
        )
        yaw_n = np.abs(
            np.deg2rad(
                pd.to_numeric(df[yaw_col], errors="coerce").to_numpy(dtype=float)
            )
        )
        gk_n = np.full_like(usva_n, np.nan, dtype=float)
        nonzero = usva_n > 1e-9
        gk_n[nonzero] = yaw_n[nonzero] / usva_n[nonzero]
        gk_series.append(gk_n)

        va_series.append(
            pd.to_numeric(df[va_col], errors="coerce").to_numpy(dtype=float)
        )
        aoa_series.append(
            pd.to_numeric(df[aoa_col], errors="coerce").to_numpy(dtype=float)
        )

    if not gk_series or not va_series or not aoa_series:
        print("No transient harmonics (n=4..9) found for VW8 overlay.")
        return {}, {}

    gk_avg = np.nanmean(np.vstack(gk_series), axis=0)
    va_avg = np.nanmean(np.vstack(va_series), axis=0)
    aoa_avg = np.nanmean(np.vstack(aoa_series), axis=0)

    points_raw = {"g_k": gk_avg, "v_a": va_avg, "alpha": aoa_avg}
    points: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    uncertainty: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for metric, y_vals in points_raw.items():
        finite = np.isfinite(u_dp) & np.isfinite(y_vals)
        if not np.any(finite):
            points[metric] = (np.array([], dtype=float), np.array([], dtype=float))
            uncertainty[metric] = (
                np.array([], dtype=float),
                np.array([], dtype=float),
                np.array([], dtype=float),
            )
            print(f"Loaded transient-average overlay for {metric}: n=0")
            continue
        grouped_stats = (
            pd.DataFrame({"u_dp": u_dp[finite], "y": y_vals[finite]})
            .groupby("u_dp", as_index=False)["y"]
            .agg(y_mean="mean", y_std="std")
            .sort_values("u_dp")
        )
        x_vals = grouped_stats["u_dp"].to_numpy(dtype=float)
        y_mean = grouped_stats["y_mean"].to_numpy(dtype=float)
        y_std = grouped_stats["y_std"].fillna(0.0).to_numpy(dtype=float)
        points[metric] = (
            x_vals,
            y_mean,
        )
        uncertainty[metric] = (
            x_vals,
            y_mean - y_std,
            y_mean + y_std,
        )
        print(
            f"Loaded transient-average overlay for {metric}: "
            f"n_raw={int(np.count_nonzero(finite))}, n_avg={len(grouped_stats)}"
        )

    return points, uncertainty


def load_vw8_turn_radius_lines(
    csv_path: Path,
    us_levels: tuple[float, ...] = (0.10, 0.15),
    us_tol: float = 1e-6,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Load turn-radius lines (x=u_dp, y=turn_radius) for selected u_s values."""
    empty = {
        float(us_level): (np.array([], dtype=float), np.array([], dtype=float))
        for us_level in us_levels
    }
    if not csv_path.is_file():
        print(f"VW8 CSV not found: {csv_path}")
        return empty

    df = _read_csv_with_header_width(csv_path)
    required = ["up", "us", "turn_radius"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {missing}")

    u_dp = np.abs(pd.to_numeric(df["up"], errors="coerce").to_numpy(dtype=float))
    us = pd.to_numeric(df["us"], errors="coerce").to_numpy(dtype=float)
    turn_radius = pd.to_numeric(df["turn_radius"], errors="coerce").to_numpy(
        dtype=float
    )

    lines = dict(empty)
    finite = np.isfinite(u_dp) & np.isfinite(us) & np.isfinite(turn_radius)
    for us_level in us_levels:
        us_target = float(us_level)
        mask = finite & np.isclose(us, us_target, atol=us_tol, rtol=0.0)
        if not np.any(mask):
            print(f"No turn-radius points found for u_s={us_target:.2f}")
            continue

        grouped = (
            pd.DataFrame({"u_dp": u_dp[mask], "turn_radius": turn_radius[mask]})
            .groupby("u_dp", as_index=False)["turn_radius"]
            .mean()
            .sort_values("u_dp")
        )
        x_vals = grouped["u_dp"].to_numpy(dtype=float)
        y_vals = grouped["turn_radius"].to_numpy(dtype=float)
        lines[us_target] = (x_vals, y_vals)
        print(f"Loaded turn-radius line for u_s={us_target:.2f}: n={len(grouped)}")

    return lines


def _panel_ylim(
    datasets: list[pd.DataFrame],
    all_bands: list[dict[str, list[Band]]],
    metric: str,
    overlay_points: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    overlay_uncertainty: (
        dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None
    ) = None,
    transient_points: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    transient_uncertainty: (
        dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None
    ) = None,
) -> tuple[float, float]:
    """Get robust y-limits including bands and representative scatter range."""
    y_vals = []
    for df in datasets:
        if metric == "g_k":
            vals = df["g_k_inst"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                p_low, p_high = np.nanpercentile(vals, [1, 99])
                y_vals.extend([float(p_low), float(p_high)])
        elif metric == "turn_radius":
            valid_turn = (
                df["turn_radius_valid"].to_numpy(dtype=bool)
                if "turn_radius_valid" in df.columns
                else np.ones(len(df), dtype=bool)
            )
            vals = df.loc[valid_turn, "turn_radius"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                p_low, p_high = np.nanpercentile(vals, [5, 95])
                y_vals.extend([float(p_low), float(p_high)])
        else:
            vals = df[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                y_vals.extend([float(np.nanmin(vals)), float(np.nanmax(vals))])

    for band_dict in all_bands:
        for band in band_dict[metric]:
            y_vals.extend([band.y_min, band.y_max])

    if overlay_points is not None and metric in overlay_points:
        overlay_y = overlay_points[metric][1]
        overlay_y = overlay_y[np.isfinite(overlay_y)]
        if overlay_y.size:
            if metric == "g_k":
                p_low, p_high = np.nanpercentile(overlay_y, [5, 95])
                y_vals.extend([float(p_low), float(p_high)])
            else:
                y_vals.extend(
                    [float(np.nanmin(overlay_y)), float(np.nanmax(overlay_y))]
                )

    if overlay_uncertainty is not None and metric in overlay_uncertainty:
        _, overlay_y_low, overlay_y_high = overlay_uncertainty[metric]
        overlay_y_low = overlay_y_low[np.isfinite(overlay_y_low)]
        overlay_y_high = overlay_y_high[np.isfinite(overlay_y_high)]
        if overlay_y_low.size:
            y_vals.extend(
                [float(np.nanmin(overlay_y_low)), float(np.nanmax(overlay_y_low))]
            )
        if overlay_y_high.size:
            y_vals.extend(
                [float(np.nanmin(overlay_y_high)), float(np.nanmax(overlay_y_high))]
            )

    if transient_points is not None and metric in transient_points:
        transient_y = transient_points[metric][1]
        transient_y = transient_y[np.isfinite(transient_y)]
        if transient_y.size:
            if metric == "g_k":
                p_low, p_high = np.nanpercentile(transient_y, [5, 95])
                y_vals.extend([float(p_low), float(p_high)])
            else:
                y_vals.extend(
                    [float(np.nanmin(transient_y)), float(np.nanmax(transient_y))]
                )

    if transient_uncertainty is not None and metric in transient_uncertainty:
        _, transient_y_low, transient_y_high = transient_uncertainty[metric]
        transient_y_low = transient_y_low[np.isfinite(transient_y_low)]
        transient_y_high = transient_y_high[np.isfinite(transient_y_high)]
        if transient_y_low.size:
            y_vals.extend(
                [float(np.nanmin(transient_y_low)), float(np.nanmax(transient_y_low))]
            )
        if transient_y_high.size:
            y_vals.extend(
                [float(np.nanmin(transient_y_high)), float(np.nanmax(transient_y_high))]
            )

    if not y_vals:
        return 0.0, 1.0

    y_min = min(y_vals)
    y_max = max(y_vals)
    if y_max <= y_min:
        y_max = y_min + 1.0
    pad = 0.08 * (y_max - y_min)
    return y_min - pad, y_max + pad


def _turn_radius_ylim(
    datasets: list[pd.DataFrame],
    all_bands: list[dict[str, list[Band]]],
    lines_by_us: dict[float, tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    """Get y-limits for turn-radius panel from experimental bands and sim lines."""
    y_exp_min, y_exp_max = _panel_ylim(datasets, all_bands, "turn_radius")
    y_vals: list[float] = [y_exp_min, y_exp_max]
    for _, (_, y_line) in lines_by_us.items():
        y_finite = y_line[np.isfinite(y_line)]
        if y_finite.size:
            y_vals.extend([float(np.nanmin(y_finite)), float(np.nanmax(y_finite))])

    if not y_vals:
        return 0.0, 1.0

    y_min = min(y_vals)
    y_max = max(y_vals)
    if y_max <= y_min:
        y_max = y_min + max(1.0, 0.05 * abs(y_min))
    pad = 0.08 * (y_max - y_min)
    return y_min - pad, y_max + pad


def _plot_metric_panel(
    ax: plt.Axes,
    metric: str,
    ylabel: str,
    datasets: list[pd.DataFrame],
    years: list[str],
    bands: list[dict[str, list[Band]]],
    year_colors: dict[str, str],
    overlay_points: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    overlay_uncertainty: (
        dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None
    ) = None,
    transient_points: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    transient_uncertainty: (
        dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None
    ) = None,
) -> None:
    """Plot one metric panel using bands (no experimental scatter points)."""

    for year_band in bands:
        # Draw upward first so downward (hatched) remains visible on top.
        bands_ordered = sorted(
            year_band[metric],
            key=lambda b: 0 if b.direction == "upward" else 1,
        )
        for band in bands_ordered:
            color = year_colors[band.year]
            line_style = "-" if band.direction == "upward" else "--"
            hatch = "///" if band.direction == "downward" else None
            x_band = np.array([band.x_min, band.x_max], dtype=float)
            ax.fill_between(
                x_band,
                np.array([band.y_min, band.y_min], dtype=float),
                np.array([band.y_max, band.y_max], dtype=float),
                facecolor=color,
                edgecolor="none",
                alpha=0.12,
                zorder=1,
            )
            if hatch is not None:
                ax.fill_between(
                    x_band,
                    np.array([band.y_min, band.y_min], dtype=float),
                    np.array([band.y_max, band.y_max], dtype=float),
                    facecolor="none",
                    edgecolor=to_rgba(color, alpha=1.0),
                    hatch=hatch,
                    linewidth=1.0,
                    alpha=1.0,
                    zorder=1.1,
                )
            ax.plot(
                x_band,
                np.array([band.y_min, band.y_min], dtype=float),
                color=color,
                linestyle=line_style,
                linewidth=1.0,
                alpha=0.9,
                zorder=2,
            )
            ax.plot(
                x_band,
                np.array([band.y_max, band.y_max], dtype=float),
                color=color,
                linestyle=line_style,
                linewidth=1.0,
                alpha=0.9,
                zorder=2,
            )

    if overlay_points is not None and metric in overlay_points:
        x_overlay, y_overlay = overlay_points[metric]
        finite = np.isfinite(x_overlay) & np.isfinite(y_overlay)
        if overlay_uncertainty is not None and metric in overlay_uncertainty:
            x_unc, y_low, y_high = overlay_uncertainty[metric]
            finite_unc = np.isfinite(x_unc) & np.isfinite(y_low) & np.isfinite(y_high)
            if np.any(finite_unc):
                ax.fill_between(
                    x_unc[finite_unc],
                    y_low[finite_unc],
                    y_high[finite_unc],
                    facecolor=to_rgba("gray", alpha=0.7),
                    edgecolor="none",
                    zorder=3,
                )
        if np.any(finite):
            ax.plot(
                x_overlay[finite],
                y_overlay[finite],
                color="black",
                linewidth=1.1,
                alpha=0.9,
                zorder=3.8,
            )
        ax.scatter(
            x_overlay[finite],
            y_overlay[finite],
            marker="o",
            s=30,
            linewidths=1.4,
            color="black",
            alpha=1.0,
            zorder=4.2,
        )

    if transient_points is not None and metric in transient_points:
        x_transient, y_transient = transient_points[metric]
        finite = np.isfinite(x_transient) & np.isfinite(y_transient)
        if transient_uncertainty is not None and metric in transient_uncertainty:
            x_unc, y_low, y_high = transient_uncertainty[metric]
            finite_unc = np.isfinite(x_unc) & np.isfinite(y_low) & np.isfinite(y_high)
            if np.any(finite_unc):
                y_center = 0.5 * (y_low + y_high)
                y_err_low = np.maximum(y_center - y_low, 0.0)
                y_err_high = np.maximum(y_high - y_center, 0.0)
                ax.errorbar(
                    x_unc[finite_unc],
                    y_center[finite_unc],
                    yerr=np.vstack([y_err_low[finite_unc], y_err_high[finite_unc]]),
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.2,
                    capsize=3.4,
                    capthick=1.2,
                    alpha=0.95,
                    zorder=4.8,
                )
        ax.scatter(
            x_transient[finite],
            y_transient[finite],
            marker="o",
            s=40,
            facecolors="none",
            edgecolors="black",
            linewidths=1.4,
            alpha=0.95,
            zorder=5,
        )

    ax.set_xlabel(r"$u_\mathrm{dp}$ (-)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _plot_turn_radius_panel(
    ax: plt.Axes,
    datasets: list[pd.DataFrame],
    years: list[str],
    bands: list[dict[str, list[Band]]],
    year_colors: dict[str, str],
    lines_by_us: dict[float, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Plot experimental turn-radius bands and simulation steering lines."""
    _plot_metric_panel(
        ax,
        metric="turn_radius",
        ylabel=r"$R_\mathrm{turn}$ (m)",
        datasets=datasets,
        years=years,
        bands=bands,
        year_colors=year_colors,
    )

    style_map = {
        0.10: {"color": "black", "marker": "o", "label": r"$u_\mathrm{s}=0.10$"},
        0.15: {"color": "black", "marker": "^", "label": r"$u_\mathrm{s}=0.15$"},
        0.20: {"color": "black", "marker": "*", "label": r"$u_\mathrm{s}=0.20$"},
    }

    plotted_any = False
    for us_level in sorted(lines_by_us):
        x_vals, y_vals = lines_by_us[us_level]
        finite = np.isfinite(x_vals) & np.isfinite(y_vals)
        if not np.any(finite):
            continue

        style = style_map.get(
            round(float(us_level), 2),
            {
                "color": "black",
                "marker": "d",
                "label": rf"$u_\mathrm{{s}}={us_level:.2f}$",
            },
        )
        ax.plot(
            x_vals[finite],
            y_vals[finite],
            marker=style["marker"],
            linestyle="-",
            color=style["color"],
            linewidth=1.6,
            markersize=4.8,
            label=style["label"],
            zorder=4,
        )
        plotted_any = True

    if plotted_any:
        ax.legend(
            loc="upper left",
            frameon=True,
            framealpha=1.0,
            borderpad=0.3,
            labelspacing=0.35,
            handlelength=1.5,
        )


def main(plot_transient: bool = False) -> None:
    set_plot_style()
    plt.rcParams["hatch.linewidth"] = 1.4

    cfg_2019 = FlightConfig(
        title="2019-10-08",
        year="2019",
        month="10",
        day="08",
        kite_model="v3",
        addition="_t26",
        time_range=(2190, 2255),
        downsample_frac=1.0,
        apply_quadrant_filter=False,
    )
    cfg_2025 = FlightConfig(
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
    )

    datasets = [
        load_experimental_dataset(cfg_2019),
        load_experimental_dataset(cfg_2025),
    ]
    years = [cfg_2019.year, cfg_2025.year]
    bands = [
        compute_bands(datasets[0], cfg_2019.year),
        compute_bands(datasets[1], cfg_2025.year),
    ]
    csv_path = Path("./data/vw8_lt270_circles_combined_all.csv")
    vw8_overlay, vw8_overlay_uncertainty = load_vw8_varying_us_points(csv_path)
    vw8_turn_radius_lines = load_vw8_turn_radius_lines(
        csv_path,
        us_levels=(0.10, 0.15, 0.20),
    )
    transient_overlay: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
    transient_overlay_uncertainty: (
        dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None
    ) = None
    if plot_transient:
        transient_overlay, transient_overlay_uncertainty = (
            load_vw8_transient_average_points(csv_path)
        )

    year_colors = {"2019": "#1f77b4", "2025": "#d62728"}

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))

    _plot_metric_panel(
        axes[0],
        metric="g_k",
        ylabel=r"$g_\mathrm{k}$ ($\mathrm{rad\,s^{-1}}$)",
        datasets=datasets,
        years=years,
        bands=bands,
        year_colors=year_colors,
        overlay_points=vw8_overlay,
        overlay_uncertainty=vw8_overlay_uncertainty,
        transient_points=transient_overlay,
        transient_uncertainty=transient_overlay_uncertainty,
    )

    _plot_metric_panel(
        axes[1],
        metric="v_a",
        ylabel=r"$v_\mathrm{a}$ ($\mathrm{m\,s^{-1}}$)",
        datasets=datasets,
        years=years,
        bands=bands,
        year_colors=year_colors,
        overlay_points=vw8_overlay,
        overlay_uncertainty=vw8_overlay_uncertainty,
        transient_points=transient_overlay,
        transient_uncertainty=transient_overlay_uncertainty,
    )
    _plot_turn_radius_panel(
        axes[2],
        datasets=datasets,
        years=years,
        bands=bands,
        year_colors=year_colors,
        lines_by_us=vw8_turn_radius_lines,
    )

    # Fixed x-limits requested for all panels.
    for ax in axes:
        ax.set_xlim(0.16, 0.44)

    y0 = _panel_ylim(
        datasets,
        bands,
        "g_k",
        overlay_points=vw8_overlay,
        overlay_uncertainty=vw8_overlay_uncertainty,
        transient_points=transient_overlay,
        transient_uncertainty=transient_overlay_uncertainty,
    )
    y1 = _panel_ylim(
        datasets,
        bands,
        "v_a",
        overlay_points=vw8_overlay,
        overlay_uncertainty=vw8_overlay_uncertainty,
        transient_points=transient_overlay,
        transient_uncertainty=transient_overlay_uncertainty,
    )
    axes[0].set_ylim(*y0)
    axes[1].set_ylim(*y1)
    axes[2].set_ylim(0, 120)

    down_2019_handle = Patch(label=r"2019 downward $g_\mathrm{k,l-r}$")
    down_2025_handle = Patch(label=r"2025 downward $g_\mathrm{k,l-r}$")
    legend_handles = [
        Patch(
            facecolor=to_rgba(year_colors["2019"], alpha=0.12),
            edgecolor="none",
            linewidth=0.0,
            label=r"2019 upward $g_\mathrm{k,l-r}$",
        ),
        down_2019_handle,
        Patch(
            facecolor=to_rgba(year_colors["2025"], alpha=0.12),
            edgecolor="none",
            linewidth=0.0,
            label=r"2025 upward $g_\mathrm{k,l-r}$",
        ),
        down_2025_handle,
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            linewidth=1.1,
            markersize=float(np.sqrt(30.0)),
            markeredgewidth=1.4,
            color="black",
            alpha=1.0,
            label=r"Sim. uniform $g_\mathrm{k}$",
        ),
        Patch(
            facecolor=to_rgba("gray", alpha=0.7),
            edgecolor="none",
            label=r"Sim. uniform variance",
        ),
    ]
    if plot_transient:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=float(np.sqrt(40.0)),
                markerfacecolor="none",
                markeredgewidth=1.4,
                color="black",
                alpha=0.95,
                label="Sim. transient",
            )
        )
    handler_map = {
        down_2019_handle: TwoLayerBandHandler(year_colors["2019"]),
        down_2025_handle: TwoLayerBandHandler(year_colors["2025"]),
    }
    axes[0].legend(
        handles=legend_handles,
        handler_map=handler_map,
        loc="upper left",
        ncol=1,
        frameon=True,
        framealpha=1,
        handlelength=1.0,
        handleheight=1.0,
        # borderpad=0.05,
        labelspacing=0.3,
        fontsize=11,
        # bbox_to_anchor=(0.9, 0.9),
    )

    axes[0].set_ylim(0, 0.9)
    axes[1].set_ylim(0, 50)
    fig.tight_layout()

    output_path = Path("./results/plots_paper") / "plot_3col_T26_gk_va_alpha.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
