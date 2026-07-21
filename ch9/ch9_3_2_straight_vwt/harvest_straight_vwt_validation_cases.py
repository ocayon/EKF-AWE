#!/usr/bin/env python3
"""Harvest straight, steady EKF intervals for Ch. 9.3.2 VWT validation."""

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
EKFAWE_ROOT = REPO_ROOT.parents[1]  # EKF-AWE repo root
SRC_ROOT = EKFAWE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from awes_ekf.load_data.read_data import read_results_from_hdf5

DEFAULT_RESULTS_2019 = EKFAWE_ROOT / "results/v3/v3_2019-10-08_t26.h5"
DEFAULT_RESULTS_2025 = EKFAWE_ROOT / "results/v3/v3_2025-10-09.h5"
DEFAULT_OUTPUT_DIR = REPO_ROOT  # CSV outputs live alongside this script
PERCENTILES = (16.0, 50.0, 84.0)
# DEFAULT_VALIDATION_2019_WINDOW = (2190.0, 2260.0)
# DEFAULT_VALIDATION_2025_WINDOW = (700.0, 800.0)
DEFAULT_VALIDATION_2019_WINDOW = None  # (1590.0, 2260.0)
DEFAULT_VALIDATION_2025_WINDOW = (700.0, 1000.0)

"""
python harvest_straight_vwt_validation_cases.py 
python plot_straight_vwt_harvest_qc.py 
"""

USE_YEAR_DEFAULT_FILTERS = True
DEFAULT_VALIDATION_FILTERS: Dict[int, Dict[str, Any]] = {
    2019: {
        "u_dp_min": 0,
        "u_dp_max": 0.2,
        "max_abs_u_s": 0.05,
        "only_reel_out": True,
        "time_range_s": DEFAULT_VALIDATION_2019_WINDOW,
        "time_bin_s": 2.0,
        "time_bin_step_s": 0.5,
        "max_path_curvature_1pm": 100.0 / 150.0,
        "max_abs_yaw_rate_deg_s": 90.0,
    },
    2025: {
        "u_dp_min": 0.0,
        "u_dp_max": 0.42,
        "max_abs_u_s": 0.05,
        "only_reel_out": True,
        "time_range_s": DEFAULT_VALIDATION_2025_WINDOW,  # 800 ->1200
        "time_bin_s": 2.0,
        "time_bin_step_s": 0.5,
        "max_path_curvature_1pm": 100.0 / 150.0,
        "max_abs_yaw_rate_deg_s": 90.0,
    },
}
DEFAULT_MIN_REELOUT_SPEED_MS = 0.0
DEFAULT_ROLLING_QUANTILE = 0.84
DEFAULT_MAX_STEERING_RATE_S = 0.20
DEFAULT_MIN_ABS_TURN_RADIUS_M = 150.0
DEFAULT_MAX_LATERAL_ACCEL_MS2 = 8.0
DEFAULT_MAX_VA_CV = 0.08
DEFAULT_MAX_FORCE_CV = 0.12
DEFAULT_MAX_DEPOWER_STD = 0.015
DEFAULT_MAX_REEL_SPEED_STD = 0.35
DEFAULT_MAX_TETHER_LENGTH_STD = 0.75
DEFAULT_MIN_VA_MS = 5.0
DEFAULT_MAX_VA_MS = 60.0
DEFAULT_MIN_FORCE_N = 50.0
DEFAULT_MIN_TETHER_LENGTH_M = 20.0
DEFAULT_MAX_TETHER_LENGTH_M = 1000.0
DEFAULT_MAX_ABS_SLACK_M = 30.0
DEFAULT_BIN_VA_WIDTH = 2.0
DEFAULT_BIN_TETHER_WIDTH = 25.0


@dataclass(frozen=True)
class CampaignConfig:
    year: int
    h5_path: Path
    campaign_label: str
    time_range: Optional[Tuple[float, float]]
    depower_conversion: str
    force_priority: Tuple[str, ...]


def parse_time_range(value: str) -> Tuple[float, float]:
    try:
        start_s, end_s = [float(part.strip()) for part in value.split(",", 1)]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "time ranges must use 'start,end' seconds"
        ) from exc
    if end_s <= start_s:
        raise argparse.ArgumentTypeError("time range end must be larger than start")
    return (start_s, end_s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Harvest straight, near-symmetric EKF intervals for Ch. 9.3.2 "
            "ASKITE VWT validation."
        )
    )
    parser.add_argument("--results-2019", type=Path, default=DEFAULT_RESULTS_2019)
    parser.add_argument("--results-2025", type=Path, default=DEFAULT_RESULTS_2025)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--disable-year-default-filters",
        dest="use_year_default_filters",
        action="store_false",
        default=USE_YEAR_DEFAULT_FILTERS,
        help=(
            "Disable top-level year defaults for u_dp, u_s, reel-out, time bins, "
            "path curvature, and yaw-rate filters."
        ),
    )
    parser.add_argument("--time-range-2019", type=parse_time_range, default=None)
    parser.add_argument(
        "--time-range-2025",
        type=parse_time_range,
        default=None,
        help=(
            "Loaded 2025 time range. Defaults to the straight-flight validation "
            "range in DEFAULT_VALIDATION_FILTERS when year defaults are enabled."
        ),
    )
    parser.add_argument(
        "--validation-2025-window",
        type=parse_time_range,
        default=DEFAULT_VALIDATION_2025_WINDOW,
        help="Window used only to check the documented 2025 depower convention.",
    )
    parser.add_argument("--window-s", type=float, default=None)
    parser.add_argument("--window-step-s", type=float, default=None)
    parser.add_argument(
        "--min-window-samples",
        type=int,
        default=None,
        help="Default is 80 percent of the expected samples in --window-s.",
    )
    parser.add_argument("--max-abs-us", type=float, default=None)
    parser.add_argument("--max-abs-yaw-rate-deg-s", type=float, default=None)
    parser.add_argument(
        "--max-steering-rate-s",
        type=float,
        default=DEFAULT_MAX_STEERING_RATE_S,
    )
    parser.add_argument(
        "--min-abs-turn-radius-m",
        type=float,
        default=DEFAULT_MIN_ABS_TURN_RADIUS_M,
    )
    parser.add_argument(
        "--rolling-quantile",
        type=float,
        default=DEFAULT_ROLLING_QUANTILE,
        help=(
            "Rolling quantile used for yaw-rate, curvature, and lateral-"
            "acceleration straightness checks."
        ),
    )
    parser.add_argument(
        "--max-lateral-accel-ms2",
        type=float,
        default=DEFAULT_MAX_LATERAL_ACCEL_MS2,
    )
    parser.add_argument("--max-va-cv", type=float, default=DEFAULT_MAX_VA_CV)
    parser.add_argument("--max-force-cv", type=float, default=DEFAULT_MAX_FORCE_CV)
    parser.add_argument(
        "--max-depower-std", type=float, default=DEFAULT_MAX_DEPOWER_STD
    )
    parser.add_argument(
        "--max-reel-speed-std",
        type=float,
        default=DEFAULT_MAX_REEL_SPEED_STD,
    )
    parser.add_argument(
        "--max-tether-length-std",
        type=float,
        default=DEFAULT_MAX_TETHER_LENGTH_STD,
    )
    parser.add_argument(
        "--min-reelout-speed-ms",
        type=float,
        default=DEFAULT_MIN_REELOUT_SPEED_MS,
        help=(
            "Fallback threshold for identifying reel-out when phase labels are "
            "not available. Positive tether_reelout_speed means reel-out."
        ),
    )
    parser.add_argument("--min-va-ms", type=float, default=DEFAULT_MIN_VA_MS)
    parser.add_argument("--max-va-ms", type=float, default=DEFAULT_MAX_VA_MS)
    parser.add_argument("--min-force-n", type=float, default=DEFAULT_MIN_FORCE_N)
    parser.add_argument(
        "--min-tether-length-m",
        type=float,
        default=DEFAULT_MIN_TETHER_LENGTH_M,
    )
    parser.add_argument(
        "--max-tether-length-m",
        type=float,
        default=DEFAULT_MAX_TETHER_LENGTH_M,
    )
    parser.add_argument(
        "--max-abs-slack-m", type=float, default=DEFAULT_MAX_ABS_SLACK_M
    )
    parser.add_argument(
        "--max-u-dp-2019",
        type=float,
        default=None,
        help=(
            "Override the 2019 u_dp upper limit from DEFAULT_VALIDATION_FILTERS. "
            "Use with --disable-year-default-filters to remove the default limit."
        ),
    )
    parser.add_argument("--bin-va-width", type=float, default=DEFAULT_BIN_VA_WIDTH)
    parser.add_argument(
        "--bin-tether-width",
        type=float,
        default=DEFAULT_BIN_TETHER_WIDTH,
    )
    parser.add_argument("--write-parquet", action="store_true")
    return parser.parse_args()


def numeric_series(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.full(len(df), default), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def clean_string_series(
    df: pd.DataFrame, column: str, default: str = "unknown"
) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=object)
    series = df[column].astype(str).str.strip()
    series = series.replace({"nan": default, "None": default, "": default})
    return series


def finite_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.isfinite(values) & (values > 0.0)


def finite_or_nan(values: Sequence[float]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def default_filter(year: int, name: str, enabled: bool) -> Any:
    if not enabled:
        return None
    return DEFAULT_VALIDATION_FILTERS[year][name]


def campaign_filter_value(
    cfg: CampaignConfig,
    args: argparse.Namespace,
    name: str,
    override: Any = None,
) -> Any:
    if override is not None:
        return override
    return default_filter(cfg.year, name, args.use_year_default_filters)


def safe_nanpercentile(values: pd.Series, percentile: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, percentile))


def safe_mean(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_std(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else float("nan")
    return float(np.std(arr, ddof=1))


def circular_mean_rad(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(math.atan2(np.mean(np.sin(arr)), np.mean(np.cos(arr))))


def circular_mean_deg(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    radians = np.deg2rad(arr)
    return float(
        np.rad2deg(math.atan2(np.mean(np.sin(radians)), np.mean(np.cos(radians))))
    )


def maybe_radians_to_degrees(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return numeric
    if float(finite.abs().quantile(0.99)) <= 2.0 * math.pi + 0.5:
        return np.rad2deg(numeric)
    return numeric


def maybe_degrees_per_second_to_radians(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    if finite.empty:
        return numeric
    if float(finite.abs().quantile(0.99)) > 10.0:
        return np.deg2rad(numeric)
    return numeric


def gradient(values: pd.Series, time_s: pd.Series) -> pd.Series:
    y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(time_s, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(y), np.nan)
    finite = np.isfinite(y) & np.isfinite(t)
    if finite.sum() < 3:
        return pd.Series(result, index=values.index, dtype=float)
    order = np.argsort(t[finite])
    y_finite = y[finite][order]
    t_finite = t[finite][order]
    if np.any(np.diff(t_finite) <= 0.0):
        return pd.Series(result, index=values.index, dtype=float)
    grad = np.gradient(y_finite, t_finite)
    finite_indices = np.flatnonzero(finite)[order]
    result[finite_indices] = grad
    result[~np.isfinite(result)] = np.nan
    return pd.Series(result, index=values.index, dtype=float)


def infer_min_window_samples(df: pd.DataFrame, window_s: float) -> int:
    dt = np.diff(pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float))
    dt = dt[np.isfinite(dt) & (dt > 0.0)]
    if dt.size == 0:
        return max(1, int(math.ceil(window_s)))
    sample_rate_hz = 1.0 / float(np.median(dt))
    return max(1, int(math.floor(0.8 * sample_rate_hz * window_s)))


def rolling_cv(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window, center=True, min_periods=min_periods).mean()
    std = series.rolling(window, center=True, min_periods=min_periods).std()
    return std / mean.abs().replace(0.0, np.nan)


def mode_label(values: pd.Series, default: str = "unknown") -> str:
    series = values.dropna().astype(str)
    series = series[(series != "") & (series != "nan") & (series != "None")]
    if series.empty:
        return default
    return str(series.mode().iloc[0])


def git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def file_metadata(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_time_utc": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
    }


def build_campaign_configs(args: argparse.Namespace) -> List[CampaignConfig]:
    return [
        CampaignConfig(
            year=2019,
            h5_path=args.results_2019,
            campaign_label="2019",
            time_range=(
                args.time_range_2019
                if args.time_range_2019 is not None
                else default_filter(2019, "time_range_s", args.use_year_default_filters)
            ),
            depower_conversion="convert_2019_to_2025_best_guess_section_9_1",
            force_priority=(
                "tether_force_kite",
                "load_cell_main_force",
                "ground_tether_force",
            ),
        ),
        CampaignConfig(
            year=2025,
            h5_path=args.results_2025,
            campaign_label="2025",
            time_range=(
                args.time_range_2025
                if args.time_range_2025 is not None
                else default_filter(2025, "time_range_s", args.use_year_default_filters)
            ),
            depower_conversion="use_2025_convention",
            force_priority=(
                "tether_force_kite",
                "ground_tether_force",
                "sensor_tether_force",
            ),
        ),
    ]


def force_sample_column(source: str) -> str:
    return {
        "tether_force_kite": "tether_force_kite_N",
        "load_cell_main_force": "load_cell_main_force_N",
        "ground_tether_force": "ground_tether_force_N",
        "sensor_tether_force": "sensor_tether_force_N",
    }[source]


def add_force_preference(df: pd.DataFrame, priority: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    df["preferred_tether_force_N"] = np.nan
    df["force_source_preferred"] = "none"
    for source in priority:
        column = force_sample_column(source)
        if column not in df.columns:
            continue
        mask = (
            df["preferred_tether_force_N"].isna()
            & np.isfinite(df[column])
            & (df[column] > 0.0)
        )
        df.loc[mask, "preferred_tether_force_N"] = df.loc[mask, column]
        df.loc[mask, "force_source_preferred"] = source
    return df


def build_working_dataframe(
    cfg: CampaignConfig, args: argparse.Namespace, warnings: List[str]
) -> pd.DataFrame:
    ekf_output, flight_data, _ = read_results_from_hdf5(str(cfg.h5_path))
    if len(ekf_output) != len(flight_data):
        raise ValueError(
            f"{cfg.h5_path}: ekf_output has {len(ekf_output)} rows but "
            f"flight_data has {len(flight_data)} rows."
        )

    df = pd.DataFrame(index=ekf_output.index)
    df["campaign"] = cfg.campaign_label
    df["year"] = cfg.year
    df["source_h5"] = str(cfg.h5_path)
    df["sample_index"] = np.arange(len(df))
    df["sample_id"] = [
        f"{cfg.campaign_label}_sample_{idx:06d}" for idx in range(len(df))
    ]

    ekf_time = numeric_series(ekf_output, "time")
    flight_time = numeric_series(flight_data, "time")
    df["time_s"] = ekf_time.where(np.isfinite(ekf_time), flight_time)
    time_delta = np.nanmax(np.abs(ekf_time - flight_time))
    if np.isfinite(time_delta) and time_delta > 1e-6:
        warnings.append(
            f"{cfg.campaign_label}: ekf_output and flight_data time columns differ "
            f"by up to {time_delta:.6g} s; ekf_output time was used where finite."
        )

    if cfg.time_range is not None:
        lo, hi = cfg.time_range
        df = df[(df["time_s"] >= lo) & (df["time_s"] <= hi)].copy()
        ekf_output = ekf_output.loc[df.index].copy()
        flight_data = flight_data.loc[df.index].copy()

    df["turn_straight_label"] = clean_string_series(flight_data, "turn_straight")
    df["powered_label"] = clean_string_series(flight_data, "powered")
    df["flight_phase_label"] = clean_string_series(flight_data, "flight_phase")
    if "flight_phase_index" in flight_data.columns:
        phase = numeric_series(flight_data, "flight_phase_index")
        df["phase_label"] = [
            "unknown" if not np.isfinite(value) else f"phase_{int(value)}"
            for value in phase
        ]
    else:
        df["phase_label"] = "unknown"

    df["u_s_existing"] = numeric_series(flight_data, "us")
    df["u_dp_existing"] = numeric_series(flight_data, "up")
    df["kcu_actual_steering_raw"] = numeric_series(flight_data, "kcu_actual_steering")
    df["kcu_actual_depower_raw"] = numeric_series(flight_data, "kcu_actual_depower")

    steering_raw = df["kcu_actual_steering_raw"]
    steering_finite = steering_raw[np.isfinite(steering_raw)]
    if not steering_finite.empty and float(steering_finite.abs().quantile(0.99)) > 1.5:
        df["u_s_ch9"] = steering_raw / 100.0
        steering_mapping = "kcu_actual_steering_percent_div_100"
    elif not steering_finite.empty:
        df["u_s_ch9"] = steering_raw
        steering_mapping = "kcu_actual_steering_already_normalized"
    else:
        df["u_s_ch9"] = df["u_s_existing"]
        steering_mapping = "flight_data_us_fallback"
        warnings.append(f"{cfg.campaign_label}: using flight_data['us'] for u_s_ch9.")
    df["u_s_conversion_rule"] = steering_mapping
    df["u_s_rate_s"] = gradient(df["u_s_ch9"], df["time_s"])

    if cfg.year == 2025:
        raw_depower = df["kcu_actual_depower_raw"]
        finite = raw_depower[np.isfinite(raw_depower)]
        if not finite.empty and float(finite.abs().quantile(0.99)) > 1.5:
            df["u_dp_ch9"] = raw_depower / 100.0
            depower_rule = "u_dp_ch9 = kcu_actual_depower / 100"
        else:
            df["u_dp_ch9"] = raw_depower
            depower_rule = "u_dp_ch9 = kcu_actual_depower"
        df["u_p_2019_raw_or_normalized"] = np.nan
        validation_mean = safe_mean(df["u_dp_ch9"])
        max_abs_u_s = campaign_filter_value(cfg, args, "max_abs_u_s", args.max_abs_us)
        if max_abs_u_s is None:
            validation_mean_strict = float("nan")
        else:
            validation_mean_strict = safe_mean(
                df.loc[df["u_s_ch9"].abs() <= max_abs_u_s, "u_dp_ch9"]
            )
        validation_window_mean = float("nan")
        if args.validation_2025_window is not None:
            start_s, end_s = args.validation_2025_window
            validation_window_mask = (df["time_s"] >= start_s) & (df["time_s"] <= end_s)
            validation_window_mean = safe_mean(
                df.loc[validation_window_mask, "u_dp_ch9"]
            )
            if (
                np.isfinite(validation_window_mean)
                and abs(validation_window_mean - 0.42) <= 0.05
            ):
                warnings.append(
                    f"2025: u_dp_ch9 mean over validation window {start_s:.1f}-{end_s:.1f} s "
                    f"is {validation_window_mean:.3f}, near the Section 9.1 value 0.42."
                )
            else:
                warnings.append(
                    f"2025: u_dp_ch9 mean over validation window {start_s:.1f}-{end_s:.1f} s "
                    f"is {validation_window_mean:.3f}; mapping kept as {depower_rule}."
                )
        if not np.isfinite(validation_mean) or abs(validation_mean - 0.42) > 0.05:
            warnings.append(
                f"2025: u_dp_ch9 mean over loaded data is {validation_mean:.3f}, "
                "not near Section 9.1 reference value 0.42. Mapping kept as "
                f"{depower_rule}."
            )
        df["u_dp_validation_mean_2025"] = validation_mean
        df["u_dp_validation_mean_2025_strict_steering"] = validation_mean_strict
        df["u_dp_validation_window_mean_2025"] = validation_window_mean
    elif cfg.year == 2019:
        raw_or_normalized = df["u_dp_existing"]
        if raw_or_normalized.notna().any():
            u_p_2019 = 1.0 - raw_or_normalized.clip(lower=0.0, upper=1.0)
            depower_rule = (
                "u_p_2019_raw_or_normalized = 1 - clipped flight_data['up']; "
                "u_dp_ch9 = 0.2564 - 0.0768 * u_p_2019_raw_or_normalized"
            )
        else:
            raw = df["kcu_actual_depower_raw"]
            span = float(raw.max() - raw.min())
            if not np.isfinite(span) or span <= 0.0:
                u_p_2019 = pd.Series(np.nan, index=df.index)
            else:
                normalized_depower = (raw - float(raw.min())) / span
                u_p_2019 = 1.0 - normalized_depower.clip(lower=0.0, upper=1.0)
            depower_rule = (
                "u_p_2019_raw_or_normalized = normalized inverse "
                "kcu_actual_depower fallback; u_dp_ch9 = 0.2564 - 0.0768 * "
                "u_p_2019_raw_or_normalized"
            )
        df["u_p_2019_raw_or_normalized"] = u_p_2019
        df["u_dp_ch9"] = 0.2564 - 0.0768 * u_p_2019
        warnings.append(
            "2019: u_dp_ch9 uses the approximate Section 9.1 conversion "
            "u_dp = 0.2564 - 0.0768 u_p_2019; treat 2019 geometry validation "
            "as trend-only."
        )
    else:
        raise ValueError(f"Unsupported campaign year: {cfg.year}")
    df["u_dp_conversion_rule"] = depower_rule
    df["depower_tape_length_m"] = 0.2 + 5.0 * df["u_dp_ch9"]

    df["V_a_ms"] = numeric_series(ekf_output, "kite_apparent_windspeed").where(
        np.isfinite(numeric_series(ekf_output, "kite_apparent_windspeed")),
        numeric_series(flight_data, "kite_apparent_windspeed"),
    )
    df["kite_position_x_m"] = numeric_series(ekf_output, "kite_position_x")
    df["kite_position_y_m"] = numeric_series(ekf_output, "kite_position_y")
    df["kite_position_z_m"] = numeric_series(ekf_output, "kite_position_z")
    df["kite_velocity_x_ms"] = numeric_series(ekf_output, "kite_velocity_x")
    df["kite_velocity_y_ms"] = numeric_series(ekf_output, "kite_velocity_y")
    df["kite_velocity_z_ms"] = numeric_series(ekf_output, "kite_velocity_z")
    df["wind_speed_horizontal_ms"] = numeric_series(ekf_output, "wind_speed_horizontal")
    df["wind_direction_rad"] = numeric_series(ekf_output, "wind_direction")
    df["wind_speed_vertical_ms"] = numeric_series(ekf_output, "wind_speed_vertical")
    df["tether_force_kite_N"] = numeric_series(ekf_output, "tether_force_kite")
    df["ground_tether_force_N"] = numeric_series(flight_data, "ground_tether_force")
    df["load_cell_main_force_N"] = numeric_series(flight_data, "load_cell_main_force")
    df["sensor_tether_force_N"] = numeric_series(flight_data, "sensor_tether_force")
    df["tether_length_m"] = numeric_series(flight_data, "tether_length").where(
        np.isfinite(numeric_series(flight_data, "tether_length")),
        numeric_series(ekf_output, "tether_length"),
    )
    df["tether_reelout_speed_ms"] = numeric_series(flight_data, "tether_reelout_speed")
    df["CL_ekf"] = numeric_series(ekf_output, "wing_lift_coefficient")
    df["CD_ekf"] = numeric_series(ekf_output, "wing_drag_coefficient")
    df["CS_ekf"] = numeric_series(ekf_output, "wing_sideforce_coefficient")
    df["L_over_D_ekf"] = df["CL_ekf"] / df["CD_ekf"].where(df["CD_ekf"] > 0.0)
    df["CL_wing_ekf"] = df["CL_ekf"]
    df["CD_wing_ekf"] = df["CD_ekf"]
    df["L_over_D_wing_ekf"] = df["L_over_D_ekf"]
    df["CD_kcu_ekf"] = numeric_series(ekf_output, "kcu_drag_coefficient", 0.0)
    df["CD_bridles_ekf"] = numeric_series(ekf_output, "bridles_drag_coefficient", 0.0)
    df["CD_tether_ekf"] = numeric_series(ekf_output, "tether_drag_coefficient", 0.0)
    df["CD_kcu_ekf"] = df["CD_kcu_ekf"].where(np.isfinite(df["CD_kcu_ekf"]), 0.0)
    df["CD_bridles_ekf"] = df["CD_bridles_ekf"].where(
        np.isfinite(df["CD_bridles_ekf"]), 0.0
    )
    df["CD_tether_ekf"] = df["CD_tether_ekf"].where(
        np.isfinite(df["CD_tether_ekf"]), 0.0
    )
    df["CL_kite_ekf"] = df["CL_wing_ekf"]
    df["CD_kite_ekf"] = df["CD_wing_ekf"] + df["CD_kcu_ekf"] + df["CD_bridles_ekf"]
    df["L_over_D_kite_ekf"] = df["CL_kite_ekf"] / df["CD_kite_ekf"].where(
        df["CD_kite_ekf"] > 0.0
    )
    df["kite_aoa_deg"] = maybe_radians_to_degrees(
        numeric_series(ekf_output, "kite_angle_of_attack")
    )
    df["wing_aoa_deg"] = maybe_radians_to_degrees(
        numeric_series(ekf_output, "wing_angle_of_attack")
    )
    df["wing_aoa_bridle_deg"] = maybe_radians_to_degrees(
        numeric_series(ekf_output, "wing_angle_of_attack_bridle")
    )
    df["elevation_deg"] = maybe_radians_to_degrees(
        numeric_series(flight_data, "kite_elevation").where(
            np.isfinite(numeric_series(flight_data, "kite_elevation")),
            numeric_series(ekf_output, "kite_elevation"),
        )
    )
    df["azimuth_deg"] = maybe_radians_to_degrees(
        numeric_series(flight_data, "kite_azimuth")
    )
    df["course_deg"] = maybe_radians_to_degrees(
        numeric_series(flight_data, "kite_course")
    )
    df["heading_deg"] = maybe_radians_to_degrees(
        numeric_series(flight_data, "kite_heading")
    )
    df["tether_elevation_deg"] = maybe_radians_to_degrees(
        numeric_series(ekf_output, "tether_elevation")
    )
    df["tether_azimuth_deg"] = maybe_radians_to_degrees(
        numeric_series(ekf_output, "tether_azimuth")
    )
    df["radius_turn_m"] = numeric_series(ekf_output, "radius_turn")
    df["omega_rad_s"] = numeric_series(ekf_output, "omega")
    df["yaw_rate_ekf_rad_s"] = maybe_degrees_per_second_to_radians(
        numeric_series(ekf_output, "yaw_rate")
    )
    if "kite_yaw_rate_1" in flight_data.columns:
        df["yaw_rate_rad_s"] = maybe_degrees_per_second_to_radians(
            numeric_series(flight_data, "kite_yaw_rate_1")
        )
        yaw_source = "flight_data.kite_yaw_rate_1"
    elif "kite_yaw_rate_0" in flight_data.columns:
        df["yaw_rate_rad_s"] = maybe_degrees_per_second_to_radians(
            numeric_series(flight_data, "kite_yaw_rate_0")
        )
        yaw_source = "flight_data.kite_yaw_rate_0"
    else:
        df["yaw_rate_rad_s"] = df["yaw_rate_ekf_rad_s"]
        yaw_source = "ekf_output.yaw_rate"
        warnings.append(f"{cfg.campaign_label}: using EKF yaw_rate for yaw filter.")
    df["yaw_rate_source"] = yaw_source
    df["yaw_rate_deg_s"] = np.rad2deg(df["yaw_rate_rad_s"])
    df["slack_m"] = numeric_series(ekf_output, "slack")

    velocity = df[["kite_velocity_x_ms", "kite_velocity_y_ms", "kite_velocity_z_ms"]]
    acceleration = pd.DataFrame(
        {
            "x": numeric_series(flight_data, "kite_acceleration_x"),
            "y": numeric_series(flight_data, "kite_acceleration_y"),
            "z": numeric_series(flight_data, "kite_acceleration_z"),
        },
        index=df.index,
    )
    speed = np.linalg.norm(velocity.to_numpy(dtype=float), axis=1)
    accel = acceleration.to_numpy(dtype=float)
    vel = velocity.to_numpy(dtype=float)
    dot_av = np.sum(accel * vel, axis=1)
    speed_sq = speed**2
    with np.errstate(invalid="ignore", divide="ignore"):
        parallel = vel * (dot_av / speed_sq)[:, None]
    lateral = np.linalg.norm(accel - parallel, axis=1)
    lateral[~np.isfinite(lateral)] = np.nan
    df["kite_speed_ms"] = speed
    df["lateral_accel_ms2"] = lateral
    with np.errstate(invalid="ignore", divide="ignore"):
        df["path_curvature_1pm"] = df["lateral_accel_ms2"] / (df["kite_speed_ms"] ** 2)

    df = add_force_preference(df, cfg.force_priority)
    for source in cfg.force_priority:
        column = force_sample_column(source)
        if column in df.columns and not finite_positive(df[column]).any():
            warnings.append(
                f"{cfg.campaign_label}: force source '{source}' has no positive "
                "finite samples and will not be selected."
            )

    return df.reset_index(drop=True)


def add_filters(
    df: pd.DataFrame,
    cfg: CampaignConfig,
    args: argparse.Namespace,
    min_window_samples: int,
) -> pd.DataFrame:
    df = df.copy()
    window_s = campaign_filter_value(cfg, args, "time_bin_s", args.window_s)
    max_abs_u_s = campaign_filter_value(cfg, args, "max_abs_u_s", args.max_abs_us)
    max_abs_yaw_rate_deg_s = campaign_filter_value(
        cfg, args, "max_abs_yaw_rate_deg_s", args.max_abs_yaw_rate_deg_s
    )
    max_path_curvature = campaign_filter_value(cfg, args, "max_path_curvature_1pm")
    u_dp_min = campaign_filter_value(cfg, args, "u_dp_min")
    u_dp_max = (
        args.max_u_dp_2019
        if cfg.year == 2019 and args.max_u_dp_2019 is not None
        else campaign_filter_value(cfg, args, "u_dp_max")
    )
    only_reel_out = bool(campaign_filter_value(cfg, args, "only_reel_out"))
    if window_s is None:
        window_s = DEFAULT_VALIDATION_FILTERS[cfg.year]["time_bin_s"]
    window_n = max(min_window_samples, int(math.ceil(window_s / 0.1)))
    min_periods = min_window_samples
    max_yaw_rad_s = (
        math.radians(max_abs_yaw_rate_deg_s)
        if max_abs_yaw_rate_deg_s is not None
        else None
    )
    max_curvature = 1.0 / args.min_abs_turn_radius_m
    if max_path_curvature is not None:
        max_curvature = max_path_curvature

    df["roll_n"] = (
        df["V_a_ms"].rolling(window_n, center=True, min_periods=1).count().astype(float)
    )
    df["roll_max_abs_u_s"] = (
        df["u_s_ch9"]
        .abs()
        .rolling(window_n, center=True, min_periods=min_periods)
        .max()
    )
    df["roll_max_abs_u_s_rate_s"] = (
        df["u_s_rate_s"]
        .abs()
        .rolling(window_n, center=True, min_periods=min_periods)
        .max()
    )
    df["roll_max_abs_yaw_rate_rad_s"] = (
        df["yaw_rate_rad_s"]
        .abs()
        .rolling(window_n, center=True, min_periods=min_periods)
        .max()
    )
    df["roll_q_abs_yaw_rate_rad_s"] = (
        df["yaw_rate_rad_s"]
        .abs()
        .rolling(window_n, center=True, min_periods=min_periods)
        .quantile(args.rolling_quantile)
    )
    df["roll_min_abs_turn_radius_m"] = (
        df["radius_turn_m"]
        .abs()
        .rolling(window_n, center=True, min_periods=min_periods)
        .min()
    )
    df["roll_median_abs_turn_radius_m"] = (
        df["radius_turn_m"]
        .abs()
        .rolling(window_n, center=True, min_periods=min_periods)
        .median()
    )
    df["roll_max_path_curvature_1pm"] = (
        df["path_curvature_1pm"]
        .abs()
        .rolling(window_n, center=True, min_periods=min_periods)
        .max()
    )
    df["roll_q_path_curvature_1pm"] = (
        df["path_curvature_1pm"]
        .abs()
        .rolling(window_n, center=True, min_periods=min_periods)
        .quantile(args.rolling_quantile)
    )
    df["roll_max_lateral_accel_ms2"] = (
        df["lateral_accel_ms2"]
        .abs()
        .rolling(window_n, center=True, min_periods=min_periods)
        .max()
    )
    df["roll_q_lateral_accel_ms2"] = (
        df["lateral_accel_ms2"]
        .abs()
        .rolling(window_n, center=True, min_periods=min_periods)
        .quantile(args.rolling_quantile)
    )
    df["roll_V_a_mean_ms"] = (
        df["V_a_ms"].rolling(window_n, center=True, min_periods=min_periods).mean()
    )
    df["roll_V_a_std_ms"] = (
        df["V_a_ms"].rolling(window_n, center=True, min_periods=min_periods).std()
    )
    df["roll_V_a_cv"] = rolling_cv(df["V_a_ms"], window_n, min_periods)
    df["roll_force_cv"] = rolling_cv(
        df["preferred_tether_force_N"], window_n, min_periods
    )
    df["roll_u_dp_std"] = (
        df["u_dp_ch9"].rolling(window_n, center=True, min_periods=min_periods).std()
    )
    df["roll_tether_length_std_m"] = (
        df["tether_length_m"]
        .rolling(window_n, center=True, min_periods=min_periods)
        .std()
    )
    df["roll_reel_speed_std_ms"] = (
        df["tether_reelout_speed_ms"]
        .rolling(window_n, center=True, min_periods=min_periods)
        .std()
    )

    df["turn_label_ok"] = df["turn_straight_label"].str.lower().eq("straight")
    df["steering_ok"] = (
        True if max_abs_u_s is None else df["u_s_ch9"].abs() <= max_abs_u_s
    )
    df["campaign_specific_straight_ok"] = True
    if u_dp_min is not None:
        df["campaign_specific_straight_ok"] &= df["u_dp_ch9"] > u_dp_min
    if u_dp_max is not None:
        df["campaign_specific_straight_ok"] &= df["u_dp_ch9"] < u_dp_max
    df["finite_physics_ok"] = (
        np.isfinite(df["V_a_ms"])
        & (df["V_a_ms"] >= args.min_va_ms)
        & (df["V_a_ms"] <= args.max_va_ms)
        & np.isfinite(df["tether_length_m"])
        & (df["tether_length_m"] >= args.min_tether_length_m)
        & (df["tether_length_m"] <= args.max_tether_length_m)
        & np.isfinite(df["preferred_tether_force_N"])
        & (df["preferred_tether_force_N"] >= args.min_force_n)
        & np.isfinite(df["CL_ekf"])
        & (df["CL_ekf"] > 0.0)
        & (df["CL_ekf"] < 3.0)
        & np.isfinite(df["CD_ekf"])
        & (df["CD_ekf"] > 0.0)
        & (df["CD_ekf"] < 2.0)
        & np.isfinite(df["u_dp_ch9"])
        & np.isfinite(df["slack_m"])
        & (df["slack_m"].abs() <= args.max_abs_slack_m)
    )
    df["window_coverage_ok"] = df["roll_n"] >= min_window_samples
    df["steering_rate_ok"] = df["roll_max_abs_u_s_rate_s"] <= args.max_steering_rate_s
    df["yaw_rate_ok"] = (
        True
        if max_yaw_rad_s is None
        else df["roll_q_abs_yaw_rate_rad_s"] <= max_yaw_rad_s
    )
    df["turn_radius_ok"] = (
        True
        if max_path_curvature is None
        else (
            (df["roll_median_abs_turn_radius_m"] >= args.min_abs_turn_radius_m)
            | (df["roll_q_path_curvature_1pm"] <= max_curvature)
        )
    )
    df["lateral_accel_ok"] = (
        df["roll_q_lateral_accel_ms2"] <= args.max_lateral_accel_ms2
    )
    df["apparent_wind_stationary_ok"] = df["roll_V_a_cv"] <= args.max_va_cv
    df["force_stationary_ok"] = df["roll_force_cv"] <= args.max_force_cv
    df["depower_stationary_ok"] = df["roll_u_dp_std"] <= args.max_depower_std
    df["tether_or_reel_stationary_ok"] = (
        df["roll_tether_length_std_m"] <= args.max_tether_length_std
    ) | (df["roll_reel_speed_std_ms"] <= args.max_reel_speed_std)

    flight_phase = df["flight_phase_label"].str.lower()
    phase_reel_out = flight_phase.isin(
        {"pp-ro", "ro", "reel-out", "reelout", "reel_out"}
    )
    phase_reel_in = flight_phase.isin({"pp-ri", "ri", "reel-in", "reelin", "reel_in"})
    phase_known = phase_reel_out | phase_reel_in

    powered = df["powered_label"].str.lower()
    powered_reel_out = powered.eq("powered")
    powered_reel_in = powered.eq("depowered")
    powered_known = powered_reel_out | powered_reel_in

    reel_speed_known = np.isfinite(df["tether_reelout_speed_ms"])
    reel_speed_out = df["tether_reelout_speed_ms"] > args.min_reelout_speed_ms
    df["reel_out_ok"] = pd.Series(not only_reel_out, index=df.index)
    if only_reel_out:
        df.loc[phase_known, "reel_out_ok"] = phase_reel_out.loc[phase_known]
        df.loc[~phase_known & powered_known, "reel_out_ok"] = powered_reel_out.loc[
            ~phase_known & powered_known
        ]
        df.loc[~phase_known & ~powered_known & reel_speed_known, "reel_out_ok"] = (
            reel_speed_out.loc[~phase_known & ~powered_known & reel_speed_known]
        )

    reason_order = [
        ("turn_label_ok", "turn_straight_not_straight"),
        ("reel_out_ok", "not_reel_out_phase"),
        ("steering_ok", "abs_u_s_above_threshold"),
        ("campaign_specific_straight_ok", "outside_campaign_specific_straight_limits"),
        ("finite_physics_ok", "nonfinite_or_implausible_physics"),
        ("window_coverage_ok", "insufficient_centered_window_coverage"),
        ("steering_rate_ok", "steering_not_stationary"),
        ("yaw_rate_ok", "yaw_rate_above_threshold"),
        ("turn_radius_ok", "turn_radius_or_curvature_above_threshold"),
        ("lateral_accel_ok", "lateral_accel_above_threshold"),
        ("apparent_wind_stationary_ok", "apparent_wind_not_stationary"),
        ("force_stationary_ok", "tether_force_not_stationary"),
        ("depower_stationary_ok", "depower_not_stationary"),
        ("tether_or_reel_stationary_ok", "tether_length_or_reel_speed_not_stationary"),
    ]
    df["hard_filters_ok"] = (
        df["turn_label_ok"]
        & df["reel_out_ok"]
        & df["steering_ok"]
        & df["campaign_specific_straight_ok"]
        & df["finite_physics_ok"]
    )
    df["straight_filter_pass"] = True
    for column, _ in reason_order:
        df["straight_filter_pass"] &= df[column].fillna(False).astype(bool)

    reasons = np.array([""] * len(df), dtype=object)
    for column, reason in reason_order:
        mask = (reasons == "") & (~df[column].fillna(False).astype(bool).to_numpy())
        reasons[mask] = reason
    reasons[df["straight_filter_pass"].to_numpy(dtype=bool)] = ""
    df["rejection_reason"] = reasons
    return df


def window_summary(
    sub: pd.DataFrame,
    cfg: CampaignConfig,
    case_id: str,
    t_start: float,
    t_end: float,
    pass_window: bool,
    rejection_reason: str,
) -> Dict[str, Any]:
    mean_cd = safe_mean(sub["CD_ekf"])
    mean_cl = safe_mean(sub["CL_ekf"])
    mean_cd_kite = safe_mean(sub["CD_kite_ekf"])
    mean_cl_kite = safe_mean(sub["CL_kite_ekf"])
    l_over_d = mean_cl / mean_cd if np.isfinite(mean_cd) and mean_cd > 0.0 else np.nan
    l_over_d_kite = (
        mean_cl_kite / mean_cd_kite
        if np.isfinite(mean_cd_kite) and mean_cd_kite > 0.0
        else np.nan
    )
    return {
        "case_id": case_id,
        "campaign": cfg.campaign_label,
        "year": cfg.year,
        "source_h5": str(cfg.h5_path),
        "t_start_s": float(t_start),
        "t_end_s": float(t_end),
        "n_samples": int(len(sub)),
        "phase_label": mode_label(sub["phase_label"]),
        "flight_phase_label": mode_label(sub["flight_phase_label"]),
        "powered_label": mode_label(sub["powered_label"]),
        "mean_u_s": safe_mean(sub["u_s_ch9"]),
        "std_u_s": safe_std(sub["u_s_ch9"]),
        "mean_u_dp_ch9": safe_mean(sub["u_dp_ch9"]),
        "std_u_dp_ch9": safe_std(sub["u_dp_ch9"]),
        "mean_depower_tape_length_m": safe_mean(sub["depower_tape_length_m"]),
        "mean_V_a_ms": safe_mean(sub["V_a_ms"]),
        "std_V_a_ms": safe_std(sub["V_a_ms"]),
        "mean_tether_length_m": safe_mean(sub["tether_length_m"]),
        "std_tether_length_m": safe_std(sub["tether_length_m"]),
        "mean_reel_speed_ms": safe_mean(sub["tether_reelout_speed_ms"]),
        "std_reel_speed_ms": safe_std(sub["tether_reelout_speed_ms"]),
        "mean_tether_force_kite_N": safe_mean(sub["tether_force_kite_N"]),
        "std_tether_force_kite_N": safe_std(sub["tether_force_kite_N"]),
        "mean_ground_tether_force_N": safe_mean(sub["ground_tether_force_N"]),
        "std_ground_tether_force_N": safe_std(sub["ground_tether_force_N"]),
        "mean_preferred_tether_force_N": safe_mean(sub["preferred_tether_force_N"]),
        "std_preferred_tether_force_N": safe_std(sub["preferred_tether_force_N"]),
        "force_source_preferred": mode_label(sub["force_source_preferred"], "none"),
        "mean_CL_ekf": mean_cl,
        "std_CL_ekf": safe_std(sub["CL_ekf"]),
        "mean_CD_ekf": mean_cd,
        "std_CD_ekf": safe_std(sub["CD_ekf"]),
        "mean_L_over_D_ekf": safe_mean(sub["L_over_D_ekf"]),
        "std_L_over_D_ekf": safe_std(sub["L_over_D_ekf"]),
        "mean_CD_kcu_ekf": safe_mean(sub["CD_kcu_ekf"]),
        "std_CD_kcu_ekf": safe_std(sub["CD_kcu_ekf"]),
        "mean_CD_bridles_ekf": safe_mean(sub["CD_bridles_ekf"]),
        "std_CD_bridles_ekf": safe_std(sub["CD_bridles_ekf"]),
        "mean_CL_kite_ekf": mean_cl_kite,
        "std_CL_kite_ekf": safe_std(sub["CL_kite_ekf"]),
        "mean_CD_kite_ekf": mean_cd_kite,
        "std_CD_kite_ekf": safe_std(sub["CD_kite_ekf"]),
        "mean_L_over_D_kite_ekf": safe_mean(sub["L_over_D_kite_ekf"]),
        "std_L_over_D_kite_ekf": safe_std(sub["L_over_D_kite_ekf"]),
        "mean_wind_speed_ms": safe_mean(sub["wind_speed_horizontal_ms"]),
        "mean_wind_direction_rad": circular_mean_rad(sub["wind_direction_rad"]),
        "mean_elevation_deg": safe_mean(sub["elevation_deg"]),
        "mean_azimuth_deg": circular_mean_deg(sub["azimuth_deg"]),
        "mean_course_deg": circular_mean_deg(sub["course_deg"]),
        "mean_aoa_deg": safe_mean(sub["wing_aoa_deg"]),
        "mean_slack_m": safe_mean(sub["slack_m"]),
        "mean_yaw_rate_deg_s": math.degrees(safe_mean(sub["yaw_rate_rad_s"])),
        "max_abs_yaw_rate_deg_s": math.degrees(
            safe_nanpercentile(sub["yaw_rate_rad_s"].abs(), 100.0)
        ),
        "mean_abs_turn_radius_m": safe_mean(sub["radius_turn_m"].abs()),
        "mean_lateral_accel_ms2": safe_mean(sub["lateral_accel_ms2"]),
        "straight_filter_pass": bool(pass_window),
        "rejection_reason": rejection_reason,
        "quality_score": (
            abs(safe_mean(sub["u_s_ch9"]))
            + safe_std(sub["u_s_ch9"])
            + safe_std(sub["V_a_ms"]) / max(abs(safe_mean(sub["V_a_ms"])), 1e-9)
            + safe_std(sub["preferred_tether_force_N"])
            / max(abs(safe_mean(sub["preferred_tether_force_N"])), 1e-9)
        ),
        "mean_CL_over_CD_from_window_means": l_over_d,
        "mean_CL_over_CD_kite_from_window_means": l_over_d_kite,
    }


def construct_windows(
    df: pd.DataFrame,
    cfg: CampaignConfig,
    args: argparse.Namespace,
    min_window_samples: int,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values("time_s").reset_index(drop=True)
    time = df["time_s"].to_numpy(dtype=float)
    t_min = float(np.nanmin(time))
    t_max = float(np.nanmax(time))
    window_s = campaign_filter_value(cfg, args, "time_bin_s", args.window_s)
    window_step_s = campaign_filter_value(
        cfg, args, "time_bin_step_s", args.window_step_s
    )
    if window_s is None:
        window_s = DEFAULT_VALIDATION_FILTERS[cfg.year]["time_bin_s"]
    if window_step_s is None:
        window_step_s = DEFAULT_VALIDATION_FILTERS[cfg.year]["time_bin_step_s"]
    starts = np.arange(t_min, t_max - window_s + 1e-9, window_step_s)
    rows: List[Dict[str, Any]] = []
    for i, start in enumerate(starts):
        end = float(start + window_s)
        lo = int(np.searchsorted(time, start, side="left"))
        hi = int(np.searchsorted(time, end, side="right"))
        sub = df.iloc[lo:hi]
        case_id = f"{cfg.campaign_label}_win_{i:06d}"
        if len(sub) < min_window_samples:
            rows.append(
                window_summary(
                    sub,
                    cfg,
                    case_id,
                    float(start),
                    end,
                    False,
                    "too_few_samples",
                )
            )
            continue
        pass_window = bool(sub["straight_filter_pass"].all())
        rejection_reason = ""
        if not pass_window:
            rejection_reason = mode_label(
                sub.loc[sub["rejection_reason"] != "", "rejection_reason"],
                "failed_sample_filter",
            )
        rows.append(
            window_summary(
                sub,
                cfg,
                case_id,
                float(start),
                end,
                pass_window,
                rejection_reason,
            )
        )
    return pd.DataFrame(rows)


def binned_summary(
    windows: pd.DataFrame,
    value_column: str,
    bin_width: float,
    bin_name: str,
) -> pd.DataFrame:
    accepted = windows[windows["straight_filter_pass"]].copy()
    accepted = accepted[np.isfinite(accepted[value_column])]
    if accepted.empty:
        return pd.DataFrame(
            columns=[
                f"{bin_name}_bin_center",
                f"{bin_name}_bin_lower",
                f"{bin_name}_bin_upper",
                "campaign",
                "year",
                "phase_label",
                "powered_label",
                "n_windows",
            ]
        )

    lo = math.floor(float(accepted[value_column].min()) / bin_width) * bin_width
    hi = math.ceil(float(accepted[value_column].max()) / bin_width) * bin_width
    if hi <= lo:
        hi = lo + bin_width
    edges = np.arange(lo, hi + bin_width * 1.0001, bin_width)
    accepted["_bin"] = pd.cut(
        accepted[value_column], bins=edges, include_lowest=True, right=False
    )

    rows: List[Dict[str, Any]] = []
    grouped = accepted.groupby(
        ["campaign", "year", "powered_label", "phase_label", "_bin"]
    )
    for (campaign, year, powered, phase, interval), group in grouped:
        if group.empty or pd.isna(interval):
            continue
        row: Dict[str, Any] = {
            "bin_variable": bin_name,
            "bin_center": float((interval.left + interval.right) / 2.0),
            "bin_lower": float(interval.left),
            "bin_upper": float(interval.right),
            f"{bin_name}_bin_center": float((interval.left + interval.right) / 2.0),
            f"{bin_name}_bin_lower": float(interval.left),
            f"{bin_name}_bin_upper": float(interval.right),
            "campaign": campaign,
            "year": int(year),
            "phase_label": phase,
            "powered_label": powered,
            "n_windows": int(len(group)),
            "mean_depower": safe_mean(group["mean_u_dp_ch9"]),
            "mean_tether_length": safe_mean(group["mean_tether_length_m"]),
            "mean_elevation": safe_mean(group["mean_elevation_deg"]),
        }
        metrics = {
            "tether_force_preferred_N": "mean_preferred_tether_force_N",
            "tether_force_kite_N": "mean_tether_force_kite_N",
            "CL_ekf": "mean_CL_ekf",
            "CD_ekf": "mean_CD_ekf",
            "L_over_D_ekf": "mean_L_over_D_ekf",
            "CD_kcu_ekf": "mean_CD_kcu_ekf",
            "CD_bridles_ekf": "mean_CD_bridles_ekf",
            "CL_kite_ekf": "mean_CL_kite_ekf",
            "CD_kite_ekf": "mean_CD_kite_ekf",
            "L_over_D_kite_ekf": "mean_L_over_D_kite_ekf",
        }
        for metric_name, column in metrics.items():
            row[f"{metric_name}_mean"] = safe_mean(group[column])
            for percentile in PERCENTILES:
                row[f"{metric_name}_p{int(percentile)}"] = safe_nanpercentile(
                    group[column], percentile
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["campaign", "powered_label", "phase_label", f"{bin_name}_bin_center"]
    )


def select_vwt_cases(windows: pd.DataFrame, binned_va: pd.DataFrame) -> pd.DataFrame:
    accepted = windows[windows["straight_filter_pass"]].copy()
    if accepted.empty or binned_va.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for _, bin_row in binned_va.iterrows():
        lower = float(bin_row["V_a_bin_lower"])
        upper = float(bin_row["V_a_bin_upper"])
        subset = accepted[
            (accepted["campaign"] == bin_row["campaign"])
            & (accepted["powered_label"] == bin_row["powered_label"])
            & (accepted["phase_label"] == bin_row["phase_label"])
            & (accepted["mean_V_a_ms"] >= lower)
            & (accepted["mean_V_a_ms"] < upper)
        ].copy()
        if subset.empty:
            continue
        center = float(bin_row["V_a_bin_center"])
        subset["_selection_score"] = (
            subset["mean_V_a_ms"] - center
        ).abs() + 0.25 * subset["quality_score"].fillna(0.0)
        selected = subset.sort_values("_selection_score").iloc[0]
        year = int(selected["year"])
        notes = f"nearest V_a bin center {center:.2f} m/s"
        if year == 2019:
            notes += "; 2019 depower conversion and bridle geometry are approximate"

        if year == 2019:
            depower_tape_length_m = 1.099
        elif year == 2025:
            depower_tape_length_m = 2.618
        else:
            print(f"error: year {year} not recognized")
            depower_tape_length_m = selected["mean_depower_tape_length_m"]

        rows.append(
            {
                "case_id": f"askite_vwt_{len(rows) + 1:03d}",
                "campaign": selected["campaign"],
                "year": year,
                "source_window_id": selected["case_id"],
                "validation_mode_requested": "apparent_wind_prescribed",
                "V_a_ms": selected["mean_V_a_ms"],
                "u_dp_ch9": selected["mean_u_dp_ch9"],
                # "depower_tape_length_m": selected["mean_depower_tape_length_m"],
                "depower_tape_length_m": depower_tape_length_m,
                "u_s_ch9": selected["mean_u_s"],
                "tether_length_m": selected["mean_tether_length_m"],
                "reel_speed_ms": selected["mean_reel_speed_ms"],
                "elevation_deg": selected["mean_elevation_deg"],
                "azimuth_deg": selected["mean_azimuth_deg"],
                "course_deg": selected["mean_course_deg"],
                "wind_speed_horizontal_ms": selected["mean_wind_speed_ms"],
                "wind_direction_rad": selected["mean_wind_direction_rad"],
                "ground_tether_force_N": selected["mean_ground_tether_force_N"],
                "tether_force_kite_N": selected["mean_tether_force_kite_N"],
                "CL_ekf": selected["mean_CL_ekf"],
                "CD_ekf": selected["mean_CD_ekf"],
                "L_over_D_ekf": selected["mean_L_over_D_ekf"],
                "CL_wing_ekf": selected["mean_CL_ekf"],
                "CD_wing_ekf": selected["mean_CD_ekf"],
                "L_over_D_wing_ekf": selected["mean_L_over_D_ekf"],
                "CD_kcu_ekf": selected["mean_CD_kcu_ekf"],
                "CD_bridles_ekf": selected["mean_CD_bridles_ekf"],
                "CL_kite_ekf": selected["mean_CL_kite_ekf"],
                "CD_kite_ekf": selected["mean_CD_kite_ekf"],
                "L_over_D_kite_ekf": selected["mean_L_over_D_kite_ekf"],
                "force_source_preferred": selected["force_source_preferred"],
                "case_weight_or_n_windows": int(bin_row["n_windows"]),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    binned_va: pd.DataFrame,
    binned_tether: pd.DataFrame,
    cases: pd.DataFrame,
    manifest: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Path]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "samples_csv": output_dir / "ch9_3_2_straight_samples.csv",
        "windows_csv": output_dir / "ch9_3_2_straight_windows.csv",
        "binned_by_va_csv": output_dir / "ch9_3_2_binned_by_va.csv",
        "binned_by_tether_length_csv": output_dir
        / "ch9_3_2_binned_by_tether_length.csv",
        "vwt_cases_csv": output_dir / "ch9_3_2_vwt_cases_for_askite.csv",
        "manifest_json": output_dir / "ch9_3_2_harvest_manifest.json",
    }
    samples.to_csv(outputs["samples_csv"], index=False)
    windows.to_csv(outputs["windows_csv"], index=False)
    binned_va.to_csv(outputs["binned_by_va_csv"], index=False)
    binned_tether.to_csv(outputs["binned_by_tether_length_csv"], index=False)
    cases.to_csv(outputs["vwt_cases_csv"], index=False)

    if args.write_parquet:
        try:
            samples.to_parquet(
                output_dir / "ch9_3_2_straight_samples.parquet", index=False
            )
            windows.to_parquet(
                output_dir / "ch9_3_2_straight_windows.parquet", index=False
            )
            binned_va.to_parquet(
                output_dir / "ch9_3_2_binned_by_va.parquet", index=False
            )
            binned_tether.to_parquet(
                output_dir / "ch9_3_2_binned_by_tether_length.parquet", index=False
            )
            cases.to_parquet(
                output_dir / "ch9_3_2_vwt_cases_for_askite.parquet", index=False
            )
        except Exception as exc:
            manifest.setdefault("warnings", []).append(
                f"Parquet output requested but failed: {exc}"
            )

    outputs["manifest_json"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    return outputs


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isfinite(value):
            return float(value)
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def manifest_for_run(
    configs: Sequence[CampaignConfig],
    args: argparse.Namespace,
    warnings: List[str],
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    binned_va: pd.DataFrame,
    binned_tether: pd.DataFrame,
    cases: pd.DataFrame,
) -> Dict[str, Any]:
    accepted_windows = windows[windows["straight_filter_pass"]]
    accepted_samples = samples[samples["straight_filter_pass"]]
    return {
        "created_time_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "git_commit": git_commit(),
        "source_h5": {
            cfg.campaign_label: file_metadata(cfg.h5_path) for cfg in configs
        },
        "campaign_config": {
            cfg.campaign_label: {
                "year": cfg.year,
                "time_range_optional": cfg.time_range,
                "depower_conversion": cfg.depower_conversion,
                "force_priority": list(cfg.force_priority),
            }
            for cfg in configs
        },
        "year_default_filters_enabled": args.use_year_default_filters,
        "year_default_filters": DEFAULT_VALIDATION_FILTERS,
        "effective_year_filters": {
            cfg.campaign_label: {
                "u_dp_min": campaign_filter_value(cfg, args, "u_dp_min"),
                "u_dp_max": (
                    args.max_u_dp_2019
                    if cfg.year == 2019 and args.max_u_dp_2019 is not None
                    else campaign_filter_value(cfg, args, "u_dp_max")
                ),
                "max_abs_u_s": campaign_filter_value(
                    cfg, args, "max_abs_u_s", args.max_abs_us
                ),
                "only_reel_out": campaign_filter_value(cfg, args, "only_reel_out"),
                "time_range_s": cfg.time_range,
                "time_bin_s": campaign_filter_value(
                    cfg, args, "time_bin_s", args.window_s
                ),
                "time_bin_step_s": campaign_filter_value(
                    cfg, args, "time_bin_step_s", args.window_step_s
                ),
                "max_path_curvature_1pm": campaign_filter_value(
                    cfg, args, "max_path_curvature_1pm"
                ),
                "max_abs_yaw_rate_deg_s": campaign_filter_value(
                    cfg, args, "max_abs_yaw_rate_deg_s", args.max_abs_yaw_rate_deg_s
                ),
            }
            for cfg in configs
        },
        "filter_thresholds": {
            "window_s": args.window_s,
            "window_step_s": args.window_step_s,
            "max_abs_us": args.max_abs_us,
            "max_abs_yaw_rate_deg_s": args.max_abs_yaw_rate_deg_s,
            "rolling_quantile": args.rolling_quantile,
            "max_steering_rate_s": args.max_steering_rate_s,
            "min_abs_turn_radius_m": args.min_abs_turn_radius_m,
            "max_lateral_accel_ms2": args.max_lateral_accel_ms2,
            "max_va_cv": args.max_va_cv,
            "max_force_cv": args.max_force_cv,
            "max_depower_std": args.max_depower_std,
            "max_reel_speed_std": args.max_reel_speed_std,
            "max_tether_length_std": args.max_tether_length_std,
            "min_reelout_speed_ms": args.min_reelout_speed_ms,
            "min_va_ms": args.min_va_ms,
            "max_va_ms": args.max_va_ms,
            "min_force_n": args.min_force_n,
            "min_tether_length_m": args.min_tether_length_m,
            "max_tether_length_m": args.max_tether_length_m,
            "max_abs_slack_m": args.max_abs_slack_m,
            "max_u_dp_2019": args.max_u_dp_2019,
        },
        "depower_conversion_rules": {
            "2025": (
                "Use documented 2025 convention. If kcu_actual_depower is stored "
                "as percent, u_dp_ch9 = kcu_actual_depower / 100. "
                "depower_tape_length_m = 0.2 + 5 u_dp_ch9."
            ),
            "2019": (
                "Retain u_p_2019_raw_or_normalized and use Section 9.1 best "
                "guess u_dp_ch9 = 0.2564 - 0.0768 u_p_2019. "
                "depower_tape_length_m = 0.2 + 5 u_dp_ch9."
            ),
        },
        "force_source_priority": {
            cfg.campaign_label: list(cfg.force_priority) for cfg in configs
        },
        "bin_definitions": {
            "V_a": {
                "width": args.bin_va_width,
                "source_column": "mean_V_a_ms",
                "percentiles": list(PERCENTILES),
            },
            "tether_length": {
                "width": args.bin_tether_width,
                "source_column": "mean_tether_length_m",
                "percentiles": list(PERCENTILES),
            },
        },
        "column_definitions": {
            "u_dp_ch9": "Dimensionless dissertation depower convention for Ch. 9.",
            "depower_tape_length_m": "0.2 + 5 u_dp_ch9.",
            "u_s_ch9": "Dimensionless steering, positive/negative signed, near zero for symmetric validation.",
            "CL_ekf/CD_ekf/L_over_D_ekf": (
                "EKF state/model estimates; not direct aerodynamic force-balance "
                "measurements. These are wing-level coefficients."
            ),
            "CL_kite_ekf/CD_kite_ekf/L_over_D_kite_ekf": (
                "Kite-level validation coefficients. CL_kite_ekf is the exported "
                "wing lift coefficient; CD_kite_ekf adds exported KCU and bridle "
                "parasitic drag coefficients to CD_ekf. Tether drag is exported "
                "separately as CD_tether_ekf and is not included."
            ),
            "CD_kcu_ekf/CD_bridles_ekf/CD_tether_ekf": (
                "Parasitic drag coefficients exported by EKF-AWE, nondimensionalized "
                "with kite reference area."
            ),
            "preferred_tether_force_N": (
                "First positive finite force source in the campaign force-source "
                "priority list."
            ),
        },
        "warnings": warnings,
        "counts": {
            "samples_total": int(len(samples)),
            "samples_retained": int(len(accepted_samples)),
            "windows_total": int(len(windows)),
            "windows_retained": int(len(accepted_windows)),
            "retained_windows_by_campaign": {
                str(key): int(value)
                for key, value in accepted_windows.groupby("campaign").size().items()
            },
            "binned_va_rows": int(len(binned_va)),
            "binned_tether_rows": int(len(binned_tether)),
            "askite_case_rows": int(len(cases)),
        },
        "acceptance_notes": {
            "coefficient_warning": (
                "EKF coefficient estimates are model states/outputs, not direct "
                "measurements."
            ),
            "expected_manual_qc": (
                "Run plot_straight_vwt_harvest_qc.py and inspect the retained "
                "intervals before using these cases for dissertation plots."
            ),
        },
    }


def print_summary(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    cases: pd.DataFrame,
    outputs: Dict[str, Path],
) -> None:
    retained_samples = samples[samples["straight_filter_pass"]]
    retained_windows = windows[windows["straight_filter_pass"]]
    print("Harvest complete")
    print(f"  retained samples: {len(retained_samples)} / {len(samples)}")
    print(f"  retained windows: {len(retained_windows)} / {len(windows)}")
    if not retained_windows.empty:
        print("  retained windows by campaign:")
        for campaign, count in retained_windows.groupby("campaign").size().items():
            print(f"    {campaign}: {count}")
    print(f"  ASKITE cases: {len(cases)}")
    print("  outputs:")
    for path in outputs.values():
        print(f"    {path}")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.rolling_quantile <= 1.0):
        raise ValueError("--rolling-quantile must be in (0, 1].")
    configs = build_campaign_configs(args)
    warnings: List[str] = []
    sample_frames: List[pd.DataFrame] = []
    window_frames: List[pd.DataFrame] = []

    for cfg in configs:
        if not cfg.h5_path.exists():
            raise FileNotFoundError(cfg.h5_path)
        df = build_working_dataframe(cfg, args, warnings)
        window_s = campaign_filter_value(cfg, args, "time_bin_s", args.window_s)
        if window_s is None:
            window_s = DEFAULT_VALIDATION_FILTERS[cfg.year]["time_bin_s"]
        min_samples = (
            args.min_window_samples
            if args.min_window_samples is not None
            else infer_min_window_samples(df, window_s)
        )
        df = add_filters(df, cfg, args, min_samples)
        windows = construct_windows(df, cfg, args, min_samples)
        sample_frames.append(df)
        window_frames.append(windows)

    samples = pd.concat(sample_frames, ignore_index=True)
    windows = pd.concat(window_frames, ignore_index=True)
    binned_va = binned_summary(windows, "mean_V_a_ms", args.bin_va_width, "V_a")
    binned_tether = binned_summary(
        windows, "mean_tether_length_m", args.bin_tether_width, "tether_length"
    )
    cases = select_vwt_cases(windows, binned_va)
    manifest = manifest_for_run(
        configs, args, warnings, samples, windows, binned_va, binned_tether, cases
    )
    outputs = write_outputs(
        samples, windows, binned_va, binned_tether, cases, manifest, args
    )
    print_summary(samples, windows, cases, outputs)


if __name__ == "__main__":
    main()
