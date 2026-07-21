"""
Shared flight-data filter settings for ch9 plot scripts.

All time windows and per-script filter dicts live here so that changing a
window in one place propagates to every script that uses it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Canonical time windows
# ---------------------------------------------------------------------------

# Tight window covering the lemniscate / circular turns analysed in ch9
TIME_RANGE_2019_CIRCLES: tuple[float, float] = (2190.0, 2255.0)
TIME_RANGE_2025_CIRCLES: tuple[float, float] = (700.0, 800.0)

# Broad window covering the full usable flight for statistics
TIME_RANGE_2019_ALL: tuple[float, float] = (1800.0, 9986.2)
TIME_RANGE_2025_ALL: tuple[float, float] = (400.0, 1000.0)

# ---------------------------------------------------------------------------
# plot_statistics.py  —  load_and_process_data kwargs
# ---------------------------------------------------------------------------

STATS_2019 = dict(time_range=TIME_RANGE_2019_ALL, downsample_frac=0.1)
STATS_2025 = dict(time_range=TIME_RANGE_2025_ALL, downsample_frac=0.5)

# ---------------------------------------------------------------------------
# plot_fitted_turn_rate_2019_and_2025.py  —  PLOT_FILTER / FIT_FILTER dicts
#
# Keys:
#   time_range               – tuple[start_s, end_s] or None (no restriction)
#   powered_only             – keep only powered-flight samples when available
#   downsample_frac          – random sample fraction in (0, 1]
#   apply_quadrant_filter    – legacy sign-consistency filter
#   y_exclude_threshold_deg  – threshold used by the quadrant filter
#   us_min / us_max          – steering filter on u_s (signed)
#   udp_min / udp_max        – depower filter on u_dp (normalized by 100)
# ---------------------------------------------------------------------------

PLOT_FILTER_DEFAULT: dict = {
    "time_range": None,
    "powered_only": True,
    "downsample_frac": 1.0,
    "apply_quadrant_filter": False,
    "y_exclude_threshold_deg": 90.0,
    "us_min": None,
    "us_max": None,
    "udp_min": None,
    "udp_max": None,
}

FIT_FILTER_DEFAULT: dict = {
    "time_range": None,
    "powered_only": True,
    "downsample_frac": 1.0,
    "apply_quadrant_filter": False,
    "y_exclude_threshold_deg": 90.0,
    "us_min": None,
    "us_max": None,
    "udp_min": None,
    "udp_max": None,
}

PLOT_FILTER_2019: dict = {
    **PLOT_FILTER_DEFAULT,
    "time_range": TIME_RANGE_2019_ALL,
    "us_min": None,
    "us_max": None,
    # "downsample_frac": 0.5, #TODO: downsampling does not work
}
PLOT_FILTER_2025: dict = {
    **PLOT_FILTER_DEFAULT,
    "time_range": TIME_RANGE_2025_ALL,
    "apply_quadrant_filter": False,
    "y_exclude_threshold_deg": 10.0,
    "us_min": None,
    "us_max": None,
}

FIT_FILTER_2019: dict = {
    **FIT_FILTER_DEFAULT,
    "time_range": TIME_RANGE_2019_ALL,
}

FIT_FILTER_2025: dict = {
    **FIT_FILTER_DEFAULT,
    "time_range": TIME_RANGE_2025_ALL,
    "apply_quadrant_filter": False,
    "y_exclude_threshold_deg": 10.0,
}


def save_plot_data_to_csv(
    df: pd.DataFrame,
    year: str,
    gk: float,
    r2: float,
    output_dir: Path,
    addendum: str = "",
) -> Path:
    """Save plotted data points to CSV with a filename encoding year, g_k and R².

    Filename pattern: ``{year}_all_gk_{gk_tag}_R_{r2_tag}{addendum}.csv``
    where ``gk_tag`` and ``r2_tag`` are the values multiplied by 100 and
    zero-padded to three digits (e.g. g_k=0.164 → "016", R²=0.850 → "085").
    """
    gk_tag = f"{round(abs(gk) * 100):03d}"
    r2_tag = f"{round(r2 * 100):03d}" if r2 == r2 else "nan"  # guard NaN
    filename = f"{year}_all_gk_{gk_tag}_R_{r2_tag}{addendum}.csv"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    df.to_csv(out_path, index=False)
    print(f"Saved plot data ({len(df)} rows) → {out_path}")
    return out_path
