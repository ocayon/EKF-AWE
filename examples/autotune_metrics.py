"""Self-contained scoring for the EKF auto-tuner (no AWETrim dependency).

Two families of metrics on a results h5:

* ``lidar_metrics``   -- agreement with a profiling lidar when the flight
  data carries ``<h>m_Wind_Speed_m_s``-style columns (EKF vs lidar at kite
  height, 60-s blocks).
* ``internal_metrics`` -- lidar-free consistency: loop-leak correlations of
  each estimated channel with the steering input (the figure-eight is
  steering-locked, real weather is not), high-passed wind energy, a
  turbulence-intensity proxy, NIS, split-half stability of the constant
  states, CD dip rates, and the wind-va loop correlation (``ro_corr``) used
  for blind pitot calibration.

Channels are scored on per-reel-out-segment linearly detrended series, so
slow real weather does not count as leak and the reel-in retraction
artifact does not contaminate the loop band.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PHASE_REELOUT = 1
MIN_SEG_SAMPLES = 300  # ~30 s at 10 Hz: at least one figure-eight


# ---------------------------------------------------------------- helpers
def load_run(path):
    from awes_ekf.load_data.read_data import read_results_from_hdf5

    ekf, fd, cfg = read_results_from_hdf5(path)
    n = min(len(ekf), len(fd))
    ekf = ekf.iloc[:n].reset_index(drop=True)
    fd = fd.iloc[:n].reset_index(drop=True)
    for c in ekf.columns:
        if ekf[c].dtype == object:
            ekf[c] = pd.to_numeric(ekf[c], errors="coerce")
    return ekf, fd, cfg


def _segments(mask, min_len=MIN_SEG_SAMPLES):
    d = np.diff(mask.astype(int))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if len(mask) and mask[0]:
        starts = [0] + starts
    if len(mask) and mask[-1]:
        ends = ends + [len(mask)]
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]


def _detrend_pool(t, series_list, segs):
    """Linear-detrend each series per segment; return pooled arrays."""
    pools = [[] for _ in series_list]
    for s, e in segs:
        tt = t[s:e]
        for out, x in zip(pools, series_list):
            xx = x[s:e]
            good = np.isfinite(xx)
            if good.sum() < MIN_SEG_SAMPLES // 3:
                out.append(np.full(e - s, np.nan))
                continue
            c = np.polyfit(tt[good], xx[good], 1)
            out.append(np.where(good, xx - np.polyval(c, tt), np.nan))
    return [np.concatenate(p) if p else np.array([]) for p in pools]


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 200 or np.nanstd(a[m]) == 0 or np.nanstd(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _split_half_drift(series):
    """|mean(2nd quarter) - mean(4th quarter)| / |final|, in percent."""
    s = pd.to_numeric(pd.Series(series), errors="coerce")
    n = len(s)
    if n < 100 or not np.isfinite(s.iloc[-1]) or abs(s.iloc[-1]) < 1e-12:
        return np.nan
    a = s.iloc[n // 4 : n // 2].mean()
    b = s.iloc[3 * n // 4 :].mean()
    return float(abs(a - b) / abs(s.iloc[-1]) * 100.0)


# ---------------------------------------------------------------- lidar
LIDAR_RE = re.compile(r"^(\d+)m_Wind_Speed_m_s$")


def lidar_heights(fd):
    return sorted(
        int(m.group(1)) for c in fd.columns for m in [LIDAR_RE.match(c)] if m
    )


def _interp_profile(z, heights, prof):
    hs = np.asarray(heights, float)
    out = np.full(len(z), np.nan)
    inside = (z >= hs[0]) & (z <= hs[-1])
    idx = np.clip(np.searchsorted(hs, z) - 1, 0, len(hs) - 2)
    w = (z - hs[idx]) / (hs[idx + 1] - hs[idx])
    vals = prof[np.arange(len(z)), idx] * (1 - w) + prof[
        np.arange(len(z)), idx + 1
    ] * w
    out[inside] = vals[inside]
    return out


def _block(t, x, s=60.0):
    b = ((t - t[0]) // s).astype(int)
    out = []
    for k in np.unique(b):
        m = (b == k) & np.isfinite(x)
        out.append(np.mean(x[m]) if m.sum() > 10 else np.nan)
    return np.array(out)


def lidar_metrics(ekf, fd):
    """EKF vs lidar at kite height, 60-s blocks. None if no lidar columns."""
    heights = lidar_heights(fd)
    if not heights:
        return None
    t = ekf["time"].to_numpy(float)
    z = ekf["kite_position_z"].to_numpy(float)
    speed = np.column_stack(
        [fd[f"{h}m_Wind_Speed_m_s"].to_numpy(float) for h in heights]
    )
    lu = _interp_profile(z, heights, speed)
    dirs = np.column_stack(
        [
            np.deg2rad(270.0 - fd[f"{h}m_Wind_Direction_deg"].to_numpy(float))
            for h in heights
        ]
    )
    ld = np.arctan2(
        _interp_profile(z, heights, np.sin(dirs)),
        _interp_profile(z, heights, np.cos(dirs)),
    )
    eu = ekf["wind_speed_horizontal"].to_numpy(float)
    ed = ekf["wind_direction"].to_numpy(float)

    ub, lb = _block(t, eu), _block(t, lu)
    m = np.isfinite(ub) & np.isfinite(lb)
    if m.sum() < 5:
        return None
    d = ub[m] - lb[m]
    dc = _block(t, np.cos(ed)) + 1j * _block(t, np.sin(ed))
    lc = _block(t, np.where(np.isfinite(lu), np.cos(ld), np.nan)) + 1j * _block(
        t, np.where(np.isfinite(lu), np.sin(ld), np.nan)
    )
    md = np.isfinite(dc) & np.isfinite(lc)
    ddir = np.angle(dc[md] / lc[md])
    return {
        "speed_bias": float(np.mean(d)),
        "speed_rms": float(np.sqrt(np.mean(d**2))),
        "speed_corr": float(np.corrcoef(ub[m], lb[m])[0, 1]),
        "dir_rms_deg": float(np.rad2deg(np.sqrt(np.mean(ddir**2)))),
        "n_blocks": int(m.sum()),
    }


# ---------------------------------------------------------------- internal
CONSTANT_STATES = [
    "k_phi_us",
    "k_cl_us",
    "k_cd_us",
    "k_cl_us_odd",
    "k_cl_up",
    "k_cd_up",
    "k_cd_cl2",
]


def internal_metrics(ekf, fd, cfg):
    t = ekf["time"].to_numpy(float)
    fpi = fd["flight_phase_index"].to_numpy(float)
    ro = fpi == PHASE_REELOUT
    segs = _segments(ro)

    us = fd["kcu_actual_steering"].to_numpy(float)
    us = us / max(np.nanmax(np.abs(us)), 1e-9)
    eu = ekf["wind_speed_horizontal"].to_numpy(float)
    ed = np.unwrap(ekf["wind_direction"].to_numpy(float))
    wz = ekf["wind_speed_vertical"].to_numpy(float)
    va = ekf["kite_apparent_windspeed"].to_numpy(float)

    def resid(plain, residual):
        col = residual if residual in ekf.columns else plain
        return ekf[col].to_numpy(float)

    cs = resid("wing_sideforce_coefficient", "wing_sideforce_coefficient_residual")
    cl = resid("wing_lift_coefficient", "wing_lift_coefficient_residual")
    cd = resid("wing_drag_coefficient", "wing_parasitic_drag_coefficient")

    (us_d, aus_d, eu_d, ed_d, wz_d, va_d, cs_d, cl_d, cd_d) = _detrend_pool(
        t, [us, np.abs(us), eu, ed, wz, va, cs, cl, cd], segs
    )

    out = {
        # loop-leak scores: |corr| of each channel with the steering input
        # (odd channels) or |steering| (even channels), reel-out detrended
        "leak_cs_us": abs(_corr(cs_d, us_d) or np.nan),
        "leak_cl_aus": abs(_corr(cl_d, aus_d) or np.nan),
        "leak_cd_aus": abs(_corr(cd_d, aus_d) or np.nan),
        "leak_dir_us": abs(_corr(ed_d, us_d) or np.nan),
        "leak_speed_aus": abs(_corr(eu_d, aus_d) or np.nan),
        "leak_wz_us": abs(_corr(wz_d, us_d) or np.nan),
        # blind pitot criterion: signed wind-va loop correlation
        "ro_corr": _corr(eu_d, va_d),
        # energy / plausibility
        "wind_hp_std": float(np.nanstd(eu_d)) if eu_d.size else np.nan,
        "ti_proxy": float(np.nanstd(eu_d) / max(np.nanmean(eu), 1e-9))
        if eu_d.size
        else np.nan,
        "nis_median": float(pd.to_numeric(ekf.get("nis"), errors="coerce").median()),
        "mean_wind": float(np.nanmean(eu)),
    }

    # CD health during reel-out (full wing CD, not the residual)
    cd_full = ekf["wing_drag_coefficient"].to_numpy(float)[ro]
    cd_full = cd_full[np.isfinite(cd_full)]
    if cd_full.size:
        med = np.median(cd_full)
        out["cd_median"] = float(med)
        out["cd_dip_frac"] = float(np.mean(cd_full < 0.35 * med))
        out["cd_neg_frac"] = float(np.mean(cd_full < 0.0))

    # split-half stability of every active constant state
    for c in CONSTANT_STATES:
        if c in ekf.columns:
            s = pd.to_numeric(ekf[c], errors="coerce")
            if s.notna().any() and 1e-9 < abs(s.iloc[-1]) < 50:
                out[f"drift_{c}"] = _split_half_drift(s)
                out[f"value_{c}"] = float(s.iloc[-1])
    return out


def score_run(path):
    ekf, fd, cfg = load_run(path)
    out = {"h5": str(Path(path))}
    out["internal"] = internal_metrics(ekf, fd, cfg)
    out["lidar"] = lidar_metrics(ekf, fd)
    return out
