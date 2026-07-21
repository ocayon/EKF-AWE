#!/usr/bin/env python3
"""Create QC plots for the Ch. 9.3.2 straight-flight EKF harvest."""

import argparse
import os
from pathlib import Path
from typing import List, Sequence, Tuple

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path("/tmp/ekf_awe_matplotlib_cache")
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent  # ch9/ch9_3_2_straight_vwt/
DEFAULT_INPUT_DIR = REPO_ROOT  # CSV harvest files live alongside this script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate QC plots for harvested straight-flight VWT cases."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <input-dir>/qc_plots",
    )
    parser.add_argument("--max-time-series-windows", type=int, default=3)
    return parser.parse_args()


def read_harvest(input_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples_path = input_dir / "ch9_3_2_straight_samples.csv"
    if not samples_path.exists():
        # only the gzipped copy is tracked; it is too large for GitHub uncompressed
        samples_path = samples_path.with_name(samples_path.name + ".gz")
    windows_path = input_dir / "ch9_3_2_straight_windows.csv"
    cases_path = input_dir / "ch9_3_2_vwt_cases_for_askite.csv"
    missing = [
        path for path in (samples_path, windows_path, cases_path) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing harvest outputs: " + ", ".join(str(path) for path in missing)
        )
    return (
        pd.read_csv(samples_path),
        pd.read_csv(windows_path),
        pd.read_csv(cases_path),
    )


def finite_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def campaigns(df: pd.DataFrame) -> List[str]:
    return sorted(str(value) for value in df["campaign"].dropna().unique())


def pass_mask(df: pd.DataFrame) -> pd.Series:
    series = df["straight_filter_pass"]
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def savefig(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def scatter_by_campaign(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    alpha: float,
    size: float,
    label_prefix: str = "",
) -> None:
    colors = {
        "2019": "tab:blue",
        "2025": "tab:orange",
    }
    for campaign in campaigns(df):
        sub = df[df["campaign"].astype(str) == campaign]
        ax.scatter(
            pd.to_numeric(sub[x_col], errors="coerce"),
            pd.to_numeric(sub[y_col], errors="coerce"),
            s=size,
            alpha=alpha,
            label=f"{label_prefix}{campaign}",
            color=colors.get(campaign, None),
            edgecolors="none",
        )


def plot_retained_rejected_path(samples: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(
        1, len(campaigns(samples)), figsize=(6 * len(campaigns(samples)), 5)
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, campaign in zip(axes, campaigns(samples)):
        sub = samples[samples["campaign"].astype(str) == campaign]
        retained_flag = pass_mask(sub)
        rejected = sub[~retained_flag]
        retained = sub[retained_flag]
        ax.scatter(
            rejected["kite_position_y_m"],
            rejected["kite_position_z_m"],
            s=2,
            color="0.75",
            alpha=0.35,
            label="rejected",
            edgecolors="none",
        )
        ax.scatter(
            retained["kite_position_y_m"],
            retained["kite_position_z_m"],
            s=4,
            color="tab:green",
            alpha=0.85,
            label="retained",
            edgecolors="none",
        )
        ax.set_title(f"{campaign} retained vs rejected")
        ax.set_xlabel("kite y [m]")
        ax.set_ylabel("kite z [m]")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
    return savefig(fig, output_dir, "qc_retained_rejected_flight_path.png")


def plot_selected_time_series(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    output_dir: Path,
    max_windows: int,
) -> List[Path]:
    paths: List[Path] = []
    metrics = [
        ("u_s_ch9", "u_s [-]"),
        ("yaw_rate_deg_s", "yaw rate [deg/s]"),
        ("V_a_ms", "V_a [m/s]"),
        ("preferred_tether_force_N", "force [N]"),
        ("u_dp_ch9", "u_dp [-]"),
    ]
    samples = samples.copy()
    samples["yaw_rate_deg_s"] = np.rad2deg(
        pd.to_numeric(samples["yaw_rate_rad_s"], errors="coerce")
    )
    retained_windows = windows[pass_mask(windows)].copy()
    retained_windows = retained_windows.sort_values(["campaign", "quality_score"])
    for campaign in campaigns(samples):
        selected = retained_windows[
            retained_windows["campaign"].astype(str) == campaign
        ].head(max_windows)
        if selected.empty:
            continue
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 9), sharex=True)
        sub_all = samples[samples["campaign"].astype(str) == campaign]
        t_min = float(selected["t_start_s"].min()) - 2.0
        t_max = float(selected["t_end_s"].max()) + 2.0
        sub = sub_all[(sub_all["time_s"] >= t_min) & (sub_all["time_s"] <= t_max)]
        retained = sub[pass_mask(sub)]
        for ax, (column, ylabel) in zip(axes, metrics):
            ax.plot(sub["time_s"], sub[column], color="0.65", linewidth=0.8)
            ax.scatter(
                retained["time_s"],
                retained[column],
                s=8,
                color="tab:green",
                alpha=0.75,
                edgecolors="none",
            )
            for _, window in selected.iterrows():
                ax.axvspan(
                    window["t_start_s"],
                    window["t_end_s"],
                    color="tab:green",
                    alpha=0.12,
                    linewidth=0,
                )
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
        axes[0].set_title(f"{campaign} selected retained intervals")
        axes[-1].set_xlabel("time [s]")
        paths.append(
            savefig(fig, output_dir, f"qc_selected_time_series_{campaign}.png")
        )
    return paths


def plot_retained_histograms(samples: pd.DataFrame, output_dir: Path) -> Path:
    retained = samples[pass_mask(samples)]
    metrics = [
        ("V_a_ms", "V_a [m/s]"),
        ("u_dp_ch9", "u_dp [-]"),
        ("tether_length_m", "tether length [m]"),
        ("preferred_tether_force_N", "preferred force [N]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (column, xlabel) in zip(axes.flat, metrics):
        for campaign in campaigns(retained):
            values = finite_values(
                retained[retained["campaign"].astype(str) == campaign][column]
            )
            if values.size:
                ax.hist(values, bins=30, histtype="step", linewidth=1.5, label=campaign)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("retained samples")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    return savefig(fig, output_dir, "qc_retained_histograms.png")


def plot_force_va_before_after(samples: pd.DataFrame, output_dir: Path) -> Path:
    retained = samples[pass_mask(samples)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    scatter_by_campaign(
        axes[0],
        samples,
        "V_a_ms",
        "preferred_tether_force_N",
        alpha=0.18,
        size=4,
    )
    axes[0].set_title("before filtering")
    scatter_by_campaign(
        axes[1],
        retained,
        "V_a_ms",
        "preferred_tether_force_N",
        alpha=0.55,
        size=8,
    )
    axes[1].set_title("after filtering")
    for ax in axes:
        ax.set_xlabel("V_a [m/s]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[0].set_ylabel("preferred tether force [N]")
    return savefig(fig, output_dir, "qc_force_vs_va_before_after.png")


def plot_coefficients_vs_va(samples: pd.DataFrame, output_dir: Path) -> Path:
    retained = samples[pass_mask(samples)]
    metrics = [
        ("CL_ekf", "C_L EKF estimate"),
        ("CD_ekf", "C_D EKF estimate"),
        ("L_over_D_ekf", "C_L/C_D EKF estimate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    for ax, (column, ylabel) in zip(axes, metrics):
        scatter_by_campaign(
            ax,
            retained,
            "V_a_ms",
            column,
            alpha=0.45,
            size=7,
        )
        ax.set_xlabel("V_a [m/s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    return savefig(fig, output_dir, "qc_coefficients_vs_va_after_filter.png")


def plot_glide_ratio_vs_tether(samples: pd.DataFrame, output_dir: Path) -> Path:
    retained = samples[pass_mask(samples)]
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter_by_campaign(
        ax,
        retained,
        "tether_length_m",
        "L_over_D_ekf",
        alpha=0.45,
        size=8,
    )
    ax.set_xlabel("tether length [m]")
    ax.set_ylabel("C_L/C_D EKF estimate")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    return savefig(fig, output_dir, "qc_glide_ratio_vs_tether_length.png")


def print_summary(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    cases: pd.DataFrame,
    paths: Sequence[Path],
) -> None:
    retained_samples = samples[pass_mask(samples)]
    retained_windows = windows[pass_mask(windows)]
    print("QC plots complete")
    print(f"  retained samples: {len(retained_samples)} / {len(samples)}")
    print(f"  retained windows: {len(retained_windows)} / {len(windows)}")
    print(f"  ASKITE cases: {len(cases)}")
    print("  generated plots:")
    for path in paths:
        print(f"    {path}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or (REPO_ROOT / "results")
    samples, windows, cases = read_harvest(input_dir)

    paths: List[Path] = []
    paths.append(plot_retained_rejected_path(samples, output_dir))
    paths.extend(
        plot_selected_time_series(
            samples, windows, output_dir, args.max_time_series_windows
        )
    )
    paths.append(plot_retained_histograms(samples, output_dir))
    paths.append(plot_force_va_before_after(samples, output_dir))
    paths.append(plot_coefficients_vs_va(samples, output_dir))
    paths.append(plot_glide_ratio_vs_tether(samples, output_dir))
    print_summary(samples, windows, cases, paths)


if __name__ == "__main__":
    main()
