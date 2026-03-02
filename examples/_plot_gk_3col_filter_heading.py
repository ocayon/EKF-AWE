"""
Two-column yaw-rate plot (2019 vs 2025), colored by heading bins.

Behavior:
- 1 row, 2 columns (2019 and 2025)
- Experimental data only
- Shared heading bins across both years
- One linear fit per panel
- Symmetric x-axis around zero (negative to positive)
- Filtering retained: time-window + powered-only + downsampling
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
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

    required = ["kcu_actual_steering", "kite_heading"]
    for col in required:
        if col not in flight_data.columns:
            raise ValueError(f"Missing column '{col}' in {cfg.title}")
    if "kite_apparent_windspeed" not in results.columns:
        raise ValueError(
            f"Missing column 'kite_apparent_windspeed' in results for {cfg.title}"
        )

    x = (
        -(flight_data["kcu_actual_steering"] / 100.0)
        * results["kite_apparent_windspeed"]
    )
    y = np.rad2deg(flight_data[yaw_rate_col])
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

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(heading)

    if cfg.apply_quadrant_filter:
        # Legacy incorrect-data filter used for 2025:
        # remove high-|yaw-rate| points whose sign conflicts with steering*va sign.
        mismatch_mask = ((y > cfg.y_exclude_threshold_deg) & (x < 0.0)) | (
            (y < -cfg.y_exclude_threshold_deg) & (x > 0.0)
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
        }
    )


def plot_heading_binned_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    heading_bin_centers_deg: np.ndarray,
    cmap,
) -> None:
    """Scatter data by heading bins and overlay one linear fit."""
    if df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Mirror left/right: keep vertical orientation (up/down), remove left-right direction.
    heading_orient = np.arctan2(
        np.sin(df["heading"].to_numpy()), np.abs(np.cos(df["heading"].to_numpy()))
    )
    heading_orient_deg = np.rad2deg(heading_orient)

    # Deterministic nearest-center assignment (tie-break: first center by argmin).
    bin_idx = np.argmin(
        np.abs(heading_orient_deg[:, None] - heading_bin_centers_deg[None, :]), axis=1
    )

    for i in range(len(heading_bin_centers_deg)):
        mask = bin_idx == i
        if np.any(mask):
            ax.scatter(
                df.loc[mask, "x"],
                df.loc[mask, "y"],
                s=40,
                alpha=0.55,
                color=cmap(i),
                marker=".",
                linewidths=0,
            )

    if len(df) > 1:
        slope, intercept, r, _, _ = linregress(df["x"], df["y"])
        x_line = np.linspace(df["x"].min(), df["x"].max(), 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="black", linestyle="--", linewidth=1.4)
        ax.text(
            0.03,
            0.97,
            rf"$g_{{\mathrm{{k}}}}={slope:.2f}$" + "\n" + rf"$R^2={r**2:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
        )

    ax.set_title(title)
    ax.set_xlabel(r"$u_\mathrm{s}v_\mathrm{a}$ ($\mathrm{m\,s^{-1}}$)")
    ax.grid(True, alpha=0.25)


def main() -> None:
    set_plot_style()

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

    heading_labels = [
        "90deg upwards\nmean=+90°, norm=[+75,+90]\nraw=[345,360] and [0,15]",
        "60deg upwards\nmean=+60°, norm=[+45,+75]\nraw=[315,345] and [15,45]",
        "30deg upwards\nmean=+30°, norm=[+15,+45]\nraw=[285,315] and [45,75]",
        "horizontal\nmean=0°, norm=[-15,+15]\nraw=[255,285] and [75,105]",
        "30deg downwards\nmean=-30°, norm=[-45,-15]\nraw=[225,255] and [105,135]",
        "60deg downwards\nmean=-60°, norm=[-75,-45]\nraw=[195,225] and [135,165]",
        "90deg downwards\nmean=-90°, norm=[-90,-75]\nraw=[165,195]",
    ]
    heading_labels = [
        "90deg upwards\nmean=+90°\nnorm=[+75,+90]",
        "60deg upwards\nmean=+60°\nnorm=[+45,+75]",
        "30deg upwards\nmean=+30°\nnorm=[+15,+45]",
        "horizontal\nmean=0°\nnorm=[-15,+15]",
        "30deg downwards\nmean=-30°\nnorm=[-45,-15]",
        "60deg downwards\nmean=-60°\nnorm=[-75,-45]",
        "90deg downwards\nmean=-90°\nnorm=[-90,-75]",
    ]
    heading_bin_centers_deg = np.array([90.0, 60.0, 30.0, 0.0, -30.0, -60.0, -90.0])

    n_heading_bins = len(heading_labels)
    cmap = plt.get_cmap("viridis", n_heading_bins)
    norm = BoundaryNorm(np.arange(-0.5, n_heading_bins + 0.5, 1), cmap.N)

    fig, axes = plt.subplots(
        1, 2, figsize=(9, 3.6), sharey=True, constrained_layout=True
    )

    for ax, cfg, df in zip(axes, datasets_cfg, datasets):
        plot_heading_binned_panel(ax, df, cfg.title, heading_bin_centers_deg, cmap)

    axes[0].set_ylabel(r"$\dot{\psi}$ ($^\circ\,\mathrm{s^{-1}}$)")
    axes[1].set_ylabel("")

    # Symmetric x-axis: enforce negative-to-positive on both panels.
    all_x = np.concatenate([df["x"].to_numpy() for df in datasets if not df.empty])
    x_abs = np.nanmax(np.abs(all_x)) if all_x.size > 0 else 1.0
    x_lim = max(1.0, 1.05 * x_abs)
    for ax in axes:
        ax.set_xlim(-x_lim, x_lim)

    y_lim = 110
    for ax in axes:
        ax.set_ylim(-y_lim, y_lim)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02)
    cbar.set_label("Heading")  # category (norm=left/right mirrored)")
    cbar.set_ticks(np.arange(n_heading_bins))
    cbar.set_ticklabels(heading_labels)
    # Ensure requested ordering appears from top to bottom.
    cbar.ax.invert_yaxis()
    cbar.ax.tick_params(labelsize=7)

    output_path = Path("./results/plots_paper") / "gk_3col_filter_heading.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
