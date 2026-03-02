"""
Create a 2x2 comparison figure:
- Row 1: identical Y-Z panels from _plot_yz_plane_lemniscate.py
- Row 2: identical g_k-heading panels from _plot_gk_3col_filter_heading.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Normalize

from awes_ekf.plotting.color_palette import set_plot_style

from _plot_gk_3col_filter_heading import (
    FlightConfig,
    load_filtered_dataset,
    plot_heading_binned_panel,
)
from _plot_yz_plane_lemniscate import (
    load_and_process_data,
    scatter_yz,
    set_shared_limits,
)


def main() -> None:
    set_plot_style()

    repo_root = Path(__file__).resolve().parents[1]

    # ---------- Row 1: Y-Z plane (same setup as _plot_yz_plane_lemniscate.py) ----------
    df_2019_yz = load_and_process_data(
        year="2019",
        month="10",
        day="08",
        kite_model="v3",
        addition="_t26",
        time_range=(2190, 2255),
        downsample_frac=1.0,
    )
    df_2025_yz = load_and_process_data(
        year="2025",
        month="10",
        day="09",
        kite_model="v3",
        addition="",
        time_range=(700, 800),
        downsample_frac=1.0,
    )

    # ---------- Row 2: g_k vs us*va by heading (same setup as _plot_gk_3col_filter_heading.py) ----------
    gk_cfgs = [
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
    gk_data = [load_filtered_dataset(cfg) for cfg in gk_cfgs]
    heading_labels = [
        "$90^\\circ$\nupwards",
        "$60^\\circ$\nupwards",
        "$30^\\circ$\nupwards",
        "$0^\\circ$\nhorizontal",
        "$30^\\circ$\ndownwards",
        "$60^\\circ$\ndownwards",
        "$90^\\circ$\ndownwards",
    ]
    heading_bin_centers_deg = np.array([90.0, 60.0, 30.0, 0.0, -30.0, -60.0, -90.0])
    n_heading_bins = len(heading_labels)
    cmap_heading = plt.get_cmap("viridis", n_heading_bins)
    norm_heading = BoundaryNorm(
        np.arange(-0.5, n_heading_bins + 0.5, 1), cmap_heading.N
    )

    # ---------- Layout: 2 rows x (2 panels + row colorbar) ----------
    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    # Reduce inter-row spacing while keeping constrained layout active.
    fig.set_constrained_layout_pads(h_pad=0.01, w_pad=0.02, hspace=0.01, wspace=0.02)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.08], wspace=0.02)

    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[0, 1], sharey=ax11)
    cax1 = fig.add_subplot(gs[0, 2])

    ax21 = fig.add_subplot(gs[1, 0])
    ax22 = fig.add_subplot(gs[1, 1], sharey=ax21)
    cax2 = fig.add_subplot(gs[1, 2])

    # ---------- Draw row 1 ----------
    speed_min = min(df_2019_yz["kite_speed"].min(), df_2025_yz["kite_speed"].min())
    speed_max = max(df_2019_yz["kite_speed"].max(), df_2025_yz["kite_speed"].max())
    norm_speed = Normalize(vmin=speed_min, vmax=speed_max)

    marker_size = 15
    scatter_yz(
        ax11,
        df_2019_yz,
        label="2019",
        marker="o",
        marker_size=marker_size,
        norm=norm_speed,
        alpha=0.7,
    )
    sc_speed = scatter_yz(
        ax12,
        df_2025_yz,
        label="2025",
        marker="o",
        marker_size=marker_size,
        norm=norm_speed,
        alpha=0.6,
    )
    ax11.set_title("2019-10-08")
    ax12.set_title("2025-10-09")
    ax11.set_ylabel(r"$z_\mathrm{W}$ (m)")
    ax11.set_xlabel(r"$y_\mathrm{W,\perp}$ (m)")
    ax12.set_xlabel(r"$y_\mathrm{W,\perp}$ (m)")
    ax12.set_ylabel("")
    ax12.tick_params(axis="y", which="both", left=False, labelleft=False)

    set_shared_limits(ax11, df_2019_yz, df_2025_yz, y_column="kite_position_y_wind")
    set_shared_limits(ax12, df_2019_yz, df_2025_yz, y_column="kite_position_y_wind")

    # Use a clean scalar mappable so colorbar is fully opaque (independent of point alpha).
    sm_speed = ScalarMappable(norm=norm_speed, cmap="viridis")
    sm_speed.set_array([])
    cbar1 = fig.colorbar(sm_speed, cax=cax1)
    cbar1.ax.set_title(r"$v_\mathrm{k}$ (ms$^{-1}$)", pad=6)

    # ---------- Draw row 2 ----------
    for ax, cfg, df in zip([ax21, ax22], gk_cfgs, gk_data):
        plot_heading_binned_panel(ax, df, "", heading_bin_centers_deg, cmap_heading)

    ax21.set_ylabel(r"$\dot{\psi}$ ($^\circ\,\mathrm{s^{-1}}$)")
    ax22.set_ylabel("")
    ax22.tick_params(axis="y", which="both", left=False, labelleft=False)

    all_x = np.concatenate([df["x"].to_numpy() for df in gk_data if not df.empty])
    x_abs = np.nanmax(np.abs(all_x)) if all_x.size > 0 else 1.0
    x_lim = max(1.0, 1.05 * x_abs)
    for ax in [ax21, ax22]:
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-110, 110)

    sm = ScalarMappable(norm=norm_heading, cmap=cmap_heading)
    sm.set_array([])
    cbar2 = fig.colorbar(sm, cax=cax2)
    cbar2.ax.set_title(r"Flight direction ($^\circ$)", pad=6)
    cbar2.set_ticks(np.arange(n_heading_bins))
    cbar2.set_ticklabels(heading_labels)
    cbar2.ax.invert_yaxis()
    cbar2.ax.tick_params(labelsize=9)

    # Match row-1 colorbar height to the actual drawn row-1 axes height.
    # Row-1 panels use equal aspect, so their visible height is smaller than the grid cell.
    fig.canvas.draw()
    pos11 = ax11.get_position()
    pos12 = ax12.get_position()
    posc1 = cax1.get_position()
    y0 = min(pos11.y0, pos12.y0) + 0.021
    y1 = max(pos11.y1, pos12.y1) + 0.023
    cax1.set_position([posc1.x0, y0, posc1.width, y1 - y0])

    output_path = repo_root / "results/plots_paper/2by2_lemniscate_gk.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
