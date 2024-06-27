from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def plot_kite_trajectory(time, x, y, z, variables=[], vecs=[], labels=None):
    # Create the figure
    n = len(variables)
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(int(n), 2, width_ratios=[1, 1], height_ratios=[0.5 for _ in range(n)])

    # 3D plot of the trajectory
    ax_3d = fig.add_subplot(gs[0:3, 0], projection="3d")
    ax_3d.plot(x, y, z, label="Trajectory")
    (red_point_3d,) = ax_3d.plot([x[0]], [y[0]], [z[0]], "ro")

    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_title("Trajectory")
    ax_3d.legend()

    # Time series plots
    ax_vars = []
    index_notnan_data = 0
    for i, var_group in enumerate(variables):
        if var_group is not None:
            ax = fig.add_subplot(gs[i, 1:])
            if i>0:
                ax.sharex(ax_prev)
            lines = []
            if not isinstance(var_group, list):
                var_group = [var_group]
            for j, var in enumerate(var_group):
                if np.all(np.isnan(var)):
                    continue
                else:
                    index_notnan_data = j
                label = f'Variable {i+1}' if labels is None or labels[i] is None else labels[i][j]
                line, = ax.plot(time, var, label=label)
                lines.append(line)
            red_point, = ax.plot([time[0]], [var_group[index_notnan_data][0]], 'ro')
            ax.set_ylabel(f'Variable {i+1}' if labels is None or labels[i] is None else labels[i][0])

            if i == len(variables) - 1:
                ax.set_xlabel("Time")
            ax.legend()
            ax.grid(True)
            ax_vars.append((lines, red_point))
            ax_prev = ax

    # Vector plots
    arrows = []
    for vec in vecs:
        arrow = ax_3d.quiver(
            x[0], y[0], z[0], vec[0][0], vec[0][1], vec[0][2], color="r", length=30
        )
        arrows.append(arrow)

    # Slider for time
    slider_ax_time = fig.add_axes(
        [0.1, 0.05, 0.3, 0.03], facecolor="lightgoldenrodyellow"
    )
    time_slider = Slider(
        slider_ax_time,
        "Time",
        time[0],
        time[-1],
        valinit=time[0],
        orientation="horizontal",
    )

    # Slider for elevation angle
    slider_ax_elev = fig.add_axes(
        [0.1, 0.1, 0.3, 0.03], facecolor="lightgoldenrodyellow"
    )
    elev_slider = Slider(
        slider_ax_elev, "Elevation", 0, 90, valinit=30, orientation="horizontal"
    )

    # Slider for azimuth angle
    slider_ax_azim = fig.add_axes(
        [0.1, 0.15, 0.3, 0.03], facecolor="lightgoldenrodyellow"
    )
    azim_slider = Slider(
        slider_ax_azim, "Azimuth", 0, 360, valinit=30, orientation="horizontal"
    )

    def update_time(val):
        current_time = time_slider.val
        idx = (np.abs(time - current_time)).argmin()

        # Update red point position in 3D plot
        red_point_3d.set_data([x[idx]], [y[idx]])
        red_point_3d.set_3d_properties([z[idx]])

        # Remove old arrows
        for arrow in arrows:
            arrow.remove()

        # Update arrow position in 3D plot
        arrows.clear()
        for vec in vecs:
            arrow = ax_3d.quiver(
                x[idx],
                y[idx],
                z[idx],
                vec[idx][0],
                vec[idx][1],
                vec[idx][2],
                color="r",
                length=30,
            )
            arrows.append(arrow)

        # Update red point position in 2D plots
        for lines, red_point in ax_vars:
            for line in lines:
                var = line.get_ydata()
                red_point.set_data([time[idx]], [var[idx]])

        fig.canvas.draw_idle()

    def update_view(val):
        ax_3d.view_init(elev=elev_slider.val, azim=azim_slider.val)
        fig.canvas.draw_idle()

    time_slider.on_changed(update_time)
    elev_slider.on_changed(update_view)
    azim_slider.on_changed(update_view)

    plt.show()
