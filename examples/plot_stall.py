import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style_no_latex
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

# Set plot style
set_plot_style_no_latex()

# Define the parameters
year = "2024"
month = "04"
day = "11"
kite_model = "v9"

# Read all four datasets in the order specified
results, flight_data, _ = read_results(year, month, day, kite_model)

# Only take the last 2000 data points
results = results.iloc[-2000:]
flight_data = flight_data.iloc[-2000:]

results = results.reset_index(drop=True)
flight_data = flight_data.reset_index(drop=True)

# Apply necessary adjustments (for wind direction correction, indexing, etc.)
def adjust_results(results):
    results.loc[results['wind_direction'] > np.radians(250), 'wind_direction'] -= np.radians(360)
    return results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

# Load flight data
# flight_data = pd.read_csv("log_2024-06-05_11-33-16.csv")

# Extract time and position data
t = flight_data['time'].values
x = flight_data['kite_position_x'].values
y = flight_data['kite_position_y'].values
z = flight_data['kite_position_z'].values

# Helper function to allow selection of variables
def select_variables():
    completer = WordCompleter(flight_data.columns, ignore_case=True)
    selected = prompt("Select variables to plot (comma-separated): ", completer=completer)
    return [var.strip() for var in selected.split(",")]

# Ask user for variables to plot
print("Available variables:")
print(flight_data.columns.to_list())
selected_vars = select_variables()

# Extract selected variables
variables = [flight_data[var].values for var in selected_vars]
labels = [(var, "Units unknown") for var in selected_vars]  # Modify this for actual units if known

# Setup figure and layout
n = len(variables)
fig = plt.figure(figsize=(14, 8))
gs = gridspec.GridSpec(n, 2, width_ratios=[1, 1], height_ratios=[0.5 for _ in range(n)])

# 3D plot of the trajectory
ax_3d = fig.add_subplot(gs[0:3, 0], projection="3d")
ax_3d.plot(x, y, z, label="Trajectory")
red_point_3d, = ax_3d.plot([x[0]], [y[0]], [z[0]], "ro")
ax_3d.set_xlabel("X")
ax_3d.set_ylabel("Y")
ax_3d.set_zlabel("Z")
ax_3d.set_title("Kite Trajectory")
ax_3d.legend()

# Time series plots for each selected variable
ax_vars = []
for i, variable in enumerate(variables):
    ax = fig.add_subplot(gs[i, 1:])
    label = labels[i][0]  # Use provided label or default to variable name
    line, = ax.plot(t, variable, label=label)
    red_point, = ax.plot([t[0]], [variable[0]], 'ro')
    ax.set_ylabel(label)
    if i == len(variables) - 1:
        ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True)
    ax_vars.append((line, red_point))

# Slider for time
slider_ax_time = fig.add_axes([0.1, 0.05, 0.3, 0.03], facecolor="lightgoldenrodyellow")
time_slider = Slider(slider_ax_time, "Time", t[0], t[-1], valinit=t[0], orientation="horizontal")

# Slider for elevation angle
slider_ax_elev = fig.add_axes([0.1, 0.1, 0.3, 0.03], facecolor="lightgoldenrodyellow")
elev_slider = Slider(slider_ax_elev, "Elevation", 0, 90, valinit=30, orientation="horizontal")

# Slider for azimuth angle
slider_ax_azim = fig.add_axes([0.1, 0.15, 0.3, 0.03], facecolor="lightgoldenrodyellow")
azim_slider = Slider(slider_ax_azim, "Azimuth", 0, 360, valinit=30, orientation="horizontal")

# Update function for the sliders
def update_time(val):
    current_time = time_slider.val
    idx = (np.abs(t - current_time)).argmin()

    # Update red point position in 3D plot
    red_point_3d.set_data([x[idx]], [y[idx]])
    red_point_3d.set_3d_properties([z[idx]])

    # Update red point position in 2D plots
    for line, red_point in ax_vars:
        var = line.get_ydata()
        red_point.set_data([t[idx]], [var[idx]])

    fig.canvas.draw_idle()

def update_view(val):
    ax_3d.view_init(elev=elev_slider.val, azim=azim_slider.val)
    fig.canvas.draw_idle()

# Connect sliders to update functions
time_slider.on_changed(update_time)
elev_slider.on_changed(update_view)
azim_slider.on_changed(update_view)

plt.show()
