import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from awes_ekf.postprocess.postprocessing import remove_offsets_IMU_data_v3
from awes_ekf.load_data.read_data import read_results
from awes_ekf.plotting.plot_utils import (
    plot_time_series,
    plot_kinetic_energy_spectrum,
    plot_forces_dimensional,
)
from awes_ekf.plotting.plot_kinematics import plot_kite_orientation
from awes_ekf.plotting.plot_tether import plot_slack_tether_force
from awes_ekf.plotting.plot_kinematics import calculate_azimuth_elevation
from awes_ekf.plotting.color_palette import (
    get_color_list,
    visualize_palette,
    set_plot_style,
    get_color,
)
from awes_ekf.setup.settings import SimulationConfig
from awes_ekf.setup.kite import PointMassEKF
from awes_ekf.utils import calculate_turn_rate_law, calculate_steering_law
from awes_ekf.setup.kcu import KCU
from scipy.stats import linregress

# Desired data from full-flight
# 1800.0 s to 9986.2 s

# Select a pre-process script by number: 1
# Data pre-processed using: process_v3_data.py
# Duration of the flight: 196.44 minutes.
# Enter the start minute for analysis or skip: 25
# Enter the end minute for analysis or skip: 170
# Filtered data from minute 25 to 170.


def cut_data(results, flight_data, range):
    results = results.iloc[range[0] : range[1]]
    flight_data = flight_data.iloc[range[0] : range[1]]
    results = results.reset_index(drop=True)
    flight_data = flight_data.reset_index(drop=True)
    return results, flight_data


set_plot_style()
year = "2019"
month = "10"
day = "08"
kite_model = "v3"
addition = "_t26"

year = "2025"
month = "10"
day = "09"
kite_model = "v3"
addition = ""

results, flight_data, config_data = read_results(
    year, month, day, kite_model, addition=addition
)
res_min, fd_min, config_data_min = read_results(
    year, month, day, kite_model, addition=addition
)

# Time-based filtering: 1800.0 s to 9986.2 s -- 2019
# time_mask = (results["time"] >= 1800.0) & (results["time"] <= 9986.2)
# Time-based filtering: 10.0 s to 1080s -- 2025
time_mask = (results["time"] >= 180.0) & (results["time"] <= 1080)

print(config_data["simulation_parameters"]["measurements"])
print(
    f"loaded results file: results/{kite_model}/{kite_model}_{year}-{month}-{day}_t26.h5"
)
print("results columns:", list(results.columns))
print("flight_data columns:", list(flight_data.columns))

circle_df = None
circle_ups = []
circle_colors = []
circle_csv_path = (
    Path(__file__).resolve().parents[1] / "data" / "circle_batch_analysis.csv"
)
if circle_csv_path.is_file():
    circle_df = pd.read_csv(circle_csv_path)
    required_cols = ["us", "v_app", "yaw_rate_paper", "cs", "up"]
    missing_cols = [col for col in required_cols if col not in circle_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing columns in {circle_csv_path}: {', '.join(missing_cols)}"
        )
    circle_mask = (
        np.isfinite(circle_df["us"])
        & np.isfinite(circle_df["v_app"])
        & np.isfinite(circle_df["yaw_rate_paper"])
        & np.isfinite(circle_df["cs"])
        & np.isfinite(circle_df["up"])
    )
    circle_df = circle_df.loc[circle_mask]
    circle_ups = np.sort(circle_df["up"].unique())
    circle_cmap = plt.get_cmap("tab10")
    circle_colors = [circle_cmap(i % circle_cmap.N) for i in range(len(circle_ups))]
else:
    print(f"circle batch csv not found: {circle_csv_path}")

# for imu in config_data["kite"]["sensor_ids"]:
#     flight_data = remove_offsets_IMU_data_v3(results, flight_data, sensor=imu)

results = results.loc[time_mask].reset_index(drop=True)
flight_data = flight_data.loc[time_mask].reset_index(drop=True)

mask = flight_data["cycle"].isin([64, 65])

colors = get_color_list()
a = results["radius_turn"]
simConfig = SimulationConfig(**config_data["simulation_parameters"])

# Create system components
kite = PointMassEKF(simConfig, **config_data["kite"])
kcu = KCU(**config_data["kcu"])
flight_data["kite_yaw_rate"] = flight_data["kite_yaw_rate_1"]

flight_data["kcu_actual_steering_delay"] = np.roll(
    flight_data["kcu_actual_steering"], int(8)
)


import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Downsample the data (e.g., use only 10% of the data)
downsample_fraction = 0.5
downsampled_data = flight_data.sample(frac=downsample_fraction, random_state=42)
downsampled_results = results.loc[downsampled_data.index]
downsampled_data = downsampled_data[downsampled_data["powered"] == "powered"]
downsampled_results = downsampled_results.loc[downsampled_data.index]

downsampled_sorted = downsampled_data.sort_values("time")
downsampled_results_sorted = downsampled_results.loc[downsampled_sorted.index]

kite_speed_norm = None
if {
    "kite_velocity_x",
    "kite_velocity_y",
    "kite_velocity_z",
}.issubset(downsampled_results_sorted.columns):
    kite_speed = np.sqrt(
        downsampled_results_sorted["kite_velocity_x"] ** 2
        + downsampled_results_sorted["kite_velocity_y"] ** 2
        + downsampled_results_sorted["kite_velocity_z"] ** 2
    )
    max_kite_speed = float(kite_speed.max())
    if max_kite_speed == 0:
        max_kite_speed = 1.0
    kite_speed_norm = kite_speed / max_kite_speed

if "kcu_actual_depower" in downsampled_data.columns:
    depower = downsampled_sorted["kcu_actual_depower"]
    depower_min = float(depower.min())
    depower_max = float(depower.max())
    depower_mean = float(depower.mean())
    print(
        "kcu_actual_depower stats (downsampled powered): "
        f"mean={depower_mean:.3f}, min={depower_min:.3f}, max={depower_max:.3f}"
    )
    fig_dep, ax_dep = plt.subplots(figsize=(6, 2.5))
    ax_dep.plot(
        downsampled_sorted["time"],
        depower,
        color=colors[0],
        marker=".",
        linestyle="None",
        alpha=0.6,
        label=r"$\mathrm{kcu\_actual\_depower}$",
    )
    ax_dep.axhline(depower_mean, color=colors[1], linestyle="--", label="depower mean")
    ax_dep.axhline(depower_min, color=colors[2], linestyle=":", label="depower min")
    ax_dep.axhline(depower_max, color=colors[3], linestyle=":", label="depower max")
    ax_dep.set_xlabel("time (s)")
    ax_dep.set_ylabel(r"$\mathrm{kcu\_actual\_depower}$")
    ax_dep.legend(frameon=True)
    fig_dep.tight_layout()
    fig_dep.savefig("./results/plots_paper/depower_cut_stats.pdf")

if "kcu_actual_steering" in downsampled_data.columns:
    steering = downsampled_sorted["kcu_actual_steering"]
    steering_min = float(steering.min())
    steering_max = float(steering.max())
    steering_mean = float(steering.mean())
    print(
        "kcu_actual_steering stats (downsampled powered): "
        f"mean={steering_mean:.3f}, min={steering_min:.3f}, max={steering_max:.3f}"
    )
    fig_steer, ax_steer = plt.subplots(figsize=(6, 2.5))
    ax_steer.plot(
        downsampled_sorted["time"],
        steering,
        color=colors[0],
        marker=".",
        linestyle="None",
        alpha=0.6,
        label=r"$\mathrm{kcu\_actual\_steering}$",
    )
    ax_steer.axhline(
        steering_mean, color=colors[1], linestyle="--", label="steering mean"
    )
    ax_steer.axhline(steering_min, color=colors[2], linestyle=":", label="steering min")
    ax_steer.axhline(steering_max, color=colors[3], linestyle=":", label="steering max")
    ax_steer.set_xlabel("time (s)")
    ax_steer.set_ylabel(r"$\mathrm{kcu\_actual\_steering}$")
    ax_steer.legend(frameon=True)
    fig_steer.tight_layout()
    fig_steer.savefig("./results/plots_paper/steering_cut_stats.pdf")


multi_row_signals = [
    (
        "tether_length",
        (
            downsampled_sorted["tether_length"]
            if "tether_length" in downsampled_sorted.columns
            else None
        ),
        r"$\mathrm{tether\_length}\;(\mathrm{m})$",
        r"\mathrm{m}",
    ),
    (
        "wind_speed_horizontal",
        (
            downsampled_results_sorted["wind_speed_horizontal"]
            if "wind_speed_horizontal" in downsampled_results_sorted.columns
            else None
        ),
        r"$\mathrm{wind\_speed\_horizontal}\;(\mathrm{m\,s^{-1}})$",
        r"\mathrm{m\,s^{-1}}",
    ),
    (
        "wing_angle_of_attack",
        (
            downsampled_results_sorted["wing_angle_of_attack"]
            if "wing_angle_of_attack" in downsampled_results_sorted.columns
            else None
        ),
        r"$\mathrm{wing\_angle\_of\_attack}\;(^\circ)$",
        r"^\circ",
    ),
    (
        "kite_apparent_windspeed",
        (
            downsampled_results_sorted["kite_apparent_windspeed"]
            if "kite_apparent_windspeed" in downsampled_results_sorted.columns
            else None
        ),
        r"$\mathrm{kite\_apparent\_windspeed}\;(\mathrm{m\,s^{-1}})$",
        r"\mathrm{m\,s^{-1}}",
    ),
    (
        "radius_turn",
        (
            downsampled_results_sorted["radius_turn"]
            if "radius_turn" in downsampled_results_sorted.columns
            else None
        ),
        r"$\mathrm{radius\_turn}\;(\mathrm{m})$",
        r"\mathrm{m}",
    ),
    (
        "kcu_actual_depower",
        (
            downsampled_sorted["kcu_actual_depower"]
            if "kcu_actual_depower" in downsampled_sorted.columns
            else None
        ),
        r"$\mathrm{kcu\_actual\_depower}\;(\%)$",
        r"\%",
    ),
    (
        "kcu_actual_steering",
        (
            downsampled_sorted["kcu_actual_steering"]
            if "kcu_actual_steering" in downsampled_sorted.columns
            else None
        ),
        r"$\mathrm{kcu\_actual\_steering}\;(\%)$",
        r"\%",
    ),
]

multi_row_signals = [entry for entry in multi_row_signals if entry[1] is not None]
if multi_row_signals:
    fig_multi, axes_multi = plt.subplots(
        len(multi_row_signals), 1, figsize=(6, 12), sharex=True
    )
    if len(multi_row_signals) == 1:
        axes_multi = [axes_multi]
    for ax, (name, series, label, unit_tex) in zip(axes_multi, multi_row_signals):
        mean_val = float(series.mean())
        min_val = float(series.min())
        max_val = float(series.max())
        ax.plot(
            downsampled_sorted["time"],
            series,
            color=colors[0],
            marker=".",
            linestyle="None",
            alpha=0.6,
            label=label,
        )
        ax.axhline(
            mean_val,
            color=colors[1],
            linestyle="--",
            label=rf"$\mathrm{{mean}} = {mean_val:.3f}\,{unit_tex}$",
        )
        ax.axhline(
            min_val,
            color=colors[2],
            linestyle=":",
            label=rf"$\mathrm{{min}} = {min_val:.3f}\,{unit_tex}$",
        )
        ax.axhline(
            max_val,
            color=colors[3],
            linestyle=":",
            label=rf"$\mathrm{{max}} = {max_val:.3f}\,{unit_tex}$",
        )
        ax.set_ylabel(label)
        ax.legend(frameon=True, loc="upper left", framealpha=1.0)
    axes_multi[-1].set_xlabel("time (s)")
    fig_multi.tight_layout()
    fig_multi.savefig("./results/plots_paper/steering_cut_stats_multi.pdf")

# plt.plot(results["radius_turn"])
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define data
max_abs_steering = flight_data["kcu_actual_steering"].abs().max()
if max_abs_steering == 0:
    max_abs_steering = 1.0
max_abs_us = flight_data["us"].abs().max()
if max_abs_us == 0:
    max_abs_us = 1.0
steering_norm = max_abs_steering / max_abs_us
steering_scale = steering_norm / 100.0

x_full_kcu = -downsampled_data["kcu_actual_steering_delay"] / 100
x_full_us = -downsampled_data["kcu_actual_steering_delay"] / steering_norm
x_no_delay_kcu = -downsampled_data["kcu_actual_steering"] / 100
x_no_delay_us = -downsampled_data["us"]
y_full = downsampled_results["wing_sideforce_coefficient"]

plt.plot(flight_data["kcu_actual_steering"])
# plt.show()
# Define the regions
# threshold = np.radians(10)
# var_threshold = downsampled_data["kite_yaw_rate"]
# mask_straight = abs(var_threshold) < threshold
# mask_left = (abs(var_threshold) > threshold) & (x_full < 0)
# mask_right = (abs(var_threshold) > threshold) & (x_full > 0)
upper_threshold = 0.08
lower_threshold = -0.06
lower_threshold_us = lower_threshold / steering_scale
upper_threshold_us = upper_threshold / steering_scale
mask_straight_kcu = x_full_kcu.between(lower_threshold, upper_threshold)
mask_left_kcu = x_full_kcu < lower_threshold
mask_right_kcu = x_full_kcu > upper_threshold
mask_straight_us = x_full_us.between(lower_threshold_us, upper_threshold_us)
mask_left_us = x_full_us < lower_threshold_us
mask_right_us = x_full_us > upper_threshold_us

# Right-turn-only version of steering_cut_stats_multi
x_full_kcu_sorted = -downsampled_sorted["kcu_actual_steering_delay"] / 100
mask_right_kcu_sorted = x_full_kcu_sorted > upper_threshold
multi_row_signals_turn = [
    (name, series.loc[mask_right_kcu_sorted], label, unit_tex)
    for (name, series, label, unit_tex) in multi_row_signals
]
multi_row_signals_turn = [
    entry
    for entry in multi_row_signals_turn
    if entry[1] is not None and not entry[1].empty
]
if multi_row_signals_turn:
    fig_multi_turn, axes_multi_turn = plt.subplots(
        len(multi_row_signals_turn), 1, figsize=(6, 12), sharex=True
    )
    if len(multi_row_signals_turn) == 1:
        axes_multi_turn = [axes_multi_turn]
    time_turn = downsampled_sorted.loc[mask_right_kcu_sorted, "time"]
    for ax, (name, series, label, unit_tex) in zip(
        axes_multi_turn, multi_row_signals_turn
    ):
        mean_val = float(series.mean())
        min_val = float(series.min())
        max_val = float(series.max())
        ax.plot(
            time_turn,
            series,
            color=colors[0],
            marker=".",
            linestyle="None",
            alpha=0.6,
            label=label,
        )
        ax.axhline(
            mean_val,
            color=colors[1],
            linestyle="--",
            label=rf"$\mathrm{{mean}} = {mean_val:.3f}\,{unit_tex}$",
        )
        ax.axhline(
            min_val,
            color=colors[2],
            linestyle=":",
            label=rf"$\mathrm{{min}} = {min_val:.3f}\,{unit_tex}$",
        )
        ax.axhline(
            max_val,
            color=colors[3],
            linestyle=":",
            label=rf"$\mathrm{{max}} = {max_val:.3f}\,{unit_tex}$",
        )
        ax.set_ylabel(label)
        ax.legend(frameon=True, loc="upper left", framealpha=1.0)
    axes_multi_turn[-1].set_xlabel("time (s)")
    fig_multi_turn.tight_layout()
    fig_multi_turn.savefig("./results/plots_paper/steering_cut_stats_turn.pdf")


# Compute linear regressions for each region (protect against too few samples)
def safe_linregress(x_series, y_series):
    if len(x_series) >= 2 and len(y_series) >= 2:
        res = linregress(x_series, y_series)
        return res.slope, res.intercept, res.rvalue
    return np.nan, np.nan, np.nan


slope_straight, intercept_straight, r_straight = safe_linregress(
    x_full_kcu[mask_straight_kcu], y_full[mask_straight_kcu]
)
slope_right, intercept_right, r_right = safe_linregress(
    x_full_kcu[mask_right_kcu], y_full[mask_right_kcu]
)
slope_left, intercept_left, r_left = safe_linregress(
    x_full_kcu[mask_left_kcu], y_full[mask_left_kcu]
)

print("R^2 Straight: ", r_straight**2 if np.isfinite(r_straight) else np.nan)
print("R^2 Right: ", r_right**2 if np.isfinite(r_right) else np.nan)
print("R^2 Left: ", r_left**2 if np.isfinite(r_left) else np.nan)


# Generate regression lines for each region when data exists
def safe_line(mask, slope, intercept):
    if mask.any() and np.isfinite(slope) and np.isfinite(intercept):
        x_values = x_full_kcu[mask]
        x_min = float(x_values.min())
        x_max = float(x_values.max())
        x_grid = np.linspace(x_min, x_max, 100)
        return x_grid, slope * x_grid + intercept
    return np.array([]), np.array([])


x_line_straight, y_line_straight = safe_line(
    mask_straight_kcu, slope_straight, intercept_straight
)
x_line_right, y_line_right = safe_line(mask_right_kcu, slope_right, intercept_right)
x_line_left, y_line_left = safe_line(mask_left_kcu, slope_left, intercept_left)

est_cs, coeff_cs = calculate_steering_law(downsampled_results, downsampled_data)
print("Coefficients steering: ", coeff_cs)
# Plot data points and regression lines
fig_cs, (ax_cs_us, ax_cs_kcu) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# u_s panel
ax_cs_us.scatter(
    x_no_delay_us,
    y_full,
    alpha=0.2,
    label="No Delay Correction",
    color="gray",
    marker=".",
)
ax_cs_us.scatter(
    x_full_us[mask_straight_us],
    y_full[mask_straight_us],
    alpha=0.4,
    label="Straight (Corrected)",
    color=colors[1],
    marker=".",
)
ax_cs_us.scatter(
    x_full_us[mask_left_us],
    y_full[mask_left_us],
    alpha=0.4,
    label="Left Turn (Corrected)",
    color=colors[2],
    marker=".",
)
ax_cs_us.scatter(
    x_full_us[mask_right_us],
    y_full[mask_right_us],
    alpha=0.4,
    label="Right Turn (Corrected)",
    color=colors[3],
    marker=".",
)
ax_cs_us.scatter(
    x_full_us,
    est_cs,
    alpha=0.2,
    label="Delay Corrected",
    color="black",
    marker=".",
)
if circle_df is not None and not circle_df.empty:
    for i, up_val in enumerate(circle_ups):
        rows = circle_df[circle_df["up"] == up_val]
        ax_cs_us.scatter(
            -rows["us"],
            rows["cs"],
            s=80,
            alpha=1.0,
            color=circle_colors[i],
            marker="x",
            label="_nolegend_",
        )
        ax_cs_kcu.scatter(
            -rows["us"],
            rows["cs"],
            s=80,
            alpha=1.0,
            color=circle_colors[i],
            marker="x",
            label="_nolegend_",
        )
ax_cs_us.set_xlabel(r"$u_s$")
ax_cs_us.set_ylabel(r"$C_S$")
ax_cs_us.legend(frameon=True)

# kcu_actual_steering panel
ax_cs_kcu.scatter(
    x_no_delay_kcu,
    y_full,
    alpha=0.2,
    label="No Delay Correction",
    color="gray",
    marker=".",
)
ax_cs_kcu.scatter(
    x_full_kcu[mask_straight_kcu],
    y_full[mask_straight_kcu],
    alpha=0.4,
    label="Straight (Corrected)",
    color=colors[1],
    marker=".",
)
ax_cs_kcu.scatter(
    x_full_kcu[mask_left_kcu],
    y_full[mask_left_kcu],
    alpha=0.4,
    label="Left Turn (Corrected)",
    color=colors[2],
    marker=".",
)
ax_cs_kcu.scatter(
    x_full_kcu[mask_right_kcu],
    y_full[mask_right_kcu],
    alpha=0.4,
    label="Right Turn (Corrected)",
    color=colors[3],
    marker=".",
)
ax_cs_kcu.scatter(
    x_full_kcu,
    est_cs,
    alpha=0.2,
    label="Delay Corrected",
    color="black",
    marker=".",
)
ax_cs_kcu.set_xlabel(r"$\mathrm{kcu\_actual\_steering}/100$")
ax_cs_kcu.legend(frameon=True)

fig_cs.tight_layout()
fig_cs.savefig("./results/plots_paper/sideforce_three_regions.pdf")
# plt.show()

# flight_data["kite_yaw_rate"] = np.gradient(np.unwrap(flight_data["kite_yaw_0"]), ts)
yaw_rate, coeffs = calculate_turn_rate_law(
    results, flight_data, model="simple", steering_offset=False
)
yaw_rate_weight, coeffs_weight = calculate_turn_rate_law(
    results, flight_data, model="simple", steering_offset=True
)

print("Coefficients: ", coeffs)
# Calculate mean errors
error = abs(np.degrees(yaw_rate) - np.degrees(flight_data["kite_yaw_rate"]))
error_weight = abs(
    np.degrees(yaw_rate_weight) - np.degrees(flight_data["kite_yaw_rate"])
)
mean_error = np.mean(error)
mean_error_weight = np.mean(error_weight)
# Calculate R^2 values
r_squared = 1 - np.sum(error**2) / np.sum(
    (
        np.degrees(flight_data["kite_yaw_rate"])
        - np.mean(np.degrees(flight_data["kite_yaw_rate"]))
    )
    ** 2
)
r_squared_weight = 1 - np.sum(error_weight**2) / np.sum(
    (
        np.degrees(flight_data["kite_yaw_rate"])
        - np.mean(np.degrees(flight_data["kite_yaw_rate"]))
    )
    ** 2
)

print("r_squared: ", r_squared)
print("r_squared_weight: ", r_squared_weight)

# Prepare data
x_kcu = x_no_delay_kcu * downsampled_results["kite_apparent_windspeed"]
x_us = x_no_delay_us * downsampled_results["kite_apparent_windspeed"]
y = downsampled_data["kite_yaw_rate"]

# Calculate the point density
xy = np.vstack([x_kcu, y])
# z = gaussian_kde(xy)(xy)
#
# Create KDE plot
fig_yaw, (ax_yaw_us, ax_yaw_kcu) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
# scatter = ax_yaw_kcu.scatter(
#     x_kcu, y,
#     c=z,  # Use density values for coloring
#     s=20,  # Adjust point size if needed
#     cmap='viridis',  # Color map (adjust to preference)
#     alpha=0.6  # Transparency level
# )

# u_s panel
ax_yaw_us.scatter(
    x_us[mask_straight_us],
    y[mask_straight_us] * 180 / np.pi,
    color=colors[1],
    alpha=0.4,
    marker=".",
    label="Straight",
)
ax_yaw_us.scatter(
    x_us[mask_left_us],
    y[mask_left_us] * 180 / np.pi,
    color=colors[2],
    marker=".",
    alpha=0.4,
    label="Left Turn",
)
ax_yaw_us.scatter(
    x_us[mask_right_us],
    y[mask_right_us] * 180 / np.pi,
    color=colors[3],
    marker=".",
    alpha=0.4,
    label="Right Turn",
)
if circle_df is not None and not circle_df.empty:
    for i, up_val in enumerate(circle_ups):
        rows = circle_df[circle_df["up"] == up_val]
        us_val = rows["us"].values[i]
        print(f"Plotting circle batch up={up_val:.3f}, us={us_val:.3f}")
        ax_yaw_us.scatter(
            rows["us"] * rows["v_app"],
            rows["yaw_rate_paper"],
            s=80,
            alpha=1.0,
            color=circle_colors[i],
            marker="x",
            label=f"Batch up={up_val:.3f}, us={us_val.values[i]:.3f}",
        )
        ax_yaw_kcu.scatter(
            rows["us"] * rows["v_app"],
            rows["yaw_rate_paper"],
            s=80,
            alpha=1.0,
            color=circle_colors[i],
            marker="x",
            label=f"Batch up={up_val:.3f}, us={us_val.values[i]:.3f}",
        )

# kcu_actual_steering panel
ax_yaw_kcu.scatter(
    x_kcu[mask_straight_kcu],
    y[mask_straight_kcu] * 180 / np.pi,
    color=colors[1],
    alpha=0.4,
    marker=".",
    label="Straight",
)
ax_yaw_kcu.scatter(
    x_kcu[mask_left_kcu],
    y[mask_left_kcu] * 180 / np.pi,
    color=colors[2],
    marker=".",
    alpha=0.4,
    label="Left Turn",
)
ax_yaw_kcu.scatter(
    x_kcu[mask_right_kcu],
    y[mask_right_kcu] * 180 / np.pi,
    color=colors[3],
    marker=".",
    alpha=0.4,
    label="Right Turn",
)

x_line_kcu = np.linspace(x_kcu.min(), x_kcu.max(), 100)
y_line_kcu = coeffs[0] * x_line_kcu
# Overlay the line plot
ax_yaw_kcu.plot(
    -x_line_kcu,
    y_line_kcu * 180 / np.pi,
    label=f"Identified Yaw Rate (R$^2$: {r_squared:.2f})",
    color=colors[0],
    linestyle="--",
)
A_kcu = np.vstack([x_line_kcu, 20 * np.ones_like(x_line_kcu)]).T
y_line_kcu = A_kcu @ coeffs_weight
ax_yaw_kcu.plot(
    -x_line_kcu,
    y_line_kcu * 180 / np.pi,
    label=f"Offset-Corrected Yaw Rate (R$^2$: {r_squared_weight:.2f})",
    color=colors[0],
    linestyle=":",
)

x_line_us = np.linspace(x_us.min(), x_us.max(), 100)
y_line_us = coeffs[0] * (x_line_us * steering_scale)
ax_yaw_us.plot(
    -x_line_us,
    y_line_us * 180 / np.pi,
    label=f"Identified Yaw Rate (R$^2$: {r_squared:.2f})",
    color=colors[0],
    linestyle="--",
)
A_us = np.vstack([x_line_us * steering_scale, 20 * np.ones_like(x_line_us)]).T
y_line_us = A_us @ coeffs_weight
ax_yaw_us.plot(
    -x_line_us,
    y_line_us * 180 / np.pi,
    label=f"Offset-Corrected Yaw Rate (R$^2$: {r_squared_weight:.2f})",
    color=colors[0],
    linestyle=":",
)

ax_yaw_us.set_xlabel(r"$u_s v_a\;(\mathrm{m\,s^{-1}})$")
ax_yaw_kcu.set_xlabel(
    r"$\mathrm{kcu\_actual\_steering}/100 \cdot v_a\;(\mathrm{m\,s^{-1}})$"
)
ax_yaw_us.set_ylabel(r"$\dot{\psi}\;(^\circ\,\mathrm{s^{-1}})$")
ax_yaw_us.legend(frameon=True)
ax_yaw_kcu.legend(frameon=True)

# Adjust layout and save the plot
fig_yaw.tight_layout()
fig_yaw.savefig("./results/plots_paper/yaw_rate_three_regions.pdf")
# plt.show()

# Right-turn yaw rate colored by tether length buckets
tether_length = downsampled_data["tether_length"]
length_buckets = [
    (225.0, 250.0, "lt = 225-250 m", colors[0]),
    (250.0, 275.0, "lt = 250-275 m", colors[1]),
    (275.0, 300.0, "lt = 275-300 m", colors[2]),
    (300.0, 325.0, "lt = 300-325 m", colors[3]),
]

fig_yaw_lt, (ax_yaw_lt_us, ax_yaw_lt_kcu) = plt.subplots(
    1, 2, figsize=(10, 4), sharey=True
)
y_deg = y * 180 / np.pi

for low, high, label, color in length_buckets:
    if high >= 600.0:
        mask_len = (tether_length >= low) & (tether_length <= high)
    else:
        mask_len = (tether_length >= low) & (tether_length < high)

    # mask_bucket_us = mask_right_us & mask_len
    mask_bucket_us = mask_len
    x_vals_us = x_us[mask_bucket_us]
    y_vals_us = y_deg[mask_bucket_us]
    finite_us = np.isfinite(x_vals_us) & np.isfinite(y_vals_us)
    x_vals_us = x_vals_us[finite_us]
    y_vals_us = y_vals_us[finite_us]
    label_us = label
    if len(x_vals_us) > 1:
        slope_us, intercept_us, r_us, _, _ = linregress(x_vals_us, y_vals_us)
        label_us = f"{label} (k={slope_us:.3f}, R$^2$={r_us**2:.2f})"
        x_line_us = np.linspace(x_vals_us.min(), x_vals_us.max(), 100)
        y_line_us = slope_us * x_line_us + intercept_us
        ax_yaw_lt_us.plot(
            x_line_us,
            y_line_us,
            color=color,
            linestyle="--",
            alpha=0.9,
            label="_nolegend_",
        )
    ax_yaw_lt_us.scatter(
        x_vals_us,
        y_vals_us,
        color=color,
        alpha=0.4,
        marker=".",
        label=label_us,
    )

    # mask_bucket_kcu = mask_right_kcu & mask_len
    mask_bucket_kcu = mask_len
    x_vals_kcu = x_kcu[mask_bucket_kcu]
    y_vals_kcu = y_deg[mask_bucket_kcu]
    finite_kcu = np.isfinite(x_vals_kcu) & np.isfinite(y_vals_kcu)
    x_vals_kcu = x_vals_kcu[finite_kcu]
    y_vals_kcu = y_vals_kcu[finite_kcu]
    label_kcu = label
    if len(x_vals_kcu) > 1:
        slope_kcu, intercept_kcu, r_kcu, _, _ = linregress(x_vals_kcu, y_vals_kcu)
        label_kcu = f"{label} (k={slope_kcu:.3f}, R$^2$={r_kcu**2:.2f})"
        x_line_kcu = np.linspace(x_vals_kcu.min(), x_vals_kcu.max(), 100)
        y_line_kcu = slope_kcu * x_line_kcu + intercept_kcu
        ax_yaw_lt_kcu.plot(
            x_line_kcu,
            y_line_kcu,
            color=color,
            linestyle="--",
            alpha=0.9,
            label="_nolegend_",
        )
    ax_yaw_lt_kcu.scatter(
        x_vals_kcu,
        y_vals_kcu,
        color=color,
        alpha=0.4,
        marker=".",
        label=label_kcu,
    )

ax_yaw_lt_us.set_xlabel(r"$u_s v_a\;(\mathrm{m\,s^{-1}})$")
ax_yaw_lt_kcu.set_xlabel(
    r"$\mathrm{kcu\_actual\_steering}/100 \cdot v_a\;(\mathrm{m\,s^{-1}})$"
)
ax_yaw_lt_us.set_ylabel(r"$\dot{\psi}\;(^\circ\,\mathrm{s^{-1}})$")
ax_yaw_lt_us.legend(frameon=True)
ax_yaw_lt_kcu.legend(frameon=True)

fig_yaw_lt.tight_layout()
fig_yaw_lt.savefig("./results/plots_paper/yaw_lt.pdf")


# # Define `x` and ensure alignment with the mask for "kcu_actual_steering_delay"
# x_delay = -flight_data.loc[mask, "kcu_actual_steering_delay"] / 100
# y = results.loc[mask, "wing_sideforce_coefficient"]

# # Define the regions for "kcu_actual_steering_delay"
# mask_straight_delay = x_delay.between(lower_threshold, upper_threshold)
# mask_left_delay = x_delay < lower_threshold
# mask_right_delay = x_delay > upper_threshold

# # Compute linear regressions for each region for "kcu_actual_steering_delay"
# slope_straight_delay, intercept_straight_delay, _, _, _ = linregress(
#     x_delay[mask_straight_delay], y[mask_straight_delay]
# )
# slope_right_delay, intercept_right_delay, _, _, _ = linregress(
#     x_delay[mask_right_delay], y[mask_right_delay]
# )
# slope_left_delay, intercept_left_delay, _, _, _ = linregress(
#     x_delay[mask_left_delay], y[mask_left_delay]
# )

# # Fit lines for each region for "kcu_actual_steering_delay"
# fit_straight_delay = pd.Series(
#     slope_straight_delay * x_delay[mask_straight_delay] + intercept_straight_delay,
#     index=x_delay[mask_straight_delay].index,
# )
# fit_right_delay = pd.Series(
#     slope_right_delay * x_delay[mask_right_delay] + intercept_right_delay,
#     index=x_delay[mask_right_delay].index,
# )
# fit_left_delay = pd.Series(
#     slope_left_delay * x_delay[mask_left_delay] + intercept_left_delay,
#     index=x_delay[mask_left_delay].index,
# )

# # Combine the fits for "kcu_actual_steering_delay"
# combined_fit_delay = pd.concat(
#     [fit_straight_delay, fit_right_delay, fit_left_delay]
# ).sort_index()

# # Define `x` and ensure alignment with the mask for "kcu_actual_steering"
# x_steering = -flight_data.loc[mask, "kcu_actual_steering"] / 100

# mask_straight_steering = x_steering.between(lower_threshold, upper_threshold)
# mask_left_steering = x_steering < lower_threshold
# mask_right_steering = x_steering > upper_threshold

# # Compute linear regressions for each region for "kcu_actual_steering"
# slope_straight_steering, intercept_straight_steering, _, _, _ = linregress(
#     x_steering[mask_straight_steering], y[mask_straight_steering]
# )
# slope_right_steering, intercept_right_steering, _, _, _ = linregress(
#     x_steering[mask_right_steering], y[mask_right_steering]
# )
# slope_left_steering, intercept_left_steering, _, _, _ = linregress(
#     x_steering[mask_left_steering], y[mask_left_steering]
# )

# # Fit lines for each region for "kcu_actual_steering"
# fit_straight_steering = pd.Series(
#     slope_straight_steering * x_steering[mask_straight_steering]
#     + intercept_straight_steering,
#     index=x_steering[mask_straight_steering].index,
# )
# fit_right_steering = pd.Series(
#     slope_right_steering * x_steering[mask_right_steering] + intercept_right_steering,
#     index=x_steering[mask_right_steering].index,
# )
# fit_left_steering = pd.Series(
#     slope_left_steering * x_steering[mask_left_steering] + intercept_left_steering,
#     index=x_steering[mask_left_steering].index,
# )

# # Combine the fits for "kcu_actual_steering"
# combined_fit_steering = pd.concat(
#     [fit_straight_steering, fit_right_steering, fit_left_steering]
# ).sort_index()
# # Plot the time series
# fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# # Plot the original time series
# plot_time_series(
#     flight_data.loc[mask],
#     y,
#     axs[0],
#     ylabel="$C_{S}$",
#     plot_phase=False,
#     color=colors[0],
#     label="EKF 0",
# )

# # Plot the combined linear fit for "kcu_actual_steering_delay"
# axs[0].plot(
#     flight_data.loc[mask, "time"],
#     combined_fit_delay,
#     label="Linear Fit (Delay)",
#     color=colors[1],
#     linestyle="--",
# )

# # Plot the combined linear fit for "kcu_actual_steering"
# axs[0].plot(
#     flight_data.loc[mask, "time"],
#     combined_fit_steering,
#     label="Linear Fit (Steering)",
#     color=colors[2],
#     linestyle=":",
# )

# c0 = np.ones_like(flight_data.loc[mask, "kcu_actual_steering_delay"])
# c1 = flight_data.loc[mask, "kcu_actual_steering_delay"] / 100
# c2 = (flight_data.loc[mask, "kcu_actual_steering_delay"] / 100) ** 2 * np.sign(
#     flight_data.loc[mask, "kcu_actual_steering_delay"]
# )
# c3 = (
#     flight_data.loc[mask, "kite_yaw_rate"]
#     / flight_data.loc[mask, "kite_apparent_windspeed"]
#     * np.sign(flight_data.loc[mask, "kcu_actual_steering_delay"])
# )

# A = np.vstack([c0, c1, c2, c3]).T
# est_cs = A @ coeff_cs
