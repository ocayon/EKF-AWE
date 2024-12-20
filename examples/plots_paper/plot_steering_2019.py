import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from awes_ekf.postprocess.postprocessing import remove_offsets_IMU_data_v3
from awes_ekf.load_data.read_data import read_results
from awes_ekf.plotting.plot_utils import plot_time_series, plot_kinetic_energy_spectrum, plot_forces_dimensional
from awes_ekf.plotting.plot_kinematics import plot_kite_orientation
from awes_ekf.plotting.plot_tether import plot_slack_tether_force
from awes_ekf.plotting.plot_kinematics import calculate_azimuth_elevation
from awes_ekf.plotting.color_palette import get_color_list, visualize_palette, set_plot_style, get_color
from awes_ekf.setup.settings import  SimulationConfig
from awes_ekf.setup.kite import PointMassEKF
from awes_ekf.utils import calculate_turn_rate_law, find_time_delay
from awes_ekf.setup.kcu import KCU
from scipy.stats import linregress

def cut_data(results, flight_data, range):
    results = results.iloc[range[0]:range[1]]
    flight_data = flight_data.iloc[range[0]:range[1]]
    results = results.reset_index(drop=True)
    flight_data = flight_data.reset_index(drop=True)
    return results, flight_data

set_plot_style()
year = "2019"
month = "10"
day = "08"
kite_model = "v3"

results, flight_data,config_data = read_results(year, month, day, kite_model,addition='_lt')
res_min, fd_min,config_data_min = read_results(year, month, day, kite_model,addition='_min')

print(config_data["simulation_parameters"]["measurements"])

# for imu in config_data["kite"]["sensor_ids"]:
#     flight_data = remove_offsets_IMU_data_v3(results, flight_data, sensor=imu)
results, flight_data = cut_data(results, flight_data, [18000, len(results)-18000])
mask = flight_data["cycle"].isin([64, 65])

colors = get_color_list()
ts =0.1
a = results["radius_turn"]
simConfig = SimulationConfig(**config_data["simulation_parameters"])

# Create system components
kite = PointMassEKF(simConfig, **config_data["kite"])
kcu = KCU(**config_data["kcu"])
flight_data['kite_yaw_rate'] = flight_data['kite_yaw_rate_1']

flight_data["kcu_actual_steering_delay"] = np.roll(flight_data["kcu_actual_steering"], int(8))


import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# Downsample the data (e.g., use only 10% of the data)
downsample_fraction = 0.1
downsampled_data = flight_data.sample(frac=downsample_fraction, random_state=42)
downsampled_results = results.loc[downsampled_data.index]
downsampled_data = downsampled_data[downsampled_data["powered"]=="powered"]
downsampled_results = downsampled_results.loc[downsampled_data.index]

# plt.plot(results["radius_turn"])
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define data
x_full = -downsampled_data["kcu_actual_steering_delay"] / 100
y_full = downsampled_results["wing_sideforce_coefficient"]

plt.plot(flight_data["us"])
plt.show()
# Define the regions
# threshold = np.radians(10)
# var_threshold = downsampled_data["kite_yaw_rate"]
# mask_straight = abs(var_threshold) < threshold
# mask_left = (abs(var_threshold) > threshold) & (x_full < 0)
# mask_right = (abs(var_threshold) > threshold) & (x_full > 0)
upper_threshold = 0.08
lower_threshold = -0.06
mask_straight = x_full.between(lower_threshold, upper_threshold)
mask_left = x_full < lower_threshold
mask_right = x_full > upper_threshold


# Compute linear regressions for each region
slope_straight, intercept_straight, r_straight, _, _ = linregress(x_full[mask_straight], y_full[mask_straight])
slope_right , intercept_right, r_right, _, _ = linregress(x_full[mask_right], y_full[mask_right])
slope_left , intercept_left, r_left, _, _ = linregress(x_full[mask_left], y_full[mask_left])

print("R^2 Straight: ", r_straight**2)
print("R^2 Right: ", r_right**2)
print("R^2 Left: ", r_left**2)

# Generate regression lines for each region
x_line_straight = np.linspace(x_full[mask_straight].min(), x_full[mask_straight].max(), 100)
y_line_straight = slope_straight * x_line_straight + intercept_straight

x_line_right = np.linspace(x_full[mask_right].min(), x_full[mask_right].max(), 100)
y_line_right = slope_right * x_line_right + intercept_right

x_line_left = np.linspace(x_full[mask_left].min(), x_full[mask_left].max(), 100)
y_line_left = slope_left * x_line_left + intercept_left


# Plot data points and regression lines
plt.figure(figsize=(5,4))

# Scatter plot for straight, left, and right regions
plt.scatter(-downsampled_data["kcu_actual_steering"] / 100, downsampled_results["wing_sideforce_coefficient"], alpha=0.2, label="No Delay Correction", color='gray', marker='.')
plt.scatter(x_full[mask_straight], y_full[mask_straight], alpha=0.4, label="Straight (Corrected)", color=colors[1], marker='.')
plt.scatter(x_full[mask_left], y_full[mask_left], alpha=0.4, label="Left Turn (Corrected)", color=colors[2], marker='.')
plt.scatter(x_full[mask_right], y_full[mask_right], alpha=0.4, label="Right Turn (Corrected)", color=colors[3], marker='.')



# Plot regression lines for each region
# plt.plot(x_line_straight, y_line_straight, label=f"Straight fit: y = {slope_straight:.2f}x + {intercept_straight:.2f}", color="blue", linestyle="-")
# plt.plot(x_line_right, y_line_right, label=f"Right fit: y = {slope_right:.2f}x + {intercept_right:.2f}", color="red", linestyle="-")
# plt.plot(x_line_left, y_line_left, label=f"Left fit: y = {slope_left:.2f}x + {intercept_left:.2f}", color="green", linestyle="-")

# Add labels and legend
plt.xlabel("$u_s$")
plt.ylabel("$C_S$")
plt.legend(frameon=True)
plt.tight_layout()
plt.savefig("./results/plots_paper/sideforce_three_regions.pdf")
# plt.show()

# flight_data["kite_yaw_rate"] = np.gradient(np.unwrap(flight_data["kite_yaw_0"]), ts)
yaw_rate, coeffs = calculate_turn_rate_law(results, flight_data, model = "simple", steering_offset=False)
yaw_rate_weight, coeffs_weight = calculate_turn_rate_law(results, flight_data, model = "simple", steering_offset=True)

# Calculate mean errors
error = abs(np.degrees(yaw_rate) - np.degrees(flight_data["kite_yaw_rate"]))
error_weight = abs(np.degrees(yaw_rate_weight) - np.degrees(flight_data["kite_yaw_rate"]))
mean_error = np.mean(error)
mean_error_weight = np.mean(error_weight)
# Calculate R^2 values
r_squared = 1 - np.sum(error**2) / np.sum((np.degrees(flight_data["kite_yaw_rate"]) - np.mean(np.degrees(flight_data["kite_yaw_rate"])))**2)
r_squared_weight = 1 - np.sum(error_weight**2) / np.sum((np.degrees(flight_data["kite_yaw_rate"]) - np.mean(np.degrees(flight_data["kite_yaw_rate"])))**2)

print("r_squared: ", r_squared)
print("r_squared_weight: ", r_squared_weight)

# Prepare data
x = -downsampled_data["kcu_actual_steering"] / 100 * downsampled_results["kite_apparent_windspeed"]
y = downsampled_data["kite_yaw_rate"]

# Calculate the point density
xy = np.vstack([x, y])
# z = gaussian_kde(xy)(xy)
# 
# Create KDE plot
plt.figure(figsize=(5,4))
# scatter = plt.scatter(
#     x, y, 
#     c=z,  # Use density values for coloring
#     s=20,  # Adjust point size if needed
#     cmap='viridis',  # Color map (adjust to preference)
#     alpha=0.6  # Transparency level
# )
plt.scatter(
    x[mask_straight],
    y[mask_straight]*180/np.pi,
    color = colors[1],
    alpha = 0.4,
    marker='.',
    label = "Straight"
)
plt.scatter(
    x[mask_left],
    y[mask_left]*180/np.pi,
    color = colors[2],
    marker='.',
    alpha = 0.4,
    label = "Left Turn"
)
plt.scatter(
    x[mask_right],
    y[mask_right]*180/np.pi,
    color = colors[3],
    alpha = 0.4,
    marker='.',
    label = "Right Turn"
)


x_line = np.linspace(x.min(), x.max(), 100)
y_line = coeffs[0] * x_line
# Overlay the line plot
plt.plot(
    -x_line, 
    y_line*180/np.pi,
    label=f'Identified Yaw Rate (R$^2$: {r_squared:.2f})',
    color=colors[0], 
    linestyle="--"
)
A = np.vstack([x_line, 20*np.ones_like(x_line)]).T
y_line = A@coeffs_weight
plt.plot(
    -x_line,
    y_line*180/np.pi,
    label=f'Offset-Corrected Yaw Rate (R$^2$: {r_squared_weight:.2f})',
    color = colors[0],
    linestyle = ':'
)

plt.xlabel(r'$u_\mathrm{s} \cdot v_\mathrm{a}$ (m s$^{-1}$)')
plt.ylabel('$\dot{\psi}$ ($^\circ$ s$^{-1}$)')
plt.legend(frameon=True)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig("./results/plots_paper/yaw_rate_three_regions.pdf")
plt.show()


# Define `x` and ensure alignment with the mask for "kcu_actual_steering_delay"
x_delay = -flight_data.loc[mask, "kcu_actual_steering_delay"] / 100
y = results.loc[mask, "wing_sideforce_coefficient"]

# Define the regions for "kcu_actual_steering_delay"
mask_straight_delay = x_delay.between(lower_threshold, upper_threshold)
mask_left_delay = x_delay < lower_threshold
mask_right_delay = x_delay > upper_threshold

# Compute linear regressions for each region for "kcu_actual_steering_delay"
slope_straight_delay, intercept_straight_delay, _, _, _ = linregress(x_delay[mask_straight_delay], y[mask_straight_delay])
slope_right_delay, intercept_right_delay, _, _, _ = linregress(x_delay[mask_right_delay], y[mask_right_delay])
slope_left_delay, intercept_left_delay, _, _, _ = linregress(x_delay[mask_left_delay], y[mask_left_delay])

# Fit lines for each region for "kcu_actual_steering_delay"
fit_straight_delay = pd.Series(slope_straight_delay * x_delay[mask_straight_delay] + intercept_straight_delay, index=x_delay[mask_straight_delay].index)
fit_right_delay = pd.Series(slope_right_delay * x_delay[mask_right_delay] + intercept_right_delay, index=x_delay[mask_right_delay].index)
fit_left_delay = pd.Series(slope_left_delay * x_delay[mask_left_delay] + intercept_left_delay, index=x_delay[mask_left_delay].index)

# Combine the fits for "kcu_actual_steering_delay"
combined_fit_delay = pd.concat([fit_straight_delay, fit_right_delay, fit_left_delay]).sort_index()

# Define `x` and ensure alignment with the mask for "kcu_actual_steering"
x_steering = -flight_data.loc[mask, "kcu_actual_steering"] / 100

mask_straight_steering = x_steering.between(lower_threshold, upper_threshold)
mask_left_steering = x_steering < lower_threshold
mask_right_steering = x_steering > upper_threshold

# Compute linear regressions for each region for "kcu_actual_steering"
slope_straight_steering, intercept_straight_steering, _, _, _ = linregress(x_steering[mask_straight_steering], y[mask_straight_steering])
slope_right_steering , intercept_right_steering, _, _, _ = linregress(x_steering[mask_right_steering], y[mask_right_steering])
slope_left_steering , intercept_left_steering, _, _, _ = linregress(x_steering[mask_left_steering], y[mask_left_steering])

# Fit lines for each region for "kcu_actual_steering"
fit_straight_steering = pd.Series(slope_straight_steering * x_steering[mask_straight_steering] + intercept_straight_steering, index=x_steering[mask_straight_steering].index)
fit_right_steering = pd.Series(slope_right_steering * x_steering[mask_right_steering] + intercept_right_steering, index=x_steering[mask_right_steering].index)
fit_left_steering = pd.Series(slope_left_steering * x_steering[mask_left_steering] + intercept_left_steering, index=x_steering[mask_left_steering].index)

# Combine the fits for "kcu_actual_steering"
combined_fit_steering = pd.concat([fit_straight_steering, fit_right_steering, fit_left_steering]).sort_index()
# Plot the time series
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

# Plot the original time series
plot_time_series(
    flight_data.loc[mask], 
    y, 
    axs[0], 
    ylabel="$C_{S}$", 
    plot_phase=False, 
    color=colors[0], 
    label="EKF 0"
)

# Plot the combined linear fit for "kcu_actual_steering_delay"
axs[0].plot(
    flight_data.loc[mask, "time"], 
    combined_fit_delay, 
    label="Linear Fit (Delay)", 
    color=colors[1], 
    linestyle="--"
)

# Plot the combined linear fit for "kcu_actual_steering"
axs[0].plot(
    flight_data.loc[mask, "time"], 
    combined_fit_steering, 
    label="Linear Fit (Steering)", 
    color=colors[2], 
    linestyle=":"
)

# Finalize the plot
plt.tight_layout()
plt.legend(frameon = True)
plt.show()

