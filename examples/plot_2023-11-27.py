import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
from awes_ekf.plotting.plot_kite_trajectory import calculate_azimuth_elevation
from awes_ekf.setup.settings import kappa, z0
from awes_ekf.plotting.color_palette import get_color_list, visualize_palette, set_plot_style, get_color

def cut_data(results, flight_data, range):
    results = results.iloc[range[0]:range[1]]
    flight_data = flight_data.iloc[range[0]:range[1]]
    results = results.reset_index(drop=True)
    flight_data = flight_data.reset_index(drop=True)
    return results, flight_data

set_plot_style()
year = "2023"
month = "11"
day = "27"
kite_model = "v9"

colors = get_color_list()

results, flight_data,config_data = read_results(year, month, day, kite_model,addition='_lt')
res_va, fd_va, _ = read_results(year, month, day, kite_model,addition='_va')
res_log, fd_log, _ = read_results(year, month, day, kite_model,addition='_log')
res_min, fd_min, config = read_results(year, month, day, kite_model,addition='_min')

print(config['simulation_parameters']['measurements'])
#%% Plot orientation
from awes_ekf.postprocess.postprocessing import remove_offsets_IMU_data
from awes_ekf.plotting.plot_orientation import plot_kite_orientation

mask = flight_data["cycle"].isin([64,65])
mask_min = fd_min["cycle"].isin([64,65])
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
yaw_rate = np.gradient(flight_data["kite_yaw_0"], config_data["simulation_parameters"]["timestep"])
yaw_rate = np.convolve(yaw_rate, np.ones(10)/10, mode='same')
plt.plot(flight_data[mask]["time"], np.degrees(yaw_rate[mask]))
pu.plot_time_series(flight_data[mask], flight_data[mask]["kcu_actual_steering"], ax,plot_phase = True)
# plt.show()
flight_data["kite_yaw_rate"] = yaw_rate
from awes_ekf.utils import calculate_turn_rate_law
yaw_rate, coeffs = calculate_turn_rate_law(results, flight_data, model = "simple", steering_offset=False)
yaw_rate_weight, coeffs_weight = calculate_turn_rate_law(results, flight_data, model = "simple_weight", steering_offset=True)


# Calculate mean errors
error = abs(np.degrees(yaw_rate) - np.degrees(flight_data["kite_yaw_rate"]))
error_weight = abs(np.degrees(yaw_rate_weight) - np.degrees(flight_data["kite_yaw_rate"]))
mean_error = np.mean(error)
mean_error_weight = np.mean(error_weight)


# Create the scatter plot
plt.figure(figsize=(6, 4))
plt.scatter(
    np.degrees(results["kite_apparent_windspeed"]*flight_data["kcu_actual_steering"]),
    np.degrees(flight_data["kite_yaw_rate"]), 
    label=f'Identified Yaw Rate (Mean Error: {mean_error:.2f} deg/s)', 
    color=colors[1], 
    alpha=0.2
)
A = np.vstack([results["kite_apparent_windspeed"]*flight_data["kcu_actual_steering"]]).T
yaw_rate = A@coeffs

plt.scatter(
    np.degrees(results["kite_apparent_windspeed"]*flight_data["kcu_actual_steering"]),
    yaw_rate, 
    label=f'Offset-Corrected Yaw Rate (Mean Error: {mean_error_weight:.2f} deg/s)', 
    color=colors[2], 
    alpha=0.2
)

# # Plot the truth line
# plt.plot(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100), color=colors[0], linestyle='--')

# Labels, legend, and grid
plt.ylabel('Identified Yaw Rate [deg/s]')
plt.xlabel('Measured Yaw Rate [deg/s]')
plt.legend(frameon = True)
plt.grid(True)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig("./results/plots_paper/yaw_rate_2023-11-27.pdf")



# Print mean errors and standard deviations
print("Mean error yaw rate: ", mean_error)
print("Mean error yaw rate weight: ", mean_error_weight)
print("Std error yaw rate: ", np.std(error))
print("Std error yaw rate weight: ", np.std(error_weight))

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
pu.plot_time_series(flight_data[mask], results[mask]["norm_epsilon_norm"], ax, plot_phase=True, color = colors[0])
ax.legend()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Norm of Normalized Residuals")
plt.tight_layout()
plt.savefig("./results/plots_paper/norm_residuals_2023-11-27.pdf")
from matplotlib.patches import Patch
# Create a new patch for the legend
reel_out_straight_patch = Patch(color=colors[5], alpha=0.2, label="Reel-out - Straight")
reel_out_turn_patch = Patch(color=colors[7], alpha=0.2, label="Reel-out - Turn")
reel_in_patch = Patch(color='white', alpha=1, label="Reel-in")

# Select starting from the second element
ax.legend(
    [reel_out_straight_patch, reel_out_turn_patch, reel_in_patch],
    ["Reel-out - Straight", "Reel-out - Turn", "Reel-in"],
    loc='upper left',
    frameon=True,
    bbox_to_anchor=(0.075, 1)  # Adjust the x-coordinate to move the legend to the right
)
plt.show()

results["kite_yaw_kin"] = np.unwrap(results["kite_yaw_kin"])
plot_kite_orientation(results[mask], flight_data[mask], kite_imus=config_data["kite"]["sensor_ids"])
plt.savefig("./results/plots_paper/kite_orientation_2023-11-27.pdf")
plt.show()

colors = get_color_list()
# Plot position and velocity
from awes_ekf.plotting.plot_kite_trajectory import plot_position_azimuth_elevation
fig, axs = plt.subplots(2, 1, figsize=(6, 10))
mean_wind_dir = np.mean(results[mask]["wind_direction"])
azimuth, elevation = calculate_azimuth_elevation(res_min[mask_min]["kite_position_x"], res_min[mask_min]["kite_position_y"], res_min[mask_min]["kite_position_z"])
axs[0].plot(np.rad2deg(azimuth%(2*np.pi)-mean_wind_dir), np.rad2deg(elevation), label="EKF 0", color = colors[0])
azimuth, elevation = calculate_azimuth_elevation(results[mask]["kite_position_x"], results[mask]["kite_position_y"], results[mask]["kite_position_z"])
axs[0].plot(np.rad2deg(azimuth%(2*np.pi)-mean_wind_dir), np.rad2deg(elevation), label="EKF 1", color = colors[1])
azimuth, elevation = calculate_azimuth_elevation(flight_data[mask]["kite_position_x"], flight_data[mask]["kite_position_y"], flight_data[mask]["kite_position_z"])
axs[0].plot(np.rad2deg(azimuth%(2*np.pi)-mean_wind_dir), np.rad2deg(elevation), label="GPS", color = colors[2])
axs[0].legend()
axs[0].set_xlabel("Azimuth [deg]")
axs[0].set_ylabel("Elevation [deg]")
r = np.sqrt(res_min[mask_min]["kite_position_x"]**2 + res_min[mask_min]["kite_position_y"]**2+ res_min[mask_min]["kite_position_z"]**2)
axs[1].plot(flight_data[mask]["time"], r, label="EKF 0 ", color = colors[0])
r = np.sqrt(results[mask]["kite_position_x"]**2 + results[mask]["kite_position_y"]**2+ results[mask]["kite_position_z"]**2)
axs[1].plot(flight_data[mask]["time"], r, label="EKF 1", color = colors[1])
r = np.sqrt(flight_data[mask]["kite_position_x"]**2 + flight_data[mask]["kite_position_y"]**2+ flight_data[mask]["kite_position_z"]**2)
axs[1].plot(flight_data[mask]["time"], r, label="GPS+IMU", color = colors[2])
axs[1].plot(flight_data[mask]["time"], flight_data[mask]["tether_length"]+15.55, label="Measured tether length", color = colors[3])
# axs[1].plot(fd_min[mask_min]["time"], res_min[mask_min]["tether_length"]+15.55, label="Min. measurements", color = colors[4])
axs[1].legend()
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Radial Distance/Tether Length [m]")
plt.tight_layout()
plt.savefig("./results/plots_paper/kite_trajectory_2023-11-27.pdf")
plt.show()

cut = 1000
results, flight_data = cut_data(results, flight_data, [cut, -cut])
res_va, fd_va = cut_data(res_va, fd_va, [cut, -cut])
res_log, fd_log = cut_data(res_log, fd_log, [cut, -cut])
res_min, fd_min = cut_data(res_min, fd_min, [cut, -cut])

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
pu.plot_kinetic_energy_spectrum(res_va, fd_va, ax, savefig=False)
plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# # pu.plot_turbulence_intensity(results, flight_data, 140, ax)
# pu.plot_turbulence_intensity(res_min, fd_min, 140, ax)
# plt.savefig("./results/plots_paper/turbulence_intensity_2023-11-27.pdf")
# plt.show()

chunk_size = 6000  # Number of rows in each subset
num_subsets = 6
# Divide the DataFrames into 6 nearly equal subsets
results_subsets = [results.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_subsets)]
flight_data_subsets = [flight_data.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_subsets)]

res_va_subsets = [res_va.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_subsets)]
fd_va_subsets = [fd_va.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_subsets)]

res_log_subsets = [res_log.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_subsets)]
fd_log_subsets = [fd_log.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_subsets)]

res_min_subsets = [res_min.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_subsets)]
fd_min_subsets = [fd_min.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_subsets)]

# Create a figure with subplots for each subset
fig, axs = plt.subplots(2, num_subsets, figsize=(16,6))  # Adjust the layout as needed

# Flatten axs for easy iteration if using a 2D grid of subplots
axs = axs.flatten()
colors = get_color_list()
# Loop through each subset and plot wind profiles
for i in range(num_subsets):
    # Plot wind profile bins
    pu.plot_wind_profile_bins(fd_min_subsets[i], res_min_subsets[i], [axs[i], axs[i+num_subsets]], step=10, color=colors[1], label="EKF 0", lidar_data=False)
    pu.plot_wind_profile_bins(flight_data_subsets[i], results_subsets[i], [axs[i], axs[i+num_subsets]], step=10, color=colors[2], lidar_data=False, label="EKF 1")
    pu.plot_wind_profile_bins(fd_log_subsets[i], res_log_subsets[i], [axs[i], axs[i+num_subsets]], step=10, color=colors[4], label="EKF 2", lidar_data=False)
    pu.plot_wind_profile_bins(fd_va_subsets[i], res_va_subsets[i], [axs[i], axs[i+num_subsets]], step=10, color=colors[3], label="EKF 3", lidar_data=True)
    
    
    
    # Extract and round time to the nearest 5-minute interval
    original_time = flight_data_subsets[i]["time_of_day"].iloc[0]
    rounded_time = (pd.to_datetime(original_time) + pd.Timedelta(minutes=2.5)).floor('5T').time()
    
    # Set the title with the rounded time
    axs[i].set_title(rounded_time.strftime("%H:%M:%S"))
    axs[i].legend().remove()
    axs[i+num_subsets].legend().remove()
    axs[i].set_xlim([5,15])
    axs[i+num_subsets].set_xlim([220,260])
    axs[i].set_ylim([0, 300])
    axs[i+num_subsets].set_ylim([0, 300])
    # Print the subset and rounded time
    print(f"Subset {i+1}: {rounded_time}")

axs[0].legend(loc="lower right", frameon=True)

# Adjust the layout and show the plot
plt.tight_layout()
# Save as a PDF file
plt.savefig("./results/plots_paper/wind_profiles_2023-11-27.pdf")
plt.show()

t0 = 12600
dt = 12000
results, flight_data = cut_data(results, flight_data, [t0, t0+dt])
flight_data = pu.interpolate_lidar_data(flight_data, results)
res_va, fd_va = cut_data(res_va, fd_va, [t0, t0+dt])
res_log, fd_log = cut_data(res_log, fd_log, [t0, t0+dt])
res_min, fd_min = cut_data(res_min, fd_min, [t0, t0+dt])



fig, axs = plt.subplots(3, 1, figsize=(16, 6), sharex=True)
axs[0].plot(fd_min["time"], res_min["wind_speed_horizontal"], label="EKF 0", color = colors[1])
axs[0].plot(flight_data["time"], results["wind_speed_horizontal"], label="EKF 1", color = colors[2])
axs[0].plot(fd_log["time"], res_log["wind_speed_horizontal"], label="EKF 2", color = colors[4])
axs[0].plot(fd_va["time"], res_va["wind_speed_horizontal"], label="EKF 3", color = colors[3])
axs[0].plot(flight_data["time"], flight_data["interp_wind_speed"], label="Lidar", color = colors[0])
axs[0].fill_between(
                    flight_data["time"],
                    flight_data["interp_wind_speed_min"],
                    flight_data["interp_wind_speed_max"],
                    color=colors[0],
                    alpha=0.3,
                )
uf = flight_data["ground_wind_speed"]*kappa/np.log(10/z0)
vel_ground = uf/kappa*np.log(results["kite_position_z"]/z0)
axs[0].plot(flight_data["time"], vel_ground, label="Ground", color = colors[5])
axs[0].set_ylabel("Wind speed [m/s]")
# axs[0].legend()

axs[1].plot(fd_min["time"], res_min["wind_direction"]*180/np.pi, label="Min. measurements", color = colors[1])
axs[1].plot(flight_data["time"], results["wind_direction"]*180/np.pi, label="Min. + $l_t$", color = colors[2])
axs[1].plot(fd_log["time"], res_log["wind_direction"]*180/np.pi, label="Min. with Log profile", color = colors[4])
axs[1].plot(fd_va["time"], res_va["wind_direction"]*180/np.pi, label="Min. + $v_a$", color = colors[3])
axs[1].plot(flight_data["time"], 270-flight_data["interp_wind_direction"], label="Lidar", color = colors[0])

axs[1].plot(flight_data["time"], flight_data["ground_wind_direction"], label="Ground", color = colors[5])
axs[1].set_ylim([160, 280])
axs[1].set_ylabel("Wind direction [deg]")


axs[2].plot(fd_min["time"], res_min["wind_speed_vertical"], label="EKF 0", color = colors[1])
axs[2].plot(flight_data["time"], results["wind_speed_vertical"], label="EKF 1", color = colors[2])
axs[2].plot(fd_log["time"], res_log["wind_speed_vertical"], label="EKF 2", color = colors[4])
axs[2].plot(fd_va["time"], res_va["wind_speed_vertical"], label="EKF 3", color = colors[3])
axs[2].plot(flight_data["time"], flight_data["interp_z_wind"], label="Lidar",   color = colors[0])
axs[2].set_ylabel("Vertical wind speed [m/s]")
axs[2].set_xlabel("Time [s]")
axs[2].legend(frameon=True)

axs[0].set_xlim([flight_data["time"].iloc[0], flight_data["time"].iloc[-1]])
time_of_day = pd.to_datetime(flight_data["time_of_day"]).dt.strftime("%H:%M")

# Determine x-ticks: 4 evenly spaced points + the last time point
tick_indices = list(range(0, len(flight_data), len(flight_data)//4))  # 4 evenly spaced ticks
if tick_indices[-1] != len(flight_data) - 1:  # Ensure the last tick is the very last time point
    tick_indices.append(len(flight_data) - 1)

# Set x-ticks and labels
axs[2].set_xticks(flight_data["time"].iloc[tick_indices])
axs[2].set_xticklabels(time_of_day.iloc[tick_indices], rotation=45, ha="right")

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig("./results/plots_paper/wind_speed_2023-11-27.pdf")
plt.show()



