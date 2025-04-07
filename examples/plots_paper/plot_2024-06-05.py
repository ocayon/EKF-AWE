import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style
from awes_ekf.plotting.plot_kinematics import calculate_azimuth_elevation

# Set plot style
set_plot_style()

# Define the parameters
year = "2024"
month = "06"
day = "05"
kite_model = "v9"

# Read all four datasets in the order specified
results_min, flight_data_min, _ = read_results(year, month, day, kite_model, addition='_min')
results_tether, flight_data_tether, _ = read_results(year, month, day, kite_model, addition='_tether')
results_vwz0, flight_data_vwz0, _ = read_results(year, month, day, kite_model, addition='_vwz0')
results_log, flight_data_log, _ = read_results(year, month, day, kite_model, addition='_log')

azimuth, elevation = calculate_azimuth_elevation(results_min["kite_position_x"], results_min["kite_position_y"], results_min["kite_position_z"])
plt.plot(azimuth, label="Kite Position Z")
plt.show()


# Apply necessary adjustments (for wind direction correction, indexing, etc.)
def adjust_results(results):
    results.loc[results['wind_direction'] > np.radians(250), 'wind_direction'] -= np.radians(360)
    return results

# Correct wind directions
results_min = adjust_results(results_min)
results_tether = adjust_results(results_tether)
results_vwz0 = adjust_results(results_vwz0)
results_log = adjust_results(results_log)

# Add cycle information from flight data
results_min["cycle"] = flight_data_min["cycle"]
results_tether["cycle"] = flight_data_tether["cycle"]
results_vwz0["cycle"] = flight_data_vwz0["cycle"]
results_log["cycle"] = flight_data_log["cycle"]

heights = [70,200]
error_wind = pu.calculate_error_10_minute(flight_data_vwz0, results_vwz0, heights=heights)
error_wind = pu.calculate_error_10_minute(flight_data_min, results_min, heights=heights)
error_wind = pu.calculate_error_10_minute(flight_data_tether, results_tether, heights=heights)
error_wind = pu.calculate_error_10_minute(flight_data_log, results_log, heights=heights)


flight_data_tether["time"] = flight_data_tether["unix_time"] - flight_data_tether["unix_time"].iloc[0]
flight_data_min["time"] = flight_data_min["unix_time"] - flight_data_min["unix_time"].iloc[0]
flight_data_vwz0["time"] = flight_data_vwz0["unix_time"] - flight_data_vwz0["unix_time"].iloc[0]
flight_data_log["time"] = flight_data_log["unix_time"] - flight_data_log["unix_time"].iloc[0]

# Cut data to reduce unwanted initial and ending data
cut = 6000
datasets = [results_min, results_tether, results_vwz0, results_log,
            flight_data_min, flight_data_tether, flight_data_vwz0, flight_data_log]

# Apply cuts and reset index
datasets = [data.iloc[cut:-cut].reset_index(drop=True) for data in datasets]
results_min, results_tether, results_vwz0, results_log, \
flight_data_min, flight_data_tether, flight_data_vwz0, flight_data_log = datasets

# Define colors
colors = get_color_list()

### Step 1: Initial Plots
# plt.figure()
# acc = np.sqrt(flight_data_min["kite_acceleration_x"]**2 + flight_data_min["kite_acceleration_y"]**2 + flight_data_min["kite_acceleration_z"]**2)
# plt.plot(flight_data_min["time"], flight_data_min["kite_acceleration_z"], label="Kite Acceleration Z")
# plt.plot(flight_data_min["time"], flight_data_min["kite_velocity_z"], label="Kite Velocity Z")
# plt.plot(flight_data_min["time"], flight_data_min["kite_position_z"], label="Kite Position Z")
# plt.xlabel("Time [s]")
# plt.legend()
# plt.show()
# print(flight_data_tether.columns)

# Calculate mean error from flight_data position and results position
error = np.sqrt((flight_data_tether["kite_position_x"] - results_tether["kite_position_x"])**2 + (flight_data_tether["kite_position_y"] - results_tether["kite_position_y"])**2 + (flight_data_tether["kite_position_z"] - results_tether["kite_position_z"])**2)
print(np.mean(error))

for col in flight_data_tether.columns:
    if "time" in col:
        print(col)
print(flight_data_tether["cycle"].iloc[-1])
mask = flight_data_tether["cycle"].isin([28])
mask_min = flight_data_min["cycle"].isin([28])
# Plot position and velocity
fig, axs = plt.subplots(2, 1, figsize=(5, 8))
mean_wind_dir = np.mean(results_tether[mask]["wind_direction"])
azimuth, elevation = calculate_azimuth_elevation(results_min[mask_min]["kite_position_x"], results_min[mask_min]["kite_position_y"], results_min[mask_min]["kite_position_z"])
axs[0].plot(np.rad2deg(azimuth-mean_wind_dir), np.rad2deg(elevation), label="EKF 0", color = colors[0])
azimuth, elevation = calculate_azimuth_elevation(results_tether[mask]["kite_position_x"], results_tether[mask]["kite_position_y"], results_tether[mask]["kite_position_z"])
axs[0].plot(np.rad2deg(azimuth-mean_wind_dir), np.rad2deg(elevation), label="EKF 5", color = colors[1])
azimuth, elevation = calculate_azimuth_elevation(flight_data_tether[mask]["kite_position_x"], flight_data_tether[mask]["kite_position_y"], flight_data_tether[mask]["kite_position_z"])
axs[0].plot(np.rad2deg(azimuth-mean_wind_dir), np.rad2deg(elevation), label="GPS", color = colors[2])
axs[0].legend()
axs[0].set_xlabel(f"Azimuth ($^\circ$)")
axs[0].set_ylabel(f"Elevation ($^\circ$)")
axs[0].set_xlim([-100, 60])
axs[0].set_ylim([10, 90])
# plt.show()
r = np.sqrt(results_min[mask_min]["kite_position_x"]**2 + results_min[mask_min]["kite_position_y"]**2+ results_min[mask_min]["kite_position_z"]**2)
axs[1].plot(flight_data_tether[mask]["unix_time"], r, label="EKF 0 ", color = colors[0])
r = np.sqrt(results_tether[mask]["kite_position_x"]**2 + results_tether[mask]["kite_position_y"]**2+ results_tether[mask]["kite_position_z"]**2)
axs[1].plot(flight_data_tether[mask]["unix_time"], r, label="EKF 5", color = colors[1])
r = np.sqrt(flight_data_tether[mask]["kite_position_x"]**2 + flight_data_tether[mask]["kite_position_y"]**2+ flight_data_tether[mask]["kite_position_z"]**2)
axs[1].plot(flight_data_tether[mask]["unix_time"], r, label="GPS+IMU", color = colors[2])
axs[1].plot(flight_data_tether[mask]["unix_time"], flight_data_tether[mask]["tether_length"]+15.55, label="Measured tether length", color = colors[3])
# axs[1].plot(fd_min[mask_min]["time"], results_min[mask_min]["tether_length"]+15.55, label="Min. measurements", color = colors[4])
axs[1].legend()
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Radial Distance/Tether Length (m)")
axs[1].set_ylim([200, 360])
plt.tight_layout()
plt.savefig("./results/plots_paper/kite_trajectory_2023-11-27.pdf")
plt.show()

# fig, ax = plt.subplots(1, 1, figsize=(10, 6))
# pu.plot_kinetic_energy_spectrum(results_min, flight_data_min, ax, savefig=False)
# plt.show()

### Step 2: Turbulence Intensity Plot
# fig, ax = plt.subplots(1, 1, figsize=(6,4))
# pu.plot_turbulence_intensity_high_res(results_min, flight_data_min, 120, ax, savefig=False)

# plt.savefig("./results/plots_paper/turbulence_intensity_2024-06-05.pdf")
# plt.show()

### Step 3: Chunking and Plotting Wind Profiles
chunk_size = 3200  # Number of rows in each subset
num_subsets = 6

def split_into_chunks(data, size, num):
    return [data.iloc[i * size:(i + 1) * size] for i in range(num)]

# Divide each dataset into subsets
subsets = {
    "results_min": split_into_chunks(results_min, chunk_size, num_subsets),
    "results_tether": split_into_chunks(results_tether, chunk_size, num_subsets),
    "results_vwz0": split_into_chunks(results_vwz0, chunk_size, num_subsets),
    "results_log": split_into_chunks(results_log, chunk_size, num_subsets),
    "flight_data_min": split_into_chunks(flight_data_min, chunk_size, num_subsets),
    "flight_data_tether": split_into_chunks(flight_data_tether, chunk_size, num_subsets),
    "flight_data_vwz0": split_into_chunks(flight_data_vwz0, chunk_size, num_subsets),
    "flight_data_log": split_into_chunks(flight_data_log, chunk_size, num_subsets)
}

# Create a figure with subplots for each subset
# Create a figure with subplots for each subset
fig, axs = plt.subplots(2, num_subsets, figsize=(10, 4), sharey=True, 
                        )
plt.subplots_adjust(left=0.05, right=0.95, wspace=0.2, hspace=0.4)
fig.text(0.5, 0.49, 'Wind speed (m s$^{-1}$)', ha='center', va='center', fontsize=12)
fig.text(0.5, 0.02, 'Wind direction ($^\circ$)', ha='center', va='center', fontsize=12)
axs = axs.flatten()
# Loop through each subset and plot wind profiles
for i in range(num_subsets):
    if i == 0:
        ylabel = "Height (m)"
    else:
        ylabel = None
    # Plot wind profile bins for all datasets
    pu.plot_wind_profile_bins(subsets["flight_data_min"][i], subsets["results_min"][i], [axs[i], axs[i + 6]], step=10, color=colors[1], lidar_data=False, label="EKF 0")
    pu.plot_wind_profile_bins(subsets["flight_data_log"][i], subsets["results_log"][i], [axs[i], axs[i + 6]], step=10, color=colors[4], lidar_data=False, label="EKF 2")
    pu.plot_wind_profile_bins(subsets["flight_data_vwz0"][i], subsets["results_vwz0"][i], [axs[i], axs[i + 6]], step=10, color=colors[3], lidar_data=False, label="EKF 4")
    pu.plot_wind_profile_bins(subsets["flight_data_tether"][i], subsets["results_tether"][i], [axs[i], axs[i + 6]], step=10, color=colors[2], lidar_data=True, label="EKF 5", ylabel=ylabel)
    
    # Extract and round time to the nearest 5-minute interval
    original_time = subsets["flight_data_min"][i]["time_of_day"].iloc[0]
    rounded_time = (pd.to_datetime(original_time) + pd.Timedelta(minutes=2.5)).floor('5T').time()
    
    # Set the title with the rounded time
    axs[i].set_title(rounded_time.strftime("%H:%M:%S"))
    axs[i].legend().remove()
    axs[i + 6].legend().remove()
    axs[i].set_xlim([5, 15])
    axs[i + 6].set_xlim([-60, 20])
    print(f"Subset {i+1}: {rounded_time}")

# Add legends and finalize layout
axs[0].legend(loc="upper center", bbox_to_anchor=(3.5, 1.5), ncol=5, frameon=False)
plt.subplots_adjust(top=0.85)  # Increase this value to push down the plot

plt.savefig("./results/plots_paper/wind_profiles_2024-06-05.pdf")
plt.show()

### Step 4: Plot Wind Speed Comparison
fig, axs = plt.subplots(3, 1, figsize=(9,6), sharex=True)
pu.plot_wind_speed(results_min, flight_data_min, axs, lidar_heights=[250,100,160], color=colors[1], label_ekf="EKF 0") 
pu.plot_wind_speed(results_log, flight_data_log, axs, lidar_data=False, color=colors[4], label_ekf="EKF 2")
pu.plot_wind_speed(results_vwz0, flight_data_vwz0, axs, lidar_data=False, color=colors[3], label_ekf="EKF 4")
pu.plot_wind_speed(results_tether, flight_data_tether, axs, lidar_data=False, color=colors[2], label_ekf="EKF 5")



# Add legend from axs[0] and position it above the entire figure
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1), ncol=4, frameon=False)

axs[1].set_ylim([-60, 20])
axs[2].set_ylim([-5, 5])
axs[0].set_xlim([flight_data_min["time"].iloc[0], flight_data_min["time"].iloc[-1]])
time_of_day = pd.to_datetime(flight_data_min["time_of_day"]).dt.strftime("%H:%M")
tick_indices = list(range(0, len(flight_data_min), len(flight_data_min)//6))
if tick_indices[-1] != len(flight_data_min) - 1:
    tick_indices.append(len(flight_data_min) - 1)

axs[2].set_xticks(flight_data_min["time"].iloc[tick_indices])
axs[2].set_xticklabels(time_of_day.iloc[tick_indices], rotation=45, ha="right")
axs[2].set_xlabel("Time of day (hh:mm)")

# Adjust layout to make space for top legend
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("./results/plots_paper/wind_speed_2024-06-05.pdf")
plt.show()
