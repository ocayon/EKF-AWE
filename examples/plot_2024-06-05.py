import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style

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
plt.figure()
acc = np.sqrt(flight_data_min["kite_acceleration_x"]**2 + flight_data_min["kite_acceleration_y"]**2 + flight_data_min["kite_acceleration_z"]**2)
plt.plot(flight_data_min["time"], flight_data_min["kite_acceleration_z"], label="Kite Acceleration Z")
plt.plot(flight_data_min["time"], flight_data_min["kite_velocity_z"], label="Kite Velocity Z")
plt.plot(flight_data_min["time"], flight_data_min["kite_position_z"], label="Kite Position Z")
plt.xlabel("Time [s]")
plt.legend()
plt.show()

### Step 2: Turbulence Intensity Plot
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
pu.plot_turbulence_intensity_high_res(results_min, flight_data_min, 120, ax, savefig=False)
ax.set_ylim([0, 0.4])
plt.savefig("./results/plots_paper/turbulence_intensity_2024-06-05.pdf")
plt.show()

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
fig, axs = plt.subplots(2, 6, figsize=(16, 6))  # Adjust the layout as needed
axs = axs.flatten()

# Loop through each subset and plot wind profiles
for i in range(num_subsets):
    # Plot wind profile bins for all datasets
    pu.plot_wind_profile_bins(subsets["flight_data_min"][i], subsets["results_min"][i], [axs[i], axs[i + 6]], step=10, color=colors[1], lidar_data=False, label="EKF 0")
    pu.plot_wind_profile_bins(subsets["flight_data_log"][i], subsets["results_log"][i], [axs[i], axs[i + 6]], step=10, color=colors[4], lidar_data=False, label="EKF 2")
    pu.plot_wind_profile_bins(subsets["flight_data_vwz0"][i], subsets["results_vwz0"][i], [axs[i], axs[i + 6]], step=10, color=colors[3], lidar_data=False, label="EKF 4")
    pu.plot_wind_profile_bins(subsets["flight_data_tether"][i], subsets["results_tether"][i], [axs[i], axs[i + 6]], step=10, color=colors[2], lidar_data=True, label="EKF 5")
    
    
    
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
axs[0].legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig("./results/plots_paper/wind_profiles_2024-06-05.pdf")
plt.show()

### Step 4: Plot Wind Speed Comparison
fig, axs = plt.subplots(3, 1, figsize=(16, 6), sharex=True)
pu.plot_wind_speed(results_min, flight_data_min, axs, lidar_heights=[250,100,160], color=colors[1], label_ekf="EKF 0") 
pu.plot_wind_speed(results_log, flight_data_log, axs, lidar_data=False, color=colors[4], label_ekf="EKF 2")
pu.plot_wind_speed(results_vwz0, flight_data_vwz0, axs, lidar_data=False, color=colors[3], label_ekf="EKF 4")
pu.plot_wind_speed(results_tether, flight_data_tether, axs, lidar_data=False, color=colors[2], label_ekf="EKF 5")



axs[1].legend(loc="right", frameon=True)
axs[1].set_ylim([-60, 20])
axs[2].set_ylim([-5, 5])
axs[0].set_xlim([flight_data_min["time"].iloc[0], flight_data_min["time"].iloc[-1]])
time_of_day = pd.to_datetime(flight_data_min["time_of_day"]).dt.strftime("%H:%M")
tick_indices = list(range(0, len(flight_data_min), len(flight_data_min)//6))
if tick_indices[-1] != len(flight_data_min) - 1:
    tick_indices.append(len(flight_data_min) - 1)

axs[2].set_xticks(flight_data_min["time"].iloc[tick_indices])
axs[2].set_xticklabels(time_of_day.iloc[tick_indices], rotation=45, ha="right")
axs[2].set_xlabel("Time of day [hh:mm]")

plt.tight_layout()
plt.savefig("./results/plots_paper/wind_speed_2024-06-05.pdf")
plt.show()
