import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
from awes_ekf.plotting.color_palette import get_color_list, visualize_palette, set_plot_style, get_color


def remove_faulty_lidar_data(flight_data):
    # Identify lidar heights from the columns
    lidar_heights = []
    for column in flight_data.columns:
        if "m_Wind_Speed_m_s" in column:
            height = "".join(filter(str.isdigit, column))
            lidar_heights.append(int(height))
    
    # Create a mask for rows that need to be updated
    faulty_indices = []

    for index in flight_data.index:
        for height in lidar_heights:
            vel0 = flight_data.loc[index, f"{height}m_Wind_Speed_m_s"]
            vel1 = flight_data.loc[index-1, f"{height}m_Wind_Speed_m_s"] if index > 0 else None
            vel2 = flight_data.loc[index+1, f"{height}m_Wind_Speed_m_s"] if index < len(flight_data) - 1 else None
            
            # Check for faulty data
            if vel0 != vel1 and vel0 != vel2:
                faulty_indices.append((index, height))
    
    # Collect all changes and apply them at once to avoid fragmentation
    for index, height in faulty_indices:
        flight_data.loc[index, [
            f"{height}m_Wind_Speed_m_s",
            f"{height}m_Wind_Speed_max_m_s",
            f"{height}m_Wind_Speed_min_m_s",
            f"{height}m_Wind_Direction_deg",
            f"{height}m_Z-wind_m_s"
        ]] = np.nan

    return flight_data


set_plot_style()
year = "2024"
month = "03"
day = "12"
kite_model = "v9"

results, flight_data,_ = read_results(year, month, day, kite_model,addition='_min')

results["cycle"] = flight_data["cycle"]

cut = 200
results = results.iloc[cut:-cut]
flight_data = flight_data.iloc[cut:-cut]



results = results.reset_index(drop=True)
flight_data = flight_data.reset_index(drop=True)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
pu.plot_kinetic_energy_spectrum(results, flight_data,ax, savefig=False)    
plt.show()
# plt.savefig("./results/plots_paper/kinetic_energy_spectrum_2019-10-08.pdf")

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
pu.plot_turbulence_intensity(results, flight_data, 140, ax, savefig=False)
plt.savefig("./results/plots_paper/turbulence_intensity_2024-03-12.pdf")
plt.show()

chunk_size = 12000  # Number of rows in each subset
num_subsets = 6
# Divide the DataFrames into 6 nearly equal subsets
results_subsets = [results.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_subsets)]
flight_data_subsets = [flight_data.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(num_subsets)]

# Create a figure with subplots for each subset
fig, axs = plt.subplots(2, 6, figsize=(12,6))  # Adjust the layout as needed

# Remove faulty lidar data
flight_data = remove_faulty_lidar_data(flight_data)
print("Removed faulty lidar data")
# Flatten axs for easy iteration if using a 2D grid of subplots
axs = axs.flatten()

# Loop through each subset and plot wind profiles
for i in range(6):
    # Plot wind profile bins
    pu.plot_wind_profile_bins(flight_data_subsets[i], results_subsets[i], [axs[i], axs[i+6]], step=10, color="Blue", lidar_data=True)
    
    # Extract and round time to the nearest 5-minute interval
    original_time = flight_data_subsets[i]["time_of_day"].iloc[0]
    rounded_time = (pd.to_datetime(original_time) + pd.Timedelta(minutes=2.5)).floor('5T').time()
    
    # Set the title with the rounded time
    axs[i].set_title(rounded_time.strftime("%H:%M:%S"))
    axs[i].legend().remove()
    axs[i+6].legend().remove()
    axs[i].set_xlim([5,15])
    axs[i+6].set_xlim([70,110])
    # Print the subset and rounded time
    print(f"Subset {i+1}: {rounded_time}")

axs[0].legend(loc="lower right", frameon=True)

# Adjust the layout and show the plot
plt.tight_layout()
# Save as a PDF file
plt.savefig("./results/plots_paper/wind_profiles_2024-03-12.pdf")

plt.show()
def cut_data(results, flight_data, range):
    results = results.iloc[range[0]:range[1]]
    flight_data = flight_data.iloc[range[0]:range[1]]
    results = results.reset_index(drop=True)
    flight_data = flight_data.reset_index(drop=True) 
    return results, flight_data

results, flight_data = cut_data(results, flight_data, [0, 6000*12])
flight_data = pu.interpolate_lidar_data(flight_data, results)



fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
axs[0].plot(flight_data["time"], results["wind_speed_horizontal"], label="Min. measurements", color = get_color("Blue"))
axs[0].plot(flight_data["time"], flight_data["interp_wind_speed"], label="Lidar", color = get_color("Dark Gray"))
axs[0].fill_between(
                    flight_data["time"],
                    flight_data["interp_wind_speed_min"],
                    flight_data["interp_wind_speed_max"],
                    color=get_color("Dark Gray"),
                    alpha=0.3,
                )

axs[0].set_ylabel("Wind speed [m/s]")
# axs[0].legend()

axs[1].plot(flight_data["time"], results["wind_direction"]*180/np.pi, label="Min. measurements", color = get_color("Blue"))
axs[1].plot(flight_data["time"], 270-flight_data["interp_wind_direction"], label="Lidar", color = get_color("Dark Gray"))

axs[1].set_ylabel("Wind direction [deg]")
axs[1].legend(frameon=True)

axs[2].plot(flight_data["time"], results["wind_speed_vertical"], label="Min. measurements$", color = get_color("Blue"))
axs[2].plot(flight_data["time"], -flight_data["interp_z_wind"], label="Lidar",   color = get_color("Dark Gray"))
axs[2].set_ylabel("Vertical wind speed [m/s]")
axs[2].set_xlabel("Time [s]")

axs[0].set_xlim([flight_data["time"].iloc[0], flight_data["time"].iloc[-1]])
time_of_day = pd.to_datetime(flight_data["time_of_day"]).dt.strftime("%H:%M")

# Determine x-ticks: 4 evenly spaced points + the last time point
tick_indices = list(range(0, len(flight_data), len(flight_data)//5))  # 4 evenly spaced ticks
if tick_indices[-1] != len(flight_data) - 1:  # Ensure the last tick is the very last time point
    tick_indices.append(len(flight_data) - 1)

# Set x-ticks and labels
axs[2].set_xticks(flight_data["time"].iloc[tick_indices])
axs[2].set_xticklabels(time_of_day.iloc[tick_indices], rotation=45, ha="right")

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig("./results/plots_paper/wind_speed_2024-03-12.pdf")
plt.show()