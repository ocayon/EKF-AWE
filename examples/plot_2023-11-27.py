import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
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

results, flight_data,config_data = read_results(year, month, day, kite_model,addition='')
res_va, fd_va, _ = read_results(year, month, day, kite_model,addition='_va')
res_log, fd_log, _ = read_results(year, month, day, kite_model,addition='_log')
res_min, fd_min, _ = read_results(year, month, day, kite_model,addition='_min')

#%% Plot orientation
from awes_ekf.postprocess.postprocessing import remove_offsets_IMU_data
from awes_ekf.plotting.plot_orientation import plot_kite_orientation

mask = flight_data["cycle"].isin([62,63,64,65,66])

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
pu.plot_time_series(flight_data[mask], results[mask]["norm_epsilon_norm"], ax, plot_phase=True)
ax.legend()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Norm of Normalized Residuals")
plt.tight_layout()
plt.savefig("./results/plots_paper/norm_residuals_2023-11-27.pdf")
plt.show()


plot_kite_orientation(results[mask], flight_data[mask], kite_imus=config_data["kite"]["sensor_ids"])
plt.savefig("./results/plots_paper/kite_orientation_2023-11-27.pdf")
plt.show()

cut = 1000
results, flight_data = cut_data(results, flight_data, [cut, -cut])
res_va, fd_va = cut_data(res_va, fd_va, [cut, -cut])
res_log, fd_log = cut_data(res_log, fd_log, [cut, -cut])
res_min, fd_min = cut_data(res_min, fd_min, [cut, -cut])

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
pu.plot_kinetic_energy_spectrum(res_va, fd_va, ax, savefig=False)
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
# pu.plot_turbulence_intensity(results, flight_data, 140, ax)
pu.plot_turbulence_intensity(res_va, fd_va, 140, ax)
plt.savefig("./results/plots_paper/turbulence_intensity_2023-11-27.pdf")
plt.show()

chunk_size = 12000  # Number of rows in each subset
num_subsets = 4
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
fig, axs = plt.subplots(2, num_subsets, figsize=(12,6))  # Adjust the layout as needed

# Flatten axs for easy iteration if using a 2D grid of subplots
axs = axs.flatten()

# Loop through each subset and plot wind profiles
for i in range(num_subsets):
    # Plot wind profile bins
    pu.plot_wind_profile_bins(fd_min_subsets[i], res_min_subsets[i], [axs[i], axs[i+num_subsets]], step=10, color="Blue", label="Min. measurements", lidar_data=False)
    pu.plot_wind_profile_bins(flight_data_subsets[i], results_subsets[i], [axs[i], axs[i+num_subsets]], step=10, color="Orange", lidar_data=False, label="Min. + $l_t$")
    pu.plot_wind_profile_bins(fd_va_subsets[i], res_va_subsets[i], [axs[i], axs[i+num_subsets]], step=10, color="Green", label="Min. + $v_a$", lidar_data=False)
    pu.plot_wind_profile_bins(fd_log_subsets[i], res_log_subsets[i], [axs[i], axs[i+num_subsets]], step=10, color="Red", label="Min. with Log profile", lidar_data=True)
    
    
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

axs[0].legend(loc="lower right", bbox_to_anchor=(0.2, 0.8), frameon=True)

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



fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
axs[0].plot(fd_min["time"], res_min["wind_speed_horizontal"], label="Min. measurements", color = get_color("Blue"))
axs[0].plot(flight_data["time"], results["wind_speed_horizontal"], label="Min. + $l_t$", color = get_color("Orange"))
axs[0].plot(fd_va["time"], res_va["wind_speed_horizontal"], label="Min. + $v_a$", color = get_color("Green"))
axs[0].plot(fd_log["time"], res_log["wind_speed_horizontal"], label="Min. with Log profile", color = get_color("Red"))
axs[0].plot(flight_data["time"], flight_data["interp_wind_speed"], label="Lidar", color = get_color("Dark Gray"))
axs[0].fill_between(
                    flight_data["time"],
                    flight_data["interp_wind_speed_min"],
                    flight_data["interp_wind_speed_max"],
                    color=get_color("Dark Gray"),
                    alpha=0.3,
                )
axs[0].plot(flight_data["time"], flight_data["ground_wind_speed"], label="Ground", color = get_color("Black"))
axs[0].set_ylabel("Wind speed [m/s]")
# axs[0].legend()

axs[1].plot(fd_min["time"], res_min["wind_direction"]*180/np.pi, label="Min. measurements", color = get_color("Blue"))
axs[1].plot(flight_data["time"], results["wind_direction"]*180/np.pi, label="Min. + $l_t$", color = get_color("Orange"))
axs[1].plot(fd_va["time"], res_va["wind_direction"]*180/np.pi, label="Min. + $v_a$", color = get_color("Green"))
axs[1].plot(fd_log["time"], res_log["wind_direction"]*180/np.pi, label="Min. with Log profile", color = get_color("Red"))
axs[1].plot(flight_data["time"], 270-flight_data["interp_wind_direction"], label="Lidar", color = get_color("Dark Gray"))
axs[1].plot(flight_data["time"], flight_data["ground_wind_direction"], label="Ground", color = get_color("Black"))
axs[1].set_ylabel("Wind direction [deg]")
axs[1].legend(frameon=True)

axs[2].plot(fd_min["time"], res_min["wind_speed_vertical"], label="Min. measurements", color = get_color("Blue"))
axs[2].plot(flight_data["time"], results["wind_speed_vertical"], label="Min. + $l_t$", color = get_color("Orange"))
axs[2].plot(fd_va["time"], res_va["wind_speed_vertical"], label="Min. + $v_a$", color = get_color("Green"))
axs[2].plot(fd_log["time"], res_log["wind_speed_vertical"], label="Min. with Log profile", color = get_color("Red"))
axs[2].plot(flight_data["time"], flight_data["interp_z_wind"], label="Lidar",   color = get_color("Dark Gray"))
axs[2].set_ylabel("Vertical wind speed [m/s]")
axs[2].set_xlabel("Time [s]")

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



