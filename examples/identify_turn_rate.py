import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
import pandas as pd
from awes_ekf.plotting.color_palette import get_color_list

def cut_data(results, flight_data, range):
    results = results.iloc[range[0]:range[1]]
    flight_data = flight_data.iloc[range[0]:range[1]]
    results = results.reset_index(drop=True)
    flight_data = flight_data.reset_index(drop=True)
    return results, flight_data

colors = get_color_list()
# Example usage
plt.close("all")
config_file_name = "v9_config.yaml"
config = load_config("examples/" + config_file_name)

# Load results and flight data and plot kite reference frame
cut = 6000
results, flight_data,config_data = read_results(
    str(config["year"]),
    str(config["month"]),
    str(config["day"]),
    config["kite"]["model_name"],
    addition="_min"
)

results = results.iloc[cut:-5000]
flight_data = flight_data.iloc[cut:-5000]

results = results.reset_index(drop=True)
flight_data = flight_data.reset_index(drop=True)


from awes_ekf.utils import calculate_turn_rate_law, find_time_delay
# Turn rate law
ts = config_data["simulation_parameters"]["timestep"]
yaw_rate = np.gradient(flight_data["kite_yaw_0"], config_data["simulation_parameters"]["timestep"])
yaw_rate = np.convolve(yaw_rate, np.ones(10)/10, mode='same')
flight_data["kite_yaw_rate"] = yaw_rate
flight_data["kcu_actual_steering"] = flight_data["kcu_actual_steering"]
signal_delay, corr = find_time_delay(flight_data["kite_yaw_rate"], -flight_data["kcu_actual_steering"])
time_delay = signal_delay*ts
print("Time delay turn rate: ", time_delay)
signal_delay, corr = find_time_delay(results["wing_sideforce_coefficient"], -flight_data["kcu_actual_steering"])
flight_data["kcu_actual_steering_delay"] = np.roll(flight_data["kcu_actual_steering"], int(signal_delay))
time_delay = signal_delay*ts
print("Time delay steering force: ", time_delay)

results, flight_data = cut_data(results, flight_data, [1000, -1000])

# flight_data["kite_yaw_rate"] = np.gradient(np.unwrap(flight_data["kite_yaw_0"]), ts)
yaw_rate, coeffs = calculate_turn_rate_law(results, flight_data, model = "simple", steering_offset=False)
yaw_rate_weight, coeffs_weight = calculate_turn_rate_law(results, flight_data, model = "simple_weight", steering_offset=True)

print("Yaw rate coeffs: ", coeffs)
print("Yaw rate coeffs weight: ", coeffs_weight)

mask = flight_data["cycle"].isin([64, 65])


# Calculate mean errors
error = abs(np.degrees(yaw_rate) - np.degrees(flight_data["kite_yaw_rate"]))
error_weight = abs(np.degrees(yaw_rate_weight) - np.degrees(flight_data["kite_yaw_rate"]))
mean_error = np.mean(error)
mean_error_weight = np.mean(error_weight)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
# Downsample the data (e.g., use only 10% of the data)
downsample_fraction = 0.1
downsampled_data = flight_data.sample(frac=downsample_fraction, random_state=42)
downsampled_results = results.loc[downsampled_data.index]
downsampled_data = downsampled_data[downsampled_data["powered"]=="powered"]
downsampled_results = downsampled_results.loc[downsampled_data.index]
# Prepare data
x = -downsampled_data["kcu_actual_steering"] / 100 * downsampled_results["kite_apparent_windspeed"]
y = downsampled_data["kite_yaw_rate"]

# Calculate the point density
xy = np.vstack([x, y])
# z = gaussian_kde(xy)(xy)
# 
# Create KDE plot
plt.figure(figsize=(6, 4))
# scatter = plt.scatter(
#     x, y, 
#     c=z,  # Use density values for coloring
#     s=20,  # Adjust point size if needed
#     cmap='viridis',  # Color map (adjust to preference)
#     alpha=0.6  # Transparency level
# )
plt.scatter(
    x,
    y,
    color = colors[1],
    alpha = 0.2,
)

# Overlay the line plot
plt.plot(
    x, 
    coeffs[0] * downsampled_data["kcu_actual_steering"] / 100 * downsampled_results["kite_apparent_windspeed"],
    label=f'Identified Yaw Rate (Mean Error: {mean_error:.2f} deg/s)', 
    color=colors[2], 
    linestyle="--"
)
y = calculate_turn_rate_law(downsampled_results, downsampled_data, model = "simple_weight", steering_offset=True, coeffs = coeffs_weight)
plt.plot(
    x,
    y,
    label=f'Offset-Corrected Yaw Rate (Mean Error: {mean_error_weight:.2f} deg/s)',
    color = colors[0],
    linestyle = '-.'
)

plt.xlabel(r'$u_\mathrm{s} \cdot v_\mathrm{a}$ [m/s]')
plt.ylabel('Kite Yaw Rate [rad/s]')
plt.legend()

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig("./results/plots_paper/yaw_rate_2019-10-08.pdf")
x = -downsampled_data["kcu_actual_steering_delay"] / 100
y = downsampled_results["wing_sideforce_coefficient"]

from scipy.stats import linregress

# Calculate the linear regression
slope, intercept, r_value, p_value, std_err = linregress(flight_data["kcu_actual_steering_delay"] / 100, results["wing_sideforce_coefficient"])

# Create the regression line
regression_line = slope * -np.linspace(-0.37,0.37,10) + intercept

plt.figure(figsize=(6, 4))
x = -downsampled_data["kcu_actual_steering"] / 100
plt.scatter(x, y, alpha=0.2, color = colors[1], label='EKF 0')
x = -downsampled_data["kcu_actual_steering_delay"] / 100
plt.scatter(x, y, alpha=0.2, color = colors[2], label='EKF 0 - Delay Corrected')
plt.plot(np.linspace(-0.37,0.37,10), regression_line, color=colors[0], label=f'Linear fit: y = {slope:.2f}x + {intercept:.2f}', linestyle='--')
plt.xlabel(r'$u_\mathrm{s}$')
plt.ylabel(r'$C_S$')
plt.legend()
plt.tight_layout()
plt.savefig("./results/plots_paper/sideforce_2019-10-08.pdf")
# plt.show()



# Print mean errors and standard deviations
print("Mean error yaw rate: ", mean_error)
print("Mean error yaw rate weight: ", mean_error_weight)
print("Std error yaw rate: ", np.std(error))
print("Std error yaw rate weight: ", np.std(error_weight))

plt.figure()
plt.plot(flight_data.time)
plt.show()

fit_sideforce = slope*flight_data[mask]["kcu_actual_steering_delay"] / 100+intercept
# Plot sideforce
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
pu.plot_time_series(flight_data[mask], results[mask]["wing_sideforce_coefficient"], axs[0], ylabel="$C_{S}$", plot_phase=False, color=colors[0], label = "EKF 0")
slope, intercept, r_value, p_value, std_err = linregress(flight_data["kcu_actual_steering"] / 100, results["wing_sideforce_coefficient"])
axs[0].plot(flight_data[mask]["time"], slope*flight_data[mask]["kcu_actual_steering"] / 100+intercept, label="Linear Fit", color=colors[1], linestyle="--")
axs[0].plot(flight_data[mask]["time"], fit_sideforce, label="Linear Fit - Delay Corrected", color=colors[2], linestyle="--")

axs[0].legend(frameon = True)
# Second subplot: Yaw Rate Comparison
axs[1].plot(flight_data[mask]["time"], np.degrees(flight_data["kite_yaw_rate"][mask]), label='Measured Yaw Rate', color=colors[0])
axs[1].plot(flight_data[mask]["time"], np.degrees(yaw_rate[mask]), label='Identified Yaw Rate', color=colors[1], linestyle='--')
axs[1].plot(flight_data[mask]["time"], np.degrees(yaw_rate_weight[mask]), label='Offset-Corrected Yaw Rate', color=colors[2], linestyle='-.')
axs[1].legend(frameon = True)
axs[1].set_ylabel("Yaw Rate [deg/s]")
# Third subplot: Steering Input
pu.plot_time_series(flight_data[mask], -flight_data[mask]["kcu_actual_steering"]/100, axs[2], ylabel="$u_s$", plot_phase=False, color=colors[0], label="Actual steering")
pu.plot_time_series(flight_data[mask], -flight_data[mask]["kcu_set_steering"]/100, axs[2], ylabel="$u_s$", plot_phase=False, color=colors[1], label="Set steering")
axs[2].legend(frameon = True)
axs[2].set_xlabel("Time [s]")
axs[2].set_ylim([-0.4,0.4])
plt.tight_layout()
plt.savefig("./results/plots_paper/turn_timeseries_2019-10-08.pdf")


yaw_rate = calculate_turn_rate_law(results, flight_data, model = "simple", steering_offset=False, span = 20, mass = 62, area=46, coeffs = coeffs) 
plt.figure()
# plt.plot(flight_data["time"], np.degrees(flight_data["kite_yaw_rate"]), label='Measured yaw rate')
plt.plot(flight_data["time"], np.degrees(yaw_rate), label='Identified yaw rate scaled v9')
plt.legend()
plt.grid(True)

plt.show()