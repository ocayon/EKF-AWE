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

for imu in config_data["kite"]["sensor_ids"]:
    flight_data = remove_offsets_IMU_data_v3(results, flight_data, sensor=imu)

mask = flight_data["cycle"].isin([64, 65])

colors = get_color_list()

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
plot_time_series(flight_data[mask], results[mask]["norm_epsilon_norm"], ax, plot_phase=True, color = colors[0])
from matplotlib.patches import Patch
# Create a new patch for the legend
reel_out_straight_patch = Patch(color=colors[5], alpha=0.2, label="Reel-out - Straight")
reel_out_turn_patch = Patch(color=colors[7], alpha=0.2, label="Reel-out - Turn")
reel_in_patch = Patch(facecolor='white', alpha=1, edgecolor='black', label="Reel-in")
ax.legend(
        [reel_out_straight_patch, reel_out_turn_patch, reel_in_patch],
        ["Reel-out - Straight", "Reel-out - Turn", "Reel-in"],
        loc='upper left',
        frameon=True,
        bbox_to_anchor=(0.075, 1)  # Adjust the x-coordinate to move the legend to the right
    )
ax.set_xlabel("Time [s]")
ax.set_ylabel("Norm of Normalized Residuals")
plt.tight_layout()
plt.savefig("./results/plots_paper/norm_residuals_2019-10-08.pdf")
plt.show()

# Plot position and velocity
fig, axs = plt.subplots(2, 1, figsize=(6, 10))
mean_wind_dir = np.mean(results[mask]["wind_direction"])
azimuth, elevation = calculate_azimuth_elevation(res_min[mask]["kite_position_x"], res_min[mask]["kite_position_y"], res_min[mask]["kite_position_z"])
axs[0].plot(np.rad2deg(azimuth-mean_wind_dir), np.rad2deg(elevation), label="EKF 0", color = colors[0])
azimuth, elevation = calculate_azimuth_elevation(results[mask]["kite_position_x"], results[mask]["kite_position_y"], results[mask]["kite_position_z"])
axs[0].plot(np.rad2deg(azimuth-mean_wind_dir), np.rad2deg(elevation), label="EKF 1", color = colors[1])
azimuth, elevation = calculate_azimuth_elevation(flight_data[mask]["kite_position_x"], flight_data[mask]["kite_position_y"], flight_data[mask]["kite_position_z"])
axs[0].plot(np.rad2deg(azimuth-mean_wind_dir), np.rad2deg(elevation), label="GPS", color = colors[2])
axs[0].legend()
axs[0].set_xlabel("Azimuth [deg]")
axs[0].set_ylabel("Elevation [deg]")
r = np.sqrt(res_min[mask]["kite_position_x"]**2 + res_min[mask]["kite_position_y"]**2+ res_min[mask]["kite_position_z"]**2)
axs[1].plot(flight_data[mask]["time"], r, label="EKF 0 ", color = colors[0])
r = np.sqrt(results[mask]["kite_position_x"]**2 + results[mask]["kite_position_y"]**2+ results[mask]["kite_position_z"]**2)
axs[1].plot(flight_data[mask]["time"], r, label="EKF 1", color = colors[1],linewidth=1)
r = np.sqrt(flight_data[mask]["kite_position_x"]**2 + flight_data[mask]["kite_position_y"]**2+ flight_data[mask]["kite_position_z"]**2)
axs[1].plot(flight_data[mask]["time"], r, label="GPS+IMU", color = colors[2])
axs[1].plot(flight_data[mask]["time"], flight_data[mask]["tether_length"]+11.5, label="Measured tether length", color = colors[3])
axs[1].legend()
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("Radial Distance/Tether Length [m]")
plt.tight_layout()
plt.savefig("./results/plots_paper/kite_trajectory_2019-10-08.pdf")
# plt.show()

simConfig = SimulationConfig(**config_data["simulation_parameters"])

# Create system components
kite = PointMassEKF(simConfig, **config_data["kite"])
kcu = KCU(**config_data["kcu"])
# Plot dimensional forces
plot_forces_dimensional(results[mask], flight_data[mask], kite,kcu)
plt.tight_layout()
plt.savefig("./results/plots_paper/awe_forces_2019-10-08.pdf")
# plt.show()

plot_slack_tether_force(results[mask], flight_data[mask], kcu)
plt.tight_layout()
plt.savefig("./results/plots_paper/slack_2019-10-08.pdf")
# plt.show()


# Plot orientation
results["kite_pitch"] = np.convolve(results["kite_pitch"], np.ones(10)/10, mode="same")
results["kite_roll"] = np.convolve(results["kite_roll"], np.ones(10)/10, mode="same")
# results["kite_yaw"] = np.convolve(results["kite_yaw"], np.ones(10)/10, mode="same")
results["kite_yaw_kin"] = results["kite_yaw_kin"]%(2*np.pi)
plot_kite_orientation(results[mask], flight_data[mask], config_data)
plt.tight_layout()
plt.savefig("./results/plots_paper/kite_orientation_2019-10-08.pdf")
# plt.show()
signal_delay, corr = find_time_delay(flight_data["kite_yaw_0"], results["kite_yaw_kin"])
time_delay = signal_delay*0.1
print("Time delay yaw: ", time_delay)
plt.show()

plt.figure()
plot_time_series(flight_data[mask], results[mask]["kite_apparent_windspeed"], plt.gca(), label="Yaw kite", plot_phase=True)
# plt.show()

fig, ax = plt.subplots(1, 1, figsize=(12, 5))
plot_time_series(
    flight_data[mask],
    np.rad2deg(results[mask]["kite_roll"] - results[mask]["tether_roll"]),
    ax,
    label="Roll kite-tether",
    plot_phase=True,
    color=colors[0],
)

plot_time_series(
    flight_data[mask],
    np.rad2deg(results[mask]["kite_pitch"] - results[mask]["tether_pitch"]),
    ax,
    label="Pitch kite-tether",
    plot_phase=False,
    color=colors[1],
)
ax.legend()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Angle [deg]")
plt.tight_layout()
plt.savefig("./results/plots_paper/kite_tether_angles_2019-10-08.pdf")
# plt.show()
#%% Plot wind energy spectrum
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
plot_kinetic_energy_spectrum(results, flight_data,ax, savefig=False)    
plt.tight_layout()
plt.savefig("./results/plots_paper/kinetic_energy_spectrum_2019-10-08.pdf")



flight_data["bridle_angle_of_attack"] = np.convolve(flight_data["bridle_angle_of_attack"], np.ones(10)/10, mode="same")
aoa_imu = (results[mask]["wing_angle_of_attack_imu_0"]+results[mask]["wing_angle_of_attack_imu_1"])/2
aoa_imu = np.convolve(aoa_imu, np.ones(10)/10, mode="same")
azimuth = np.arctan2(results["kite_position_y"], results["kite_position_x"])
# Plot aerodynamic coefficients
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
plot_time_series(flight_data[mask], results[mask]["wing_lift_coefficient"], axs[0], ylabel="$C_L$", plot_phase=True, color=colors[0])
axs[0].legend()
plot_time_series(flight_data[mask], results[mask]["wing_drag_coefficient"], axs[1],  label="$C_\mathrm{D}$", color=colors[0])
plot_time_series(flight_data[mask], results[mask]["kcu_drag_coefficient"], axs[1],  label="$C_\mathrm{D,kcu}$", color=colors[1])
plot_time_series(flight_data[mask], results[mask]["tether_drag_coefficient"], axs[1],  label="$C_\mathrm{D,t}$", plot_phase=True, color=colors[2], ylabel="$C_D$")
axs[1].legend()
plot_time_series(flight_data[mask], flight_data[mask]["bridle_angle_of_attack"], axs[2], color=colors[1])
plot_time_series(flight_data[mask], aoa_imu, axs[2], color=colors[2])
plot_time_series(flight_data[mask], results[mask]["wing_angle_of_attack_bridle"], axs[2],  ylabel=r"$\alpha$ [$^\circ$]", plot_phase=True, color=colors[0])
axs[2].legend([r"$\alpha_\mathrm{b}$ measured", r"$\alpha_\mathrm{w}$ from IMU", r"$\alpha_\mathrm{w}$ from bridle"], frameon=True)
axs[2].set_xlabel("Time [s]")
plt.tight_layout()
plt.savefig("./results/plots_paper/aero_coefficients_2019-10-08.pdf")
# plt.show()

mask_polar = (flight_data["cycle"]>10)&(flight_data["cycle"]<70)
mask_turn = (flight_data["turn_straight"]=="turn")&mask_polar
mask_straight = (flight_data["turn_straight"]=="straight")&mask_polar
# Plot curves 
from awes_ekf.plotting.plot_utils import plot_cl_curve
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
cl_roullier = np.loadtxt("./processed_data/previous_analysis/cl_roullier_mean.csv", delimiter=",")
cd_roullier = np.loadtxt("./processed_data/previous_analysis/cd_roullier_mean.csv", delimiter=",")
cl_rans = np.loadtxt("./processed_data/previous_analysis/RANS_CL_alpha_struts.csv", delimiter=",")
cd_rans = np.loadtxt("./processed_data/previous_analysis/RANS_CD_alpha_struts.csv", delimiter=",")
VSM_coeffs = pd.read_csv("./processed_data/previous_analysis/VSM_aero_coeffs_V3.csv")

plot_cl_curve(np.sqrt((results["wing_lift_coefficient"]**2+results["wing_sideforce_coefficient"]**2)), results["wing_drag_coefficient"], results['wing_angle_of_attack_bridle'], mask_polar,axs, label = "Wing", color=colors[0])
# plot_cl_curve(np.sqrt((results["wing_lift_coefficient"]**2+results["wing_sideforce_coefficient"]**2)), results["wing_drag_coefficient"], results['wing_angle_of_attack_bridle'], mask_turn,axs, label = "Wing Turn", color=colors[2])
# plot_cl_curve(np.sqrt((results["wing_lift_coefficient"]**2+results["wing_sideforce_coefficient"]**2)), results["wing_drag_coefficient"], results['wing_angle_of_attack_bridle'], mask_straight,axs, label = "Wing Straight", color=colors[3])
plot_cl_curve(np.sqrt((results["wing_lift_coefficient"]**2+results["wing_sideforce_coefficient"]**2)), results["wing_drag_coefficient"]+results["kcu_drag_coefficient"]+results["tether_drag_coefficient"], results['wing_angle_of_attack_bridle'], mask_polar,axs, label = "Wing+KCU+tether", color=colors[1])
axs[0].plot(cl_roullier[:,0], cl_roullier[:,1], label="Exp. Roullier", linewidth=1.5, color = colors[2])
axs[1].plot(cd_roullier[:,0], cd_roullier[:,1], label="Exp. Roullier", linewidth=1.5,color = colors[2])
axs[0].plot(cl_rans[:,0], cl_rans[:,1], label="RANS",linewidth=1.5, color = colors[3])
axs[1].plot(cd_rans[:,0], cd_rans[:,1], label="RANS", linewidth=1.5,color = colors[3])
axs[0].plot(VSM_coeffs["alpha"], VSM_coeffs["cl_powered"], label="VSM - Powered",linewidth=1.5, color = colors[5])
axs[0].plot(VSM_coeffs["alpha"], VSM_coeffs["cl_depowered"], label="VSM - Depowered",linewidth=1.5, color = colors[5], linestyle = '--')
axs[0].plot(VSM_coeffs["alpha"], VSM_coeffs["cl_powered_max_steering"], label="VSM - Powered Turn",linewidth=1.5, color = colors[5], linestyle = '-.')
axs[1].plot(VSM_coeffs["alpha"], VSM_coeffs["cd_powered"], label="VSM - Powered",linewidth=1.5, color = colors[5])
axs[1].plot(VSM_coeffs["alpha"], VSM_coeffs["cd_depowered"], label="VSM - Depowered",linewidth=1.5, color = colors[5], linestyle = '--')
axs[1].plot(VSM_coeffs["alpha"], VSM_coeffs["cd_powered_max_steering"], label="VSM - Powered Turn",linewidth=1.5, color = colors[5], linestyle = '-.')
mean_aoa_pow = np.mean(results["wing_angle_of_attack_bridle"][flight_data["powered"]=="powered"])
mean_aoa_dep = np.mean(results["wing_angle_of_attack_bridle"][flight_data["powered"]=="depowered"])
axs[0].axvline(x = mean_aoa_pow, color = colors[6],linestyle = '--', label = 'Mean reel-out angle of attack')
axs[0].axvline(x = mean_aoa_dep, color = colors[7],linestyle = '--', label = 'Mean reel-in angle of attack')
axs[1].axvline(x = mean_aoa_pow, color = colors[6],linestyle = '--', label = 'Mean reel-out angle of attack')
axs[1].axvline(x = mean_aoa_dep, color = colors[7],linestyle = '--', label = 'Mean reel-in angle of attack')
axs[0].legend(loc = "lower right")
plt.tight_layout()
plt.savefig("./results/plots_paper/polars_2019-10-08.pdf")
# plt.show()



results, flight_data = cut_data(results, flight_data, [18000, len(results)-18000])


# Turn rate law
ts = config_data["simulation_parameters"]["timestep"]
flight_data["kite_yaw_rate"] = flight_data["kite_yaw_rate_1"]
flight_data["kcu_actual_steering"] = flight_data["kcu_actual_steering"]
signal_delay, corr = find_time_delay(flight_data["kite_yaw_rate"], -flight_data["kcu_actual_steering"])
time_delay = signal_delay*ts
print("Time delay turn rate: ", time_delay)
signal_delay, corr = find_time_delay(results["wing_sideforce_coefficient"], -flight_data["kcu_actual_steering"])
flight_data["kcu_actual_steering_delay"] = np.roll(flight_data["kcu_actual_steering"], int(signal_delay))
time_delay = signal_delay*ts
print("Time delay steering force: ", time_delay)


# flight_data["kite_yaw_rate"] = np.gradient(np.unwrap(flight_data["kite_yaw_0"]), ts)
yaw_rate, coeffs = calculate_turn_rate_law(results, flight_data, model = "simple", steering_offset=False)
yaw_rate_weight, coeffs_weight = calculate_turn_rate_law(results, flight_data, model = "simple", steering_offset=True)

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
A = np.vstack([x, downsampled_results["kite_apparent_windspeed"]]).T
y = A@coeffs_weight
plt.plot(
    -x,
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




fit_sideforce = slope*flight_data[mask]["kcu_actual_steering_delay"] / 100+intercept
# Plot sideforce
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
plot_time_series(flight_data[mask], results[mask]["wing_sideforce_coefficient"], axs[0], ylabel="$C_{S}$", plot_phase=False, color=colors[0], label = "EKF 0")
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
plot_time_series(flight_data[mask], -flight_data[mask]["kcu_actual_steering"]/100, axs[2], ylabel="$u_s$", plot_phase=False, color=colors[0], label="Actual steering")
plot_time_series(flight_data[mask], -flight_data[mask]["kcu_set_steering"]/100, axs[2], ylabel="$u_s$", plot_phase=False, color=colors[1], label="Set steering")
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



