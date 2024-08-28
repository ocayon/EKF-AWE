import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from awes_ekf.postprocess.postprocessing import remove_offsets_IMU_data_v3
from awes_ekf.load_data.read_data import read_results
from awes_ekf.plotting.plot_utils import plot_time_series
from awes_ekf.plotting.plot_orientation import plot_kite_orientation
from awes_ekf.plotting.color_palette import get_color_list, visualize_palette, set_plot_style, get_color

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

results, flight_data,config_data = read_results(year, month, day, kite_model,addition='')

for imu in config_data["kite"]["sensor_ids"]:
    flight_data = remove_offsets_IMU_data_v3(results, flight_data, sensor=imu)

mask = flight_data["cycle"].isin([64, 65])

# Plot orientation
results["kite_pitch"] = np.convolve(results["kite_pitch"], np.ones(10)/10, mode="same")
results["kite_roll"] = np.convolve(results["kite_roll"], np.ones(10)/10, mode="same")
# results["kite_yaw"] = np.convolve(results["kite_yaw"], np.ones(10)/10, mode="same")
plot_kite_orientation(results[mask], flight_data[mask], kite_imus=[0, 1])
plt.savefig("./results/plots_paper/kite_orientation_2019-10-08.pdf")
plt.show()

# Plot position and velocity
from awes_ekf.plotting.plot_kite_trajectory import plot_position_azimuth_elevation
plot_position_azimuth_elevation(results[mask], flight_data[mask])
plt.savefig("./results/plots_paper/kite_trajectory_2019-10-08.pdf")
plt.show()

flight_data["bridle_angle_of_attack"] = np.convolve(flight_data["bridle_angle_of_attack"], np.ones(10)/10, mode="same")
aoa_imu = (results[mask]["wing_angle_of_attack_imu_0"]+results[mask]["wing_angle_of_attack_imu_1"])/2
aoa_imu = np.convolve(aoa_imu, np.ones(10)/10, mode="same")
azimuth = np.arctan2(results["kite_position_y"], results["kite_position_x"])
# Plot aerodynamic coefficients
fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
plot_time_series(flight_data[mask], results[mask]["wing_lift_coefficient"], axs[0], ylabel="$C_L$", plot_phase=True)
axs[0].legend()
plot_time_series(flight_data[mask], results[mask]["wing_drag_coefficient"], axs[1], label="Estimated", ylabel="$C_D$")
plot_time_series(flight_data[mask], results[mask]["kcu_drag_coefficient"], axs[1], label="Estimated", ylabel="$C_{D_{KCU}}$")
plot_time_series(flight_data[mask], results[mask]["tether_drag_coefficient"], axs[1], label="Estimated", ylabel="$C_{D_{tether}}$", plot_phase=True)
plot_time_series(flight_data[mask], flight_data[mask]["bridle_angle_of_attack"], axs[2])
plot_time_series(flight_data[mask], aoa_imu, axs[2])
plot_time_series(flight_data[mask], results[mask]["wing_angle_of_attack_bridle"], axs[2],  ylabel=r"$\alpha$ [$^\circ$]", plot_phase=True)
axs[2].legend(["Measured at bridle", "Wing from IMUs", "Wing from bridle"], frameon=True)
plt.savefig("./results/plots_paper/aero_coefficients_2019-10-08.pdf")
# plt.show()

mask_polar = (flight_data["cycle"]>10)&(flight_data["cycle"]<70)
# Plot curves 
from awes_ekf.plotting.plot_utils import plot_cl_curve
fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
plot_cl_curve(np.sqrt((results["wing_lift_coefficient"]**2+results["wing_sideforce_coefficient"]**2)), results["wing_drag_coefficient"], results['wing_angle_of_attack_bridle'], mask_polar,axs, label = "Wing")
plot_cl_curve(np.sqrt((results["wing_lift_coefficient"]**2+results["wing_sideforce_coefficient"]**2)), results["wing_drag_coefficient"]+results["kcu_drag_coefficient"]+results["tether_drag_coefficient"], results['wing_angle_of_attack_bridle'], mask_polar,axs, label = "Wing+KCU+tether")
axs[0,0].legend(loc = "upper left")
plt.savefig("./results/plots_paper/polars_2019-10-08.pdf")
# plt.show()

# Plot sideforce
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
plot_time_series(flight_data[mask], results[mask]["wing_sideforce_coefficient"], axs[0], ylabel="$C_{S}$", plot_phase=True)
plot_time_series(flight_data[mask], flight_data[mask]["us"], axs[1], ylabel="$u_s$", plot_phase=True)
plt.savefig("./results/plots_paper/sideforce_2019-10-08.pdf")
plt.show()



