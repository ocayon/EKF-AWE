import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style

set_plot_style()
# Example usage
plt.close("all")
config_file_name = "v9_config.yaml"
config = load_config("examples/" + config_file_name)

# Load results and flight data and plot kite reference frame
results, flight_data = read_results(
    str(config["year"]),
    str(config["month"]),
    str(config["day"]),
    config["kite"]["model_name"],
    addition="",
)


cycles_plotted = np.arange(6, 10, step=1)
# %% Plot results aerodynamic coefficients
pu.plot_aero_coeff_vs_aoa_ss(
    results, flight_data, cycles_plotted, IMU_0=False, savefig=False
)  # Plot aero coeff vs aoa_ss
pu.plot_aero_coeff_vs_up_us(
    results, flight_data, cycles_plotted, IMU_0=False, savefig=False
)  # Plot aero coeff vs up_used
plt.show()

# %% Polars
aoa_plot = results["kite_aoa"]
# aoa_plot = results['aoa_IMU_0']
# aoa_plot = flight_data['kite_angle_of_attack']
mask = np.any([flight_data["cycle"] == cycle for cycle in cycles_plotted], axis=0)
mask_angles = mask  # &((aoa_plot>0) & (aoa_plot<15))

# mask_angles =(results['aoa']>0) & (results['aoa']<20)
fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
mask = (flight_data["turn_straight"] == "straight") & mask_angles
pu.plot_cl_curve(
    np.sqrt((results["cl_wing"] ** 2 + results["cs_wing"] ** 2)),
    results["cd_wing"],
    aoa_plot,
    mask,
    axs,
    label="Straight",
)
mask = (flight_data["turn_straight"] == "turn") & mask_angles
pu.plot_cl_curve(
    np.sqrt((results["cl_wing"] ** 2 + results["cs_wing"] ** 2)),
    results["cd_wing"],
    aoa_plot,
    mask,
    axs,
    label="Turn",
)

axs[0, 0].axvline(
    x=np.mean(aoa_plot[flight_data["powered"] == "powered"]),
    color="k",
    linestyle="--",
    label="Mean reel-out angle of attack",
)
axs[0, 0].axvline(
    x=np.mean(aoa_plot[flight_data["powered"] == "depowered"]),
    color="b",
    linestyle="--",
    label="Mean reel-in angle of attack",
)


axs[0, 0].legend()
fig.suptitle("cl_wing vs cd_wing of the kite wing (without KCU and tether drag)")
# plt.show()

# %%
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
aoa_plot = flight_data["kite_angle_of_attack"]
mask = (flight_data["turn_straight"] == "straight") & mask_angles & (
    flight_data["up"] > 0.9
) | (flight_data["up"] < 0.1)
pu.plot_cl_curve(
    np.sqrt((results["cl_wing"] ** 2 + results["cs_wing"] ** 2)),
    results["cd_wing"] + results["cd_tether"] + results["cd_kcu"],
    aoa_plot,
    mask,
    axs,
    label="Straight",
)
mask = (flight_data["turn_straight"] == "turn") & mask_angles
pu.plot_cl_curve(
    np.sqrt((results["cl_wing"] ** 2 + results["cs_wing"] ** 2)),
    results["cd_wing"] + results["cd_tether"] + results["cd_kcu"],
    aoa_plot,
    mask,
    axs,
    label="Turn",
)
axs[0, 0].axvline(
    x=np.mean(aoa_plot[flight_data["powered"] == "powered"]),
    color="k",
    linestyle="--",
    label="Mean reel-out angle of attack",
)
axs[0, 0].axvline(
    x=np.mean(aoa_plot[flight_data["powered"] == "depowered"]),
    color="b",
    linestyle="--",
    label="Mean reel-in angle of attack",
)
axs[0, 0].legend()
fig.suptitle("cl_wing vs cd_wing of the system (incl. KCU and tether drag)")

# %%
plt.figure()
plt.plot(
    flight_data["time"], flight_data["ground_tether_force"], label="Measured ground"
)
plt.plot(results["time"], results["tether_force_kite"], label="Estimated at kite")
for column in flight_data.columns:
    if "load_cell" in column:
        plt.plot(flight_data["time"], flight_data[column] * 9.81, label=column)
plt.plot(
    flight_data["time"],
    flight_data["ground_tether_reelout_speed"],
    label="Reelout speed",
)
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.legend()
plt.title("Tether force comparison")
# plt.show()

plt.show()

# %% Find delay cs_wing with us


def find_time_delay(signal_1, signal_2):
    # Compute the cross-correlation
    cross_corr = np.correlate(signal_2, signal_1, mode="full")

    # Find the index of the maximum value in the cross-correlation
    max_corr_index = np.argmax(cross_corr)

    # Compute the time delay
    time_delay = (max_corr_index - (len(signal_1) - 1)) * 0.1

    # Print the time delay
    print(f"Time delay between the two signals is {time_delay} seconds.")

    return time_delay, cross_corr


signal_1 = -flight_data["us"]
signal_2 = results["cs_wing"]

time_delay, cross_corr = find_time_delay(signal_1, signal_2)
# Plot the signals and their cross-correlation
fig, axs = plt.subplots(
    3, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1, 1.5]}
)

# Share x-axis between the first two subplots
axs[0].plot(signal_1, label="us")
axs[0].legend()
axs[0].set_title("Signal 1")

axs[1].plot(signal_2, label="cs_wing")
axs[1].legend()
axs[1].set_title("Signal 2")
axs[1].sharex(axs[0])

# For cross-correlation, set the x-axis to match the number of samples
x_corr = np.arange(-len(signal_1) + 1, len(signal_1))
axs[2].plot(x_corr, cross_corr, label="Cross-correlation")
axs[2].axvline(
    x=time_delay * 10, color="r", linestyle="--", label="Max correlation index"
)
axs[2].legend()
axs[2].set_title("Cross-correlation")

plt.tight_layout()

# %% Plot kite velocity
plt.figure()
kite_speed = np.sqrt(
    results["kite_vel_x"] ** 2 + results["kite_vel_y"] ** 2 + results["kite_vel_z"] ** 2
)
meas_kite_speed = np.sqrt(
    flight_data["kite_velocity_east_s0"] ** 2
    + flight_data["kite_velocity_north_s0"] ** 2
    + flight_data["kite_velocity_up_s0"] ** 2
)
plt.plot(results["time"], kite_speed, label="Estimated")
plt.plot(flight_data["time"], meas_kite_speed, label="Measured")
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.legend()
plt.title("Kite speed comparison")

# %% Plot apparent wind speed
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
pu.plot_time_series(
    flight_data,
    results["apparent_windspeed"],
    ax,
    ylabel="Apparent wind speed (m/s)",
    label="Estimated",
)
pu.plot_time_series(
    flight_data,
    flight_data["kite_apparent_windspeed"],
    ax,
    ylabel="Apparent wind speed (m/s)",
    label="Measured",
)


plt.show()
