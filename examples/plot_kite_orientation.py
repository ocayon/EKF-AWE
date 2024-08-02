import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
import seaborn as sns
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style
from awes_ekf.postprocess.postprocessing import remove_offsets_IMU_data

set_plot_style()

# Example usage
plt.close("all")
config_file_name = "v3_config.yaml"
config = load_config("examples/" + config_file_name)

# Load results and flight data and plot kite reference frame
results, flight_data = read_results(
    str(config["year"]),
    str(config["month"]),
    str(config["day"]),
    config["kite"]["model_name"],
    addition="",
)

results, flight_data = remove_offsets_IMU_data(results, flight_data,sensor=0)


imus = [0]

for column in results.columns:
    if "pitch" in column or "roll" in column or "yaw" in column:
        results[column] = np.degrees(results[column])

for column in flight_data.columns:
    if "pitch" in column or "roll" in column or "yaw" in column:
        flight_data[column] = np.degrees(flight_data[column])


# Calculate errors
pitch_error = abs(results["kite_pitch_s0"] - results["kite_pitch"])
roll_error = abs(results["kite_roll_s0"] - results["kite_roll"])
yaw_error = abs(results["kite_yaw_s0"] - results["kite_yaw"])

mean_pitch_error = np.mean(pitch_error)
mean_roll_error = np.mean(roll_error)
mean_yaw_error = np.mean(yaw_error)

std_pitch_error = np.std(pitch_error)
std_roll_error = np.std(roll_error)
std_yaw_error = np.std(yaw_error)

print(f"Mean pitch error: {mean_pitch_error:.2f} deg, std: {std_pitch_error:.2f} deg")
print(f"Mean roll error: {mean_roll_error:.2f} deg, std: {std_roll_error:.2f} deg")
print(f"Mean yaw error: {mean_yaw_error:.2f} deg, std: {std_yaw_error:.2f} deg")


# %%
fig, ax = plt.subplots()
for imu in imus:
    pu.plot_time_series(flight_data, flight_data['kite_pitch_s'+str(imu)], ax, label='Measured',plot_phase=False)
    pu.plot_time_series(
        flight_data,
        results["kite_pitch_s" + str(imu)],
        ax,
        label="Measured Corrected",
        plot_phase=False,
    )

pu.plot_time_series(
    flight_data, results["kite_pitch"], ax, label="Estimated", plot_phase=False
)
ax.legend()

fig, ax = plt.subplots()
for imu in imus:
    # pu.plot_time_series(flight_data, flight_data['kite_roll_s'+str(imu)], ax, label='Measured',plot_phase=False)
    pu.plot_time_series(
        flight_data,
        results["kite_roll_s" + str(imu)],
        ax,
        label="Measured",
        plot_phase=False,
    )
pu.plot_time_series(
    flight_data, results["kite_roll"], ax, label="Estimated", plot_phase=False
)
ax.legend()


fig, ax = plt.subplots()
for imu in imus:
    # pu.plot_time_series(flight_data, flight_data['kite_yaw_s'+str(imu)], ax, label='Measured',plot_phase=False)
    pu.plot_time_series(
        flight_data,
        results["kite_yaw_s" + str(imu)],
        ax,
        label="Measured",
        plot_phase=False,
    )

pu.plot_time_series(
    flight_data, results["kite_yaw"], ax, label="Estimated", plot_phase=False
)
ax.legend()

fig, ax = plt.subplots()
pu.plot_time_series(
    flight_data,
    results["kite_roll"] - results["kcu_roll"],
    ax,
    label="Roll kite-tether",
    plot_phase=False,
)

pu.plot_time_series(
    flight_data,
    results["kite_pitch"] - results["kcu_pitch"],
    ax,
    label="Pitch kite-tether",
    plot_phase=True,
)

ax.legend()

plt.show()
