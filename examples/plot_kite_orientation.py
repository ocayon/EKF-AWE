import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
import seaborn as sns
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style, get_color
from awes_ekf.postprocess.postprocessing import (
    remove_offsets_IMU_data,
    calculate_offset_pitch_depower_turn,
)

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
    addition="_va",
)

flight_data = remove_offsets_IMU_data(results, flight_data, sensor=0)

imus = [0,1]

for column in results.columns:
    if "pitch" in column or "roll" in column or "yaw" in column:
        results[column] = np.degrees(results[column])

for column in flight_data.columns:
    if "pitch" in column or "roll" in column or "yaw" in column:
        flight_data[column] = np.degrees(flight_data[column])


# Calculate errors
pitch_error = abs(results["kite_pitch_s0"] - results["kite_pitch"])
roll_error = abs(flight_data["kite_roll_s0"] - results["kite_roll"])
yaw_error = abs(flight_data["kite_yaw_s0"] - results["kite_yaw"])

mean_pitch_error = np.mean(pitch_error)
mean_roll_error = np.mean(roll_error)
mean_yaw_error = np.mean(yaw_error)

std_pitch_error = np.std(pitch_error)
std_roll_error = np.std(roll_error)
std_yaw_error = np.std(yaw_error)

print(f"Mean pitch error: {mean_pitch_error:.2f} deg, std: {std_pitch_error:.2f} deg")
print(f"Mean roll error: {mean_roll_error:.2f} deg, std: {std_roll_error:.2f} deg")
print(f"Mean yaw error: {mean_yaw_error:.2f} deg, std: {std_yaw_error:.2f} deg")

colors = get_color_list()


plot_mask = flight_data["cycle"] == 60
results = results[plot_mask]
flight_data = flight_data[plot_mask]
# %%
import matplotlib.pyplot as plt

# Create a figure and three subplots
fig, axs = plt.subplots(3, 1, figsize=(10,7))  # 3 rows, 1 column

# Define the titles for each subplot
titles = ['Roll', 'Pitch', 'Yaw']

# Define colors for IMUs
color_imu = ["Green", "Orange"]

# Iterate over each subplot
for i, ax in enumerate(axs):
    # Plot estimated data
    if titles[i].lower() == "yaw":
        pu.plot_time_series(
            flight_data, results[f"kite_{titles[i].lower()}"], ax, label="Following $v_a$", color=colors[0]
        )
    else:
        pu.plot_time_series(
            flight_data, results[f"kite_{titles[i].lower()}"], ax, label="Bridle tether element", color=colors[0]
        )
    
    # Plot measured and corrected data for each IMU
    for imu in imus:
        # if titles[i].lower() == "pitch":
            
        #     pu.plot_time_series(
        #         flight_data,
        #         results[f"kite_{titles[i].lower()}_s" + str(imu)],
        #         ax,
        #         label=f"IMU {imu} Corrected Deformation",
        #         color=get_color("Dark " + color_imu[imu]),
        #     )
        pu.plot_time_series(
            flight_data,
            flight_data[f"kite_{titles[i].lower()}_s" + str(imu)],
            ax,
            label=f"IMU {imu} Measured",
            color=get_color(color_imu[imu]),
        )
    
    # Set the title for each subplot
    
    ax.set_ylabel(titles[i])
    ax.legend(loc = "best")
ax.set_xlabel("Time (s)")
# Adjust layout for better readability
plt.tight_layout()

# Save the figure if needed
plt.savefig("euler_angles.pdf", dpi=300)

# Display the plot
# plt.show()


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



#Plot yaw rate
fig, ax = plt.subplots()
pu.plot_time_series(
    flight_data,
    flight_data["kite_yaw_rate_s0"]%360/180*np.pi,
    ax,
    label="Yaw rate",
    plot_phase=False,
)
plt.show()

