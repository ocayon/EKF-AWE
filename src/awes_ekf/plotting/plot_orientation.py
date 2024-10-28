from awes_ekf.plotting.plot_utils import plot_time_series
import matplotlib.pyplot as plt
import numpy as np
from awes_ekf.plotting.color_palette import get_color_list, visualize_palette, set_plot_style, get_color

def plot_kite_orientation(results, flight_data, kite_imus = [0], kcu_imus = []):
    """
    Plot the kite orientation from the results and the flight data.
    The orientation is plotted as euler angles.
    """
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)  # 3 rows, 1 column

    colors = get_color_list()
    for column in results.columns:
        if "pitch" in column or "roll" in column or "yaw" in column:
            results.loc[:, column] = np.degrees(results[column])

    for column in flight_data.columns:
        if "pitch" in column or "roll" in column or "yaw" in column:
            flight_data.loc[:, column] = np.degrees(flight_data[column])

    
    # Plot roll
    for imu in kite_imus:
        plot_time_series(
            flight_data,
            flight_data["kite_roll_" + str(imu)],
            axs[0],
            label="IMU Srut " + str(imu),
            plot_phase=False,
            color= colors[imu+1]
        )
    plot_time_series(
        flight_data, results["kite_roll"], axs[0], label="EKF 1", plot_phase=False, color= colors[0]
    )
    axs[0].legend(loc = "lower center", frameon=True)
    axs[0].set_ylabel('Roll Angle [$^\circ$]')

    # Plot pitch
    
    for imu in kite_imus:
        plot_time_series(
            flight_data,
            flight_data["kite_pitch_" + str(imu)],
            axs[1],
            label="IMU Srut " + str(imu),
            plot_phase=False,
            color= colors[imu+1]
        )
    plot_time_series(
        flight_data, results["kite_pitch"], axs[1], label="EKF 1", plot_phase=False, color= colors[0]
    )
    
    axs[1].legend(loc = "lower center", frameon=True)
    axs[1].set_ylabel('Pitch Angle [$^\circ$]')

    # Plot yaw
    
    
    for imu in kite_imus:
        plot_time_series(
            flight_data,
            flight_data["kite_yaw_" + str(imu)]-360,
            axs[2],
            label="IMU Srut " + str(imu),
            plot_phase=False,
            color= colors[imu+1]
        )
    plot_time_series(
        flight_data, results["kite_yaw"]-360, axs[2], label=r"Aligned with $v_\mathrm{a}$", plot_phase=False, color= colors[0]
    )
    plot_time_series(flight_data, results["kite_yaw_kin"]-360, axs[2], label=r"Aligned with $v_\mathrm{k}$", plot_phase=False, color= colors[3])
    axs[2].legend(loc = "lower center", frameon=True)
    axs[2].set_ylabel('Yaw Angle [$^\circ$]')

    # Common x-label for the bottom plot
    axs[2].set_xlabel('Time')

    plt.tight_layout()  # Adjusts spacing between subplots to prevent overlap
    