import matplotlib.pyplot as plt
import numpy as np
from awes_ekf.plotting.color_palette import set_plot_style_no_latex
from awes_ekf.plotting.plot_utils import plot_time_series

def calculate_azimuth_elevation(x, y, z):
    # Calculate the azimuth angle (in radians)
    azimuth = np.arctan2(y, x)
    
    # Calculate the elevation angle (in radians)
    distance = np.sqrt(x**2 + y**2 + z**2)
    elevation = np.arcsin(z / distance)
    
    return azimuth, elevation

def plot_kinematics(results, flight_data, config_data):
    set_plot_style_no_latex()

    # Plot the azimuth and elevation angles
    fig, axs = plt.subplots(2, 1, figsize=(6, 10))
    mean_wind_dir = np.mean(results["wind_direction"])

    azimuth, elevation = calculate_azimuth_elevation(results["kite_position_x"], results["kite_position_y"], results["kite_position_z"])
    axs[0].plot(np.rad2deg(azimuth-mean_wind_dir), np.rad2deg(elevation), label="EKF")
    azimuth, elevation = calculate_azimuth_elevation(flight_data["kite_position_x"], flight_data["kite_position_y"], flight_data["kite_position_z"])
    axs[0].plot(np.rad2deg(azimuth-mean_wind_dir), np.rad2deg(elevation), label="GPS")
    axs[0].legend()
    axs[0].set_xlabel("Azimuth [deg]")
    axs[0].set_ylabel("Elevation [deg]")
    # Plot the radial distance
    r = np.sqrt(results["kite_position_x"]**2 + results["kite_position_y"]**2+ results["kite_position_z"]**2)
    axs[1].plot(flight_data["time"], r, label="EKF",linewidth=1)
    r = np.sqrt(flight_data["kite_position_x"]**2 + flight_data["kite_position_y"]**2+ flight_data["kite_position_z"]**2)
    axs[1].plot(flight_data["time"], r, label="Measured")
    axs[1].plot(flight_data["time"], flight_data["tether_length"]+11.5, label="Measured tether length")
    axs[1].legend()
    axs[1].set_xlabel("Time [s]")
    axs[1].set_ylabel("Radial Distance/Tether Length [m]")
    plt.tight_layout()

    # Plot the kite velocity and apparent wind velocity
    plt.figure()
    kite_speed = np.sqrt(results['kite_velocity_x']**2+results['kite_velocity_y']**2+results['kite_velocity_z']**2)
    meas_kite_speed = np.sqrt(flight_data['kite_velocity_x']**2+flight_data['kite_velocity_y']**2+flight_data['kite_velocity_z']**2)
    plt.plot(results['time'], kite_speed, label='EKF kite speed')
    plt.plot(flight_data['time'], meas_kite_speed, label='Measured kite speed')
    plt.plot(flight_data['time'], results["kite_apparent_windspeed"], label='EKF apparent wind speed')
    try:
        plt.plot(flight_data['time'], flight_data["kite_apparent_windspeed"], label='Measured apparent wind speed')
    except:
        pass
    plt.xlabel('Time [s]')
    plt.ylabel('Speed [m/s]')
    plt.legend()
    plt.title('Kite speed comparison')

    # Plot the kite orientation
    plot_kite_orientation(results, flight_data, config_data)

    plt.show()


def plot_kite_orientation(results, flight_data, config_data):
    """
    Plot the kite orientation from the results and the flight data.
    The orientation is plotted as euler angles.
    """
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)  # 3 rows, 1 column

    for column in results.columns:
        if "pitch" in column or "roll" in column or "yaw" in column:
            results.loc[:, column] = np.unwrap((results[column]) % (2*np.pi))

    for column in flight_data.columns:
        if "pitch" in column or "roll" in column or "yaw" in column:
            flight_data.loc[:, column] = np.unwrap((flight_data[column]) % (2*np.pi))

    
    kite_imus = config_data["kite"].get("sensor_ids", [])
    kcu_imus = config_data["kcu"].get("sensor_ids", [])
    # Plot roll
    for imu in kite_imus:
        plot_time_series(
            flight_data,
            np.degrees(flight_data["kite_roll_" + str(imu)]),
            axs[0],
            label="IMU Srut " + str(imu),
            plot_phase=False
        )
    for imu in kcu_imus:
        plot_time_series(
            flight_data,
            np.degrees(flight_data["kcu_roll_" + str(imu)]),
            axs[0],
            label="KCU " + str(imu),
            plot_phase=False
        )
    plot_time_series(
        flight_data, np.degrees(results["kite_roll"]), axs[0], label="EKF 1", plot_phase=False
    )
    axs[0].legend(loc = "lower center", frameon=True)
    axs[0].set_ylabel('Roll Angle [$^\circ$]')

    # Plot pitch
    
    for imu in kite_imus:
        plot_time_series(
            flight_data,
            np.degrees(flight_data["kite_pitch_" + str(imu)]),
            axs[1],
            label="IMU Srut " + str(imu),
            plot_phase=False
        )
    for imu in kcu_imus:
        plot_time_series(
            flight_data,
            np.degrees(flight_data["kcu_pitch_" + str(imu)]),
            axs[1],
            label="KCU " + str(imu),
            plot_phase=False
        )
    plot_time_series(
        flight_data, np.degrees(results["kite_pitch"]), axs[1], label="EKF 1", plot_phase=False
    )
    
    axs[1].legend(loc = "lower center", frameon=True)
    axs[1].set_ylabel('Pitch Angle [$^\circ$]')

    # Plot yaw
    
    
    for imu in kite_imus:
        plot_time_series(
            flight_data,
            np.degrees(flight_data["kite_yaw_" + str(imu)]),
            axs[2],
            label="IMU Srut " + str(imu),
            plot_phase=False
        )
    for imu in kcu_imus:
        plot_time_series(
            flight_data,
            np.degrees(flight_data["kcu_yaw_" + str(imu)]),
            axs[2],
            label="KCU " + str(imu),
            plot_phase=False
        )
    plot_time_series(
        flight_data, np.degrees(results["kite_yaw"]), axs[2], label=r"Aligned with $v_\mathrm{a}$", plot_phase=False
    )
    plot_time_series(flight_data, np.degrees(results["kite_yaw_kin"]), axs[2], label=r"Aligned with $v_\mathrm{k}$", plot_phase=False)
    axs[2].legend(loc = "lower center", frameon=True)
    axs[2].set_ylabel('Yaw Angle [$^\circ$]')

    # Common x-label for the bottom plot
    axs[2].set_xlabel('Time')

    plt.tight_layout()  # Adjusts spacing between subplots to prevent overlap