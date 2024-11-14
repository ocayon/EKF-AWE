from awes_ekf.setup.settings import kappa, z0
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from awes_ekf.plotting.color_palette import get_color_list

colors = get_color_list()

def plot_wind_velocity(results, flight_data, config_data):

    for col in flight_data.columns:
        lidar_data = False
        if "Wind_Speed_m_s" in col:
            lidar_data = True
            break
    if lidar_data:
        flight_data = interpolate_lidar_data(flight_data, results)

    plot_wind_timeseries(results, flight_data)
    plt.show()
    
def plot_wind_timeseries(results, flight_data):
    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(16, 6), sharex=True)
    
    # Plot horizontal wind speed
    axs[0].plot(flight_data["time"], results["wind_speed_horizontal"], label="EKF", color=colors[0])

    # Plot wind direction
    axs[1].plot(flight_data["time"], results["wind_direction"] * 180 / np.pi, label="EKF", color=colors[0])

    # Plot vertical wind speed
    axs[2].plot(flight_data["time"], results["wind_speed_vertical"], label="EKF", color=colors[0])    
    

    if "ground_wind_speed" in flight_data.columns:
        uf = flight_data["ground_wind_speed"] * kappa / np.log(10 / z0)
        vel_ground = uf / kappa * np.log(results["kite_position_z"] / z0)
        axs[0].plot(flight_data["time"], vel_ground, label="Ground", color=colors[2])
        axs[1].plot(flight_data["time"], flight_data["ground_wind_direction"], label="Ground", color=colors[2])

    if "interp_wind_speed" in flight_data.columns:
        axs[0].plot(flight_data["time"], flight_data["interp_wind_speed"], label="Lidar", color=colors[1])
        axs[0].fill_between(
            flight_data["time"],
            flight_data["interp_wind_speed_min"],
            flight_data["interp_wind_speed_max"],
            color=colors[1],
            alpha=0.3
        )
        axs[1].plot(flight_data["time"], 270 - flight_data["interp_wind_direction"], label="Lidar", color=colors[1])
        axs[2].plot(flight_data["time"], flight_data["interp_z_wind"], label="Lidar", color=colors[1])

    # Set x-axis limits
    axs[0].set_xlim([flight_data["time"].iloc[0], flight_data["time"].iloc[-1]])
    
    # Set axis labels
    axs[2].set_xlabel("Time (s)")
    axs[0].set_ylabel("Wind speed [m/s]")
    axs[1].set_ylabel("Wind direction [deg]")
    axs[2].set_ylabel("Vertical wind speed [m/s]")
    axs[2].set_xlabel("Time of day [hh:mm]")
    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    axs[0].legend(frameon=True)
    axs[1].legend(frameon=True)
    axs[2].legend(frameon=True)

    # Adjust layout
    plt.tight_layout()



def interpolate_lidar_data(flight_data, results):

    lidar_heights = []

    for column in flight_data.columns:
        if "Wind_Speed_m_s" in column:
            height = "".join(filter(str.isdigit, column))
            lidar_heights.append(int(height))
    # Sort the lidar heights
    lidar_heights = sorted(lidar_heights)
    # lidar_heights = lidar_heights[::-1]

    interp_wind_speeds = []
    interp_wind_speeds_max = []
    interp_wind_speeds_min = []
    interp_wind_directions = []
    interp_z_winds = []
    for index in flight_data.index:
        wind_speeds = []
        wind_speeds_max = []
        wind_speeds_min = []
        wind_directions = []
        z_wind = []
        for height in lidar_heights:
            wind_speeds.append(flight_data[str(height)+"m_Wind_Speed_m_s"].iloc[index])
            if str(height)+"m_Wind_Speed_max_m_s" not in flight_data.columns:
                flight_data[str(height)+"m_Wind_Speed_max_m_s"] = flight_data[str(height)+"m_Wind_Speed_m_s"]
                flight_data[str(height)+"m_Wind_Speed_min_m_s"] = flight_data[str(height)+"m_Wind_Speed_m_s"]
               
            wind_speeds_max.append(flight_data[str(height)+"m_Wind_Speed_max_m_s"].iloc[index])
            wind_speeds_min.append(flight_data[str(height)+"m_Wind_Speed_min_m_s"].iloc[index])
            wind_directions.append(flight_data[str(height)+"m_Wind_Direction_deg"].iloc[index])
            z_wind.append(flight_data[str(height)+"m_Z-wind_m_s"].iloc[index])

        z_pos = results["kite_position_z"].iloc[index]

        interp_wind_speeds.append(log_interp(z_pos, lidar_heights, wind_speeds))
        interp_wind_speeds_max.append(log_interp(z_pos, lidar_heights, wind_speeds_max))
        interp_wind_speeds_min.append(log_interp(z_pos, lidar_heights, wind_speeds_min))
        interp_wind_directions.append(np.interp(z_pos, lidar_heights, wind_directions))
        interp_z_winds.append(np.interp(z_pos, lidar_heights, z_wind))

    flight_data["interp_wind_speed"] = interp_wind_speeds
    flight_data["interp_wind_speed_max"] = interp_wind_speeds_max
    flight_data["interp_wind_speed_min"] = interp_wind_speeds_min
    flight_data["interp_wind_direction"] = interp_wind_directions
    flight_data["interp_z_wind"] = interp_z_winds

    return flight_data

def log_interp(height, heights, values):
    """
    Interpolate or extrapolate values on a log scale.
    
    :param height: Height to interpolate or extrapolate.
    :param heights: Known heights corresponding to the values.
    :param values: Known values at the given heights.
    :return: Interpolated or extrapolated value at the specified height.
    """
    # Ensure inputs are numpy arrays
    heights = np.array(heights)
    values = np.array(values)
    
    # Sort heights and values just in case they are not sorted
    sorted_indices = np.argsort(heights)
    heights = heights[sorted_indices]
    values = values[sorted_indices]
    
    if height <= heights.min():
        # Extrapolate below the minimum height
        uf = values[0]*kappa/np.log(heights[0]/z0)
        return uf/kappa*np.log(height/z0)
    elif height >= heights.max():
        # Extrapolate above the maximum height
       uf = values[-1]*kappa/np.log(heights[-1]/z0)
       return uf/kappa*np.log(height/z0)
    else:
        # Interpolate within the range
        lower_idx = np.where(heights <= height)[0].max()
        upper_idx = np.where(heights >= height)[0].min()
        z1, z2 = heights[lower_idx], heights[upper_idx]
        u1, u2 = values[lower_idx], values[upper_idx]
        return u1 + (np.log(height) - np.log(z1)) / (np.log(z2) - np.log(z1)) * (u2 - u1)
    