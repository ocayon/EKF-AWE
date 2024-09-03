# Plotting utilities for the project
# Author: Oriol Cayon
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.utils import calculate_reference_frame_euler
import seaborn as sns
from awes_ekf.plotting.color_palette import get_color_list, get_color
from awes_ekf.setup.settings import kappa, z0

def find_turn_law(flight_data):

    window_size = 10
    yaw_rate = np.diff(flight_data["kite_0_yaw"]) / np.diff(flight_data["time"])
    yaw_rate = np.concatenate((yaw_rate, [0]))
    yaw_rate = np.convolve(
        yaw_rate / 180 * np.pi, np.ones(window_size) / window_size, mode="same"
    )
    mask = yaw_rate < -1.8
    yaw_rate[mask] += np.pi
    mask = yaw_rate > 1.8
    yaw_rate[mask] += -np.pi

    # opt_res = least_squares(get_tether_end_position, list(calculate_polar_coordinates(np.array(kite_pos))), args=args,
    #                         kwargs={'find_force': False}, verbose=0)


# def calculate_yaw_rate(x, us, va, beta, yaw,v_kite,radius,forces):
#     norm_va = va
#     norm_v = v_kite
#     yaw_rate = x[0]*norm_va**2*(us)

#     if 'weight' in forces:
#         yaw_rate += -x[1]*np.cos(beta)*np.sin(yaw)
#     if 'centripetal' in forces:
#         yaw_rate += x[2]*norm_v**2/radius

#     # if 'sideslip' in forces:
#     #     yaw_rate += x[4]*(np.cos(beta)*np.cos(azimuth)*np.sin(yaw))*norm_va
#     if 'tether' in forces:
#         yaw_rate += x[4]*norm_va**2
#     if 'centripetal' in forces:
#         yaw_rate = yaw_rate/(x[3]*norm_va)
#     else:
#         yaw_rate = yaw_rate/(x[3]*norm_va+x[2]*norm_v)


#     return yaw_rate


def plot_wind_speed(
    results,
    flight_data,
    axs,
    lidar_heights=[],
    IMU_0=False,
    IMU_1=False,
    EKF_tether=False,
    EKF=True,
    savefig=False,
):
    """
    Plot wind speed based on kite and KCU IMU data
    :param flight_data: flight data
    :return: wind speed plot
    """
    palette = get_color_list()
    

    i = 1
    for column in flight_data.columns:
        if "Wind_Speed_m_s" in column:
            height = "".join(filter(str.isdigit, column))
            vw_max_col = height + "m_Wind_Speed_max_m_s"
            vw_min_col = height + "m_Wind_Speed_min_m_s"
            label = "Lidar " + height + "m height"
            height = int(height)

            if height in lidar_heights or lidar_heights == []:

                axs[0].fill_between(
                    flight_data["time"],
                    flight_data[vw_min_col],
                    flight_data[vw_max_col],
                    color=palette[i % len(palette)],
                    alpha=0.3,
                )
                axs[0].plot(
                    flight_data["time"],
                    flight_data[column],
                    color=palette[i % len(palette)],
                    label=label,
                )

        if "Wind_Direction" in column:
            height = "".join(filter(str.isdigit, column))
            label = "Lidar " + height + "m height"
            height = int(height)
            if height in lidar_heights or lidar_heights == []:
                axs[1].plot(
                    flight_data["time"],
                    360 - 90 - flight_data[column],
                    color=palette[i % len(palette)],
                    label=label,
                )

        if "Z-wind_m_s" in column:
            height = "".join(filter(str.isdigit, column))
            label = "Lidar " + height + "m height"
            height = int(height)
            if height in lidar_heights or lidar_heights == []:
                axs[2].plot(
                    flight_data["time"],
                    -flight_data[column],
                    color=palette[i % len(palette)],
                    label=label,
                )
                i += 1

    if IMU_0:
        from awes_ekf.postprocess.postprocessing import calculate_wind_speed_airborne_sensors
        flight_data = calculate_wind_speed_airborne_sensors(results, flight_data)
        vw_mod = np.sqrt(
            flight_data["vwx_IMU_0"] ** 2
            + flight_data["vwy_IMU_0"] ** 2
            + flight_data["vwz_IMU_0"] ** 2
        )
        axs[0].plot(flight_data["time"], vw_mod, label="Pitot+Vanes", alpha=0.8)
        vw_dir = np.arctan2(flight_data["vwy_IMU_0"], flight_data["vwx_IMU_0"])
        axs[1].plot(flight_data["time"], np.degrees(vw_dir) % 360, alpha=0.8)
        axs[2].plot(
            flight_data["time"], flight_data["vwz_IMU_0"], label="IMU_0", alpha=0.5
        )
    if IMU_1:
        vw_mod = np.sqrt(
            flight_data["vwx_IMU_1"] ** 2
            + flight_data["vwy_IMU_1"] ** 2
            + flight_data["vwz_IMU_1"] ** 2
        )
        axs[0].plot(flight_data["time"], vw_mod, label="IMU_1", alpha=0.5)
        vw_dir = np.arctan2(flight_data["vwy_IMU_1"], flight_data["vwx_IMU_1"])
        axs[1].plot(flight_data["time"], np.degrees(vw_dir) % 360, alpha=0.5)
        axs[2].plot(
            flight_data["time"], flight_data["vwz_IMU_1"], label="IMU_1", alpha=0.5
        )
    if EKF_tether:
        vw_mod = np.sqrt(
            flight_data["vwx_EKF"] ** 2
            + flight_data["vwy_EKF"] ** 2
            + flight_data["vwz_EKF"] ** 2
        )
        axs[0].plot(flight_data["time"], vw_mod, label="EKF_tether", alpha=0.5)
        vw_dir = np.arctan2(flight_data["vwy_EKF"], flight_data["vwx_EKF"])
        axs[1].plot(flight_data["time"], np.degrees(vw_dir) % 360, alpha=0.5)
        axs[2].plot(
            flight_data["time"], flight_data["vwz_EKF"], label="EKF_tether", alpha=0.5
        )
    if EKF:
        wvel = results["wind_speed_horizontal"]
        vw = np.vstack(
            (
                wvel * np.cos(results["wind_direction"]),
                wvel * np.sin(results["wind_direction"]),
                np.zeros(len(results)),
            )
        ).T
        axs[0].plot(
            flight_data["time"], np.linalg.norm(vw, axis=1), label="EKF", alpha=0.8
        )
        axs[1].plot(
            flight_data["time"],
            np.degrees(results["wind_direction"]),
            label="EKF",
            alpha=0.8,
        )
        axs[2].plot(flight_data["time"], results["wind_speed_vertical"], label="EKF", alpha=0.8)

    min_wdir = min(flight_data["ground_wind_direction"])
    max_wdir = max(flight_data["ground_wind_direction"])

    y1 = np.full(len(flight_data), min_wdir - 20)
    y2 = np.full(len(flight_data), max_wdir + 20)
    # mask = (flight_data['turn_straight'] == 'straight') & (flight_data['powered'] == 'powered')
    # axs[1].fill_between(flight_data['time'], y1,y2,where=mask, color='blue', alpha=0.2, label='Straight')
    # mask = (flight_data['turn_straight'] == 'turn') & (flight_data['powered'] == 'powered')
    # axs[1].fill_between(flight_data['time'], y1,y2,where=mask, color='red', alpha=0.2, label='Turn')
    # mask = (flight_data['powered'] == 'depowered')
    # axs[1].fill_between(flight_data['time'], y1,y2,where=mask, color='green', alpha=0.2, label='Depowered')

    axs[0].plot(
        flight_data["time"],
        flight_data["ground_wind_speed"],
        label="Ground",
        color="grey",
        alpha=0.8,
    )
    axs[1].plot(
        flight_data["time"],
        flight_data["ground_wind_direction"],
        label="Ground",
        color="grey",
        alpha=0.8,
    )

    axs[0].set_ylim([0, 20])
    # axs[0].legend()
    axs[0].set_ylabel("Wind speed (m/s)")
    axs[0].set_xlabel("Time (s)")
    # axs[0].grid()
    axs[1].legend()
    axs[1].set_ylabel("Wind direction (deg)")
    axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylim([175,275])
    # axs[1].grid()
    # axs[2].legend()
    axs[2].set_ylabel("Vertical Wind speed (m/s)")
    axs[2].set_xlabel("Time (s)")
    # axs[2].grid()
    sns.set(style="whitegrid")
    # Enhance overall aesthetics
    for ax in axs:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

    if savefig:
        plt.tight_layout()
        plt.savefig("wind_speed.png", dpi=300)


def plot_wind_speed_height_bins(results, flight_data, lidar_heights=[], savefig=False):
    """
    Plot wind speed based on kite and KCU IMU data
    :param flight_data: flight data
    :return: wind speed plot
    """
    palette = get_color_list()
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    wvel = results["wind_speed_horizontal"]
    wdir = np.degrees(results["wind_direction"])
    i0 = 0
    wvel100 = []
    wdir100 = []
    wvel150 = []
    wdir150 = []
    wvel200 = []
    wdir200 = []
    wvel250 = []
    wdir250 = []
    t_lidar = []
    i_change = []
    for column in flight_data.columns:
        if "Wind_Speed_m_s" in column:
            break
    for i in range(len(flight_data) - 1):
        if flight_data[column].iloc[i] != flight_data[column].iloc[i + 1]:
            wvel = results["wind_speed_horizontal"].iloc[i0:i]
            wdir = np.degrees(results["wind_direction"].iloc[i0:i])
            wvel100.append(np.mean(wvel[results.kite_position_z < 100]))
            wdir100.append(np.mean(wdir[results.kite_position_z < 100]))
            wvel150.append(
                np.mean(wvel[(results.kite_position_z < 150) & (results.kite_position_z > 100)])
            )
            wdir150.append(
                np.mean(wdir[(results.kite_position_z < 150) & (results.kite_position_z > 100)])
            )
            wvel200.append(
                np.mean(wvel[(results.kite_position_z > 150) & (results.kite_position_z < 200)])
            )
            wdir200.append(
                np.mean(wdir[(results.kite_position_z > 150) & (results.kite_position_z < 200)])
            )
            wvel250.append(np.mean(wvel[results.kite_position_z > 200]))
            wdir250.append(np.mean(wdir[results.kite_position_z > 200]))
            t_lidar.append(flight_data["time"].iloc[i0])
            i_change.append(i0)
            i0 = i + 1

    i = 0
    for column in flight_data.columns:
        if "Wind_Speed_m_s" in column:
            height = "".join(filter(str.isdigit, column))
            vw_max_col = height + "m Wind Speed max (m/s)"
            vw_min_col = height + "m Wind Speed min (m/s)"
            label = "Lidar " + height + "m height"
            height = int(height)
            col_save = column
            if height in lidar_heights or lidar_heights == []:
                selected_values = [flight_data.iloc[j][column] for j in i_change]
                # axs[0].fill_between(flight_data['time'], flight_data[vw_min_col], flight_data[vw_max_col], color=palette[i % len(palette)], alpha=0.3)
                axs[0].plot(
                    t_lidar,
                    selected_values,
                    color=palette[i % len(palette)],
                    label=label,
                )

                i += 1
    i = 0
    for column in flight_data.columns:
        if "Wind_Direction" in column:
            height = "".join(filter(str.isdigit, column))
            label = "Lidar " + height + "m height"
            height = int(height)
            if height in lidar_heights or lidar_heights == []:
                selected_values = [flight_data.iloc[j][column] for j in i_change]
                # axs[0].fill_between(flight_data['time'], flight_data[vw_min_col], flight_data[vw_max_col], color=palette[i % len(palette)], alpha=0.3)
                axs[1].plot(
                    t_lidar,
                    360 - 90 - np.array(selected_values),
                    color=palette[i % len(palette)],
                    label=label,
                )
                i += 1
    i = 0
    for column in flight_data.columns:
        if "Z-wind (m/s)" in column:
            height = "".join(filter(str.isdigit, column))
            label = "Lidar " + height + "m height"
            height = int(height)
            if height in lidar_heights or lidar_heights == []:
                axs[2].plot(
                    flight_data["time"],
                    flight_data[column],
                    color=palette[i % len(palette)],
                    label=label,
                )
                i += 1

    axs[0].plot(
        t_lidar, wvel100, label="EKF 100m bin", color=palette[0], linestyle="--"
    )
    axs[0].plot(
        t_lidar, wvel150, label="EKF 150m bin", color=palette[1], linestyle="--"
    )
    axs[0].plot(
        t_lidar, wvel200, label="EKF 200m bin", color=palette[2], linestyle="--"
    )
    axs[0].plot(
        t_lidar, wvel250, label="EKF 250m bin", color=palette[3], linestyle="--"
    )
    axs[1].plot(t_lidar, wdir100, label="100m", color=palette[0], linestyle="--")
    axs[1].plot(t_lidar, wdir150, label="150m", color=palette[1], linestyle="--")
    axs[1].plot(t_lidar, wdir200, label="200m", color=palette[2], linestyle="--")
    axs[1].plot(t_lidar, wdir250, label="250m", color=palette[3], linestyle="--")

    # axs[0].set_ylim([0,20])
    axs[0].legend()
    axs[0].set_ylabel("Wind speed (m/s)")
    axs[0].set_xlabel("Time (s)")
    axs[0].grid()
    # axs[1].legend()
    axs[1].set_ylabel("Wind direction (deg)")
    axs[1].set_xlabel("Time (s)")
    axs[1].grid()
    # axs[2].legend()
    axs[2].set_ylabel("Wind speed (m/s)")
    axs[2].grid()
    if savefig:
        plt.tight_layout()
        plt.savefig("wind_speed_bins.png", dpi=300)


def plot_aero_coeff_vs_aoa_ss(
    results,
    flight_data,
    cycles_plotted,
    kite_sensors,
    savefig=False,
):
    """
    Plot wind speed based on kite and KCU IMU data
    :param flight_data: flight data
    :return: wind speed plot
    """
    palette = get_color_list()
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Aero coefficients vs aoa and ss")

    mask_cycle = np.any(
        [flight_data["cycle"] == cycle for cycle in cycles_plotted], axis=0
    )

    # Define time bounds for x-axis
    time_min = flight_data[mask_cycle]["time"].min()
    time_max = flight_data[mask_cycle]["time"].max()

    # Plotting each aerodynamic coefficient
    axs[0].plot(results[mask_cycle]["time"], results[mask_cycle]["wing_lift_coefficient"])
    axs[1].plot(
        results[mask_cycle]["time"], results[mask_cycle]["wing_drag_coefficient"], label="cd_wing (Total)"
    )
    axs[1].plot(
        results[mask_cycle]["time"], results[mask_cycle]["kcu_drag_coefficient"], label="cd_wing (KCU)"
    )
    axs[1].plot(
        results[mask_cycle]["time"],
        results[mask_cycle]["tether_drag_coefficient"],
        label="cd_wing (Tether)",
    )
    axs[2].plot(results[mask_cycle]["time"], results[mask_cycle]["wing_sideforce_coefficient"], label="cs_wing")

    # AOA and Side Slip plots with conditions
    axs[3].plot(
        results[mask_cycle]["time"],
        results[mask_cycle]["wing_angle_of_attack"],
        label="aoa EKF",
    )
    axs[4].plot(
        results[mask_cycle]["time"],
        results[mask_cycle]["wing_sideslip_angle"],
        label="ss EKF",
    )
    aoa_imu = np.zeros(len(results[mask_cycle]))
    ss_imu = np.zeros(len(results[mask_cycle]))
    for sensor in kite_sensors:
        aoa_imu += results[mask_cycle]["wing_angle_of_attack_imu_" + str(sensor)]
        ss_imu += results[mask_cycle]["wing_sideslip_angle_imu_" + str(sensor)]
    aoa_imu /= len(kite_sensors)
    ss_imu /= len(kite_sensors)
    axs[3].plot(
        results[mask_cycle]["time"],
        aoa_imu,
        label="aoa IMU",
    )

    # Vane data
    if "wing_angle_of_attack" in flight_data.columns:
        axs[3].plot(
            flight_data[mask_cycle]["time"],
            flight_data[mask_cycle]["wing_angle_of_attack"],
            label="aoa vane",
        )
        axs[4].plot(
            flight_data[mask_cycle]["time"],
            flight_data[mask_cycle]["wing_sideslip_angle"],
            label="ss vane",
        )

    # Highlight operational modes
    i = 0
    for ax in axs:
        mask_straight = (flight_data["turn_straight"] == "straight") & mask_cycle
        mask_turn = (flight_data["turn_straight"] == "turn") & mask_cycle
        if "powered" in flight_data.columns:
            mask_depowered = (flight_data["powered"] == "depowered") & mask_cycle
        else:
            mask_depowered = np.zeros(len(flight_data), dtype=bool)
        if i == 0:
            ax.fill_between(
                flight_data["time"],
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                where=mask_straight,
                color="blue",
                alpha=0.2,
                label="Straight",
            )
            ax.fill_between(
                flight_data["time"],
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                where=mask_turn,
                color="red",
                alpha=0.2,
                label="Turn",
            )
            ax.fill_between(
                flight_data["time"],
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                where=mask_depowered,
                color="green",
                alpha=0.2,
                label="Reel-in",
            )
        else:
            ax.fill_between(
                flight_data["time"],
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                where=mask_straight,
                color="blue",
                alpha=0.2,
            )
            ax.fill_between(
                flight_data["time"],
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                where=mask_turn,
                color="red",
                alpha=0.2,
            )
            ax.fill_between(
                flight_data["time"],
                ax.get_ylim()[0],
                ax.get_ylim()[1],
                where=mask_depowered,
                color="green",
                alpha=0.2,
            )

        ax.set_xlim([time_min, time_max])  # Set x-axis limits to fit the data
        ax.grid(True)  # Enable grid for better data visualization

        i += 1
    # Labels, legends, and layout
    axs[0].set_ylabel

    axs[0].legend()
    axs[1].legend()
    axs[3].legend()
    axs[4].legend()
    axs[0].set_ylabel("cl_wing")
    axs[1].set_ylabel("cd_wing")
    axs[2].set_ylabel("cs_wing")
    axs[3].set_ylabel("Angle of attack (deg)")
    axs[4].set_ylabel("Side slip (deg)")
    axs[4].set_xlabel("Time (s)")


def plot_aero_coeff_vs_up_us(
    results,
    flight_data,
    cycles_plotted,
    IMU_0=False,
    IMU_1=False,
    EKF_tether=False,
    EKF=True,
    savefig=False,
):
    """
    Plot wind speed based on kite and KCU IMU data
    :param flight_data: flight data
    :return: wind speed plot
    """
    palette = get_color_list()
    fig, axs = plt.subplots(5, 1, figsize=(20, 12), sharex=True)
    fig.suptitle("Aero coefficients vs up and us")
    mask_cycle = np.any(
        [flight_data["cycle"] == cycle for cycle in cycles_plotted], axis=0
    )

    axs[0].plot(results[mask_cycle]["time"], results[mask_cycle]["wing_lift_coefficient"])
    axs[1].plot(results[mask_cycle]["time"], results[mask_cycle]["wing_drag_coefficient"])
    axs[2].plot(results[mask_cycle]["time"], results[mask_cycle]["wing_sideforce_coefficient"])

    axs[3].plot(flight_data[mask_cycle]["time"], flight_data[mask_cycle]["up"])
    axs[4].plot(flight_data[mask_cycle]["time"], flight_data[mask_cycle]["us"])

    mask = (
        (flight_data["turn_straight"] == "straight")
        & (flight_data["powered"] == "powered")
        & mask_cycle
    )
    axs[0].fill_between(
        flight_data["time"],
        0,
        1.5,
        where=mask,
        color="blue",
        alpha=0.2,
        label="Straight",
    )
    axs[1].fill_between(
        flight_data["time"],
        0,
        0.3,
        where=mask,
        color="blue",
        alpha=0.2,
        label="Straight",
    )
    axs[2].fill_between(
        flight_data["time"],
        -0.2,
        0.2,
        where=mask,
        color="blue",
        alpha=0.2,
        label="Straight",
    )
    axs[3].fill_between(
        flight_data["time"], -5, 30, where=mask, color="blue", alpha=0.2
    )
    axs[4].fill_between(
        flight_data["time"], -15, 15, where=mask, color="blue", alpha=0.2
    )
    mask = (
        (flight_data["turn_straight"] == "turn")
        & (flight_data["powered"] == "powered")
        & mask_cycle
    )
    axs[0].fill_between(
        flight_data["time"], 0, 1.5, where=mask, color="red", alpha=0.2, label="Turn"
    )
    axs[1].fill_between(
        flight_data["time"], 0, 0.3, where=mask, color="red", alpha=0.2, label="Turn"
    )
    axs[2].fill_between(
        flight_data["time"], -0.2, 0.2, where=mask, color="red", alpha=0.2, label="Turn"
    )
    axs[3].fill_between(flight_data["time"], -5, 30, where=mask, color="red", alpha=0.2)
    axs[4].fill_between(
        flight_data["time"], -15, 15, where=mask, color="red", alpha=0.2
    )
    mask = (flight_data["powered"] == "depowered") & mask_cycle
    axs[0].fill_between(
        flight_data["time"],
        0,
        1.5,
        where=mask,
        color="green",
        alpha=0.2,
        label="Depowered",
    )
    axs[1].fill_between(
        flight_data["time"],
        0,
        0.3,
        where=mask,
        color="green",
        alpha=0.2,
        label="Depowered",
    )
    axs[2].fill_between(
        flight_data["time"],
        -0.2,
        0.2,
        where=mask,
        color="green",
        alpha=0.2,
        label="Depowered",
    )
    axs[3].fill_between(
        flight_data["time"], -5, 30, where=mask, color="green", alpha=0.2
    )
    axs[4].fill_between(
        flight_data["time"], -15, 15, where=mask, color="green", alpha=0.2
    )

    axs[0].set_ylim([0, 1.2])
    axs[1].set_ylim([0, 0.3])
    axs[2].set_ylim([-0.12, 0.12])
    axs[3].set_ylim([0, 1])
    axs[4].set_ylim([-1, 1])
    axs[0].set_ylabel("cl_wing")
    axs[1].set_ylabel("cd_wing")
    axs[2].set_ylabel("cs_wing")
    axs[3].set_ylabel("up")
    axs[4].set_ylabel("us")
    # axs[0].grid()
    # axs[1].grid()
    # axs[2].grid()
    # axs[3].grid()
    # axs[4].grid()
    axs[0].legend()

    if savefig:
        plt.savefig("aero_coeff_vs_up_us.png", dpi=300)


def determine_turn_straight(row, threshold_us=0.4):

    if abs(row["us"]) > threshold_us:
        return "turn"
    else:
        return "straight"


def determine_powered_depowered(row, threshold_up=0.25):

    if row["up"] > threshold_up:
        return "depowered"
    else:
        return "powered"


def determine_left_right(row, threshold_azimuth=0):
    if row["kite_azimuth"] < threshold_azimuth:
        return "right"
    else:
        return "left"


from scipy.stats import gaussian_kde


# Function to calculate densities
def calculate_densities(x, y):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    return z


def plot_probability_density(x, y, fig, axs, xlabel=None, ylabel=None):
    z1 = calculate_densities(x, y)
    sc1 = axs.scatter(x, y, c=z1, cmap="viridis", label="Sensor Fusion")
    fig.colorbar(sc1, ax=axs, label="Probability Density")
    axs.set_ylabel(ylabel)
    axs.set_xlabel(xlabel)
    axs.grid()
    axs.legend()


def plot_hexbin_density(x, y, xlabel=None, ylabel=None):
    fig, ax = plt.subplots()
    hb = ax.hexbin(x, y, gridsize=50, cmap="viridis", bins="log")
    cb = fig.colorbar(hb, ax=ax, label="log10(N)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    # plt.show()


def plot_cl_wing_cd_wing_aoa(results, flight_data, mask, aoa_method, savefig=False):

    if aoa_method == "IMU_0":
        aoa = results["aoa_IMU_0"]
    elif aoa_method == "IMU_1":
        aoa = results["aoa_IMU_1"]
    elif aoa_method == "EKF":
        aoa = results["kite_aoa"]
    else:
        aoa = flight_data["kite_angle_of_attack"]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("cl_wing and cd_wing vs aoa")
    plot_probability_density(aoa[mask], results["wing_lift_coefficient"][mask], fig, axs[0], ylabel="cl_wing")
    plot_probability_density(aoa[mask], results["wing_drag_coefficient"][mask], fig, axs[1], "aoa", "cd_wing")

    if savefig == True:
        plt.tight_layout()
        plt.savefig("wind_profile.png", dpi=300)


def plot_cl_wing_cd_wing_up(results, flight_data, mask, aoa_method, savefig=False):

    up = flight_data["up"]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("cl_wing and cd_wing vs aoa")
    plot_probability_density(up[mask], results["wing_lift_coefficient"][mask], fig, axs[0], ylabel="cl_wing")
    plot_probability_density(up[mask], results["wing_drag_coefficient"][mask], fig, axs[1], "up", "cd_wing")
    if savefig == True:
        plt.tight_layout()
        plt.savefig("wind_profile.png", dpi=300)


def plot_cl_wing_cd_wing_ss(results, flight_data, mask, ss_method):

    if ss_method == "IMU_0":
        ss = results["ss_IMU_0"]
    elif ss_method == "IMU_1":
        ss = results["ss_IMU_1"]
    elif ss_method == "EKF":
        ss = results["kite_sideslip"]
    else:
        ss = flight_data["kite_sideslip_angle"]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.suptitle("cl_wing and cd_wing vs ss")
    plot_probability_density(ss[mask], results["wing_lift_coefficient"][mask], fig, axs[0], ylabel="cl_wing")
    plot_probability_density(ss[mask], results["wing_drag_coefficient"][mask], fig, axs[1], "ss", "cd_wing")


def plot_prob_coeff_vs_aoa_ss(results, coeff, mask, aoa_method):

    if aoa_method == "IMU_0":
        aoa = results["aoa_IMU_0"]
        ss = results["ss_IMU_0"]
    elif aoa_method == "IMU_1":
        aoa = results["aoa_IMU_1"]
        ss = results["ss_IMU_1"]
    elif aoa_method == "EKF":
        aoa = results["kite_aoa"]
        ss = results["kite_sideslip"]
    else:
        aoa = results["kite_aoa"]
        ss = np.zeros(len(results))

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle("Probability Density vs aoa and ss")
    plot_probability_density(aoa[mask], coeff[mask], fig, axs[0], xlabel="aoa")
    plot_probability_density(ss[mask], coeff[mask], fig, axs[1], "ss", "")


def plot_time_series(
    flight_data, y, ax, color=None, ylabel=None, label=None, plot_phase=False, xlabel=None
):
    t = flight_data.time
    ax.plot(t, y, color=color, label=label)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Set x-axis limits to match the time data range
    ax.set_xlim([t.min(), t.max()])

    if plot_phase:
        # Set y1 and y2 to cover the entire y-axis
        y1, y2 = ax.get_ylim()

        # Mask and fill for "straight and powered" phase
        mask = (abs(flight_data["us"]) < 0.4) & (
            flight_data["powered"] == "powered"
        )
        ax.fill_between(
            t,
            y1,
            y2,
            where=mask,
            color="blue",
            alpha=0.2,
            label="Straight",
        )

        # Mask and fill for "turn and powered" phase
        mask = (abs(flight_data["us"]) > 0.4) & (
            flight_data["powered"] == "powered"
        )
        ax.fill_between(
            t,
            y1,
            y2,
            where=mask,
            color="red",
            alpha=0.2,
            label="Turn",
        )

        # Mask and fill for "depowered" phase
        mask = flight_data["powered"] == "depowered"
        ax.fill_between(
            t,
            y1,
            y2,
            where=mask,
            color="green",
            alpha=0.2,
            label="Depowered",
        )

        # Ensure that y-axis limits are reset correctly after fill_between
        ax.set_ylim([y1, y2])


def plot_wind_profile(flight_data, results, savefig=False):

    palette = get_color_list()
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    i = 1
    lidar_heights = []
    min_vel = []
    max_vel = []
    min_dir = []
    max_dir = []
    for column in flight_data.columns:
        if "Wind_Speed_m_s" in column:
            height = "".join(filter(str.isdigit, column))
            vw_max_col = height + "m Wind Speed max (m/s)"
            vw_min_col = height + "m Wind Speed min (m/s)"
            label = "Lidar " + height + "m height"
            height = int(height)
            lidar_heights.append(height)
            min_vel.append(min(flight_data[column]))
            max_vel.append(max(flight_data[column]))

            i += 1
        if "Wind_Direction" in column:
            height = "".join(filter(str.isdigit, column))
            label = "Lidar " + height + "m height"
            height = int(height)
            min_dir.append(min(360 - 90 - flight_data[column]))
            max_dir.append(max(360 - 90 - flight_data[column]))

    axs[0].fill_betweenx(
        lidar_heights, min_vel, max_vel, color=palette[0], alpha=0.3, label="Lidar"
    )
    axs[1].fill_betweenx(
        lidar_heights, min_dir, max_dir, color=palette[0], alpha=0.3, label="Lidar"
    )

    wvelEKF = results["wind_speed_horizontal"]
    plot_hexbin_density(wvelEKF, results["kite_position_z"], fig, axs[0])
    # axs[0].scatter( wvelEKF, results['kite_position_z'], color=palette[1], label='EKF', alpha = 0.1)
    axs[1].scatter(
        np.degrees(results["wind_direction"]),
        results["kite_position_z"],
        color=palette[1],
        label="EKF",
        alpha=0.1,
    )

    axs[0].legend()
    axs[0].set_xlabel("Wind speed (m/s)")
    axs[0].set_ylabel("Height (m)")
    axs[0].grid()
    axs[0].set_xlim([0, 17])
    axs[1].legend()
    axs[1].set_xlabel("Wind direction (deg)")
    axs[1].grid()

    if savefig:
        plt.tight_layout()
        plt.savefig("wind_profile.png", dpi=300)


def plot_wind_profile_bins(flight_data, results, axs, step=20, savefig=False, color=None, label=None, lidar_data=False):
    # Extract data
    height = results["kite_position_z"]
    wvel = results["wind_speed_horizontal"]
    wdir = np.degrees(results["wind_direction"])

    if color is None:
        color = "Blue"
    if label is None:
        label = "EKF"

    # Define bins and calculate statistics
    bins = np.arange(int(height.min()) - step / 2, int(height.max()) + step / 2, step)
    bin_indices = np.digitize(height, bins)
    num_bins = len(bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Compute means and standard deviations
    wvel_means = [wvel[bin_indices == i].mean() for i in range(1, num_bins)]
    wdir_means = [wdir[bin_indices == i].mean() for i in range(1, num_bins)]
    wvel_stds = [wvel[bin_indices == i].std() for i in range(1, num_bins)]
    wdir_stds = [wdir[bin_indices == i].std() for i in range(1, num_bins)]

    

    # Set up plot-
    sns.set(style="whitegrid")

    # fig.suptitle('Wind Profile Analysis', fontsize=16)
    # fontsize = 16

    # Wind velocity plot
    axs[0].errorbar(
        wvel_means,
        bin_centers,
        xerr=wvel_stds,
        fmt="o-",
        color=get_color(color),
        ecolor=get_color(color, alpha=0.5),
        elinewidth=3,
        capsize=0,
        label=label,
    )

    axs[0].set_xlabel("Wind Speed ($\mathrm{m}/\mathrm{s}$)")
    axs[0].set_ylabel("Height ($\mathrm{m}$)")
    axs[0].legend()
    axs[0].set_xlim([min(wvel_means) - 3, max(wvel_means) + 3])

    # Wind direction plot
    axs[1].errorbar(
        wdir_means,
        bin_centers,
        xerr=wdir_stds,
        fmt="o-",
        color=get_color(color),
        ecolor=get_color(color, alpha=0.5),
        elinewidth=3,
        capsize=0,
        label=label,
    )
    if lidar_data:
        i = 1
        lidar_heights = []
        min_vel = []
        max_vel = []
        wvel = []
        min_dir = []
        max_dir = []
        wdir = []
        std_vel = []
        std_dir = []
        for column in flight_data.columns:
            if "Wind_Speed_m_s" in column:
                height = "".join(filter(str.isdigit, column))
                stdvw_col = height + "m_Wind_Speed_Dispersion_m_s"
                height = int(height)
                lidar_heights.append(height)
                min_vel.append(np.mean(flight_data[column] - flight_data[stdvw_col]))
                max_vel.append(np.mean(flight_data[column] + flight_data[stdvw_col]))
                wvel.append(np.mean(flight_data[column]))
                std_vel.append(np.std(flight_data[column]))
                i += 1

            if "Wind_Direction" in column:
                height = "".join(filter(str.isdigit, column))
                height = int(height)
                min_dir.append(min(360 - 90 - flight_data[column]))
                max_dir.append(max(360 - 90 - flight_data[column]))
                wdir.append(np.mean(360 - 90 - flight_data[column]))
                std_dir.append(np.std(flight_data[column]))

        sorted_indices = np.argsort(lidar_heights)
        lidar_heights = np.array(lidar_heights)[sorted_indices]
        wvel = np.array(wvel)[sorted_indices]
        std_vel = np.array(std_vel)[sorted_indices]
        wdir = np.array(wdir)[sorted_indices]
        std_dir = np.array(std_dir)[sorted_indices]

        axs[0].errorbar(
            wvel,
            lidar_heights,
            xerr=std_vel,
            fmt="o-",
            color=get_color("Light Gray"),
            ecolor=get_color("Light Gray", alpha=0.5),
            elinewidth=3,
            capsize=0,
            label="Lidar",
            )
        
        axs[1].errorbar(
            wdir,
            lidar_heights,
            xerr=std_dir,
            fmt="o-",
            color=get_color("Light Gray"),
            ecolor=get_color("Light Gray", alpha=0.5),
            elinewidth=3,
            capsize=0,
            label="Lidar",
        )
    axs[1].set_xlabel("Wind Direction ($^\circ$)")
    axs[1].set_ylabel("Height ($\mathrm{m}$)")
    axs[1].legend()
    axs[1].set_xlim([min(wdir_means) - 15, max(wdir_means) + 15])

    # Enhance overall aesthetics
    for ax in axs:
        ax.set_ylim([0, bin_centers.max() + step])

        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

    
    # plt.show()
    if savefig:
        plt.savefig("wind_profile_bins.png", dpi=300)

    return axs

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
    


def plot_wind_speed_log_interpolation(results, flight_data, axs,savefig=False):
    """
    Plot wind speed based on kite and KCU IMU data
    :param flight_data: flight data
    :return: wind speed plot
    """
    palette = get_color_list()
    
    

    axs[0].plot(flight_data["time"], flight_data["interp_wind_speed"], label="Interpolated", color="grey", alpha=0.8)
    axs[1].plot(flight_data["time"], 360-90-flight_data["interp_wind_direction"], label="Interpolated", color="grey", alpha=0.8)
    axs[2].plot(flight_data["time"], flight_data["interp_z_wind"], label="Interpolated", color="grey", alpha=0.8)

    axs[0].set_ylim([0, 20])
    # axs[0].legend()
    axs[0].set_ylabel("Wind speed (m/s)")
    axs[0].set_xlabel("Time (s)")
    # axs[0].grid()
    axs[1].legend()
    axs[1].set_ylabel("Wind direction (deg)")
    axs[1].set_xlabel("Time (s)")
    # axs[1].set_ylim([175,275])
    # axs[1].grid()
    # axs[2].legend()
    axs[2].set_ylabel("Vertical Wind speed (m/s)")
    axs[2].set_xlabel("Time (s)")
    # axs[2].grid()
    sns.set(style="whitegrid")
    # Enhance overall aesthetics
    for ax in axs:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)

    if savefig:
        plt.tight_layout()
        plt.savefig("wind_speed.png", dpi=300)


def plot_kite_reference_frame(results, flight_data, imus, frame_axis="xyz"):
    ## Create 3d plot of kite reference frame
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X_N")
    ax.set_ylabel("Y_E")
    ax.set_zlabel("Z_U")
    spacing = 40
    for i in np.arange(0, len(flight_data), spacing):
        len_arrow = 30
        # Calculate EKF tether orientation based on euler angles and plot it
        dcm = calculate_reference_frame_euler(
            results.kite_roll.iloc[i],
            results.kite_pitch.iloc[i],
            results.kite_yaw.iloc[i],
            eulerFrame="NED",
            outputFrame="ENU",
        )
        ex = dcm[:, 0]
        ey = dcm[:, 1]
        ez = dcm[:, 2]
        if "x" in frame_axis:
            ax.quiver(
                results["kite_position_x"].iloc[i],
                results["kite_position_y"].iloc[i],
                results["kite_position_z"].iloc[i],
                ex[0],
                ex[1],
                ex[2],
                color="green",
                length=len_arrow,
            )
        if "y" in frame_axis:
            ax.quiver(
                results["kite_position_x"].iloc[i],
                results["kite_position_y"].iloc[i],
                results["kite_position_z"].iloc[i],
                ey[0],
                ey[1],
                ey[2],
                color="green",
                length=len_arrow,
            )
        if "z" in frame_axis:
            ax.quiver(
                results["kite_position_x"].iloc[i],
                results["kite_position_y"].iloc[i],
                results["kite_position_z"].iloc[i],
                ez[0],
                ez[1],
                ez[2],
                color="green",
                length=len_arrow,
            )
        # Calculate IMU tether orientation based on euler angles and plot it
        for imu in imus:
            ex, ey, ez = calculate_reference_frame_euler(
                flight_data["kite_roll_s" + str(imu)].iloc[i],
                flight_data["kite_pitch_s" + str(imu)].iloc[i],
                flight_data["kite_yaw_s" + str(imu)].iloc[i],
                eulerFrame="NED",
                outputFrame="ENU",
            )

            ax.quiver(
                results["kite_position_x"].iloc[i],
                results["kite_position_y"].iloc[i],
                results["kite_position_z"].iloc[i],
                ex[0],
                ex[1],
                ex[2],
                color="b",
                length=len_arrow,
            )
            ax.quiver(
                results["kite_position_x"].iloc[i],
                results["kite_position_y"].iloc[i],
                results["kite_position_z"].iloc[i],
                ey[0],
                ey[1],
                ey[2],
                color="b",
                length=len_arrow,
            )
            ax.quiver(
                results["kite_position_x"].iloc[i],
                results["kite_position_y"].iloc[i],
                results["kite_position_z"].iloc[i],
                ez[0],
                ez[1],
                ez[2],
                color="b",
                length=len_arrow,
            )

    ax.plot(
        results.kite_position_x,
        results.kite_position_y,
        results.kite_position_z,
        color="grey",
        linestyle="--",
    )
    ax.scatter(
        results["kite_position_x"].iloc[0:spacing:-1],
        results["kite_position_y"].iloc[0:spacing:-1],
        results["kite_position_z"].iloc[0:spacing:-1],
        color="r",
    )
    ax.legend()
    ax.quiver(0, 0, 0, 0, 0, 1, color="black", length=len_arrow)
    ax.quiver(0, 0, 0, 0, 1, 0, color="black", length=len_arrow)
    ax.quiver(0, 0, 0, 1, 0, 0, color="black", length=len_arrow)
    ax.set_box_aspect([1, 1, 1])
    # plt.show()


def plot_cl_curve(cl, cd, aoa, mask, axs, label=None, savefig=False):
    cl_wing = cl[mask]
    cd_wing = cd[mask]
    alpha = aoa[mask]

    step = 1
    bins = np.arange(int(alpha.min()) - step / 2, int(alpha.max()) + step / 2, step)
    bin_indices = np.digitize(alpha, bins)  # Find the bin index for each alpha value
    num_bins = len(bins)

    cl_wing_means = np.array([cl_wing[bin_indices == i].mean() for i in range(1, num_bins)])
    cd_wing_means = np.array([cd_wing[bin_indices == i].mean() for i in range(1, num_bins)])

    cl_wing_stds = np.array([cl_wing[bin_indices == i].std() for i in range(1, num_bins)])
    cd_wing_stds = np.array([cd_wing[bin_indices == i].std() for i in range(1, num_bins)])
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Plot cl_wing and shade the area for std deviation
    axs[0, 0].plot(bin_centers, cl_wing_means, "o-", markersize=8, linewidth=2, label=label)
    axs[0, 0].fill_between(
        bin_centers, cl_wing_means - cl_wing_stds, cl_wing_means + cl_wing_stds, alpha=0.2
    )

    axs[0, 0].set_ylabel("$C_L$", fontsize=14)
    axs[0, 0].grid(True)

    # Plot cd_wing and shade the area for std deviation
    axs[0, 1].plot(bin_centers, cd_wing_means, "o-", markersize=8, linewidth=2, label=label)
    axs[0, 1].fill_between(
        bin_centers, cd_wing_means - cd_wing_stds, cd_wing_means + cd_wing_stds, alpha=0.2
    )

    axs[0, 1].set_ylabel("$C_{D}$", fontsize=14)
    axs[0, 1].grid(True)

    axs[1, 0].plot(bin_centers, cl_wing_means**3 / cd_wing_means**2, "o-", label=label)
    axs[1, 0].fill_between(
        bin_centers,
        (cl_wing_means - cl_wing_stds) ** 3 / (cd_wing_means + cd_wing_stds) ** 2,
        (cl_wing_means + cl_wing_stds) ** 3 / (cd_wing_means - cd_wing_stds) ** 2,
        alpha=0.2,
    )
    axs[1, 0].set_xlabel(r"$\alpha_{w,b}$ [$^\circ$]", fontsize=14)
    axs[1, 0].set_ylabel("$C_L^3/C_D^2$", fontsize=14)
    axs[1, 0].grid(True)

    axs[1, 1].plot(bin_centers, cl_wing_means / cd_wing_means, "o-", label=label)
    axs[1, 1].fill_between(
        bin_centers,
        (cl_wing_means - cl_wing_stds) / (cd_wing_means + cd_wing_stds),
        (cl_wing_means + cl_wing_stds) / (cd_wing_means - cd_wing_stds),
        alpha=0.2,
    )
    axs[1, 1].set_xlabel(r"$\alpha_{w,b}$ [$^\circ$]", fontsize=14)
    axs[1, 1].set_ylabel("$C_L/C_D$", fontsize=14)
    axs[1, 1].grid(True)


def plot_kinetic_energy_spectrum(results, flight_data, ax, savefig=False):
    signal = np.array(results["wind_speed_horizontal"])
    from scipy.stats import linregress
    # Assuming your signal is stored in `signal`
    fs = 10  # Sampling frequency in Hz, adjusted to 10 Hz based on the timestep of 0.1s
    T = len(signal) / fs  # Duration in seconds, calculated based on your signal length
    t = np.linspace(0, T, int(T*fs), endpoint=False)  # Time vector, adjusted

    # Subtract the mean to focus on fluctuations
    signal_fluctuations = signal - np.mean(signal)

    # Compute FFT and frequencies
    fft_result = np.fft.fft(signal_fluctuations)
    fft_freq = np.fft.fftfreq(signal_fluctuations.size, d=1/fs)

    # Compute energy spectrum (magnitude squared)
    energy_spectrum = np.abs(fft_result)**2/flight_data['time'].iloc[-1]

    # Select positive frequencies for plotting and analysis
    pos_freq = fft_freq > 0
    pos_energy_spectrum = energy_spectrum[pos_freq]
    pos_fft_freq = fft_freq[pos_freq]

    # Log-log plot
    ax.loglog(pos_fft_freq, pos_energy_spectrum, label='Energy Spectrum')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density ($m^2/s^2$)')

    # Determine an appropriate subrange and calculate the slope
    # Adjust subrange_start and subrange_end based on your data
    subrange_start = 1e-2  # Example start frequency
    subrange_end = 2e-1   # Example end frequency
    subrange_mask = (pos_fft_freq > subrange_start) & (pos_fft_freq < subrange_end)

    if np.any(subrange_mask):
        slope, intercept, r_value, p_value, std_err = linregress(np.log(pos_fft_freq[subrange_mask]), np.log(pos_energy_spectrum[subrange_mask]))
        plt.plot(pos_fft_freq[subrange_mask], np.exp(intercept) * pos_fft_freq[subrange_mask] ** slope, 'r--', label=f'Fitted Slope: {slope:.2f}')
        print(f"The calculated slope is: {slope:.2f}")
    else:
        print("No data in the specified subrange. Please adjust the subrange criteria.")

    slope = -5/3
    plt.plot(pos_fft_freq[subrange_mask], np.exp(intercept) * pos_fft_freq[subrange_mask] ** slope, 'g--', label=f'Kolmogorov: -5/3')
    plt.legend()

def plot_turbulence_intensity(results,flight_data, height, ax, savefig=False):
    mask = (results['kite_position_z']>height-10)&(results['kite_position_z']<height+10)
    TI_160m = []
    for i in range(len(results)):
        if i<600:
            std = np.std(results["wind_speed_horizontal"].iloc[0:i][mask])
            mean = np.mean(results["wind_speed_horizontal"].iloc[0:i][mask])
        else:
            std = np.std(results["wind_speed_horizontal"].iloc[i-600:i][mask])
            mean = np.mean(results["wind_speed_horizontal"].iloc[i-600:i][mask])

        TI_160m.append(std/mean)
    #%%
    TI_160m_lidar = flight_data[str(height)+'m_Wind_Speed_Dispersion_m_s'][mask]/flight_data[str(height)+'m_Wind_Speed_m_s'][mask]
    ax.plot(TI_160m)
    ax.plot(TI_160m_lidar)
    ax.legend(['EKF','Lidar'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Turbulence intensity')