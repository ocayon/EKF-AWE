import numpy as np
import pandas as pd
from awes_ekf.ekf.ekf_input import EKFInput
from awes_ekf.setup.tether import TetherInput
from awes_ekf.setup.settings import kappa, z0


def create_input_from_csv(
    flight_data, kite, kcu, tether, simConfig, kite_sensor=0, kcu_sensor=None
):
    """Create input classes and initial state vector from flight data"""
    n_intervals = len(flight_data)
    # Kite measurements
    kite_pos = np.array(
        [
            flight_data["kite_position_east"],
            flight_data["kite_position_north"],
            flight_data["kite_position_up"],
        ]
    ).T
    kite_vel = np.array(
        [
            flight_data["kite_velocity_east_s" + str(kite_sensor)],
            flight_data["kite_velocity_north_s" + str(kite_sensor)],
            flight_data["kite_velocity_up_s" + str(kite_sensor)],
        ]
    ).T
    kite_acc = np.array(
        [
            flight_data["kite_acceleration_east_s" + str(kite_sensor)],
            flight_data["kite_acceleration_north_s" + str(kite_sensor)],
            flight_data["kite_acceleration_up_s" + str(kite_sensor)],
        ]
    ).T
    # KCU measurements
    if kcu_sensor is not None:
        kcu_vel = np.array(
            [
                flight_data["kite_velocity_east_s" + str(kcu_sensor)],
                flight_data["kite_velocity_north_s" + str(kcu_sensor)],
                flight_data["kite_velocity_up_s" + str(kcu_sensor)],
            ]
        ).T
        kcu_acc = np.array(
            [
                flight_data["kite_acceleration_east_s" + str(kcu_sensor)],
                flight_data["kite_acceleration_north_s" + str(kcu_sensor)],
                flight_data["kite_acceleration_up_s" + str(kcu_sensor)],
            ]
        ).T
    else:
        kcu_vel = np.zeros((n_intervals, 3))
        kcu_acc = np.zeros((n_intervals, 3))
    # Tether measurements
    tether_force = np.array(flight_data["ground_tether_force"])
    tether_length = np.array(flight_data["ground_tether_length"])

    # Airflow measurements
    ground_windspeed = np.array(flight_data["ground_wind_velocity"])
    ground_winddir = np.array(flight_data["ground_wind_direction"])
    try:
        apparent_windspeed = np.array(flight_data["kite_apparent_windspeed"])
    except:
        apparent_windspeed = np.zeros(n_intervals)

    up = np.array(flight_data["kcu_actual_depower"]) / max(
        abs(flight_data["kcu_actual_depower"])
    )
    try:
        kite_aoa = np.array(flight_data["kite_angle_of_attack"])
        kite_aoa_mean_v9 = 10
        offset_aoa = -0.590496147373921
        ########!!!!!!! ADD automation to calculate offset_aoa
        offset_dep = -0.89
        offset_dep += -0.4
        kite_aoa = kite_aoa + offset_aoa + up * offset_dep

    except:
        kite_aoa = np.zeros(n_intervals)
    relout_speed = np.array(flight_data["ground_tether_reelout_speed"])
    kite_elevation = np.arcsin(kite_pos[:, 2] / np.linalg.norm(kite_pos, axis=1))
    kite_azimuth = np.arctan2(kite_pos[:, 1], kite_pos[:, 0])
    try:
        thrust_force = np.array(
            [
                flight_data["thrust_force_east"],
                flight_data["thrust_force_north"],
                flight_data["thrust_force_up"],
            ]
        ).T
    except:
        thrust_force = np.zeros((n_intervals, 3))
    try:
        kite_yaw = np.unwrap(
            np.array(flight_data["kite_yaw_s" + str(kite_sensor)] - np.pi / 2)
        )
    except:
        kite_yaw = np.zeros(n_intervals)

    init_wind_dir = np.mean(ground_winddir[0:3000])
    init_wind_vel = np.mean(ground_windspeed[0])

    if np.isnan(init_wind_dir):
        for column in flight_data.columns:
            if "Wind Speed (m/s)" in column:
                init_wind_vel = flight_data[column].iloc[1400]
                break
        for column in flight_data.columns:
            if "Wind Direction" in column:
                init_wind_dir = np.deg2rad(360 - 90 - flight_data[column].iloc[1400])
                break

    try:
        us = (flight_data["kcu_actual_steering"]) / max(
            abs(flight_data["kcu_actual_steering"])
        )
    except:
        us = np.zeros(n_intervals)
    timestep = flight_data["time"].iloc[1] - flight_data["time"].iloc[0]

    # Find initial wind velocity
    uf = init_wind_vel * kappa / np.log(10 / z0)
    wvel0 = uf / kappa * np.log(kite_pos[0][2] / z0)
    if np.isnan(wvel0):
        raise ValueError("Initial wind velocity is NaN")
    vw0 = [
        wvel0 * np.cos(init_wind_dir),
        wvel0 * np.sin(init_wind_dir),
        0,
    ]  # Initial wind velocity

    ekf_input_list = []
    for i in range(len(flight_data)):
        ekf_input_list.append(
            EKFInput(
                kite_pos=kite_pos[i],
                kite_vel=kite_vel[i],
                kite_acc=kite_acc[i],
                kcu_acc=kcu_acc[i],
                tether_force=tether_force[i],
                apparent_windspeed=apparent_windspeed[i],
                tether_length=tether_length[i],
                kite_aoa=kite_aoa[i],
                kcu_vel=kcu_vel[i],
                reelout_speed=relout_speed[i],
                elevation_first_element=kite_elevation[i],
                azimuth_first_element=kite_azimuth[i],
                ts=timestep,
                kite_yaw=kite_yaw[i],
                steering_input=us[i],
                thrust_force=thrust_force[i],
                depower_input=up[i],
            )
        )

    return ekf_input_list


def find_initial_state_vector(tether, ekf_input, simConfig, offset_aoa=0):
    tether_input = TetherInput(
        kite_pos=ekf_input.kite_pos,
        kite_vel=ekf_input.kite_vel,
        kite_acc=ekf_input.kite_acc,
        kcu_acc=ekf_input.kcu_acc,
        kcu_vel=ekf_input.kcu_vel,
        tether_force=ekf_input.tether_force,
        tether_elevation=ekf_input.elevation_first_element,
        tether_azimuth=ekf_input.azimuth_first_element,
        tether_length=ekf_input.tether_length,
    )

    tether_input = tether.solve_tether_shape(tether_input)

    # %% Find the initial state vector for the EKF
    args = tether_input.create_input_tuple(simConfig.obsData)

    CL = float(tether.CL(*args))
    CD = float(tether.CD(*args))
    CS = float(tether.CS(*args))
    x0 = np.vstack((tether_input.kite_pos, tether_input.kite_vel))

    if simConfig.log_profile:
        uf = np.linalg.norm(tether_input.wind_vel) * kappa / np.log(10 / z0)
        ground_winddir = np.arctan2(tether_input.wind_vel[1], tether_input.wind_vel[0])
        x0 = np.append(
            x0, [uf, ground_winddir, 0]
        )  # Initial wind velocity and direction
    else:
        x0 = np.append(x0, tether_input.wind_vel)  # Initial wind velocity
    x0 = np.append(
        x0,
        [
            CL,
            CD,
            CS,
            tether_input.tether_length,
            tether_input.tether_elevation,
            tether_input.tether_azimuth,
        ],
    )  # Initial state vector (Last two elements are bias, used if needed)
    if simConfig.model_yaw:
        x0 = np.append(
            x0, [ekf_input.kite_yaw, 0]
        )  # Initial wind velocity and direction
    if simConfig.tether_offset:
        x0 = np.append(x0, 0)  # Initial tether offset

    return x0
