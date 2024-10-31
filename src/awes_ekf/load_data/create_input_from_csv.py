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
    try:
        kite_position = np.array(
            [
                flight_data["kite_position_x"],
                flight_data["kite_position_y"],
                flight_data["kite_position_z"],
            ]
        ).T
    except KeyError:
        raise ValueError("No kite position data found")
    
    try:
        kite_velocity = np.array(
            [
                flight_data["kite_velocity_x"],
                flight_data["kite_velocity_y"],
                flight_data["kite_velocity_z"],
            ]
        ).T
    except KeyError:
        raise ValueError("No kite velocity data found")

    try:
        kite_acceleration = np.array(
            [
                flight_data["kite_acceleration_x"],
                flight_data["kite_acceleration_y"],
                flight_data["kite_acceleration_z"],
            ]
        ).T
    except KeyError:
        if simConfig.obsData.kite_acceleration:
            raise ValueError("No kite acceleration data found, but required by the config file")
        kite_acceleration = np.zeros((n_intervals, 3))

    # KCU measurements
    try:
        kcu_velocity = np.array(
            [
                flight_data["kcu_velocity_x"],
                flight_data["kcu_velocity_y"],
                flight_data["kcu_velocity_z"],
            ]
        ).T
    except KeyError:
        if simConfig.obsData.kcu_velocity:
            raise ValueError("No KCU velocity data found, but required by the config file")
        kcu_velocity = np.zeros((n_intervals, 3))
    try:
        kcu_acceleration = np.array(
            [
                flight_data["kcu_acceleration_x"],
                flight_data["kcu_acceleration_y"],
                flight_data["kcu_acceleration_z"],
            ]
        ).T
    except KeyError:
        if simConfig.obsData.kcu_acceleration:
            raise ValueError("No KCU acceleration data found, but required by the config file")
        kcu_acceleration = np.zeros((n_intervals, 3))


    # Tether measurements
    try:
        tether_force = np.array(flight_data["ground_tether_force"])
    except KeyError:
        raise ValueError("No tether force data found")
    
    try:
        tether_length = np.array(flight_data["tether_length"])
    except KeyError:
        if simConfig.obsData.tether_length:
            raise ValueError("No tether length data found, but required by the config file")
        tether_length = np.zeros(n_intervals)

    # Airflow measurements
    try:
        ground_windspeed = np.array(flight_data["ground_wind_speed"])
        ground_winddir = np.array(flight_data["ground_wind_direction"])
    except KeyError:
        print("No ground wind speed or direction data found, initializing to zero")
        ground_windspeed = np.zeros(n_intervals)
        ground_winddir = np.zeros(n_intervals)
        
    try:
        kite_apparent_windspeed = np.array(flight_data["kite_apparent_windspeed"])
    except KeyError:
        if simConfig.obsData.kite_apparent_windspeed:
            raise ValueError("No apparent wind speed data found, but required by the config file")
        kite_apparent_windspeed = np.zeros(n_intervals)

    if kcu is not None:
        try:
            depower_input = np.array(flight_data["kcu_actual_depower"]) / max(
                abs(flight_data["kcu_actual_depower"])
            )
        except:
            print("No depower input data found")
            depower_input = np.zeros(n_intervals)

        try:
            steering_input = flight_data["kcu_actual_steering"] / max(
                abs(flight_data["kcu_actual_steering"])
            )
        except KeyError:
            print("No steering input data found")
            steering_input = np.zeros(n_intervals)

    try:
        kite_angle_of_attack = np.array(flight_data["bridle_angle_of_attack"])
    except KeyError:
        if simConfig.obsData.kite_angle_of_attack:
            raise ValueError("No angle of attack data found, but required by the config file")
        kite_angle_of_attack = np.zeros(n_intervals)

    try:
        tether_reelout_speed = np.array(flight_data["tether_reelout_speed"])
    except KeyError:
        raise ValueError("No tether reelout speed data found, but required by the config file")

    try:
        tether_elevation_ground = np.array(flight_data["tether_elevation_ground"])
    except KeyError:
        if simConfig.obsData.tether_elevation_ground:
            raise ValueError("No tether elevation data found, but required by the config file")
        tether_elevation_ground = np.zeros(n_intervals)
    try:
        tether_azimuth_ground = np.array(flight_data["tether_azimuth_ground"])
    except KeyError:
        if simConfig.obsData.tether_azimuth_ground:
            raise ValueError("No tether azimuth data found, but required by the config file")
        tether_azimuth_ground = np.zeros(n_intervals)
        
    try:
        kite_thrust_force = np.array(
            [
                flight_data["thrust_force_x"],
                flight_data["thrust_force_y"],
                flight_data["thrust_force_z"],
            ]
        ).T
    except KeyError:
        if simConfig.obsData.kite_thrust_force:
            raise ValueError("No thrust force data found, but required by the config file")
        kite_thrust_force = np.zeros((n_intervals, 3))
        

    try:
        kite_yaw = np.unwrap(
            np.array(flight_data["kite_yaw_" + str(kite_sensor)] - np.pi / 2)
        )
    except KeyError:
        if simConfig.model_yaw:
            raise ValueError("No kite yaw data found, but required by the config file")
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



    timestep = np.gradient(flight_data["time"].values)

    # Find initial wind velocity
    uf = init_wind_vel * kappa / np.log(10 / z0)
    wvel0 = uf / kappa * np.log(kite_position[0][2] / z0)
    if np.isnan(wvel0):
        raise ValueError("Initial wind velocity is NaN")

    ekf_input_list = []
    for i in range(len(flight_data)):
        ekf_input_list.append(
            EKFInput(
                kite_position=kite_position[i],
                kite_velocity=kite_velocity[i],
                kite_acceleration=kite_acceleration[i],
                kcu_acceleration=kcu_acceleration[i],
                tether_force=tether_force[i],
                kite_apparent_windspeed=kite_apparent_windspeed[i],
                tether_length=tether_length[i],
                bridle_angle_of_attack=kite_angle_of_attack[i],
                kcu_velocity=kcu_velocity[i],
                tether_reelout_speed=tether_reelout_speed[i],
                tether_elevation_ground = tether_elevation_ground[i],
                tether_azimuth_ground = tether_azimuth_ground[i],
                timestep=timestep[i],
                kite_yaw=kite_yaw[i],
                steering_input=steering_input[i],
                kite_thrust_force=kite_thrust_force[i],
                depower_input=depower_input[i],
            )
        )

    return ekf_input_list


def find_initial_state_vector(tether, ekf_input, simConfig, wind_velocity=np.array([1e-3, 1e-3, 0]), CL=None, CD=None, CS=None):
    tether_input = TetherInput(
        kite_position=ekf_input.kite_position,
        kite_velocity=ekf_input.kite_velocity,
        kite_acceleration=ekf_input.kite_acceleration,
        kcu_acceleration=ekf_input.kcu_acceleration,
        kcu_velocity=ekf_input.kcu_velocity,
        tether_force=ekf_input.tether_force,
        tether_elevation=ekf_input.tether_elevation_ground,
        tether_azimuth=ekf_input.tether_azimuth_ground,
        tether_length=ekf_input.tether_length,
        wind_velocity=wind_velocity,
    )

    tether_input = tether.solve_tether_shape(tether_input)

    # %% Find the initial state vector for the EKF
    args = tether_input.create_input_tuple(simConfig.obsData)
    if CL is None:
        CL = float(tether.CL(*args))
    if CD is None:
        CD = float(tether.CD(*args))
    if CS is None:
        CS = float(tether.CS(*args))

    x0 = np.vstack((tether_input.kite_position, tether_input.kite_velocity))

    if simConfig.log_profile:
        uf = np.linalg.norm(tether_input.wind_velocity) * kappa / np.log(10 / z0)
        ground_winddir = np.arctan2(tether_input.wind_velocity[1], tether_input.wind_velocity[0])
        x0 = np.append(
            x0, [uf, ground_winddir, 0]
        )  # Initial wind velocity and direction
    else:
        x0 = np.append(x0, tether_input.wind_velocity)  # Initial wind velocity
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
    if simConfig.obsData.tether_length:
        x0 = np.append(x0, 0)  # Initial tether offset
    if simConfig.obsData.tether_elevation:
        x0 = np.append(x0, 0)  # Initial tether elevation offset
    if simConfig.obsData.tether_azimuth:
        x0 = np.append(x0, 0)

    return x0
