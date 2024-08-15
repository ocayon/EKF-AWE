from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List
from awes_ekf.setup.settings import z0, kappa
from awes_ekf.utils import (
    calculate_euler_from_reference_frame,
    calculate_airflow_angles,
    calculate_reference_frame_euler,
    rotate_ENU2NED,
    calculate_log_wind_velocity,
)


@dataclass
class EKFOutput:
    """Dataclass for storing output data from the Extended Kalman Filter"""

    # Kite dynamics
    kite_position_x: float = None  # Kite position in ENU coordinates (m)
    kite_position_y: float = None  # Kite position in ENU coordinates (m)
    kite_position_z: float = None  # Kite position in ENU coordinates (m)
    kite_velocity_x: float = None  # Kite velocity in ENU coordinates (m/s)
    kite_velocity_y: float = None  # Kite velocity in ENU coordinates (m/s)
    kite_velocity_z: float = None  # Kite velocity in ENU coordinates (m/s)
    kite_roll: float = None  # Bridle element roll angle in radians, NED frame (rad)
    kite_pitch: float = None  # Bridle element pitch angle in radians, NED frame (rad)
    kite_yaw: float = None  # Bridle element yaw angle in radians, NED frame (rad) (towards the kite velocity direction)
    kite_thrust_force: float = None  # Thrust force applied to the kite (N)

    # Wind dynamics
    wind_speed_horizontal: float = None  # Horizontal wind speed magnitude in ENU (m/s)
    wind_direction: float = None  # Downwind direction in ENU (rad)
    wind_speed_vertical: float = (
        None  # Vertical wind component (positive is up) in ENU (m/s)
    )

    # Tether parameters
    tether_length: float = None  # Tether length (m)
    tether_elevation_angle: float = (
        None  # Elevation angle of the first tether element from the ground(rad)
    )
    tether_azimuth_angle: float = (
        None  # Azimuth angle of the first tether element from the ground (rad)
    )
    tether_length_offset: float = None  # Tether length measurement offset (m)
    tether_force_kite: float = None  # Tether force at the bridle point (N)
    tether_roll: float = None  # Roll of the first tether element below KCU (rad)
    tether_pitch: float = (
        None  # Pitch of the first tether element below KCU (rad)
    )
    tether_yaw: float = None  # Yaw of the first tether element below KCU (rad) (Velocity direction)

    # Aerodynamic parameters
    kite_apparent_windspeed: float = None  # Apparent wind speed at the kite (m/s)
    kite_angle_of_attack: float = (
        None  # Angle of attack of the bridle tether element (rad)
    )
    kite_sideslip_angle: float = None  # Sideslip angle in degrees (rad) (Angle between velocity and apparent velocity)
    wing_lift_coefficient: float = None  # Lift coefficient of the wing (-)
    wing_drag_coefficient: float = None  # Drag coefficient of the wing (-)
    wing_sideforce_coefficient: float = None  # Side force coefficient of the wing (-)
    tether_angle_of_attack: float = (
        None  # Angle of attack of the tether element below the KCU (rad)
    )
    tether_sideslip_angle: float = (
        None  # Sideslip angle of the tether element below the KCU (rad)
    )
    tether_drag_coefficient: float = (
        None  # Drag coefficient of the tether nondimensionalized with kite area
    )
    kcu_drag_coefficient: float = (
        None  # Drag coefficient of the KCU nondimensionalized with kite area
    )

    # Steering and control parameters
    steering_law_coefficient: float = None  # Steering law parameter (-)

    # Performance metrics
    normalized_innovation_squared: float = None  # Normalized innovation squared
    mahalanobis_distance: float = None  # Mahalanobis distance
    normalized_residual_norm: float = None  # Norm of the normalized residuals by the stdv (0 is best fit) (more than 1 is bad)


def create_ekf_output(x, u, ekf_input, tether, kite, simConfig):
    """Store results in a list of instances of the class EKFOutput"""

    state_index_map = kite.state_index_map
    input_index_map = kite.input_index_map

    # Store kite position and velocity from state vector
    r_kite = np.array([x[state_index_map[f"r_{i}"]] for i in range(3)])
    v_kite = np.array([x[state_index_map[f"v_{i}"]] for i in range(3)])

    # Calculate wind velocity based on configuration
    if simConfig.log_profile:
        vw = calculate_log_wind_velocity(
            x[state_index_map["uf"]],
            x[state_index_map["wdir"]],
            x[state_index_map["vw_2"]],
            x[state_index_map["r_2"]],
        )
    else:
        vw = np.array([x[state_index_map[f"vw_{i}"]] for i in range(3)])

    wind_speed_horizontal = np.linalg.norm(vw[:2])
    wind_direction = np.arctan2(vw[1], vw[0])
    wind_speed_vertical = vw[2]

    # Tether force and geometry
    tension_ground = u[input_index_map["ground_tether_force"]]
    tether_length = x[state_index_map["tether_length"]]
    elevation_0 = x[state_index_map["elevation_first_tether_element"]]
    azimuth_0 = x[state_index_map["azimuth_first_tether_element"]]
    args = (elevation_0, azimuth_0, tether_length, tension_ground, r_kite, v_kite, vw)

    # Add optional accelerations if observed
    if simConfig.obsData.kite_acceleration:
        args += (ekf_input.kite_acceleration,)
    if simConfig.obsData.kcu_acceleration:
        args += (ekf_input.kcu_acceleration,)
    if simConfig.obsData.kcu_velocity:
        args += (ekf_input.kcu_velocity,)

    # Frame rotations and euler angle calculations
    dcm_b2w = np.array(tether.bridle_frame_va(*args))
    dcm_b2vel = np.array(tether.bridle_frame_vk(*args))
    dcm_t2w = np.array(tether.tether_frame(*args))

    euler_angles = calculate_euler_from_reference_frame(rotate_ENU2NED(dcm_b2w))
    euler_angles1 = calculate_euler_from_reference_frame(rotate_ENU2NED(dcm_t2w))
    drag_coefficient_kcu = float(tether.cd_kcu(*args))
    drag_coefficient_tether = float(tether.cd_tether(*args))
    tether_force_kite = np.linalg.norm(tether.tether_force_kite(*args))

    # Airflow angles based on configuration
    if simConfig.model_yaw:
        dcm = calculate_reference_frame_euler(
            euler_angles[0], euler_angles[1], x[15], eulerFrame="NED", outputFrame="ENU"
        )
        airflow_angles = calculate_airflow_angles(dcm, vw - v_kite)
    else:
        airflow_angles = calculate_airflow_angles(dcm_b2vel, vw - v_kite)

    # Unpack position and velocity vectors
    kite_position_x, kite_position_y, kite_position_z = r_kite
    kite_velocity_x, kite_velocity_y, kite_velocity_z = v_kite

    if simConfig.obsData.tether_length:
        tether_offset = x[state_index_map["tether_offset"]]
    else:
        tether_offset = None

    kite_apparent_windspeed = np.linalg.norm(vw - v_kite)

    # Create an instance of EKFOutput with unpacked vectors and calculated parameters
    ekf_output = EKFOutput(
        kite_position_x=kite_position_x,
        kite_position_y=kite_position_y,
        kite_position_z=kite_position_z,
        kite_velocity_x=kite_velocity_x,
        kite_velocity_y=kite_velocity_y,
        kite_velocity_z=kite_velocity_z,
        wind_speed_horizontal=wind_speed_horizontal,
        wind_direction=wind_direction,
        wind_speed_vertical=wind_speed_vertical,
        kite_roll=euler_angles[0],
        kite_pitch=euler_angles[1],
        kite_yaw=euler_angles[2],
        tether_length=tether_length,
        kite_angle_of_attack=airflow_angles[0],
        kite_sideslip_angle=airflow_angles[1],
        wing_lift_coefficient=x[state_index_map["CL"]],
        wing_drag_coefficient=x[state_index_map["CD"]],
        wing_sideforce_coefficient=x[state_index_map["CS"]],
        tether_elevation_angle=elevation_0,
        tether_azimuth_angle=azimuth_0,
        kcu_drag_coefficient=drag_coefficient_kcu,
        tether_drag_coefficient=drag_coefficient_tether,
        kite_thrust_force=x[state_index_map.get("kite_thrust_force", 0)],
        tether_roll=euler_angles1[0],
        tether_pitch=euler_angles1[1],
        tether_yaw=euler_angles1[2],
        tether_offset=tether_offset,
        tether_force_kite=tether_force_kite,
        kite_apparent_windspeed=kite_apparent_windspeed,
    )

    # Optional yaw modeling
    if simConfig.model_yaw:
        ekf_output.steering_law_coefficient = x[16]

    return ekf_output


def convert_ekf_output_to_df(ekf_output_list: List[EKFOutput]) -> pd.DataFrame:
    """Convert list of EKFOutput instances to DataFrame by using the __dict__ of each instance."""
    # List comprehension that converts each instance to a dictionary
    data_dicts = [vars(output) for output in ekf_output_list]

    # Create DataFrame directly from list of dictionaries
    return pd.DataFrame(data_dicts)
