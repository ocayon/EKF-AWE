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
)


@dataclass
class EKFOutput:
    kite_pos_x: float = None
    kite_pos_y: float = None
    kite_pos_z: float = None
    kite_vel_x: float = None
    kite_vel_y: float = None
    kite_vel_z: float = None
    wind_velocity: float = None # m/s
    wind_direction: float = None # Downwind direction ENU
    kite_roll: float = None #[rad] NED
    kite_pitch: float = None
    kite_yaw: float = None
    tether_length: float = None # without offset and bridle?
    kite_aoa: float = None
    kite_sideslip: float = None
    CL: float = None # Wing only
    CD: float = None # Wing only
    CS: float = None # WIng only
    elevation_first_element: float = None # rad relative to horizon
    azimuth_first_element: float = None # rad relative to ENU
    thrust_force: float = None
    cd_tether: float = None # Non dimensionalized with area of the kite
    cd_kcu: float = None # Non dimensionalized with area of the kite
    z_wind: float = None # positive values are up (ENU)
    k_steering_law: float = None
    kcu_roll: float = None # Orientation of the first tether element below KCU
    kcu_pitch: float = None # Orientation of the first tether element below KCU
    kcu_yaw: float = None # Orientation of the first tether element below KCU
    tether_offset: float = None # incl bridle?
    tether_force_kite: float = None # [N]


def create_ekf_output(x, u, ekf_input, tether, kcu, simConfig):
    """Store results in a list of instances of the class EKFOutput"""
    # Store tether force and tether model results
    r_kite = x[0:3]
    v_kite = x[3:6]
    if simConfig.log_profile:
        wind_vel = x[6] / kappa * np.log(x[2] / z0)
        wind_dir = x[7]
        z_wind = x[8]
        vw = np.array(
            [wind_vel * np.cos(wind_dir), wind_vel * np.sin(wind_dir), z_wind]
        )
    else:
        vw = x[6:9]
        wind_vel = np.linalg.norm(vw)
        wind_dir = np.arctan2(vw[1], vw[0])
        z_wind = vw[2]
    tension_ground = u[1]
    tether_length = x[12]
    elevation_0 = x[13]
    azimuth_0 = x[14]
    args = (elevation_0, azimuth_0, tether_length, tension_ground, r_kite, v_kite, vw)

    if simConfig.obsData.kite_acc:
        args += (ekf_input.kite_acc,)
    if simConfig.obsData.kcu_acc:
        args += (ekf_input.kcu_acc,)
    if simConfig.obsData.kcu_vel:
        args += (ekf_input.kcu_vel,)

    dcm_b2w = np.array(tether.bridle_frame_va(*args))
    dcm_b2vel = np.array(tether.bridle_frame_vk(*args))
    dcm_t2w = np.array(tether.tether_frame(*args))
    dcm_b2w = rotate_ENU2NED(dcm_b2w)
    dcm_t2w = rotate_ENU2NED(dcm_t2w)
    euler_angles = calculate_euler_from_reference_frame(dcm_b2w)
    euler_angles1 = calculate_euler_from_reference_frame(dcm_t2w)
    cd_kcu = float(tether.cd_kcu(*args))
    cd_tether = float(tether.cd_tether(*args))
    tether_force_kite = np.linalg.norm(tether.tether_force_kite(*args))

    if simConfig.model_yaw:
        dcm = calculate_reference_frame_euler(
            euler_angles[0], euler_angles[1], x[15], eulerFrame="NED", outputFrame="ENU"
        )
        airflow_angles = calculate_airflow_angles(dcm, vw - v_kite)
    else:
        airflow_angles = calculate_airflow_angles(dcm_b2vel, vw - v_kite)

    # Unpack position and velocity vectors
    kite_pos_x, kite_pos_y, kite_pos_z = x[0:3]
    kite_vel_x, kite_vel_y, kite_vel_z = x[3:6]

    if simConfig.tether_offset:
        tether_offset = x[15]
    else:
        tether_offset = None

    # Create an instance of EKFOutput with unpacked vectors
    ekf_output = EKFOutput(
        kite_pos_x=kite_pos_x,
        kite_pos_y=kite_pos_y,
        kite_pos_z=kite_pos_z,
        kite_vel_x=kite_vel_x,
        kite_vel_y=kite_vel_y,
        kite_vel_z=kite_vel_z,
        wind_velocity=wind_vel,
        wind_direction=wind_dir,
        kite_roll=euler_angles[0],
        kite_pitch=euler_angles[1],
        kite_yaw=euler_angles[2],
        tether_length=x[12],
        kite_aoa=airflow_angles[0],
        kite_sideslip=airflow_angles[1],
        CL=x[9],
        CD=x[10],
        CS=x[11],
        elevation_first_element=x[13],
        azimuth_first_element=x[14],
        cd_kcu=cd_kcu,
        cd_tether=cd_tether,
        z_wind=z_wind,
        kcu_roll=euler_angles1[0],
        kcu_pitch=euler_angles1[1],
        kcu_yaw=euler_angles1[2],
        tether_offset=tether_offset,
        tether_force_kite=tether_force_kite,
    )

    if simConfig.model_yaw:
        ekf_output.k_steering_law = x[16]
        ekf_output.yaw = x[15]

    return ekf_output


def convert_ekf_output_to_df(ekf_output_list: List[EKFOutput]) -> pd.DataFrame:
    """Convert list of EKFOutput instances to DataFrame by using the __dict__ of each instance."""
    # List comprehension that converts each instance to a dictionary
    data_dicts = [vars(output) for output in ekf_output_list]

    # Create DataFrame directly from list of dictionaries
    return pd.DataFrame(data_dicts)
