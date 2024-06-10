from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List
from awes_ekf.setup.settings import z0, kappa
from awes_ekf.utils import calculate_euler_from_reference_frame, calculate_airflow_angles, calculate_reference_frame_euler, rotate_ENU2NED


@dataclass
class EKFOutput:
    kite_pos_x: float
    kite_pos_y: float
    kite_pos_z: float
    kite_vel_x: float
    kite_vel_y: float
    kite_vel_z: float
    wind_velocity: float
    wind_direction: float
    kite_roll: float
    kite_pitch: float
    kite_yaw: float
    tether_length: float
    kite_aoa: float
    kite_sideslip: float
    CL: float
    CD: float
    CS: float
    elevation_first_element: float
    azimuth_first_element: float
    thrust_force: float = None
    cd_tether: float = None
    cd_kcu: float = None
    z_wind: float = None
    k_steering_law: float = None
    kcu_roll: float = None
    kcu_pitch: float = None
    kcu_yaw: float = None
    
def create_ekf_output(x, u, kite, tether,kcu, model_specs):
    """Store results in a list of instances of the class EKFOutput"""
    # Store tether force and tether model results
    kite_pos = x[0:3]
    kite_vel = x[3:6]
    if model_specs.log_profile:
        wind_vel  = x[6]/kappa*np.log(x[2]/z0)
        wind_dir = x[7]
        z_wind = x[8]
        vw = np.array([wind_vel*np.cos(wind_dir), wind_vel*np.sin(wind_dir), z_wind])
    else:
        vw = x[6:9]
        wind_vel = np.linalg.norm(vw)
        wind_dir = np.arctan2(vw[1],vw[0])
        z_wind = vw[2]
    tension_ground = u[1]
    tether_length = x[12]
    elevation_0 = x[13]
    azimuth_0 = x[14]

    if kcu is not None:
        if kcu.data_available:
            a_kcu = u[2:5]
            v_kcu = u[5:8]
            a_kite = None
        else:
            a_kite = u[2:5]
            a_kcu = None
            v_kcu = None
    else:
        a_kite = None
        a_kcu = None
        v_kcu = None

    args = (kite_pos, kite_vel, vw, kite, kcu, tension_ground )
    opt_guess = [elevation_0, azimuth_0, tether_length]
    info_tether = tether.calculate_tether_shape_symbolic(elevation_0,
        azimuth_0,
        tether_length,
        tension_ground,
        kite_pos,
        kite_vel,
        vw,
        kite,
        kcu,
        tether,
        a_kite=a_kite,
        a_kcu=a_kcu,
        v_kcu=v_kcu,
        return_results=True,
    )
    dcm_b2w = info_tether['bridle_frame_va']
    dcm_b2vel = info_tether['bridle_frame_vk']
    dcm_t2w = info_tether['tether_frame']
    dcm_b2w = rotate_ENU2NED(dcm_b2w)
    dcm_t2w = rotate_ENU2NED(dcm_t2w)
    euler_angles = calculate_euler_from_reference_frame(dcm_b2w)
    euler_angles1 = calculate_euler_from_reference_frame(dcm_t2w)
    cd_kcu = info_tether['cd_kcu']
    cd_tether = info_tether['cd_tether']
    
    if model_specs.model_yaw:
        dcm = calculate_reference_frame_euler( euler_angles[0], 
                                                     euler_angles[1], 
                                                     x[15], 
                                                     eulerFrame='NED',
                                                     outputFrame='ENU')
        airflow_angles = calculate_airflow_angles(dcm, vw-kite_vel)
    else:
        airflow_angles = calculate_airflow_angles(dcm_b2vel, vw-kite_vel)

    # Unpack position and velocity vectors
    kite_pos_x, kite_pos_y, kite_pos_z = x[0:3]
    kite_vel_x, kite_vel_y, kite_vel_z = x[3:6]

    # Create an instance of EKFOutput with unpacked vectors
    ekf_output = EKFOutput(
        kite_pos_x = kite_pos_x,
        kite_pos_y = kite_pos_y,
        kite_pos_z = kite_pos_z,
        kite_vel_x = kite_vel_x,
        kite_vel_y = kite_vel_y,
        kite_vel_z = kite_vel_z,
        wind_velocity = wind_vel,
        wind_direction = wind_dir,
        kite_roll = euler_angles[0],
        kite_pitch = euler_angles[1],
        kite_yaw = euler_angles[2],
        tether_length = x[12],
        kite_aoa = airflow_angles[0],
        kite_sideslip = airflow_angles[1],
        CL = x[9],
        CD = x[10],
        CS = x[11],
        elevation_first_element = x[13],
        azimuth_first_element = x[14],
        cd_kcu = cd_kcu,
        cd_tether = cd_tether,
        z_wind = z_wind,
        kcu_roll = euler_angles1[0],
        kcu_pitch = euler_angles1[1],
        kcu_yaw = euler_angles1[2]
    )
    
    if model_specs.model_yaw:
        ekf_output.k_steering_law = x[16]
        ekf_output.yaw = x[15]
                            
    return ekf_output

def convert_ekf_output_to_df(ekf_output_list:List[EKFOutput])->pd.DataFrame:
    """Convert list of EKFOutput instances to DataFrame by using the __dict__ of each instance."""
    # List comprehension that converts each instance to a dictionary
    data_dicts = [vars(output) for output in ekf_output_list]
    
    # Create DataFrame directly from list of dictionaries
    return pd.DataFrame(data_dicts)