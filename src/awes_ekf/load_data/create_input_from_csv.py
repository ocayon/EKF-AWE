import numpy as np
import pandas as pd
from awes_ekf.ekf.ekf_input import EKFInput
from awes_ekf.setup.kite import Kite
from awes_ekf.setup.tether import Tether
from awes_ekf.setup.kcu import KCU
from awes_ekf.setup.settings import kappa,z0

def create_input_from_csv(flight_data, kite,kcu,tether, model_specs, kite_sensor = 0, kcu_sensor = 1, thrust_force = False):
    """Create input classes and initial state vector from flight data"""
    n_intervals = len(flight_data)
    # Kite measurements
    kite_pos = np.array([flight_data['kite_position_east'],flight_data['kite_position_north'], flight_data['kite_position_up']]).T
    kite_vel = np.array([flight_data['kite_velocity_east_s'+str(kite_sensor)],flight_data['kite_velocity_north_s'+str(kite_sensor)],flight_data['kite_velocity_up_s'+str(kite_sensor)]]).T
    kite_acc = np.array([flight_data['kite_acceleration_east_s'+str(kite_sensor)],flight_data['kite_acceleration_north_s'+str(kite_sensor)],flight_data['kite_acceleration_up_s'+str(kite_sensor)]]).T
    # KCU measurements
    if kcu_sensor is not None:
        kcu_vel = np.array([flight_data['kite_velocity_east_s'+str(kcu_sensor)],flight_data['kite_velocity_north_s'+str(kcu_sensor)],flight_data['kite_velocity_up_s'+str(kcu_sensor)]]).T
        kcu_acc = np.array([flight_data['kite_acceleration_east_s'+str(kcu_sensor)],flight_data['kite_acceleration_north_s'+str(kcu_sensor)],flight_data['kite_acceleration_up_s'+str(kcu_sensor)]]).T
    else:
        kcu_vel = np.zeros((n_intervals,3))
        kcu_acc = np.zeros((n_intervals,3))
    # Tether measurements
    tether_force = np.array(flight_data['ground_tether_force'])
    tether_length = np.array(flight_data['ground_tether_length'])
      
    # Airflow measurements
    ground_windspeed = np.array(flight_data['ground_wind_velocity'])
    ground_winddir = np.array(flight_data['ground_wind_direction'])
    apparent_windspeed = np.array(flight_data['kite_apparent_windspeed'])
    kite_aoa = np.array(flight_data['kite_angle_of_attack'])
    relout_speed = np.array(flight_data['ground_tether_reelout_speed'])
    kite_elevation = np.arcsin(kite_pos[:,2]/np.linalg.norm(kite_pos,axis=1))
    kite_azimuth = np.arctan2(kite_pos[:,1],kite_pos[:,0])
    
    kite_yaw = np.unwrap(np.array(flight_data['kite_yaw_s'+str(kite_sensor)]-90)/180*np.pi)

    init_wind_dir = np.mean(ground_winddir[0:3000])/180*np.pi
    init_wind_vel = np.mean(ground_windspeed[0])
    
    if np.isnan(init_wind_dir):
        for column in flight_data.columns:
            if 'Wind Speed (m/s)' in column:
                init_wind_vel = flight_data[column].iloc[1400]
                break
        for column in flight_data.columns:
            if 'Wind Direction' in column:
                init_wind_dir = np.deg2rad(360-90-flight_data[column].iloc[1400])
                break
                
        
    us = (flight_data['kcu_actual_steering'])/max(abs(flight_data['kcu_actual_steering']))
    timestep = flight_data['time'].iloc[1]-flight_data['time'].iloc[0]
    ekf_input_list = []
    for i in range(len(flight_data)):
        if kcu_sensor is not None:
            ekf_input_list.append(EKFInput(kite_pos = kite_pos[i], 
                                    kite_vel = kite_vel[i], 
                                    kite_acc = kite_acc[i],
                                    kcu_acc = kcu_acc[i], 
                                    tether_force = tether_force[i],
                                    apparent_windspeed = apparent_windspeed[i], 
                                    tether_length = tether_length[i],
                                    kite_aoa = kite_aoa[i], 
                                    kcu_vel = kcu_vel[i], 
                                    reelout_speed = relout_speed[i], 
                                    elevation = kite_elevation[i],
                                    azimuth = kite_azimuth[i], 
                                    ts = timestep,
                                    kite_yaw = kite_yaw[i], 
                                    steering_input = us[i]))
        elif thrust_force is not None:
            ekf_input_list.append(EKFInput(kite_pos = kite_pos[i], 
                                    kite_vel = kite_vel[i], 
                                    kite_acc = kite_acc[i],
                                    tether_force = tether_force[i],
                                    apparent_windspeed = apparent_windspeed[i], 
                                    tether_length = tether_length[i],
                                    kite_aoa = kite_aoa[i], 
                                    reelout_speed = relout_speed[i], 
                                    elevation = kite_elevation[i],
                                    azimuth = kite_azimuth[i], 
                                    ts = timestep,
                                    kite_yaw = kite_yaw[i], 
                                    steering_input = us[i],
                                    thrust_force = thrust_force[i]))


    x0 = find_initial_state_vector(kite_pos[0], kite_vel[0], kite_acc[0], 
                                       init_wind_dir, init_wind_vel, tether_force[0], 
                                       tether_length[0], kite_elevation[0], kite_azimuth[0], kite, kcu,tether, model_specs)
    if model_specs.model_yaw:
        x0 = np.append(x0,[kite_yaw[0],0])  # Initial wind velocity and direction
    if model_specs.tether_offset:
        x0 = np.append(x0,0)
    return ekf_input_list, x0

def find_initial_state_vector(kite_pos, kite_vel, kite_acc, ground_winddir, ground_windspeed, tether_force, tether_length,
                              elevation, azimuth, kite, kcu,tether, model_specs):

    # Solve for the tether shape
    uf = ground_windspeed*kappa/np.log(10/z0)
    wvel0 = uf/kappa*np.log(kite_pos[2]/z0)
    if np.isnan(wvel0):
        raise ValueError('Initial wind velocity is NaN')
    vw = [wvel0*np.cos(ground_winddir),wvel0*np.sin(ground_winddir),0] # Initial wind velocity

    tether.solve_tether_shape( kite_pos, kite_vel, vw, kite, kcu, tension_ground = tether_force, tether_length = tether_length,
                                a_kite = kite_acc)
    x0 = np.vstack((kite_pos,kite_vel))
    
    if model_specs.log_profile:
        x0 = np.append(x0,[uf,ground_winddir,0])  # Initial wind velocity and direction
    else:
        x0 = np.append(x0,vw)   # Initial wind velocity
    x0 = np.append(x0,[tether.CL,tether.CD,tether.CS,tether_length, elevation, azimuth])     # Initial state vector (Last two elements are bias, used if needed)

    return x0

