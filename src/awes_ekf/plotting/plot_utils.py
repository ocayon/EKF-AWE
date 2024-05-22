# Plotting utilities for the project
# Author: Oriol Cayon

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from setup.settings import kappa, z0
from utils import  calculate_angle, project_onto_plane,calculate_reference_frame_euler
import seaborn as sns



def find_turn_law(flight_data):

    window_size = 10
    yaw_rate = np.diff(flight_data['kite_0_yaw']) / np.diff(flight_data['time'])
    yaw_rate = np.concatenate((yaw_rate, [0]))
    yaw_rate=np.convolve(yaw_rate/180*np.pi, np.ones(window_size)/window_size, mode='same')
    mask = yaw_rate<-1.8
    yaw_rate[mask] += np.pi
    mask = yaw_rate>1.8
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


def remove_offsets_IMU_data(flight_data, offset):
    """
    Remove offsets from IMU data
    :param data: IMU data
    :param offsets: IMU offsets
    :return: IMU data without offsets
    """
    flight_data['kite_0_pitch'] = flight_data['kite_0_pitch']+offset['pitch0']
    flight_data['kite_1_pitch'] = flight_data['kite_1_pitch']+offset['pitch1']
    flight_data['kite_0_roll'] = flight_data['kite_0_roll']+offset['roll0']
    flight_data['kite_1_roll'] = flight_data['kite_1_roll']+offset['roll1']
    flight_data['kite_0_yaw'] = flight_data['kite_0_yaw']+offset['yaw0']
    flight_data['kite_1_yaw'] = flight_data['kite_1_yaw']+offset['yaw1']

    return flight_data


def postprocess_results(results,flight_data, kite, IMU_0=False, IMU_1=False, EKF_tether=False):
    """
    Calculate angle of attack and sideslip based on kite and KCU IMU data
    :param results: results from the simulation
    :param kite: kite object
    :param IMU_0: IMU data from the kite
    :param IMU_1: IMU data from the KCU
    :param EKF_tether: EKF data from the tether orientation and IMU yaw
    :return: results with aoa and ss va radius omega and slack
    """
    # Calculate apparent speed based on EKF results
    wvel = results['wind_velocity']
    vw = np.vstack((wvel*np.cos(results['wind_direction']),wvel*np.sin(results['wind_direction']),np.zeros(len(results)))).T
    r_kite = np.vstack((np.array(results['kite_pos_x']),np.array(results['kite_pos_y']),np.array(results['kite_pos_z']))).T
    v_kite = np.vstack((np.array(results['vx']),np.array(results['vy']),np.array(results['vz']))).T
    # Calculate a_kite with diff of v_kite
    dt = flight_data['time'].iloc[1]-flight_data['time'].iloc[0]
    a_kite = np.vstack((np.concatenate((np.diff(v_kite[:,0])/dt, [0])),np.concatenate((np.diff(v_kite[:,1])/dt, [0])),np.concatenate((np.diff(v_kite[:,2])/dt, [0])))).T
    
    if 'kite_sideslip_angle' not in flight_data.columns:
        flight_data['kite_sideslip_angle'] = np.zeros(len(flight_data))

    results['time'] = flight_data['time']
    # Smooth a_kite
    window_size = 10
    a_kite[:,0] = np.convolve(a_kite[:,0], np.ones(window_size)/window_size, mode='same')
    a_kite[:,1] = np.convolve(a_kite[:,1], np.ones(window_size)/window_size, mode='same')
    a_kite[:,2] = np.convolve(a_kite[:,2], np.ones(window_size)/window_size, mode='same')

    va_kite = vw-v_kite
    results['va_kite'] = np.linalg.norm(va_kite,axis=1)

    if IMU_0:
        results['aoa_IMU_0'] = np.zeros(len(results))
        results['ss_IMU_0'] = np.zeros(len(results))
    if IMU_1:
        results['aoa_IMU_1'] = np.zeros(len(results))
        results['ss_IMU_1'] = np.zeros(len(results))
    if EKF_tether:
        results['aoa_EKF_tether'] = np.zeros(len(results))
        results['ss_EKF_tether'] = np.zeros(len(results))     
        
    min_depower = min(flight_data['kcu_actual_depower'])
    max_depower = max(flight_data['kcu_actual_depower'])
    flight_data['us'] =  (flight_data['kcu_actual_steering'])/max(abs(flight_data['kcu_actual_steering'])) 
    flight_data['up'] = (flight_data['kcu_actual_depower']-min_depower)/(max_depower-min_depower)
    # Identify flight phases
    flight_data['turn_straight'] = flight_data.apply(determine_turn_straight, axis=1)
    flight_data['right_left'] = flight_data.apply(determine_turn_straight, axis=1)
    flight_data['powered'] = flight_data.apply(determine_powered_depowered, axis=1)

    
    # Correct angle of attack for depowered phase on EKF mean pitch during depower
    mask_turn = abs(flight_data['us'])>0.9
    mask_dep = flight_data['up']>0.9
    pitch_EKF = np.array(results['pitch'])
    pitch_IMU_0 = np.array(flight_data['kite_0_pitch'])
    offset_dep = np.mean(-pitch_EKF[mask_dep]-pitch_IMU_0[mask_dep])
    offset_turn = np.mean(-pitch_EKF[mask_turn]-pitch_IMU_0[mask_turn])
    offset_pitch = offset_dep*np.array(flight_data['up'])#+offset_turn*np.array(abs(flight_data['us']))
    flight_data['offset_pitch'] = offset_pitch
    print('Offset pitch depower: ', offset_dep, 'Offset pitch turn: ', offset_turn)   
    flight_data['kite_0_pitch'] = flight_data['kite_0_pitch']+offset_pitch
    
    flight_data['cycle'] = np.zeros(len(flight_data))
    cycle_count = 0
    in_cycle = False
    ip = 0
    slack = []
    radius_turn = []
    omega = []
    for i in range(len(results)):
        res = results.iloc[i]
        fd = flight_data.iloc[i]
        # Calculate tether orientation based on euler angles
        q = 0.5*1.225*kite.area*res['va_kite']**2
        slack.append(fd['ground_tether_length']+kite.distance_kcu_kite-np.sqrt(res.x**2+res.y**2+res.z**2))
        if IMU_0:
            ex, ey, ez = calculate_reference_frame_euler(fd['kite_0_roll'], fd['kite_0_pitch'], fd['kite_0_yaw'])
            # Calculate wind velocity based on KCU orientation and wind speed and direction
            va_proj = project_onto_plane(va_kite[i], ey)           # Projected apparent wind velocity onto kite y axis
            results['aoa_IMU_0'].iloc[i] = 90-calculate_angle(ez,va_proj)            # Angle of attack
            va_proj = project_onto_plane(va_kite[i], ez)           # Projected apparent wind velocity onto kite z axis
            results['ss_IMU_0'].iloc[i] = 90-calculate_angle(ey,va_proj)        # Sideslip angle
        if IMU_1:
            ex, ey, ez = calculate_reference_frame_euler(fd['kite_1_roll'], fd['kite_1_pitch'], fd['kite_1_yaw'])
            # Calculate wind velocity based on KCU orientation and wind speed and direction
            va_proj = project_onto_plane(va_kite[i], ey)           # Projected apparent wind velocity onto kite y axis
            results['aoa_IMU_1'].iloc[i] = 90-calculate_angle(ez,va_proj)            # Angle of attack
            va_proj = project_onto_plane(va_kite[i], ez)           # Projected apparent wind velocity onto kite z axis
            results['ss_IMU_1'].iloc[i] = 90-calculate_angle(ey,va_proj)        # Sideslip angle
        if EKF_tether:
            ex, ey, ez = calculate_reference_frame_euler(-res['roll'], -res['pitch'], fd['kite_0_yaw'])
            # Calculate wind velocity based on KCU orientation and wind speed and direction
            va_proj = project_onto_plane(va_kite[i], ey)           # Projected apparent wind velocity onto kite y axis
            results['aoa_EKF_tether'].iloc[i] = 90-calculate_angle(ez,va_proj)            # Angle of attack
            va_proj = project_onto_plane(va_kite[i], ez)           # Projected apparent wind velocity onto kite z axis
            results['ss_EKF_tether'].iloc[i] = 90-calculate_angle(ey,va_proj)        # Sideslip angle
            
        at = np.dot(a_kite[i],np.array(v_kite[i])/np.linalg.norm(v_kite[i]))*np.array(v_kite[i])/np.linalg.norm(v_kite[i])
        omega_kite = np.cross(a_kite[i]-at,v_kite[i])/(np.linalg.norm(v_kite[i])**2)
        ICR = np.cross(v_kite[i],omega_kite)/(np.linalg.norm(omega_kite)**2)    
        
        radius_turn.append(np.linalg.norm(ICR))
        omega.append(np.linalg.norm(omega_kite))

        if fd['powered']=='depowered' and not in_cycle:
            flight_data.loc[ip:i, 'cycle'] = cycle_count
            ip = i
            # Entering a new cycle
            cycle_count += 1
            in_cycle = True
        elif fd['powered']=='powered' and in_cycle:
            # Exiting the current cycle
            in_cycle = False

    print("Number of cycles:", cycle_count)
    results['slack'] = slack
    results['radius_turn'] = radius_turn
    results['omega'] = omega

    return results, flight_data

def calculate_wind_speed_airborne_sensors(results, flight_data, IMU_0 =False, IMU_1=False, EKF_tether=False):
    """
    Calculate wind speed based on kite and KCU IMU data
    :param flight_data: flight data
    :return: flight data with wind speed
    """
    
    
    # Calculate wind speed based on kite sensor measurements
    if IMU_0:
        flight_data['wind_speed_kite'] = np.zeros(len(flight_data))
        v_0 = np.array([flight_data['kite_0_vx'],flight_data['kite_0_vy'],flight_data['kite_0_vz']]).T
        vw_0 = np.zeros((len(flight_data),3))
    if IMU_1:
        flight_data['wind_speed_kcu'] = np.zeros(len(flight_data))
        v_1 = np.array([flight_data['kite_1_vx'],flight_data['kite_1_vy'],flight_data['kite_1_vz']]).T
        vw_1 = np.zeros((len(flight_data),3))
    if EKF_tether:
        flight_data['wind_speed_tether'] = np.zeros(len(flight_data))
        v_EKF = np.array([results['vx'],results['vy'],results['vz']]).T
        vw_EKF = np.zeros((len(flight_data),3))

    measured_va = flight_data['kite_apparent_windspeed']
    measured_aoa = flight_data['kite_angle_of_attack']
    measured_ss = flight_data['kite_sideslip_angle']

        
    measured_va = results['va_kite']
    for i in range(len(flight_data)):
        if IMU_0:
            ex_kite, ey_kite, ez_kite = calculate_reference_frame_euler(flight_data['kite_0_roll'][i], flight_data['kite_0_pitch'][i], flight_data['kite_0_yaw'][i])
            # Calculate apparent wind velocity based on KCU orientation and apparent wind speed and aoa and ss
            va = ex_kite*measured_va[i]*np.cos(measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ey_kite*measured_va[i]*np.sin(measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ez_kite*measured_va[i]*np.sin(measured_aoa[i]/180*np.pi)
            # Calculate wind velocity based on KCU orientation and wind speed and direction
            vw_0[i,:] = va+v_0[i,:]
            
        if IMU_1:
            ex_kcu, ey_kcu, ez_kcu = calculate_reference_frame_euler(flight_data['kite_1_roll'][i], flight_data['kite_1_pitch'][i], flight_data['kite_1_yaw'][i])
            # Calculate apparent wind velocity based on KCU orientation and apparent wind speed and aoa and ss
            va = ex_kcu*measured_va[i]*np.cos(measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ey_kcu*measured_va[i]*np.sin(measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ez_kcu*measured_va[i]*np.sin(measured_aoa[i]/180*np.pi)
            # Calculate wind velocity based on KCU orientation and wind speed and direction
            vw_1[i,:] = va+v_1[i,:]
        if EKF_tether:
            ex_kcu, ey_kcu, ez_kcu = calculate_reference_frame_euler(-results['roll'][i], -results['pitch'][i], flight_data['kite_0_yaw'][i])
            # Calculate apparent wind velocity based on KCU orientation and apparent wind speed and aoa and ss
            va = ex_kcu*measured_va[i]*np.cos(measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ey_kcu*measured_va[i]*np.sin(measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ez_kcu*measured_va[i]*np.sin(measured_aoa[i]/180*np.pi)
            # Calculate wind velocity based on KCU orientation and wind speed and direction
            vw_EKF[i,:] = va+v_EKF[i,:]

        if IMU_0:
            flight_data['vwx_IMU_0'] = vw_0[:,0]
            flight_data['vwy_IMU_0'] = vw_0[:,1]
            flight_data['vwz_IMU_0'] = vw_0[:,2]
        if IMU_1:
            flight_data['vwx_IMU_1'] = vw_1[:,0]
            flight_data['vwy_IMU_1'] = vw_1[:,1]
            flight_data['vwz_IMU_1'] = vw_1[:,2]
        if EKF_tether:
            flight_data['vwx_EKF'] = vw_EKF[:,0]
            flight_data['vwy_EKF'] = vw_EKF[:,1]
            flight_data['vwz_EKF'] = vw_EKF[:,2]
    

    return flight_data

def plot_wind_speed(results, flight_data, lidar_heights, IMU_0 = False, IMU_1=False, EKF_tether=False, EKF = True, savefig = False):
    """
    Plot wind speed based on kite and KCU IMU data
    :param flight_data: flight data
    :return: wind speed plot
    """
    palette = sns.color_palette("tab10")
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    i = 1
    for column in flight_data.columns:
        if 'Wind Speed (m/s)' in column:
            height = ''.join(filter(str.isdigit, column))
            vw_max_col = height + 'm Wind Speed max (m/s)'
            vw_min_col = height + 'm Wind Speed min (m/s)'
            label = 'Lidar ' + height +'m height'
            height = int(height)
            
            if height in lidar_heights:
                
                axs[0].fill_between(flight_data['time'], flight_data[vw_min_col], flight_data[vw_max_col], color=palette[i], alpha=0.3)
                axs[0].plot(flight_data['time'], flight_data[column],color=palette[i], label=label)

                
        if 'Wind Direction' in column:
            height = ''.join(filter(str.isdigit, column))
            label = 'Lidar ' + height +'m height'
            height = int(height)
            if height in lidar_heights:
                axs[1].plot(flight_data['time'], 360-90-flight_data[column],color=palette[i], label=label)

        if 'Z-wind (m/s)' in column:
            height = ''.join(filter(str.isdigit, column))
            label = 'Lidar ' + height +'m height'
            height = int(height)
            if height in lidar_heights:
                axs[2].plot(flight_data['time'], -flight_data[column],color=palette[i], label=label)
                i +=1
                
        
                
    if IMU_0:
        vw_mod = np.sqrt(flight_data['vwx_IMU_0']**2+flight_data['vwy_IMU_0']**2+flight_data['vwz_IMU_0']**2)
        axs[0].plot(flight_data['time'], vw_mod, label='Pitot+Vanes', alpha = 0.8)
        vw_dir = np.arctan2(flight_data['vwy_IMU_0'],flight_data['vwx_IMU_0'])
        axs[1].plot(flight_data['time'], np.degrees(vw_dir)%360, alpha = 0.8)
        axs[2].plot(flight_data['time'], flight_data['vwz_IMU_0'], label='IMU_0', alpha = 0.5)
    if IMU_1:
        vw_mod = np.sqrt(flight_data['vwx_IMU_1']**2+flight_data['vwy_IMU_1']**2+flight_data['vwz_IMU_1']**2)
        axs[0].plot(flight_data['time'], vw_mod, label='IMU_1', alpha = 0.5)
        vw_dir = np.arctan2(flight_data['vwy_IMU_1'],flight_data['vwx_IMU_1'])
        axs[1].plot(flight_data['time'], np.degrees(vw_dir)%360, alpha = 0.5)
        axs[2].plot(flight_data['time'], flight_data['vwz_IMU_1'], label='IMU_1', alpha = 0.5)
    if EKF_tether:
        vw_mod = np.sqrt(flight_data['vwx_EKF']**2+flight_data['vwy_EKF']**2+flight_data['vwz_EKF']**2)
        axs[0].plot(flight_data['time'], vw_mod, label='EKF_tether', alpha = 0.5)
        vw_dir = np.arctan2(flight_data['vwy_EKF'],flight_data['vwx_EKF'])
        axs[1].plot(flight_data['time'], np.degrees(vw_dir)%360, alpha = 0.5)
        axs[2].plot(flight_data['time'], flight_data['vwz_EKF'], label='EKF_tether', alpha = 0.5)
    if EKF:
        wvel = results['wind_velocity']
        vw = np.vstack((wvel*np.cos(results['wind_direction']),wvel*np.sin(results['wind_direction']),np.zeros(len(results)))).T
        axs[0].plot(flight_data['time'], np.linalg.norm(vw,axis=1), label='EKF', alpha = 0.8)
        axs[1].plot(flight_data['time'], np.degrees(results['wind_direction']),label='EKF', alpha = 0.8)
        axs[2].plot(flight_data['time'], results['z_wind'], label='EKF', alpha = 0.8)
    

    min_wdir = min(flight_data['ground_wind_direction'])
    max_wdir = max(flight_data['ground_wind_direction'])
    
    y1 = np.full(len(flight_data), min_wdir-20)
    y2 = np.full(len(flight_data), max_wdir+20)
    # mask = (flight_data['turn_straight'] == 'straight') & (flight_data['powered'] == 'powered')
    # axs[1].fill_between(flight_data['time'], y1,y2,where=mask, color='blue', alpha=0.2, label='Straight')
    # mask = (flight_data['turn_straight'] == 'turn') & (flight_data['powered'] == 'powered')
    # axs[1].fill_between(flight_data['time'], y1,y2,where=mask, color='red', alpha=0.2, label='Turn')
    # mask = (flight_data['powered'] == 'depowered')
    # axs[1].fill_between(flight_data['time'], y1,y2,where=mask, color='green', alpha=0.2, label='Depowered')
     
    axs[0].plot(flight_data['time'], flight_data['ground_wind_velocity'], label='Ground', color = 'grey', alpha = 0.8)
    axs[1].plot(flight_data['time'],flight_data['ground_wind_direction'], label='Ground', color = 'grey', alpha = 0.8)

    axs[0].set_ylim([0,20])
    # axs[0].legend()
    axs[0].set_ylabel('Wind speed (m/s)')
    axs[0].set_xlabel('Time (s)')
    # axs[0].grid()
    axs[1].legend()
    axs[1].set_ylabel('Wind direction (deg)')
    axs[1].set_xlabel('Time (s)')
    # axs[1].set_ylim([175,275])
    # axs[1].grid()
    # axs[2].legend()
    axs[2].set_ylabel('Vertical Wind speed (m/s)')
    axs[2].set_xlabel('Time (s)')
    # axs[2].grid()
    sns.set(style="whitegrid")
    # Enhance overall aesthetics
    for ax in axs:
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

    if savefig:
        plt.tight_layout()
        plt.savefig('wind_speed.png', dpi=300)

def plot_wind_speed_height_bins(results, flight_data, lidar_heights, savefig = False):
    """
    Plot wind speed based on kite and KCU IMU data
    :param flight_data: flight data
    :return: wind speed plot
    """
    palette = sns.color_palette("tab10")
    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    wvel = results['wind_velocity']
    wdir = np.degrees(results['wind_direction'])
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
        if 'Wind Speed (m/s)' in column:
            break
    for i in range(len(flight_data)-1):
        if flight_data[column].iloc[i] != flight_data[column].iloc[i+1]:
            wvel = results['wind_velocity'].iloc[i0:i]
            wdir = np.degrees(results['wind_direction'].iloc[i0:i])
            wvel100.append(np.mean(wvel[results.kite_pos_z<100]))
            wdir100.append(np.mean(wdir[results.kite_pos_z<100]))
            wvel150.append(np.mean(wvel[(results.kite_pos_z<150)&(results.kite_pos_z>100)]))
            wdir150.append(np.mean(wdir[(results.kite_pos_z<150)&(results.kite_pos_z>100)]))
            wvel200.append(np.mean(wvel[(results.kite_pos_z>150)&(results.kite_pos_z<200)]))
            wdir200.append(np.mean(wdir[(results.kite_pos_z>150)&(results.kite_pos_z<200)]))
            wvel250.append(np.mean(wvel[results.kite_pos_z>200]))
            wdir250.append(np.mean(wdir[results.kite_pos_z>200]))
            t_lidar.append(flight_data['time'].iloc[i0])
            i_change.append(i0)
            i0 = i+1
            

    i = 0
    for column in flight_data.columns:
        if 'Wind Speed (m/s)' in column:
            height = ''.join(filter(str.isdigit, column))
            vw_max_col = height + 'm Wind Speed max (m/s)'
            vw_min_col = height + 'm Wind Speed min (m/s)'
            label = 'Lidar ' + height +'m height'
            height = int(height)
            col_save = column            
            if height in lidar_heights:
                selected_values = [flight_data.iloc[j][column] for j in i_change] 
                # axs[0].fill_between(flight_data['time'], flight_data[vw_min_col], flight_data[vw_max_col], color=palette[i], alpha=0.3)
                axs[0].plot(t_lidar, selected_values,color=palette[i], label=label)

                i +=1
    i = 0
    for column in flight_data.columns:      
        if 'Wind Direction' in column:
            height = ''.join(filter(str.isdigit, column))
            label = 'Lidar ' + height +'m height'
            height = int(height)
            if height in lidar_heights:
                selected_values = [flight_data.iloc[j][column] for j in i_change] 
                # axs[0].fill_between(flight_data['time'], flight_data[vw_min_col], flight_data[vw_max_col], color=palette[i], alpha=0.3)
                axs[1].plot(t_lidar, 360-90-np.array(selected_values),color=palette[i], label=label)
                i +=1
    i = 0
    for column in flight_data.columns:
        if 'Z-wind (m/s)' in column:
            height = ''.join(filter(str.isdigit, column))
            label = 'Lidar ' + height +'m height'
            height = int(height)
            if height in lidar_heights:
                axs[2].plot(flight_data['time'], flight_data[column],color=palette[i], label=label)
                i +=1

    
            
    axs[0].plot(t_lidar, wvel100, label='EKF 100m bin', color = palette[0],linestyle='--')
    axs[0].plot(t_lidar, wvel150, label='EKF 150m bin', color = palette[1],linestyle='--')
    axs[0].plot(t_lidar, wvel200, label='EKF 200m bin', color = palette[2],linestyle='--')
    axs[0].plot(t_lidar, wvel250, label='EKF 250m bin', color = palette[3],linestyle='--')
    axs[1].plot(t_lidar, wdir100, label='100m', color = palette[0],linestyle='--')
    axs[1].plot(t_lidar, wdir150, label='150m', color = palette[1],linestyle='--')
    axs[1].plot(t_lidar, wdir200, label='200m', color = palette[2],linestyle='--')
    axs[1].plot(t_lidar, wdir250, label='250m', color = palette[3],linestyle='--')
    
    # axs[0].set_ylim([0,20])
    axs[0].legend()
    axs[0].set_ylabel('Wind speed (m/s)')
    axs[0].set_xlabel('Time (s)')
    axs[0].grid()
    # axs[1].legend()
    axs[1].set_ylabel('Wind direction (deg)')
    axs[1].set_xlabel('Time (s)')
    axs[1].grid()
    # axs[2].legend()
    axs[2].set_ylabel('Wind speed (m/s)')
    axs[2].grid()
    if savefig:
        plt.tight_layout()
        plt.savefig('wind_speed_bins.png', dpi=300)
def correct_aoa_ss_measurements(results,flight_data):

    # Correct angle of attack and sideslip angle based on EKF mean angle of attack
    aoa_EKF = np.array(results['kite_aoa'])
    aoa_vane = np.array(flight_data['kite_angle_of_attack'])
    aoa_vane = np.convolve(aoa_vane, np.ones(10)/10, mode='same')
    mask_pow = (flight_data['ground_tether_reelout_speed'] > 0) & (flight_data['up']<0.2)
    aoa_trim = np.mean(aoa_EKF[mask_pow])
    offset_aoa = aoa_trim - np.mean(flight_data['kite_angle_of_attack'][mask_pow])
    
    # Correct sideslip angle based on EKF mean sideslip angle of 0
    ss_vane = np.array(flight_data['kite_sideslip_angle'])
    ss_vane = np.convolve(ss_vane, np.ones(10)/10, mode='same')
    offset_ss = -np.mean(flight_data['kite_sideslip_angle'][mask_pow])

    

    print('Offset aoa: ', offset_aoa, 'Offset ss: ', offset_ss)
    # Correct angle of attack and sideslip angle based on kite deployment
    aoa_vane = aoa_vane + offset_aoa
    # aoa_vane = aoa_vane + offset_dep*np.array(flight_data['up'])
    # aoa_vane = aoa_vane + offset_turn*np.array(flight_data['us'])
    flight_data['kite_angle_of_attack'] = aoa_vane
    ss_vane = ss_vane + offset_ss
    flight_data['kite_sideslip_angle'] = ss_vane

    return flight_data

def calculate_masks(flight_data):
    up = (flight_data['kcu_actual_depower']-min(flight_data['kcu_actual_depower']))/(max(flight_data['kcu_actual_depower'])-min(flight_data['kcu_actual_depower']))
    us = (flight_data['kcu_actual_steering'])/max(abs(flight_data['kcu_actual_steering']))  
    vz = flight_data['kite_0_vz']
    azimuth = flight_data['kite_azimuth']
    azimuth_rate = np.concatenate((np.diff(azimuth), [0]))

    dep = (up>0.25)
    pow = (flight_data['ground_tether_reelout_speed'] > 0) & (up<0.25)
    trans = ~pow & ~dep
    turn = pow & (vz<0)
    turn = pow & (abs(us) > 0.3)
    straight = pow & ~turn
    turn_right = turn  & (azimuth<0)
    turn_left = turn  & (azimuth>0)
    straight_right = straight  & (azimuth_rate<0)
    straight_left = straight  & (azimuth_rate>0)

    return pow,dep,turn
def plot_aero_coeff_vs_aoa_ss(results, flight_data, cycles_plotted, IMU_0=False, IMU_1=False, EKF_tether=False, EKF=True, savefig=False):
    """
    Plot wind speed based on kite and KCU IMU data
    :param flight_data: flight data
    :return: wind speed plot
    """
    palette = sns.color_palette("tab10")
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Aero coefficients vs aoa and ss')
    
    mask_cycle = np.any([flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
    
    # Define time bounds for x-axis
    time_min = flight_data[mask_cycle]['time'].min()
    time_max = flight_data[mask_cycle]['time'].max()
    
    # Plotting each aerodynamic coefficient
    axs[0].plot(results[mask_cycle]['time'], results[mask_cycle]['CL'])
    axs[1].plot(results[mask_cycle]['time'], results[mask_cycle]['CD'], label='CD (Total)')
    axs[1].plot(results[mask_cycle]['time'], results[mask_cycle]['cd_kcu'], label='CD (KCU)')
    axs[1].plot(results[mask_cycle]['time'], results[mask_cycle]['cd_tether'], label='CD (Tether)')
    axs[2].plot(results[mask_cycle]['time'], results[mask_cycle]['CS'], label='CS')

    # AOA and Side Slip plots with conditions
    if EKF:
        axs[3].plot(results[mask_cycle]['time'], results[mask_cycle]['kite_aoa'], label='aoa EKF')
        axs[4].plot(results[mask_cycle]['time'], results[mask_cycle]['kite_sideslip'], label='ss EKF')
    if IMU_0:
        axs[3].plot(results[mask_cycle]['time'], results[mask_cycle]['aoa_IMU_0'], label='aoa IMU 0')
        axs[4].plot(results[mask_cycle]['time'], results[mask_cycle]['ss_IMU_0'], label='ss IMU 0')
    if IMU_1:
        axs[3].plot(results[mask_cycle]['time'], results[mask_cycle]['aoa_IMU_1'], label='aoa IMU 1')
        axs[4].plot(results[mask_cycle]['time'], results[mask_cycle]['ss_IMU_1'], label='ss IMU 1')
    if EKF_tether:
        axs[3].plot(results[mask_cycle]['time'], results[mask_cycle]['aoa_EKF_tether'], label='aoa EKF tether')
        axs[4].plot(results[mask_cycle]['time'], results[mask_cycle]['ss_EKF_tether'], label='ss EKF tether')

    # Vane data
    axs[3].plot(flight_data[mask_cycle]['time'], flight_data[mask_cycle]['kite_angle_of_attack'], label='aoa vane')
    axs[4].plot(flight_data[mask_cycle]['time'], flight_data[mask_cycle]['kite_sideslip_angle'], label='ss vane')

    # Highlight operational modes
    i = 0
    for ax in axs:
        mask_straight = (flight_data['turn_straight'] == 'straight') & (flight_data['powered'] == 'powered') & mask_cycle
        mask_turn = (flight_data['turn_straight'] == 'turn') & (flight_data['powered'] == 'powered') & mask_cycle
        mask_depowered = (flight_data['powered'] == 'depowered') & mask_cycle
        if i == 0:
            ax.fill_between(flight_data['time'], ax.get_ylim()[0], ax.get_ylim()[1], where=mask_straight, color='blue', alpha=0.2, label = 'Straight')
            ax.fill_between(flight_data['time'], ax.get_ylim()[0], ax.get_ylim()[1], where=mask_turn, color='red', alpha=0.2, label = 'Turn')
            ax.fill_between(flight_data['time'], ax.get_ylim()[0], ax.get_ylim()[1], where=mask_depowered, color='green', alpha=0.2, label = 'Reel-in')
        else:
            ax.fill_between(flight_data['time'], ax.get_ylim()[0], ax.get_ylim()[1], where=mask_straight, color='blue', alpha=0.2)
            ax.fill_between(flight_data['time'], ax.get_ylim()[0], ax.get_ylim()[1], where=mask_turn, color='red', alpha=0.2)
            ax.fill_between(flight_data['time'], ax.get_ylim()[0], ax.get_ylim()[1], where=mask_depowered, color='green', alpha=0.2)
        
        ax.set_xlim([time_min, time_max])  # Set x-axis limits to fit the data
        ax.grid(True)  # Enable grid for better data visualization
        
        i += 1
    # Labels, legends, and layout
    axs[0].set_ylabel
    
    axs[0].legend()
    axs[1].legend()
    axs[3].legend()
    axs[4].legend()
    axs[0].set_ylabel('CL')
    axs[1].set_ylabel('CD')
    axs[2].set_ylabel('CS')
    axs[3].set_ylabel('Angle of attack (deg)')
    axs[4].set_ylabel('Side slip (deg)')
    axs[4].set_xlabel('Time (s)')

        
def plot_aero_coeff_vs_up_us(results,flight_data,cycles_plotted, IMU_0 = False, IMU_1=False, EKF_tether=False, EKF = True,savefig=False):
    """
    Plot wind speed based on kite and KCU IMU data
    :param flight_data: flight data
    :return: wind speed plot
    """
    palette = sns.color_palette("tab10")
    fig, axs = plt.subplots(5, 1, figsize=(20, 12), sharex=True)
    fig.suptitle('Aero coefficients vs up and us')
    mask_cycle = np.any([flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)

    axs[0].plot(results[mask_cycle]['time'], results[mask_cycle]['CL'])
    axs[1].plot(results[mask_cycle]['time'], results[mask_cycle]['CD'])
    axs[2].plot(results[mask_cycle]['time'], results[mask_cycle]['CS'])
    
    axs[3].plot(flight_data[mask_cycle]['time'], flight_data[mask_cycle]['up'])
    axs[4].plot(flight_data[mask_cycle]['time'], flight_data[mask_cycle]['us'])

    mask = (flight_data['turn_straight'] == 'straight') & (flight_data['powered'] == 'powered')&mask_cycle
    axs[0].fill_between(flight_data['time'], 0,1.5,where=mask, color='blue', alpha=0.2, label='Straight')
    axs[1].fill_between(flight_data['time'], 0,0.3,where=mask, color='blue', alpha=0.2, label='Straight')
    axs[2].fill_between(flight_data['time'], -0.2,0.2,where=mask, color='blue', alpha=0.2, label='Straight')
    axs[3].fill_between(flight_data['time'], -5,30,where=mask, color='blue', alpha=0.2)
    axs[4].fill_between(flight_data['time'], -15,15,where=mask, color='blue', alpha=0.2)
    mask = (flight_data['turn_straight'] == 'turn') & (flight_data['powered'] == 'powered')&mask_cycle
    axs[0].fill_between(flight_data['time'], 0,1.5,where=mask, color='red', alpha=0.2, label='Turn')
    axs[1].fill_between(flight_data['time'], 0,0.3,where=mask, color='red', alpha=0.2, label='Turn')
    axs[2].fill_between(flight_data['time'], -0.2,0.2,where=mask, color='red', alpha=0.2, label='Turn')
    axs[3].fill_between(flight_data['time'], -5,30,where=mask, color='red', alpha=0.2)
    axs[4].fill_between(flight_data['time'], -15,15,where=mask, color='red', alpha=0.2)
    mask = (flight_data['powered'] == 'depowered')&mask_cycle
    axs[0].fill_between(flight_data['time'],0,1.5,where=mask, color='green', alpha=0.2, label='Depowered')
    axs[1].fill_between(flight_data['time'],0,0.3,where=mask, color='green', alpha=0.2, label='Depowered')
    axs[2].fill_between(flight_data['time'],-0.2,0.2,where=mask, color='green', alpha=0.2, label='Depowered')
    axs[3].fill_between(flight_data['time'],-5,30,where=mask, color='green', alpha=0.2)
    axs[4].fill_between(flight_data['time'],-15,15,where=mask, color='green', alpha=0.2)

    axs[0].set_ylim([0,1.2])
    axs[1].set_ylim([0,0.3])
    axs[2].set_ylim([-0.12,0.12])
    axs[3].set_ylim([0,1])
    axs[4].set_ylim([-1,1])
    axs[0].set_ylabel('CL')
    axs[1].set_ylabel('CD')
    axs[2].set_ylabel('CS')
    axs[3].set_ylabel('up')
    axs[4].set_ylabel('us')
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    axs[3].grid()
    axs[4].grid()
    axs[0].legend()  

    if savefig:
        plt.savefig('aero_coeff_vs_up_us.png', dpi=300)

    
def determine_turn_straight(row, threshold_us = 0.4):
    
    if (abs(row['us']) >threshold_us):
        return 'turn'
    else:
        return 'straight'
    
def determine_powered_depowered(row, threshold_up = 0.25):
    
    if (row['up']>threshold_up):
        return 'depowered'
    else:
        return 'powered'

def determine_left_right(row, threshold_azimuth = 0):
    if (row['kite_azimuth']<threshold_azimuth):
        return 'right'
    else:
        return 'left'
    
from scipy.stats import gaussian_kde
# Function to calculate densities
def calculate_densities(x, y):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    return z

def plot_probability_density(x,y,fig,axs,xlabel=None,ylabel=None):
    z1 = calculate_densities(x, y)
    sc1 = axs.scatter(x, y, c=z1, cmap='viridis', label = 'Sensor Fusion')
    fig.colorbar(sc1, ax=axs, label='Probability Density')
    axs.set_ylabel(ylabel)
    axs.set_xlabel(xlabel)
    axs.grid()
    axs.legend()
    
def plot_hexbin_density(x, y, xlabel=None, ylabel=None):
    fig, ax = plt.subplots()
    hb = ax.hexbin(x, y, gridsize=50, cmap='viridis', bins='log')
    cb = fig.colorbar(hb, ax=ax, label='log10(N)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    plt.show()

def plot_CL_CD_aoa(results,flight_data,mask,aoa_method,savefig = False):

    if aoa_method == 'IMU_0':
        aoa = results['aoa_IMU_0']
    elif aoa_method == 'IMU_1':
        aoa = results['aoa_IMU_1']
    elif aoa_method == 'EKF':
        aoa = results['kite_aoa']
    else:
        aoa = flight_data['kite_angle_of_attack']
    
    
    fig, axs = plt.subplots(2,1, figsize=(10, 10), sharex=True)
    fig.suptitle('CL and CD vs aoa')
    plot_probability_density(aoa[mask],results['CL'][mask],fig,axs[0],ylabel='CL')
    plot_probability_density(aoa[mask],results['CD'][mask],fig,axs[1],'aoa','CD')
    
    if savefig == True:
        plt.tight_layout()
        plt.savefig('wind_profile.png', dpi=300)
    
def plot_CL_CD_up(results,flight_data,mask,aoa_method,savefig = False):

    up = flight_data['up']
    
    
    fig, axs = plt.subplots(2,1, figsize=(10, 10), sharex=True)
    fig.suptitle('CL and CD vs aoa')
    plot_probability_density(up[mask],results['CL'][mask],fig,axs[0],ylabel='CL')
    plot_probability_density(up[mask],results['CD'][mask],fig,axs[1],'up','CD')
    if savefig == True:
        plt.tight_layout()
        plt.savefig('wind_profile.png', dpi=300)

def plot_CL_CD_ss(results,flight_data,mask,ss_method):

    if ss_method == 'IMU_0':
        ss = results['ss_IMU_0']
    elif ss_method == 'IMU_1':
        ss = results['ss_IMU_1']
    elif ss_method == 'EKF':
        ss = results['kite_sideslip']
    else:
        ss = flight_data['kite_sideslip_angle']

    fig, axs = plt.subplots(2,1, figsize=(10, 10), sharex=True)
    fig.suptitle('CL and CD vs ss')
    plot_probability_density(ss[mask],results['CL'][mask],fig,axs[0],ylabel='CL')
    plot_probability_density(ss[mask],results['CD'][mask],fig,axs[1],'ss','CD')

def plot_prob_coeff_vs_aoa_ss(results,coeff,mask,aoa_method):

    if aoa_method == 'IMU_0':
        aoa = results['aoa_IMU_0']
        ss = results['ss_IMU_0']
    elif aoa_method == 'IMU_1':
        aoa = results['aoa_IMU_1']
        ss = results['ss_IMU_1']
    elif aoa_method == 'EKF':
        aoa = results['kite_aoa']
        ss = results['kite_sideslip']
    else:
        aoa = results['kite_aoa']
        ss = np.zeros(len(results))
    
    fig, axs = plt.subplots(1,2, figsize=(10, 6))
    fig.suptitle('Probability Density vs aoa and ss')
    plot_probability_density(aoa[mask],coeff[mask],fig,axs[0],xlabel='aoa')
    plot_probability_density(ss[mask],coeff[mask],fig,axs[1],'ss','')

def plot_time_series(flight_data, y, ax, color='blue',  ylabel = None, label=None,plot_phase=False):
    t = flight_data.time
    ax.plot(t, y, color=color, label=label)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time (s)')
    
    if plot_phase:
        y1 = min(y)-0.1*(max(y)-min(y))
        y2 = max(y)+0.1*(max(y)-min(y))
        mask = (flight_data['turn_straight'] == 'straight') & (flight_data['powered'] == 'powered')
        ax.fill_between(flight_data['time'], y1,y2,where=mask, color='blue', alpha=0.2, label='Straight')
        mask = (flight_data['turn_straight'] == 'turn') & (flight_data['powered'] == 'powered')
        ax.fill_between(flight_data['time'], y1,y2,where=mask, color='red', alpha=0.2, label='Turn')
        mask = (flight_data['powered'] == 'depowered')
        ax.fill_between(flight_data['time'], y1,y2,where=mask, color='green', alpha=0.2, label='Depowered')
        ax.legend()
        
        # def plot_wind_profile(flight_data, 


# def plot_kinematic_yaw(flight_data, results):
def plot_wind_profile(flight_data, results,savefig=False):

    palette = sns.color_palette("tab10")
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    i = 1
    lidar_heights = []
    min_vel = []
    max_vel = []
    min_dir = []
    max_dir = []
    for column in flight_data.columns:
        if 'Wind Speed (m/s)' in column:
            height = ''.join(filter(str.isdigit, column))
            vw_max_col = height + 'm Wind Speed max (m/s)'
            vw_min_col = height + 'm Wind Speed min (m/s)'
            label = 'Lidar ' + height +'m height'
            height = int(height)
            lidar_heights.append(height)
            min_vel.append(min(flight_data[column]))
            max_vel.append(max(flight_data[column]))

            i +=1
        if 'Wind Direction' in column:
            height = ''.join(filter(str.isdigit, column))
            label = 'Lidar ' + height +'m height'
            height = int(height)
            min_dir.append(min(360-90-flight_data[column]))
            max_dir.append(max(360-90-flight_data[column]))

            
    axs[0].fill_betweenx(lidar_heights, min_vel, max_vel, color=palette[0], alpha=0.3, label='Lidar')
    axs[1].fill_betweenx(lidar_heights, min_dir, max_dir, color=palette[0], alpha=0.3, label='Lidar')

    wvelEKF = results['wind_velocity']
    plot_hexbin_density(wvelEKF,results['kite_pos_z'],fig,axs[0])
    # axs[0].scatter( wvelEKF, results['kite_pos_z'], color=palette[1], label='EKF', alpha = 0.1)
    axs[1].scatter(np.degrees(results['wind_direction']), results['kite_pos_z'], color=palette[1], label='EKF', alpha = 0.1)

    axs[0].legend()
    axs[0].set_xlabel('Wind speed (m/s)')
    axs[0].set_ylabel('Height (m)')
    axs[0].grid()
    axs[0].set_xlim([0, 17])
    axs[1].legend()
    axs[1].set_xlabel('Wind direction (deg)')
    axs[1].grid()

    if savefig:
        plt.tight_layout()
        plt.savefig('wind_profile.png', dpi=300)

def plot_wind_profile_bins(flight_data, results, step = 20, savefig = False):
    # Extract data
    height = results['kite_pos_z']
    wvel = results['wind_velocity']
    wdir = np.degrees(results['wind_direction'])

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
        if 'Wind Speed (m/s)' in column:
            height = ''.join(filter(str.isdigit, column))
            stdvw_col = height + 'm Wind Speed Dispersion (m/s)'
            label = 'Lidar ' + height +'m height'
            height = int(height)
            lidar_heights.append(height)
            min_vel.append(np.mean(flight_data[column]-flight_data[stdvw_col]))
            max_vel.append(np.mean(flight_data[column]+flight_data[stdvw_col]))
            wvel.append(np.mean(flight_data[column]))
            std_vel.append(np.std(flight_data[column]))
            i +=1
            
        if 'Wind Direction' in column:
            height = ''.join(filter(str.isdigit, column))
            label = 'Lidar ' + height +'m height'
            height = int(height)
            min_dir.append(min(360-90-flight_data[column]))
            max_dir.append(max(360-90-flight_data[column]))
            wdir.append(np.mean(360-90-flight_data[column]))
            std_dir.append(np.std(flight_data[column]))

    # Set up plot-
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    # fig.suptitle('Wind Profile Analysis', fontsize=16)
    # fontsize = 16

    # Wind velocity plot
    axs[0].errorbar(wvel_means, bin_centers, xerr=wvel_stds, fmt='o-', color='#00B8C8', ecolor='lightgray', elinewidth=3, capsize=0, label='EKF ± SD')
    axs[0].errorbar(wvel, lidar_heights,xerr = std_vel , fmt='o-', color='#0076C2', ecolor='gray', elinewidth=3, capsize=0, label='Lidar ± SD')
    axs[0].set_xlabel('Wind Velocity (m/s)')
    axs[0].set_ylabel('Height (m)')
    axs[0].legend()
    axs[0].set_xlim([min(wvel_means) - 2, max(wvel_means) + 2])

    # Wind direction plot
    axs[1].errorbar(wdir_means, bin_centers, xerr=wdir_stds, fmt='o-', color='#6CC24A', ecolor='lightgray', elinewidth=3, capsize=0, label='EKF ± SD')
    axs[1].errorbar(wdir, lidar_heights, xerr = std_dir, fmt='o-', color='#009B77', ecolor='gray', elinewidth=3, capsize=0, label='Lidar ± SD')
    axs[1].set_xlabel('Wind Direction (degrees)')
    axs[1].legend()
    axs[1].set_xlim([min(wdir_means) - 5, max(wdir_means) + 5])
    

    # Enhance overall aesthetics
    for ax in axs:
        ax.set_ylim([0, bin_centers.max() + step])
        
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for title
    plt.show()
    if savefig:
        plt.savefig('wind_profile_bins.png', dpi=300)


    return axs



def plot_kite_reference_frame(results, flight_data, imus):
    ## Create 3d plot of kite reference frame
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X_N')
    ax.set_ylabel('Y_E')
    ax.set_zlabel('Z_U')
    spacing = 20
    for i in np.arange(0,len(flight_data), spacing):
        len_arrow = 30
        # Calculate EKF tether orientation based on euler angles and plot it
        ex, ey, ez = calculate_reference_frame_euler( np.radians(results.kite_roll.iloc[i]), 
                                                     np.radians(results.kite_pitch.iloc[i]), 
                                                     np.radians(results.kite_yaw.iloc[i]), 
                                                     bodyFrame='ENU')
        ax.quiver(results['kite_pos_x'].iloc[i], results['kite_pos_y'].iloc[i], results['kite_pos_z'].iloc[i], ex[0],  \
                    ex[1], ex[2],
                color='green', length=len_arrow)
        ax.quiver(results['kite_pos_x'].iloc[i], results['kite_pos_y'].iloc[i], results['kite_pos_z'].iloc[i], ey[0],  \
                    ey[1], ey[2],
                color='black', length=len_arrow)
        ax.quiver(results['kite_pos_x'].iloc[i], results['kite_pos_y'].iloc[i], results['kite_pos_z'].iloc[i], ez[0],  \
                    ez[1], ez[2],
                color='r', length=len_arrow)
        # Calculate IMU tether orientation based on euler angles and plot it
        # for imu in imus:
        #     ex, ey, ez = calculate_reference_frame_euler(flight_data['kite_'+str(imu)+'_roll'].iloc[i], 
        #                                                 flight_data['kite_'+str(imu)+'_pitch'].iloc[i], 
        #                                                 flight_data['kite_'+str(imu)+'_yaw'].iloc[i])    
        #     ax.quiver(flight_data['kite_'+str(imu)+'_rx'].iloc[i], 
        #                 flight_data['kite_'+str(imu)+'_ry'].iloc[i], 
        #                 flight_data['kite_'+str(imu)+'_rz'].iloc[i], 
        #                 ex[0],
        #                 ex[1],
        #                 ex[2],
        #                 color='b', length=len_arrow)
        #     ax.quiver(flight_data['kite_'+str(imu)+'_rx'].iloc[i],
        #                 flight_data['kite_'+str(imu)+'_ry'].iloc[i], 
        #                 flight_data['kite_'+str(imu)+'_rz'].iloc[i], 
        #                 ey[0], 
        #                 ey[1], 
        #                 ey[2],
        #                 color='b', length=len_arrow)
        #     ax.quiver(flight_data['kite_'+str(imu)+'_rx'].iloc[i],
        #                 flight_data['kite_'+str(imu)+'_ry'].iloc[i], 
        #                 flight_data['kite_'+str(imu)+'_rz'].iloc[i], 
        #                 ez[0], 
        #                 ez[1], 
        #                 ez[2],
        #                 color='b', length=len_arrow)

    ax.plot(results.kite_pos_x, results.kite_pos_y, results.kite_pos_z,color='grey')
    ax.scatter(results['kite_pos_x'].iloc[0:spacing:-1], results['kite_pos_y'].iloc[0:spacing:-1], results['kite_pos_z'].iloc[0:spacing:-1],color='r')
    ax.legend()
    ax.quiver(0,0,0, 
                0, 
                0, 
                1,
                color='black', length=len_arrow)
    ax.quiver(0,0,0, 
                0, 
                1, 
                0,
                color='black', length=len_arrow)
    ax.quiver(0,0,0, 
                1, 
                0, 
                0,
                color='black', length=len_arrow)
    ax.set_box_aspect([1,1,1])

def plot_cl_curve(cl,cd,aoa,mask,axs,label = None, savefig=False):
    CL = cl[mask]
    CD = cd[mask]
    alpha = aoa[mask]


    step = 1
    bins = np.arange(int(alpha.min())-step/2, int(alpha.max())+step/2, step)
    bin_indices = np.digitize(alpha, bins)  # Find the bin index for each alpha value
    num_bins = len(bins)

    CL_means = np.array([CL[bin_indices == i].mean() for i in range(1, num_bins)])
    CD_means = np.array([CD[bin_indices == i].mean() for i in range(1, num_bins)])

    CL_stds = np.array([CL[bin_indices == i].std() for i in range(1, num_bins)])
    CD_stds = np.array([CD[bin_indices == i].std() for i in range(1, num_bins)])
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Plot CL and shade the area for std deviation
    axs[0,0].plot(bin_centers, CL_means, 'o-', markersize=8, linewidth=2, label=label)
    axs[0,0].fill_between(bin_centers, CL_means - CL_stds, CL_means + CL_stds, alpha=0.2)
    axs[0,0].set_xlabel('Angle of Attack (degrees)', fontsize=14)
    axs[0,0].set_ylabel('Lift Coefficient (CL)',  fontsize=14)
    axs[0,0].grid(True)

    # Plot CD and shade the area for std deviation
    axs[0,1].plot(bin_centers, CD_means, 'o-', markersize=8, linewidth=2, label=label)
    axs[0,1].fill_between(bin_centers, CD_means - CD_stds, CD_means + CD_stds, alpha=0.2)
    axs[0,1].set_xlabel('Angle of Attack (degrees)', fontsize=14)
    axs[0,1].set_ylabel('Drag Coefficient (CD)', fontsize=14)
    axs[0,1].grid(True)

    axs[1,0].plot(bin_centers, CL_means**3/CD_means**2, 'o-', label=label)
    axs[1,0].fill_between(bin_centers, (CL_means - CL_stds)**3/(CD_means + CD_stds)**2, (CL_means + CL_stds)**3/(CD_means - CD_stds)**2, alpha=0.2)
    axs[1,0].set_xlabel('Angle of Attack (degrees)', fontsize=14)
    axs[1,0].set_ylabel('Power Coefficient (CL^3/CD^2)', fontsize=14)
    axs[1,0].grid(True)
    
    axs[1,1].plot(bin_centers, CL_means/CD_means, 'o-', label=label)
    axs[1,1].fill_between(bin_centers, (CL_means - CL_stds)/(CD_means + CD_stds), (CL_means + CL_stds)/(CD_means - CD_stds), alpha=0.2)
    axs[1,1].set_xlabel('Angle of Attack (degrees)', fontsize=14)
    axs[1,1].set_ylabel('L/D Ratio', fontsize=14)
    axs[1,1].grid(True)

    


    