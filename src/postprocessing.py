import numpy as np
from config import kappa, z0
from utils import  R_EG_Body, calculate_angle, project_onto_plane

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
    wvel = np.array(results['uf']/kappa*np.log(results['z']/z0))
    vw = np.vstack((wvel*np.cos(results['wdir']),wvel*np.sin(results['wdir']),np.zeros(len(results)))).T
    r_kite = np.vstack((np.array(results['x']),np.array(results['y']),np.array(results['z']))).T
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
        
    min_depower = min(flight_data['kcu_actual_depower'].iloc[5000:-5000])
    max_depower = max(flight_data['kcu_actual_depower'].iloc[5000:-5000])
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

def calculate_reference_frame_euler(roll, pitch, yaw):
    """
    Calculate the reference frame based on euler angles
    :param roll: roll angle
    :param pitch: pitch angle
    :param yaw: yaw angle
    :return: ex, ey, ez
    """

    ######################### DO IT WELL #########################################
    # Calculate tether orientation based on euler angles
    Transform_Matrix=R_EG_Body(roll/180*np.pi,pitch/180*np.pi,(yaw)/180*np.pi)
    #    Transform_Matrix=R_EG_Body(kite_roll[i]/180*np.pi,kite_pitch[i]/180*np.pi,kite_yaw_modified[i])
    Transform_Matrix=Transform_Matrix.T
    #X_vector
    ex_kite=Transform_Matrix.dot(np.array([-1,0,0]))
    #Y_vector
    ey_kite=Transform_Matrix.dot(np.array([0,-1,0]))
    #Z_vector
    ez_kite=Transform_Matrix.dot(np.array([0,0,1]))

    return ex_kite, ey_kite, ez_kite

def determine_turn_straight(row):
    
    if (abs(row['us']) > 0.3):
        return 'turn'
    else:
        return 'straight'
    
def determine_powered_depowered(row):
    
    if (row['up']>0.25):
        return 'depowered'
    else:
        return 'powered'

def determine_left_right(row):
    if (row['kite_azimuth']<0):
        return 'right'
    else:
        return 'left'