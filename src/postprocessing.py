import numpy as np
from config import kappa, z0
from utils import  R_EG_Body, calculate_angle, project_onto_plane

def unwrap_degrees(signal):
    for i in range(1,len(signal)):
        if abs(signal[i]-signal[i-1])>180:
            signal[i::] += signal[i-1]-signal[i]
    
    return signal

def normalize_angles(signal):
    """
    Normalizes an array of angles so that all angles are within the range 0 to 360 degrees.

    Parameters:
    signal (numpy.ndarray): The input array of angles in degrees.

    Returns:
    numpy.ndarray: The normalized array of angles within the range 0 to 360 degrees.
    """
    normalized_signal = np.mod(signal, 360)
    normalized_signal[normalized_signal < 0] += 360
    return normalized_signal

def compute_mse(sig1, sig2, offset):
    shifted_sig2 = sig2 + offset
    mse = np.mean((sig1 - shifted_sig2) ** 2)
    return mse

def find_offset(signal1, signal2, offset_range=[-360, 360]):
    """
    Find the offset between two signals
    :param signal1: first signal
    :param signal2: second signal
    :return: offset
    """
    offsets = np.linspace(offset_range[0], offset_range[1], 1000)
    mse_values = [compute_mse(signal1, signal2, offset) for offset in offsets]
    offset = offsets[np.argmin(mse_values)]
    return offset

def construct_transformation_matrix(e_x_b, e_y_b, e_z_b):
    # Construct the matrix by arranging the unit vectors as columns
    R = np.array([e_x_b, e_y_b, e_z_b]).T
    return R

def remove_offsets_IMU_data(results, flight_data, sensor=0):
    """ Remove offsets of IMU euler angles based on EKF results"""
    unwrapped_angles = np.unwrap(np.radians(flight_data['kite_'+str(sensor)+'_yaw']))
    flight_data['kite_'+str(sensor)+'_yaw' ] = np.degrees(unwrapped_angles)
    unwrapped_angles = np.unwrap(np.radians(results['yaw']))
    results['yaw'] = np.degrees(unwrapped_angles)
    # Roll
    roll_offset = find_offset(results['roll'], flight_data['kite_'+str(sensor)+'_roll'])
    flight_data['kite_0_roll'] = flight_data['kite_'+str(sensor)+'_roll'] + roll_offset
    print('Roll offset: ', roll_offset)
    # Pitch
    mask_pitch = (flight_data['turn_straight'] == 'straight')&(flight_data['powered'] == 'powered')
    pitch_offset = find_offset(results[mask_pitch]['pitch'], flight_data[mask_pitch]['kite_'+str(sensor)+'_pitch'])
    flight_data['kite_0_pitch'] = flight_data['kite_'+str(sensor)+'_pitch'] + pitch_offset
    print('Pitch offset: ', pitch_offset)
    # Yaw
    yaw_offset = find_offset(results['yaw'], flight_data['kite_'+str(sensor)+'_yaw'])
    flight_data['kite_0_yaw'] = flight_data['kite_'+str(sensor)+'_yaw'] + yaw_offset
    print('Yaw offset: ', yaw_offset)

    return flight_data



def postprocess_results(results,flight_data, kite, imus = [0], remove_IMU_offsets=True, correct_IMU_deformation = False, remove_vane_offsets=False, estimate_kite_angle=False):
    """
    Calculate angle of attack and sideslip based on kite and KCU IMU data
    :param results: results from the simulation
    :param kite: kite object
    :param IMU_0: IMU data from the kite
    :param IMU_1: IMU data from the KCU
    :param EKF_tether: EKF data from the tether orientation and IMU yaw
    :return: results with aoa and ss va radius omega and slack
    """
    
    min_depower = min(flight_data['kcu_actual_depower'])
    max_depower = max(flight_data['kcu_actual_depower'])
    flight_data['us'] =  (flight_data['kcu_actual_steering'])/max(abs(flight_data['kcu_actual_steering'])) 
    flight_data['up'] = (flight_data['kcu_actual_depower']-min_depower)/(max_depower-min_depower)
    # Identify flight phases
    flight_data['turn_straight'] = flight_data.apply(determine_turn_straight, axis=1)
    flight_data['right_left'] = flight_data.apply(determine_turn_straight, axis=1)
    flight_data['powered'] = flight_data.apply(determine_powered_depowered, axis=1)
    
    
    results['wind_direction'] = results['wind_direction']%(2*np.pi)
    
    if remove_IMU_offsets:
        for imu in imus:
            flight_data = remove_offsets_IMU_data(results, flight_data, sensor=imu)
    


    # Calculate apparent speed based on EKF results
    wvel = results['wind_velocity']
    vw = np.vstack((wvel*np.cos(results['wind_direction']),wvel*np.sin(results['wind_direction']),np.zeros(len(results)))).T
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


    for imu in imus:
        results['aoa_IMU_'+str(imu)] = np.zeros(len(results))
        results['ss_IMU_'+str(imu)] = np.zeros(len(results))
 
        
    

    if correct_IMU_deformation:
        # Correct angle of attack for depowered phase on EKF mean pitch during depower
        mask_turn = abs(flight_data['us'])>0.9
        mask_dep = flight_data['up']>0.9
        pitch_EKF = np.array(results['pitch'])
        pitch_IMU_0 = np.array(flight_data['kite_0_pitch'])
        offset_dep = np.mean(pitch_EKF[mask_dep]-pitch_IMU_0[mask_dep])
        offset_turn = np.mean(pitch_EKF[mask_turn]-pitch_IMU_0[mask_turn])
        offset_pitch = offset_dep*np.array(flight_data['up'])#-offset_turn*np.array(abs(flight_data['us']))
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
        for imu in imus:
            ex, ey, ez = calculate_reference_frame_euler(flight_data['kite_'+str(imu)+'_roll'].iloc[i], 
                                                            flight_data['kite_'+str(imu)+'_pitch'].iloc[i], 
                                                            flight_data['kite_'+str(imu)+'_yaw'].iloc[i])  
            # Calculate wind velocity based on KCU orientation and wind speed and direction
            va_proj = project_onto_plane(va_kite[i], ey)           # Projected apparent wind velocity onto kite y axis
            results.loc[i,'aoa_IMU_'+str(imu)] = calculate_angle(ez,va_proj)-90            # Angle of attack
            va_proj = project_onto_plane(va_kite[i], ez)           # Projected apparent wind velocity onto kite z axis
            results.loc[i,'ss_IMU_'+str(imu)] = 90-calculate_angle(ey,va_proj)        # Sideslip angle
            
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
    # results['slack'] = slack
    results['radius_turn'] = radius_turn
    results['omega'] = omega

        
    if remove_vane_offsets:
        flight_data = correct_aoa_ss_measurements(results, flight_data)
    
    if estimate_kite_angle:
        for imu in imus:
            results['aoa_IMU_'+str(imu)] = results['aoa_IMU_'+str(imu)]-flight_data['offset_pitch']
        results['aoa'] = results['aoa']-flight_data['offset_pitch']
        flight_data['kite_angle_of_attack'] = flight_data['kite_angle_of_attack']-flight_data['offset_pitch']
    
    return results, flight_data

def calculate_wind_speed_airborne_sensors(results, flight_data, imus = [0]):
    """
    Calculate wind speed based on kite and KCU IMU data
    :param flight_data: flight data
    :return: flight data with wind speed
    """
    for imu in imus:
        flight_data['vwx_IMU_'+str(imu)] = np.zeros(len(flight_data))
        flight_data['vwy_IMU_'+str(imu)] = np.zeros(len(flight_data))
        flight_data['vwz_IMU_'+str(imu)] = np.zeros(len(flight_data))
    
    measured_va = flight_data['kite_apparent_windspeed']
    measured_aoa = flight_data['kite_angle_of_attack']
    measured_ss = -flight_data['kite_sideslip_angle']
    
    # measured_aoa = results['aoa_IMU_0']
    # measured_ss =  results['ss_IMU_0']
        
    measured_va = results['va_kite']
    for i in range(len(flight_data)):
        for imu in imus:
            ex_kite, ey_kite, ez_kite = calculate_reference_frame_euler(flight_data['kite_'+str(imu)+'_roll'][i], 
                                                                        flight_data['kite_'+str(imu)+'_pitch'][i], 
                                                                        flight_data['kite_'+str(imu)+'_yaw'][i],
                                                                        bodyFrame='ENU')
            # Calculate apparent wind velocity based on KCU orientation and apparent wind speed and aoa and ss
            va = -ex_kite*measured_va[i]*np.cos(measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ey_kite*measured_va[i]*np.sin(measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ez_kite*measured_va[i]*np.sin(measured_aoa[i]/180*np.pi)
            # Calculate wind velocity based on KCU orientation and wind speed and direction
            flight_data.loc[i, 'vwx_IMU_'+str(imu)] = va[0]+results['vx'][i]
            flight_data.loc[i, 'vwy_IMU_'+str(imu)] = va[1]+results['vy'][i]
            flight_data.loc[i, 'vwz_IMU_'+str(imu)] = va[2]+results['vz'][i]
                
    return flight_data

def calculate_reference_frame_euler(roll, pitch, yaw, bodyFrame='NED'):
    """
    Calculate the reference frame based on euler angles
    :param roll: roll angle
    :param pitch: pitch angle
    :param yaw: yaw angle
    :param eulerFrame: euler frame
    :return: ex, ey, ez
    """
    if bodyFrame == 'NED':
        roll = -roll+180
    elif bodyFrame == 'ENU':
        roll = -roll
 
    
    # Calculate tether orientation based on euler angles
    Transform_Matrix=R_EG_Body(roll/180*np.pi,pitch/180*np.pi,(yaw)/180*np.pi)
    #    Transform_Matrix=R_EG_Body(kite_roll[i]/180*np.pi,kite_pitch[i]/180*np.pi,kite_yaw_modified[i])
    Transform_Matrix=Transform_Matrix.T
    #X_vector
    ex_kite=Transform_Matrix.dot(np.array([1,0,0]))
    #Y_vector
    ey_kite=Transform_Matrix.dot(np.array([0,1,0]))
    #Z_vector
    ez_kite=Transform_Matrix.dot(np.array([0,0,1]))

    # Transform from ENU to NED and return        
    return ex_kite, ey_kite, ez_kite


def determine_turn_straight(row):
    
    if (abs(row['us']) > 0.3):
        return 'turn'
    else:
        return 'straight'
    
def determine_powered_depowered(row):
    
    if (row['up']>0.6):
        return 'depowered'
    else:
        return 'powered'

def determine_left_right(row):
    if (row['kite_azimuth']<0):
        return 'right'
    else:
        return 'left'
    
def correct_aoa_ss_measurements(results,flight_data):

    # Correct angle of attack and sideslip angle based on EKF mean angle of attack
    aoa_EKF = np.array(results['aoa'])
    aoa_vane = np.array(flight_data['kite_angle_of_attack'])
    aoa_vane = np.convolve(aoa_vane, np.ones(10)/10, mode='same')
    mask_pow = (flight_data['ground_tether_reelout_speed'] > 0) & (flight_data['up']<0.2)
    aoa_trim = np.mean(aoa_EKF[mask_pow])
    offset_aoa = aoa_trim - np.mean(flight_data['kite_angle_of_attack'][mask_pow])

    # Correct sideslip angle based on EKF mean sideslip angle of 0
    ss_vane = np.array(flight_data['kite_sideslip_angle'])
    ss_vane = np.convolve(ss_vane, np.ones(10)/10, mode='same')
    offset_ss = np.mean(results['ss'])-np.mean(flight_data['kite_sideslip_angle'][mask_pow])



    print('Offset aoa: ', offset_aoa, 'Offset ss: ', offset_ss)
    # Correct angle of attack and sideslip angle based on kite deployment
    aoa_vane = aoa_vane + offset_aoa
    # aoa_vane = aoa_vane + offset_dep*np.array(flight_data['up'])
    # aoa_vane = aoa_vane + offset_turn*np.array(flight_data['us'])
    flight_data['kite_angle_of_attack'] = aoa_vane
    ss_vane = ss_vane + offset_ss
    flight_data['kite_sideslip_angle'] = ss_vane

    return flight_data  