
import numpy as np
import pandas as pd
def process_KP_data(df):
    #%%
    
    # Smooth radius
    window_size = 20
    dt = df['time'].iloc[1] - df['time'].iloc[0]  # Time step
    df = df.reset_index() # Reset the index
    df = df.interpolate()   # Interpolate the missing data

    flight_data = pd.DataFrame()    # Create a new dataframe for the flight data


    
    sensors = [0,1]
    #%% Add the data to the flight data dataframe
    # Add position data
    flight_data['kite_position_east'] = df['kite_pos_east']
    flight_data['kite_position_north'] = df['kite_pos_north']
    flight_data['kite_position_up'] = df['kite_height']

    for sensor in sensors:
        # Velocity data
        flight_data['kite_velocity_east_s'+str(sensor)] = df['kite_'+str(sensor)+'_vy']
        flight_data['kite_velocity_north_s'+str(sensor)] = df['kite_'+str(sensor)+'_vx']
        flight_data['kite_velocity_up_s'+str(sensor)] = -df['kite_'+str(sensor)+'_vz'] 
        # Euler angles data
        flight_data['kite_pitch_s'+str(sensor)] = np.deg2rad(df['kite_'+str(sensor)+'_pitch'])
        flight_data['kite_roll_s'+str(sensor)] = np.deg2rad(df['kite_'+str(sensor)+'_roll'])
        flight_data['kite_yaw_s'+str(sensor)] = np.deg2rad(df['kite_'+str(sensor)+'_yaw'])
        # Acceleration data
        if df['kite_'+str(sensor)+'_ax'].isnull().all():
            # Differentiate velocity to get acceleration
            ax = np.diff(flight_data['kite_velocity_east_s'+str(sensor)]) / dt
            ay = np.diff(flight_data['kite_velocity_north_s'+str(sensor)]) / dt
            az = np.diff(flight_data['kite_velocity_up_s'+str(sensor)]) / dt
            # Add the last value as 0 to keep the same length
            flight_data['kite_acceleration_east_s'+str(sensor)] = np.concatenate((ax, [0]))
            flight_data['kite_acceleration_north_s'+str(sensor)] = np.concatenate((ay, [0]))
            flight_data['kite_acceleration_up_s'+str(sensor)] = np.concatenate((az, [0]))
            # Smooth the acceleration data
            flight_data['kite_acceleration_east_s'+str(sensor)]=np.convolve(flight_data['kite_acceleration_east_s'+str(sensor)], np.ones(window_size)/window_size, mode='same')
            flight_data['kite_acceleration_north_s'+str(sensor)]=np.convolve(flight_data['kite_acceleration_north_s'+str(sensor)], np.ones(window_size)/window_size, mode='same')
            flight_data['kite_acceleration_up_s'+str(sensor)]=np.convolve(flight_data['kite_acceleration_up_s'+str(sensor)], np.ones(window_size)/window_size, mode='same')
        else:
            flight_data['kite_acceleration_east_s'+str(sensor)] = df['kite_0_ay']
            flight_data['kite_acceleration_north_s'+str(sensor)] = df['kite_0_ax']
            flight_data['kite_acceleration_up_s'+str(sensor)] = -df['kite_0_az']

        # Add angular velocity data
        if df['kite_'+str(sensor)+'_yaw_rate'].isnull().all():
            # Differentiate orientation to get angular velocity
            roll_rate = np.diff(flight_data['kite_roll_s'+str(sensor)]) / dt
            pitch_rate = np.diff(flight_data['kite_pitch_s'+str(sensor)]) / dt
            yaw_rate = np.diff(flight_data['kite_yaw_s'+str(sensor)]) / dt
            # Add the last value as 0 to keep the same length
            flight_data['kite_roll_rate_s'+str(sensor)] = np.concatenate((roll_rate, [0]))
            flight_data['kite_pitch_rate_s'+str(sensor)] = np.concatenate((pitch_rate, [0]))
            flight_data['kite_yaw_rate_s'+str(sensor)] = np.concatenate((yaw_rate, [0]))
            # Smooth the yaw rate data
            flight_data['kite_roll_rate_s'+str(sensor)]=np.convolve(flight_data['kite_roll_rate_s'+str(sensor)], np.ones(window_size)/window_size, mode='same')
            flight_data['kite_pitch_rate_s'+str(sensor)]=np.convolve(flight_data['kite_pitch_rate_s'+str(sensor)], np.ones(window_size)/window_size, mode='same')
            flight_data['kite_yaw_rate_s'+str(sensor)]=np.convolve(flight_data['kite_yaw_rate_s'+str(sensor)], np.ones(window_size)/window_size, mode='same')

    # Add the ground station data
    flight_data['ground_tether_force'] = df['ground_tether_force']*9.81 # Convert to N
    flight_data['ground_wind_velocity'] = df['ground_wind_velocity']    
    flight_data['ground_wind_direction'] = 360-90-df['ground_upwind_direction']     # Convert from NED clockwise to ENU counter-clockwise
    flight_data['ground_tether_length'] = df['ground_tether_length']            # Tether length
    flight_data['ground_tether_reelout_speed'] = df['ground_tether_reelout_speed']      # Tether reelout speed

    # Add the KCU data
    flight_data['kcu_set_depower'] = df['kite_set_depower']
    flight_data['kcu_set_steering'] = df['kite_set_steering']
    flight_data['kcu_actual_steering'] = df['kite_actual_steering']
    flight_data['kcu_actual_depower'] = df['kite_actual_depower']

    # Add the airspeed data
    flight_data['kite_apparent_windspeed'] = df['airspeed_apparent_windspeed']
    flight_data['kite_angle_of_attack'] = df['airspeed_angle_of_attack']
    if 'v9' in file_path: 
        flight_data['kite_sideslip_angle'] = df['airspeed_sideslip_angle']
    flight_data['kite_airspeed_temperature'] = df['airspeed_temperature']

    kite_radius = np.linalg.norm(flight_data[['kite_position_east','kite_position_north','kite_position_up']],axis=1)
    offset_tether_length = np.mean(kite_radius-flight_data['ground_tether_length'])
    flight_data['ground_tether_length'] = flight_data['ground_tether_length'] + offset_tether_length
    print('Date: ',df['date'].iloc[0], 'Flight length: ',round(len(flight_data)/10/60, 1), 'min')
    print('Offset tether length: ',offset_tether_length)
    # Add the time data
    flight_data['time'] = df['time'] - df['time'].iloc[0]
    flight_data['time_of_day'] = df['time_of_day']
    flight_data['unix_time'] = df['time']
        
    # Add heading and course
    flight_data['kite_course'] = df['kite_course']
    flight_data['kite_heading'] = df['kite_heading']

    # Add azimuth and elevation
    flight_data['kite_azimuth'] = df['kite_azimuth']
    flight_data['kite_elevation'] = df['kite_elevation']

    # Add time of day
    flight_data['time_of_day'] = df['time_of_day']

    
    columns_to_copy = []
    flight_data_add = False
    for column in df.columns:
        if 'Wind ' in column:
            columns_to_copy.append(column)
        if 'Z-wind' in column:
            columns_to_copy.append(column)
        if 'load_cell' in column:
            columns_to_copy.append(column)
    if len(columns_to_copy) > 0:
        flight_data_add = pd.concat([df[col] for col in columns_to_copy], axis=1)
        flight_data = pd.concat([flight_data, flight_data_add], axis=1)

    return flight_data

# model = 'v3'
# file_path = './data/'+model+'/'
# file_name = '2019-10-08_11.csv'

file_name = []
model = 'v9'
file_path = './data/'+model+'/'

file_name.append('2023-11-27_13-37-48_ProtoLogger_lidar.csv')
file_name.append('2023-11-16_12-47-08_ProtoLogger_lidar.csv')
file_name.append('2023-10-26_12-21-08_ProtoLogger_lidar.csv')
file_name.append('2023-06-21_11-48-37_ProtoLogger.csv')
file_name.append('2023-03-29_15-56-58_ProtoLogger.csv')
file_name.append('2021-10-07_19-38-15_ProtoLogger.csv')
file_name.append('2023-12-11_13-15-42_ProtoLogger_lidar.csv')
file_name.append('2024-01-09_12-28-51_ProtoLogger_lidar.csv')
file_name.append('2024-06-05_11-33-16_ProtoLogger_lidar.csv')

# file_name.append('tether_force_kite/2021-06-03_10-19-52_ProtoLogger.csv')



# file_path = './'+model+'/v9.60.F-different-bridle/'
# file_name.append('2024-03-26_15-00-03_ProtoLogger_lidar.csv')

# file_path = './'+model+'/more-depowered-reelin/'
# file_name.append('2024-03-25_14-08-47_ProtoLogger_lidar.csv')
# file_name.append('2024-03-12_13-36-02_ProtoLogger_lidar.csv')
# file_path = './'+model+'/front_stall/'
# file_name.append('2024-04-11_12-51-46_ProtoLogger_lidar.csv')
# file_path = './'+model+'/more-depowered-reelin/'
# # file_name.append('2024-03-25_14-08-47_ProtoLogger_lidar.csv)'
# file_name.append('2024-03-12_13-36-02_ProtoLogger_lidar.csv')
# file_path = './data/'+model+'/front_stall/'
# # file_name.append('2024-04-11_12-51-46_ProtoLogger_lidar.csv')
# file_name.append('2024-05-01_11-04-02_ProtoLogger_lidar.csv')



for file in file_name:
    #%%
    df = pd.read_csv(file_path+file,delimiter = ' ',low_memory=False)
    df = df[df['kite_height'] >80] #Select the indexes where the kite is flying

    flight_data = process_KP_data(df)
    date = str(df['date'].iloc[0])
    #%%
    csv_filepath = './processed_data/flight_data/'+model+'/'
    csv_filename = model +'_'+date+'.csv'
    flight_data.to_csv(csv_filepath+csv_filename, index=False)

