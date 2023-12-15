
import numpy as np
import pandas as pd

model = 'v3'
file_path = './'+model+'/'
file_name = '2019-10-08_11.csv'

# file_path = './v9/'
# file_name = '2023-11-27_13-37-48_ProtoLogger_lidar.csv'
# file_name = '2023-11-16_12-47-08_ProtoLogger_lidar.csv'
# file_name = '2023-10-26_12-21-08_ProtoLogger_lidar.csv'


#%%

df = pd.read_csv(file_path+file_name,delimiter = ' ')



df = df[df['kite_height'] > 30] #Select the indexes where the kite is flying

dt = df['time'].iloc[1] - df['time'].iloc[0]  # Time step
df = df.reset_index() # Reset the index
df = df.interpolate()   # Interpolate the missing data

flight_data = pd.DataFrame()    # Create a new dataframe for the flight data


date = str(df['date'].iloc[0])

#%% Add the data to the flight data dataframe

# Add GPS data
flight_data['kite_0_rx'] = df['kite_pos_east']
flight_data['kite_0_ry'] = df['kite_pos_north']
flight_data['kite_0_rz'] = df['kite_height']

# Add GPS + IMU data Sensor 0
flight_data['kite_0_vx'] = df['kite_0_vy']
flight_data['kite_0_vy'] = df['kite_0_vx']
flight_data['kite_0_vz'] = -df['kite_0_vz']
flight_data['kite_0_pitch'] = df['kite_0_pitch']
flight_data['kite_0_roll'] = df['kite_0_roll']
flight_data['kite_0_yaw'] = df['kite_0_yaw']

window_size = 20
if df['kite_0_ax'].isnull().all():
    ax = np.diff(df['kite_0_vy']) / dt
    ay = np.diff(df['kite_0_vx']) / dt
    az = -np.diff(df['kite_0_vz']) / dt
    flight_data['kite_0_ax'] = np.concatenate((ax, [0]))
    flight_data['kite_0_ay'] = np.concatenate((ay, [0]))
    flight_data['kite_0_az'] = np.concatenate((az, [0]))
    flight_data['kite_0_ax']=np.convolve(flight_data['kite_0_ax'], np.ones(window_size)/window_size, mode='same')
    flight_data['kite_0_ay']=np.convolve(flight_data['kite_0_ay'], np.ones(window_size)/window_size, mode='same')
    flight_data['kite_0_az']=np.convolve(flight_data['kite_0_az'], np.ones(window_size)/window_size, mode='same')
else:
    flight_data['kite_0_ax'] = df['kite_0_ay']
    flight_data['kite_0_ay'] = df['kite_0_ax']
    flight_data['kite_0_az'] = -df['kite_0_az']

# Add GPS + IMU data Sensor 1
flight_data['kite_1_vx'] = df['kite_1_vy']
flight_data['kite_1_vy'] = df['kite_1_vx']
flight_data['kite_1_vz'] = -df['kite_1_vz']
flight_data['kite_1_pitch'] = df['kite_1_pitch']
flight_data['kite_1_roll'] = df['kite_1_roll']
flight_data['kite_1_yaw'] = df['kite_1_yaw']
flight_data['kite_1_ax'] = df['kite_1_ay']
flight_data['kite_1_ay'] = df['kite_1_ax']
flight_data['kite_1_az'] = -df['kite_1_az']
flight_data['kite_1_yaw_rate'] = df['kite_1_yaw_rate']
flight_data['kite_1_pitch_rate'] = df['kite_1_pitch_rate']
flight_data['kite_1_roll_rate'] = df['kite_1_roll_rate']
if df['kite_1_ax'].isnull().all():
    ax = np.diff(df['kite_1_vy']) / dt
    ay = np.diff(df['kite_1_vx']) / dt
    az = -np.diff(df['kite_1_vz']) / dt
    flight_data['kite_1_ax'] = np.concatenate((ax, [0]))
    flight_data['kite_1_ay'] = np.concatenate((ay, [0]))
    flight_data['kite_1_az'] = np.concatenate((az, [0]))
    flight_data['kite_1_ax']=np.convolve(flight_data['kite_1_ax'], np.ones(window_size)/window_size, mode='same')
    flight_data['kite_1_ay']=np.convolve(flight_data['kite_1_ay'], np.ones(window_size)/window_size, mode='same')
    flight_data['kite_1_az']=np.convolve(flight_data['kite_1_az'], np.ones(window_size)/window_size, mode='same')
else:   
    flight_data['kite_1_ax'] = df['kite_1_ay']
    flight_data['kite_1_ay'] = df['kite_1_ax']
    flight_data['kite_1_az'] = -df['kite_1_az']
    flight_data['kite_1_ax']=np.convolve(flight_data['kite_1_ax'], np.ones(window_size)/window_size, mode='same')
    flight_data['kite_1_ay']=np.convolve(flight_data['kite_1_ay'], np.ones(window_size)/window_size, mode='same')
    flight_data['kite_1_az']=np.convolve(flight_data['kite_1_az'], np.ones(window_size)/window_size, mode='same')




# Add the ground station data
flight_data['ground_tether_force'] = df['ground_tether_force']*9.81
flight_data['ground_wind_velocity'] = df['ground_wind_velocity']
flight_data['ground_wind_direction'] = 360-90-df['ground_upwind_direction'] 
flight_data['ground_tether_length'] = df['ground_tether_length']
flight_data['ground_tether_reelout_speed'] = df['ground_tether_reelout_speed']

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

if 'lidar' in file_name:

    for column in df.columns:
        if 'Wind Direction' in column:
            # Use regular expression to extract a 3-digit number from the column name
            height = ''.join(filter(str.isdigit, column))
            
            flight_data[column] = df[column]
        if 'Wind' in column:
            
            flight_data[column] = df[column]
            
        if 'Z-wind' in column:
            
            flight_data[column] = df[column]



#%%
csv_filepath = '../processed_data/flight_data/'+model+'/'
csv_filename = model +'_'+date+'.csv'
flight_data.to_csv(csv_filepath+csv_filename, index=False)

