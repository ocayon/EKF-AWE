
import numpy as np
import pandas as pd

model = 'v3'
file_path = './'+model+'/'
file_name = '2019-10-08_11.csv'

model = 'v9'
file_path = './'+model+'/'
file_name = '2023-11-27_13-37-48_ProtoLogger_lidar.csv'
# file_name = '2023-11-16_12-47-08_ProtoLogger_lidar.csv'
# file_name = '2023-10-26_12-21-08_ProtoLogger_lidar.csv'
# file_name = '2023-06-21_11-48-37_ProtoLogger.csv'
# file_name = '2023-03-29_15-56-58_ProtoLogger.csv'
# file_name = '2021-10-07_19-38-15_ProtoLogger.csv'
# file_name = '2023-12-11_13-15-42_ProtoLogger_lidar.csv'
# file_name = '2024-01-09_12-28-51_ProtoLogger_lidar.csv'
file_name = '2024-06-05_11-33-16_ProtoLogger_lidar.csv'

# file_path = './'+model+'/v9.60.F-different-bridle/'
# file_name = '2024-03-26_15-00-03_ProtoLogger_lidar.csv'

# file_path = './'+model+'/more-depowered-reelin/'
# file_name = '2024-03-25_14-08-47_ProtoLogger_lidar.csv'
# # file_name = '2024-03-12_13-36-02_ProtoLogger_lidar.csv'
# file_path = './'+model+'/front_stall/'
# file_name = '2024-04-11_12-51-46_ProtoLogger_lidar.csv'


# Smooth radius
window_size = 20


#%%

df = pd.read_csv(file_path+file_name,delimiter = ' ')

#%%

df = df[df['kite_height'] > 80] #Select the indexes where the kite is flying

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

# Add GPS + IMU data Sensor 0 - Convert from NED to ENU
flight_data['kite_0_vx'] = df['kite_0_vy']
flight_data['kite_0_vy'] = df['kite_0_vx']
flight_data['kite_0_vz'] = -df['kite_0_vz']
flight_data['kite_0_pitch'] = df['kite_0_pitch']
flight_data['kite_0_roll'] = df['kite_0_roll']
flight_data['kite_0_yaw'] = df['kite_0_yaw']

# Add acceleration data Sensor 0 - Convert from NED to ENU and differentiate velocity if needed
window_size = 5
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

if 'v9' in file_path:
    radius_kcu = []
    radius_kite = []
    for i in range(len(flight_data)):
        row = flight_data.iloc[i]
        # Calculate KCU omega and turn radius from acceleration and velocity
        a_kcu = np.array([row['kite_1_ax'],row['kite_1_ay'],row['kite_1_az']]).T
        v_kcu = np.array([row['kite_1_vx'],row['kite_1_vy'],row['kite_1_vz']]).T
        at = np.dot(a_kcu,np.array(v_kcu)/np.linalg.norm(v_kcu))*np.array(v_kcu)/np.linalg.norm(v_kcu)
        omega_kcu = np.cross(a_kcu-at,v_kcu)/(np.linalg.norm(v_kcu)**2)
        ICR_kcu = np.cross(v_kcu,omega_kcu)/(np.linalg.norm(omega_kcu)**2)      
        alpha = np.cross(at,ICR_kcu)/np.linalg.norm(ICR_kcu)**2
        radius_kcu.append(np.linalg.norm(ICR_kcu))
        
        # Calculate kite omega and turn radius from acceleration and velocity
        a_kite = np.array([row['kite_0_ax'],row['kite_0_ay'],row['kite_0_az']]).T
        v_kite = np.array([row['kite_0_vx'],row['kite_0_vy'],row['kite_0_vz']]).T
        at = np.dot(a_kite,np.array(v_kite)/np.linalg.norm(v_kite))*np.array(v_kite)/np.linalg.norm(v_kite)
        omega_kite = np.cross(a_kite-at,v_kite)/(np.linalg.norm(v_kite)**2)
        ICR_kite = np.cross(v_kite,omega_kite)/(np.linalg.norm(omega_kite)**2)
        alpha = np.cross(at,ICR_kite)/np.linalg.norm(ICR_kite)**2
        radius_kite.append(np.linalg.norm(ICR_kite))

# # Smooth radius
# window_size = 10
# radius_kcu = np.convolve(radius_kcu, np.ones(window_size)/window_size, mode='same')
# radius_kite = np.convolve(radius_kite, np.ones(window_size)/window_size, mode='same')

#%% Find tether length offset
# Function to compute mean squared error for a given offset
def compute_mse(sig1, sig2, offset):
    shifted_sig2 = sig2 + offset
    mse = np.mean((sig1 - shifted_sig2) ** 2)
    return mse

up = (flight_data['kcu_actual_depower']-min(flight_data['kcu_actual_depower']))/(max(flight_data['kcu_actual_depower'])-min(flight_data['kcu_actual_depower']))
us = (flight_data['kcu_actual_steering'])/max(abs(flight_data['kcu_actual_steering']))
dep = (up>0.25)
pow = (flight_data['ground_tether_reelout_speed'] > 0) & (up<0.25)
trans = ~pow & ~dep
turn = pow & (flight_data['kite_0_vz']<0)
straight = pow & ~turn

# Tether length
r = np.sqrt(flight_data['kite_0_rx']**2+flight_data['kite_0_ry']**2+flight_data['kite_0_rz']**2)
# Try different offsets and find the one with the minimum MSE
offsets = np.linspace(-50,50,500)  # Adjust range and step size as needed
mse_values = [compute_mse( r[pow], flight_data['ground_tether_length'][pow],offset) for offset in offsets]
off_lt = offsets[np.argmin(mse_values)]


flight_data['ground_tether_length'] = flight_data['ground_tether_length']+off_lt

print('Offset tether length:',off_lt)

#%%
csv_filepath = '../processed_data/flight_data/'+model+'/'
csv_filename = model +'_'+date+'.csv'
flight_data.to_csv(csv_filepath+csv_filename, index=False)

