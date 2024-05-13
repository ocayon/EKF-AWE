import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import kappa, z0, kite_model, year, month, day
from utils import get_tether_end_position,  R_EG_Body, calculate_angle,project_onto_plane, create_kite
import seaborn as sns
from scipy.signal import correlate

#%%
plt.close('all')


path = '../results/'+kite_model+'/'
file_name = kite_model+'_'+year+'-'+month+'-'+day
date = year+'-'+month+'-'+day

results = pd.read_csv(path+file_name+'_res_GPS.csv')
flight_data = pd.read_csv(path+file_name+'_fd.csv')

offset_df = pd.read_csv('../processed_data/IMU_offset.csv')

meas_pitch = flight_data['kite_0_pitch']
meas_pitch1 = flight_data['kite_1_pitch']
meas_roll = flight_data['kite_0_roll']
meas_roll1 = flight_data['kite_1_roll']
meas_yaw = flight_data['kite_0_yaw']
meas_yaw1 = flight_data['kite_1_yaw']
# Convert to NED from ENU
roll = -results.roll
pitch = -results.pitch
yaw = -results.yaw-180

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

# Unwrap the phase
unwrapped_angles = np.unwrap(np.radians(meas_yaw))
meas_yaw = np.degrees(unwrapped_angles)
# meas_yaw = normalize_angles(meas_yaw)


# Unwrap the phase
unwrapped_angles = np.unwrap(np.radians(meas_yaw1))
meas_yaw1 = np.degrees(unwrapped_angles)
# meas_yaw1 = normalize_angles(meas_yaw1)

# Unwrap the phase
unwrapped_angles = np.unwrap(np.radians(yaw))
yaw = np.degrees(unwrapped_angles)
# yaw = normalize_angles(yaw)

# Unwrap the phase
unwrapped_angles = np.unwrap(np.radians(meas_roll1))
meas_roll1 = np.degrees(unwrapped_angles)


meas_ax = flight_data.kite_0_ax
meas_ay = flight_data.kite_0_ay
meas_az = flight_data.kite_0_az
acc = np.vstack((np.array(meas_ax),np.array(meas_ay),np.array(meas_az))).T





#%% Find offsets
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
turn = pow & (results.vz<0)
straight = pow & ~turn

# Tether length
r = np.sqrt(flight_data['kite_0_rx']**2+flight_data['kite_0_ry']**2+flight_data['kite_0_rz']**2)
# Try different offsets and find the one with the minimum MSE
offsets = np.linspace(-40,40,500)  # Adjust range and step size as needed
mse_values = [compute_mse( r[pow], flight_data['ground_tether_length'][pow],offset) for offset in offsets]
tether_length = offsets[np.argmin(mse_values)]


offsets = np.linspace(-360, 360, 360*4)
# Roll
mask = pow & (np.linalg.norm(acc,axis = 1)<9.81)
mse_values = [compute_mse( roll[mask],meas_roll[mask],offset) for offset in offsets]
roll0 = offsets[np.argmin(mse_values)]


# mask = (np.linalg.norm(acc,axis = 1)<9.81)
mse_values = [compute_mse( roll[mask],meas_roll1[mask],offset) for offset in offsets]

roll1 = offsets[np.argmin(mse_values)]

# Pitch
mask = pow & (np.linalg.norm(acc,axis = 1)<9.81)
mse_values = [compute_mse( pitch[mask],meas_pitch[mask],offset) for offset in offsets]
pitch0 = offsets[np.argmin(mse_values)]


mask = (np.linalg.norm(acc,axis = 1)<9.81)
mse_values = [compute_mse( pitch[mask],meas_pitch1[mask],offset) for offset in offsets]
pitch1 = offsets[np.argmin(mse_values)]


# Yaw
mask = pow & straight
mse_values = [compute_mse( yaw[mask],meas_yaw[mask],offset) for offset in offsets]
yaw0 = offsets[np.argmin(mse_values)]


# mask = (np.linalg.norm(acc,axis = 1)<9.81)
mse_values = [compute_mse( yaw[mask],meas_yaw1[mask],offset) for offset in offsets] 

yaw1 = offsets[np.argmin(mse_values)]

# Values to add as a new row
values_to_add = {'date': date, 'roll0': roll0, 'roll1': roll1, 'pitch0': pitch0, 'pitch1': pitch1, 'yaw0': yaw0, 'yaw1': yaw1}
for i in range(len(offset_df)):
    if offset_df['date'].iloc[i] == date:
        offset_df = offset_df.drop(i)
        break
offset_df = pd.concat([offset_df, pd.DataFrame([values_to_add])], ignore_index=True)
print(offset_df)
# Save the new offset values
offset_df.to_csv('../processed_data/IMU_offset.csv', index=False)


#%% Plot euler angles

t = flight_data['time']



# Plot pitch
plt.figure()
plt.plot(t, pitch, label='Pitch EKF respect to v_kite')
plt.plot(t, meas_pitch+pitch0, label='Pitch kite sensor')
plt.plot(t, meas_pitch1+pitch1, label='Pitch KCU sensor')


plt.xlabel('Time')
plt.ylabel('Pitch [deg]')
plt.grid()
plt.legend()

plt.figure()
plt.plot(t, roll, label='Roll EKF respect to v_kite')
plt.plot(t, meas_roll+roll0, label='Roll kite sensor')
plt.plot(t, meas_roll1+roll1, label='Roll KCU sensor')

plt.xlabel('Time')
plt.ylabel('Roll [deg]')
plt.grid()
plt.legend()

# Plot yaw
plt.figure()
plt.plot(t, yaw, label='Yaw EKF respect to v_kite')
plt.plot(t, meas_yaw+yaw0, label='Yaw kite sensor')
plt.plot(t, meas_yaw1+yaw1, label='Yaw KCU sensor')

plt.xlabel('Time')
plt.ylabel('Yaw [deg]')
plt.grid()
plt.legend()

