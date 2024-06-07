
import numpy as np
import pandas as pd
import numpy as np
from awes_ekf.utils import  Rx, Ry, Rz

file_path = './kitekraft/'


file_name = 'kitekraft_meridional.csv'

model = 'kitekraft'

day = '07'
month = '06'
year = '2024'

# Smooth radius
window_size = 20


#%%

df = pd.read_csv(file_path+file_name,delimiter = ',')

df = df.iloc[55000:61000]
df = df.reset_index() # Reset the index
#%%

time = df['time']
azimuth_center = df['positionKiteAzimuthCenterDemand']
kite_pos_x = df['positionKiteIdentified_0']
kite_pos_y = df['positionKiteIdentified_1']
kite_pos_z = df['positionKiteIdentified_2']

kite_pos = np.array([kite_pos_x,kite_pos_y,kite_pos_z]).T



kite_vel_x = df['velocityKiteIdentifiedByXsens_0']
kite_vel_y = df['velocityKiteIdentifiedByXsens_1']
kite_vel_z = df['velocityKiteIdentifiedByXsens_2']

kite_acc_x = df['accelerationMeasuredInHoverFrame_0']
kite_acc_y = df['accelerationMeasuredInHoverFrame_1']
kite_acc_z = df['accelerationMeasuredInHoverFrame_2']

kite_euler_0 = df['eulerAnglesKiteIdentified_0']
kite_euler_1 = df['eulerAnglesKiteIdentified_1']
kite_euler_2 = df['eulerAnglesKiteIdentified_2']

force_tether = df['forceTetherMeasured']
length_tether = df['lengthTetherRolledOutIdentified']

whinch_speed = df['speedWinchIdentified']
perch_speed = df['speedPerchIdentified']

thrust_force = df['thrustSet']
ground_wind_speed = df['velocityWindMagnitudeAtGroundMeasuredFused']
ground_wind_direction = df['velocityWindAzimuthAtGroundMeasuredFused']


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X_E')
ax.set_ylabel('Y_N')
ax.set_zlabel('Z_U')
spacing = 300
thrust_force_vector = []
for i in np.arange(0,len(time)):
    len_arrow = 30
    # Calculate EKF tether orientation based on euler angles and plot it
    R = np.dot(np.dot(np.dot(Rz(kite_euler_2[i]), Ry(kite_euler_1[i])), Rx(kite_euler_0[i])), Ry(np.deg2rad(90)))
    ex = R[:,0]
    ey = R[:,1]
    ez = R[:,2]

    thrust_force_vector.append(-ex*thrust_force[i])
#     ax.quiver(kite_pos[i][0], kite_pos[i][1],kite_pos[i][2], ex[0],  \
#                 ex[1], ex[2],
#             color='green', length=len_arrow)
#     ax.quiver(kite_pos_x[i], kite_pos_y[i], kite_pos_z[i], ey[0],  \
#                 ey[1], ey[2],
#             color='blue', length=len_arrow)
#     ax.quiver(kite_pos_x[i], kite_pos_y[i], kite_pos_z[i], ez[0],  \
#                 ez[1], ez[2],
#             color='r', length=len_arrow)
#     ax.scatter(kite_pos_x[i], kite_pos_y[i], kite_pos_z[i])
# ax.plot(kite_pos_x, kite_pos_y, kite_pos_z,color='grey')

thrust_force_vector = np.array(thrust_force_vector)
#%% Add the data to the flight data dataframe
flight_data = pd.DataFrame()
# Add position data
flight_data['kite_position_east'] = kite_pos_x
flight_data['kite_position_north'] = kite_pos_y
flight_data['kite_position_up'] = kite_pos_z

# Velocity data
flight_data['kite_velocity_east_s0'] = kite_vel_x
flight_data['kite_velocity_north_s0'] = kite_vel_y
flight_data['kite_velocity_up_s0'] = kite_vel_z

# Acceleration data
flight_data['kite_acceleration_east_s0'] = kite_acc_x
flight_data['kite_acceleration_north_s0'] = kite_acc_y
flight_data['kite_acceleration_up_s0'] = kite_acc_z

# Add the ground station data
flight_data['ground_tether_force'] = force_tether
flight_data['ground_tether_length'] = length_tether
flight_data['ground_tether_reelout_speed'] = whinch_speed
flight_data['ground_wind_velocity'] = ground_wind_speed
flight_data['ground_wind_direction'] = ground_wind_direction

# Add thrust data
flight_data['thrust_force_east'] = thrust_force_vector[:,0]
flight_data['thrust_force_north'] = thrust_force_vector[:,1]
flight_data['thrust_force_up'] = thrust_force_vector[:,2]

flight_data['time'] = df['time']

# Save the data
date = year+month+day
csv_filepath = '../processed_data/flight_data/'+model+'/'
csv_filename = f"{model}_{year}-{month}-{day}.csv"
flight_data.to_csv(csv_filepath+csv_filename,index=False)