
import numpy as np
import pandas as pd
import numpy as np


def calculate_euler_from_reference_frame(dcm):
    
    # Calculate the roll, pitch and yaw angles from a direction cosine matrix, in NED coordinates   
    Teb = dcm
    pitch = np.arcsin(-Teb[2,0])
    roll = np.arctan(Teb[2,1]/Teb[2,2])
    yaw = np.arctan2(Teb[1,0],Teb[0,0])

    return roll, pitch, yaw

def Rx(theta):
    """Generate a rotation matrix for a rotation about the x-axis by `theta` radians."""
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def Ry(theta):
    """Generate a rotation matrix for a rotation about the y-axis by `theta` radians."""
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def Rz(theta):
    """Generate a rotation matrix for a rotation about the z-axis by `theta` radians."""
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

def R_EG_Body(roll, pitch, yaw):
    """Create the total rotation matrix from Earth-fixed to body reference frame in ENU coordinate system."""
    # Perform rotation about x-axis (roll), then y-axis (pitch), then z-axis (yaw)
    return Rz(yaw).dot(Ry(pitch).dot(Rx(roll)))

def calculate_reference_frame_euler(roll, pitch, yaw, bodyFrame='default'):
    """
    Calculate the Earth reference frame vectors based on Euler angles for a specified body frame.
    
    Parameters:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.
        bodyFrame (str): Type of body frame 
        'default' - Aircraft body frame (x-forward, y-right, z-down)
    
    Returns:
        tuple: Transformed unit vectors along the x, y, and z axes of the kite/body in Earth coordinates.
    """
    # Unit vectors in Earth frame
    if bodyFrame == 'default':
         # Calculate transformation matrix and its transpose
        Transform_Matrix = R_EG_Body(roll, pitch, yaw)
        ex_kite = Transform_Matrix[:,0]
        ey_kite = Transform_Matrix[:,1]
        ez_kite = Transform_Matrix[:,2]
    elif bodyFrame == 'kitekraft':
       Transform_Matrix = Rz(yaw).dot(Ry(pitch).dot(Rx(roll).dot(Ry(np.radians(90)))))
       ex_kite = Transform_Matrix[:,0]
       ey_kite = Transform_Matrix[:,1]
       ez_kite = Transform_Matrix[:,2]
    return ex_kite, ey_kite, ez_kite

file_path = './kitekraft/'


file_name = 'kitekraft_meridional.csv'

# Smooth radius
window_size = 20


#%%

df = pd.read_csv(file_path+file_name,delimiter = ',')

df = df.iloc[55000:57000]
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



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X_E')
ax.set_ylabel('Y_N')
ax.set_zlabel('Z_U')
spacing = 300
thrust_force_vector = []
for i in np.arange(0,len(time), spacing):
    len_arrow = 30
    # Calculate EKF tether orientation based on euler angles and plot it
    ex, ey, ez = calculate_reference_frame_euler( kite_euler_0[i], 
                                                 kite_euler_1[i], 
                                                 kite_euler_2[i],
                                                 bodyFrame='kitekraft')
    kite_pos[i] = Rz(azimuth_center[i])@kite_pos[i]
    thrust_force_vector.append(ex*thrust_force[i])
    ax.quiver(kite_pos[i][0], kite_pos[i][1],kite_pos[i][2], ex[0],  \
                ex[1], ex[2],
            color='green', length=len_arrow)
    ax.quiver(kite_pos_x[i], kite_pos_y[i], kite_pos_z[i], ey[0],  \
                ey[1], ey[2],
            color='blue', length=len_arrow)
    ax.quiver(kite_pos_x[i], kite_pos_y[i], kite_pos_z[i], ez[0],  \
                ez[1], ez[2],
            color='r', length=len_arrow)
    ax.scatter(kite_pos_x[i], kite_pos_y[i], kite_pos_z[i])
ax.plot(kite_pos_x, kite_pos_y, kite_pos_z,color='grey')