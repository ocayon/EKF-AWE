import numpy as np
import casadi as ca
from config import kappa, z0
import pandas as pd


def get_measurement_vector(input_class, model_specs):
    opt_measurements = model_specs.opt_measurements
    z = np.array([])  # Initialize an empty NumPy array

    # Append values to the NumPy array
    z = np.append(z, input_class.kite_pos)
    z = np.append(z, input_class.kite_vel)
    z = np.append(z,input_class.tether_length)
    z = np.append(z,input_class.elevation)
    z = np.append(z,input_class.azimuth)
    z = np.append(z, np.zeros(3))  # Add zeros for the least-squares problem
    if model_specs.model_yaw:
        z = np.append(z,input_class.kite_yaw)
    if model_specs.enforce_z_wind:
        z = np.append(z,0)
    if 'apparent_windspeed' in opt_measurements:
        z = np.append(z, input_class.apparent_windspeed)

    return z

class SimulationConfig:
    def __init__(self,timestep, n_tether_elements, opt_measurements = [],  
                 kcu_data = False, doIEKF = True, epsilon = 1e-6, max_iterations = 200,
                 log_profile = False, tether_offset = True, enforce_z_wind = True,
                 model_yaw = False):
        self.ts = timestep
        self.n_tether_elements = n_tether_elements
        self.opt_measurements = opt_measurements
        self.kcu_data = kcu_data
        self.doIEKF = doIEKF
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.log_profile = log_profile
        self.tether_offset = tether_offset
        self.enforce_z_wind = enforce_z_wind
        self.model_yaw = model_yaw


class SystemParameters:
    # Class to store the system specifications
    def __init__(self, kite_model, kcu_model, tether_material, tether_diameter, meas_stdv, model_stdv, model_specs):
        self.kite_model = kite_model
        self.kcu_model = kcu_model
        self.tether_material = tether_material
        self.tether_diameter = tether_diameter

        model_stdv = model_stdv[kite_model]
        if model_specs.log_profile is True:
            self.stdv_dynamic_model = np.array([model_stdv['x'], model_stdv['x'], model_stdv['x'], 
                       model_stdv['v'], model_stdv['v'], model_stdv['v'], 
                       model_stdv['uf'], model_stdv['wdir'], model_stdv['vwz'],
                       model_stdv['CL'], model_stdv['CD'], model_stdv['CS'],
                       model_stdv['tether_length'], model_stdv['elevation'], model_stdv['azimuth']])  # Standard deviations for the dynamic model
        else:
            self.stdv_dynamic_model = np.array([model_stdv['x'], model_stdv['x'], model_stdv['x'], 
                       model_stdv['v'], model_stdv['v'], model_stdv['v'], 
                       model_stdv['vw'], model_stdv['vw'], model_stdv['vwz'],
                       model_stdv['CL'], model_stdv['CD'], model_stdv['CS'],
                       model_stdv['tether_length'], model_stdv['elevation'], model_stdv['azimuth']])  # Standard deviations for the dynamic model
        if model_specs.model_yaw:
            self.stdv_dynamic_model = np.append(self.stdv_dynamic_model,[model_stdv['yaw'],1e-6])
        
        stdv_y = []
        for _ in range(3):
            stdv_y.append(meas_stdv['x'])
        for _ in range(3):
            stdv_y.append(meas_stdv['v'])
        stdv_y.append(meas_stdv['tether_length'])
        stdv_y.append(meas_stdv['elevation'])
        stdv_y.append(meas_stdv['azimuth'])
        for _ in range(3):
            stdv_y.append(meas_stdv['least_squares']) 
        if model_specs.model_yaw:
            stdv_y.append(meas_stdv['yaw'])
        if model_specs.enforce_z_wind:
            stdv_y.append(meas_stdv['z_wind'])
            
        for key in model_specs.opt_measurements:
            if key == 'apparent_windspeed':
                stdv_y.append(meas_stdv['va'])


        stdv_y = np.array(stdv_y)
        self.stdv_measurements = stdv_y



def find_initial_state_vector(kite_pos, kite_vel, kite_acc, ground_winddir, ground_windspeed, tether_force, tether_length, n_tether_elements,
                              elevation, azimuth, kite, kcu,tether, model_specs):

    # Solve for the tether shape
    uf = ground_windspeed*kappa/np.log(10/z0)
    wvel0 = uf/kappa*np.log(kite_pos[2]/z0)
    if np.isnan(wvel0):
        raise ValueError('Initial wind velocity is NaN')
    vw = [wvel0*np.cos(ground_winddir),wvel0*np.sin(ground_winddir),0] # Initial wind velocity

    tether.solve_tether_shape(n_tether_elements, kite_pos, kite_vel, vw, kite, kcu, tension_ground = tether_force, tether_length = tether_length,
                                a_kite = kite_acc)
    x0 = np.vstack((kite_pos,kite_vel))
    
    if model_specs.log_profile:
        x0 = np.append(x0,[uf,ground_winddir,0])  # Initial wind velocity and direction
    else:
        x0 = np.append(x0,vw)   # Initial wind velocity
    x0 = np.append(x0,[tether.CL,tether.CD,tether.CS,tether_length, elevation, azimuth])     # Initial state vector (Last two elements are bias, used if needed)
    u0 = tether.Ft_kite

    return x0, u0

#%% Function definitions

def project_onto_plane(vector, plane_normal):
    if type(vector) == ca.SX:
        return vector - ca.dot(vector, plane_normal) * plane_normal
    
    return vector - np.dot(vector, plane_normal) * plane_normal


def project_onto_plane_sym(vector, plane_normal):
    return vector - ca.dot(vector, plane_normal) * plane_normal

def rotate_vector(v, u, theta):
    # Normalize vectors
    v = v 
    u = u / ca.norm_2(u)

    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)

    v_rot = v * cos_theta + ca.cross(u, v) * sin_theta + u * ca.dot(u, v) * (1 - cos_theta)

    return v_rot    

def calculate_angle(vector_a, vector_b, deg=True):
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = np.arccos(cos_theta)

    # # Determine the sign of the angle
    # cross_product = np.cross(vector_a, vector_b)
    # if cross_product[2] < 0:
    #     angle_rad = -angle_rad

    angle_deg = np.degrees(angle_rad)

    if deg:
        return angle_deg
    else:
        return angle_rad

def calculate_angle_2vec(vector_a, vector_b, reference_vector=None):
    
    if type(vector_a) == ca.SX:
        dot_product = ca.dot(vector_a, vector_b)
        magnitude_a = ca.norm_2(vector_a)
        magnitude_b = ca.norm_2(vector_b)
        
        cos_theta = dot_product / (magnitude_a * magnitude_b)
        angle_rad = ca.arccos(cos_theta)
        
        return angle_rad
    
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # Determine the sign of the angle
    if reference_vector is not None:
        reference_cross = np.cross(reference_vector, vector_a)
        if np.dot(reference_cross, vector_b) < 0:
            angle_rad = -angle_rad

    return angle_rad


def rank_observability_matrix(A,C):
    # Construct the observability matrix O_numeric
    n = A.shape[1]  # Number of state variables
    m = C.shape[0]  # Number of measurements
    O = np.zeros((m * n, n))
    
    for i in range(n):
        power_of_A = np.linalg.matrix_power(A, i)
        O[i * m: (i + 1) * m, :] = C @ power_of_A

    # Compute the rank of O using NumPy
    rank_O = np.linalg.matrix_rank(O)
    return rank_O

def Rx(theta):
  return np.array([[ 1, 0           , 0           ],
                   [ 0, np.cos(theta),-np.sin(theta)],
                   [ 0, np.sin(theta), np.cos(theta)]])
  
def Ry(theta):
  return np.array([[ np.cos(theta), 0, np.sin(theta)],
                   [ 0           , 1, 0           ],
                   [-np.sin(theta), 0, np.cos(theta)]])
  
def Rz(theta):
  return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                   [ np.sin(theta), np.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])

def R_EG_Body(roll, pitch, yaw):  # !!In radians!!
    
    # Rotation matrix from earth fixed to body reference frame (ENU)
    # Roll 0 flying parallel to ground (yaxis parallel to ground)
    # Pitch 0 x-axis parallel to ground
    # Yaw 0 in east direction
    
        
    # Rotational matrix for Roll
    R_Roll = Rx(roll)

    # Rotational matrix for Pitch
    R_Pitch = Ry(pitch)

    # Rotational matrix for Yaw
    R_Yaw = Rz(yaw)

    # Total Rotational Matrix
    return R_Roll.dot(R_Pitch.dot(R_Yaw))


def calculate_polar_coordinates(r):
    # Calculate azimuth and elevation angles from a vector.
    r_mod = np.linalg.norm(r)
    az = np.arctan2(r[1], r[0])
    el = np.arcsin(r[2]/r_mod)
    return el, az, r_mod

def calculate_vw_loglaw(uf, z0, z, wdir, kappa = 0.4,vz = 0):
    wvel = uf/kappa*np.log(z/z0)
    vw = np.array([wvel*np.cos(wdir),wvel*np.sin(wdir),vz])
    return vw

def calculate_euler_from_reference_frame(dcm):
    
    # Calculate the roll, pitch and yaw angles from a direction cosine matrix, in NED coordinates   
    Teb = dcm
    pitch = np.arcsin(Teb[2,0])*180/np.pi
    roll = np.arctan(Teb[2,1]/Teb[2,2])*180/np.pi
    yaw = -np.arctan2(Teb[1,0],Teb[0,0])*180/np.pi

    return roll, pitch, yaw

def calculate_airflow_angles(dcm, v_kite, vw):
    ey_kite = dcm[:,1]      # Kite y axis perpendicular to v and tether
    ez_kite = dcm[:,2]      # Kite z axis pointing in the direction of the tension
    va = vw-v_kite
    va_proj = project_onto_plane(va, ey_kite)           # Projected apparent wind velocity onto kite y axis
    aoa = calculate_angle(ez_kite,va_proj)-90             # Angle of attack
    va_proj = project_onto_plane(va, ez_kite)           # Projected apparent wind velocity onto kite z axis
    sideslip = 90-calculate_angle(ey_kite,va_proj)         # Sideslip angle
    return aoa, sideslip


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