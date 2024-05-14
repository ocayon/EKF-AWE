import numpy as np
import casadi as ca
from setup.settings import kappa, z0
import pandas as pd

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



def calculate_airflow_angles(dcm, v_kite, vw):
    ey_kite = dcm[:,1]      # Kite y axis perpendicular to v and tether
    ez_kite = dcm[:,2]      # Kite z axis pointing in the direction of the tension
    va = vw-v_kite
    va_proj = project_onto_plane(va, ey_kite)           # Projected apparent wind velocity onto kite y axis
    aoa = calculate_angle(ez_kite,va_proj)-90             # Angle of attack
    va_proj = project_onto_plane(va, ez_kite)           # Projected apparent wind velocity onto kite z axis
    sideslip = 90-calculate_angle(ey_kite,va_proj)         # Sideslip angle
    return aoa, sideslip

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

def calculate_reference_frame_euler(roll, pitch, yaw, bodyFrame='NED'):
    """
    Calculate the Earth reference frame vectors based on Euler angles for a specified body frame.
    
    Parameters:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.
        bodyFrame (str): Type of body frame ('NED' or 'ENU').
    
    Returns:
        tuple: Transformed unit vectors along the x, y, and z axes of the kite/body in Earth coordinates.
    """
    # Adjust roll for different coordinate systems
    # if bodyFrame == 'NED':
    #     roll = np.radians(np.degrees(-roll + np.pi))  # Convert to degrees, adjust, convert back to radians
    # if bodyFrame == 'ENU':
    #     roll = -roll
    
    
    # Calculate transformation matrix and its transpose
    Transform_Matrix = R_EG_Body(roll, pitch, yaw)
    
    # Unit vectors in Earth frame
    if bodyFrame == 'NED':
        ex_kite = Transform_Matrix[:,0]
        ey_kite = Transform_Matrix[:,1]
        ez_kite = Transform_Matrix[:,2]
    elif bodyFrame == 'ENU':
        ex_kite = -Transform_Matrix[:,0]
        ey_kite = Transform_Matrix[:,1]
        ez_kite = -Transform_Matrix[:,2]
    
    

    return ex_kite, ey_kite, ez_kite