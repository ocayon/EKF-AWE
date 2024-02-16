import numpy as np
import casadi as ca
from config import kite_models,kcu_cylinders, tether_materials, kappa, z0, rho, g, n_tether_elements
from scipy.interpolate import splrep, splev
#%% Class definitions
class KiteModel:
    def __init__(self, model_name, mass, area, distance_kcu_kite, total_length_bridle_lines, diameter_bridle_lines, KCU):
        self.model_name = model_name
        self.mass = mass
        self.area = area
        self.distance_kcu_kite = distance_kcu_kite
        self.total_length_bridle_lines = total_length_bridle_lines
        self.diameter_bridle_lines = diameter_bridle_lines
        self.KCU = KCU
class KCUModel:
    # Exracted from Applied fluid dynamics handbook
    ldt_data = np.array([0,0.5,1.0,1.5,2.0,3.0,4.0,5.0])  # L/D values
    cdt_data = np.array([1.15,1.1,0.93,0.85,0.83,0.85,0.85,0.85])  # Cd values for tangential flow
    ldp_data = np.array([1,1.98,2.96,5,10,20,40,1e6])  # L/D values
    cdp_data = np.array([0.64,0.68,0.74,0.74,0.82,0.91,0.98,1.2])  # Cd values perpendicular flow

    cdt_cone_data= np.array([0.43, 0.35, 0.22, 0.19, 0.20, 0.21, 0.22, 0.24])
    ldt_cone = np.array([0.01, 0.64, 1.98, 4.57, 8.91, 12.02, 13.69, 15.29])
    # Create spline interpolations
    spline_t = splrep(ldt_data, cdt_data, s=0)
    spline_p = splrep(ldp_data, cdp_data, s=0)
    
    def __init__(self,length,diameter,mass):
        
        self.length = length
        self.diameter = diameter
        self.mass = mass

        # Example: Interpolate Cd for tangential flow at a specific L/D
        self.cdt = splev(self.length/self.diameter, KCUModel.spline_t)
        self.cdp = splev(self.length/self.diameter, KCUModel.spline_p)

        self.At = np.pi*(self.diameter/2)**2  # Calculate area of the KCU
        self.Ap = self.diameter*self.length  # Calculate area of the KCU
        
class EKFInput:
    def __init__(self, kite_pos,kite_vel,kite_acc,tether_force,apparent_windspeed = None, tether_length = None,kite_aoa = None, kcu_vel = None, kcu_acc = None):
        self.kite_pos = kite_pos
        self.kite_vel = kite_vel
        self.kite_acc = kite_acc
        self.apparent_windspeed = apparent_windspeed
        self.tether_length = tether_length
        self.tether_force = tether_force
        self.kite_aoa = kite_aoa
        self.kcu_acc = kcu_acc
        self.kcu_vel = kcu_vel

def get_measurement_vector(input_class, opt_measurements):
    z = np.array([])  # Initialize an empty NumPy array

    # Append values to the NumPy array
    z = np.append(z, input_class.kite_pos)
    z = np.append(z, input_class.kite_vel)
    if 'kite_acc' in opt_measurements:
        z = np.append(z, input_class.kite_acc)
    if 'apparent_windspeed' in opt_measurements:
        z = np.append(z, input_class.apparent_windspeed)
    if 'tether_length' in opt_measurements:
        z = np.append(z, input_class.tether_length)
    if 'kite_aoa' in opt_measurements:
        z = np.append(z, input_class.kite_aoa)

    return z

def tether_input(input_class, model_specs):
    if model_specs.correct_height:
        tether_length = input_class.tether_length
    else:
        tether_length = None
    
    if model_specs.kcu_data:
        kcu_acc = input_class.kcu_acc
        kcu_vel = input_class.kcu_vel
    else:
        kcu_acc = None
        kcu_vel = None

    return input_class.kite_acc, input_class.tether_force, tether_length, kcu_acc, kcu_vel
class ModelSpecs:
    def __init__(self,timestep, n_tether_elements, opt_measurements = [], correct_height = False,  kcu_data = False, doIEKF = True, epsilon = 1e-6, max_iterations = 100):
        self.ts = timestep
        self.n_tether_elements = n_tether_elements
        self.opt_measurements = opt_measurements
        self.correct_height = correct_height
        self.kcu_data = kcu_data
        self.doIEKF = doIEKF
        self.epsilon = epsilon
        self.max_iterations = max_iterations

class SystemSpecs:
    # Class to store the system specifications
    def __init__(self, kite_model, kcu_model, tether_material, tether_diameter, stdv_dynamic_model, stdv_measurements):
        self.kite_model = kite_model
        self.kcu_model = kcu_model
        self.tether_material = tether_material
        self.tether_diameter = tether_diameter
        self.stdv_dynamic_model = stdv_dynamic_model
        self.stdv_measurements = stdv_measurements



def find_initial_state_vector(kite_pos, kite_vel, kite_acc, ground_winddir, ground_windspeed, tether_force, tether_length, n_tether_elements, kite, kcu,tether):

    # Solve for the tether shape

    
    uf = ground_windspeed*kappa/np.log(10/z0)
    wvel0 = uf/kappa*np.log(kite_pos[2]/z0)
    if np.isnan(wvel0):
        raise ValueError('Initial wind velocity is NaN')
    vw = [wvel0*np.cos(ground_winddir),wvel0*np.sin(ground_winddir),0] # Initial wind velocity

    tether.solve_tether_shape(n_tether_elements, kite_pos, kite_vel, vw, kite, kcu, tension_ground = tether_force, tether_length = tether_length,
                                a_kite = kite_acc)
    x0 = np.vstack((kite_pos,kite_vel))
    x0 = np.append(x0,[uf,ground_winddir,tether.CL,tether.CD,tether.CS,0,0])     # Initial state vector (Last two elements are bias, used if needed)
    u0 = tether.Ft_kite

    return x0, u0

#%% Function definitions

def project_onto_plane(vector, plane_normal):
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

def calculate_angle_2vec(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = np.arccos(cos_theta)

    # Determine the sign of the angle
    cross_product = np.cross(vector_a, vector_b)
    if cross_product[2] < 0:
        angle_rad = -angle_rad

    return angle_rad

def calculate_angle_2vec_sym(vector_a, vector_b):
    dot_product = ca.dot(vector_a, vector_b)
    magnitude_a = ca.norm_2(vector_a)
    magnitude_b = ca.norm_2(vector_b)
    
    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = ca.arccos(cos_theta)
    
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

def R_EG_Body(Roll,Pitch,Yaw):#!!In radians!!
    
    #Rotational matrix for Roll
    R_Roll=np.array([[1, 0, 0],[0,np.cos(Roll),np.sin(Roll)],[0,-np.sin(Roll),np.cos(Roll)]])
    
    #Rotational matrix for Pitch
    R_Pitch=np.array([[np.cos(Pitch), 0, np.sin(Pitch)],[0,1,0],[-np.sin(Pitch), 0, np.cos(Pitch)]])

    #Rotational matrix for Roll
    R_Yaw= np.array([[np.cos(Yaw),-np.sin(Yaw),0],[np.sin(Yaw),np.cos(Yaw),0],[0,0,1]])
    
    #Total Rotational Matrix
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
    ex_kite = dcm[:,0]      # Kite x axis 
    ey_kite = dcm[:,1]      # Kite y axis perpendicular to va and tether
    ez_kite = dcm[:,2]      # Kite z axis pointing in the direction of the tension
    pitch = 90-calculate_angle(ex_kite, [0,0,1])          # Pitch angle
    x_se = project_onto_plane([0,0,1], ez_kite)
    # yaw = calculate_angle_2vec(x_se, ex_kite)
    yaw = np.arctan2(ex_kite[1],ex_kite[0])*180/np.pi   # Yaw angle       
    roll = 90-calculate_angle(ey_kite, [0,0,1])            # Roll angle

    return roll, pitch, yaw

def calculate_airflow_angles(dcm, v_kite, vw):
    ex_kite = dcm[:,0]      # Kite x axis
    ey_kite = dcm[:,1]      # Kite y axis perpendicular to v and tether
    ez_kite = dcm[:,2]      # Kite z axis pointing in the direction of the tension
    va = vw-v_kite
    va_proj = project_onto_plane(va, ey_kite)           # Projected apparent wind velocity onto kite y axis
    aoa = 90-calculate_angle(ez_kite,va_proj)             # Angle of attack
    va_proj = project_onto_plane(va, ez_kite)           # Projected apparent wind velocity onto kite z axis
    sideslip = 90-calculate_angle(ey_kite,va_proj)         # Sideslip angle
    return aoa, sideslip

