import numpy as np
import casadi as ca
from config import kite_models,kcu_cylinders, tether_materials, kappa, z0, rho, g, n_tether_elements
from scipy.interpolate import splrep, splev
from scipy.optimize import least_squares
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

#%% Function definitions
def create_kite(model_name):
    if model_name in kite_models:
        model_params = kite_models[model_name]
        return KiteModel(model_name, model_params["mass"], model_params["area"], model_params["distance_kcu_kite"],
                     model_params["total_length_bridle_lines"], model_params["diameter_bridle_lines"],model_params['KCU'])
    else:
        raise ValueError("Invalid kite model")
    
def create_kcu(model_name):
    if model_name in kcu_cylinders:
        model_params = kcu_cylinders[model_name]
        return KCUModel(model_params["length"], model_params["diameter"], model_params["mass"])
    else:
        raise ValueError("Invalid KCU model")

def project_onto_plane(vector, plane_normal):
    return vector - np.dot(vector, plane_normal) * plane_normal


def project_onto_plane_sym(vector, plane_normal):
    return vector - ca.dot(vector, plane_normal) * plane_normal


def get_measurements(df, measurements,multiple_GPS = True):
    meas_dict = {}
    Z = []
    for meas in measurements:
        if meas == 'GPS_pos':
            col_rx = [col for col in df.columns if 'rx' in col]
            col_ry = [col for col in df.columns if 'ry' in col]
            col_rz = [col for col in df.columns if 'rz' in col]
            for i in range(len(col_rx)):
                Z.append(df[col_rx[i]].values)
                Z.append(df[col_ry[i]].values)
                Z.append(df[col_rz[i]].values)
            meas_dict[meas] = len(col_rx)

        elif meas == 'GPS_vel':
            col_vx = [col for col in df.columns if 'vx' in col]
            col_vy = [col for col in df.columns if 'vy' in col]
            col_vz = [col for col in df.columns if 'vz' in col]
            if multiple_GPS:
                for i in range(len(col_vx)):
                    Z.append(df[col_vx[i]].values)
                    Z.append(df[col_vy[i]].values)
                    Z.append(df[col_vz[i]].values)
                meas_dict[meas] = len(col_vx)
            else:
                Z.append(df['kite_0_vx'].values)
                Z.append(df['kite_0_vy'].values)
                Z.append(df['kite_0_vz'].values)
                meas_dict[meas] = 1
        
        elif meas == 'GPS_acc':
            col_ax = [col for col in df.columns if 'ax' in col]
            col_ay = [col for col in df.columns if 'ay' in col]
            col_az = [col for col in df.columns if 'az' in col]
            if multiple_GPS:
                for i in range(len(col_ax)):
                    Z.append(df[col_ax[i]].values)
                    Z.append(df[col_ay[i]].values)
                    Z.append(df[col_az[i]].values)
                meas_dict[meas] = len(col_ax)
            else:
                Z.append(df['kite_0_ax'].values)
                Z.append(df['kite_0_ay'].values)
                Z.append(df['kite_0_az'].values)
                meas_dict[meas] = 1
                
        
        elif meas == 'ground_wvel':
            uf = df['ground_wind_velocity']*kappa/np.log(10/z0)
            Z.append(uf.values)
            meas_dict[meas] = 1

        elif meas == 'apparent_wvel':
            col_va = [col for col in df.columns if 'apparent' in col]
            for i in range(len(col_va)):
                Z.append(df[col_va[i]].values)
                meas_dict[meas] = len(col_va)

        elif meas == 'tether_len':
            Z.append(df['ground_tether_length'].values)

        elif meas == 'aoa':
            Z.append(df['kite_angle_of_attack'].values)

    Z = np.array(Z)
    Z = Z.T

    return meas_dict, Z

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
    R_Roll=np.array([[1, 0, 0],[0,np.cos(Roll),np.sin(Roll)],[0,-np.sin(Roll),np.cos(Roll)]])#OK checked with Blender
    
    #Rotational matrix for Pitch
    R_Pitch=np.array([[np.cos(Pitch), 0, np.sin(Pitch)],[0,1,0],[-np.sin(Pitch), 0, np.cos(Pitch)]])#Checked with blender
    
    #Rotational matrix for Roll
    R_Yaw= np.array([[np.cos(Yaw),-np.sin(Yaw),0],[np.sin(Yaw),np.cos(Yaw),0],[0,0,1]])#Checked with Blender
    
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