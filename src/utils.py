import numpy as np
import casadi as ca
from config import kite_models,kcu_cylinders, tether_materials, kappa, z0, rho, g, n_tether_elements
from scipy.interpolate import splrep, splev
from scipy.optimize import least_squares
import pandas as pd
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

class EKF_input:
    def __init__(self, Z, ts,x0,u0):
        self.Z = Z
        self.ts = ts
        self.x0 = x0
        self.u0 = u0
    def current_z(self, current_index):
        return self.Z[current_index]
    
class tether_model_input:
    def __init__(self, n_tether_elements, kite_acc, tether_force, tether_length=None, kcu_vel = None, kcu_acc = None):
        self.n_tether_elements = n_tether_elements
        self.kite_acc = kite_acc
        self.tether_force = tether_force
        self.tether_length = tether_length
        self.kcu_vel = kcu_vel
        self.kcu_acc = kcu_acc

    def current_state(self, current_index):
        if self.kcu_vel is None:
            kcu_vel = None
        else:
            kcu_vel = self.kcu_vel[current_index]
        if self.kcu_acc is None:
            kcu_acc = None
        else:
            kcu_acc = self.kcu_acc[current_index]
        if self.tether_length is None:
            tether_length = None
        else:
            tether_length = self.tether_length[current_index]

        return self.kite_acc[current_index], self.tether_force[current_index], tether_length, kcu_vel, kcu_acc


    
def create_input_from_KP_csv(file_path, measurements, kite,kcu,tether,kite_sensor = 0, kcu_sensor = None, correct_height = False):
    flight_data = pd.read_csv(file_path)
    flight_data = flight_data.reset_index()
    Z = []

    ## Get measurements

    # Kite measurements
    kite_pos = np.array([flight_data['kite_'+str(kite_sensor)+'_rx'],flight_data['kite_'+str(kite_sensor)+'_ry'],flight_data['kite_'+str(kite_sensor)+'_rz']]).T
    kite_vel = np.array([flight_data['kite_'+str(kite_sensor)+'_vx'],flight_data['kite_'+str(kite_sensor)+'_vy'],flight_data['kite_'+str(kite_sensor)+'_vz']]).T
    kite_acc = np.array([flight_data['kite_'+str(kite_sensor)+'_ax'],flight_data['kite_'+str(kite_sensor)+'_ay'],flight_data['kite_'+str(kite_sensor)+'_az']]).T
    # KCU measurements
    if kcu_sensor is not None:
        kcu_vel = np.array([flight_data['kite_'+str(kcu_sensor)+'_vx'],flight_data['kite_'+str(kcu_sensor)+'_vy'],flight_data['kite_'+str(kcu_sensor)+'_vz']]).T
        kcu_acc = np.array([flight_data['kite_'+str(kcu_sensor)+'_ax'],flight_data['kite_'+str(kcu_sensor)+'_ay'],flight_data['kite_'+str(kcu_sensor)+'_az']]).T
    else:
        kcu_vel = None
        kcu_acc = None
    # Tether measurements
    tether_force = np.array(flight_data['ground_tether_force'])
    if correct_height:
        tether_length = np.array(flight_data['ground_tether_length'])
    else:
        tether_length = None        
    # Airflow measurements
    ground_windspeed = np.array(flight_data['ground_wind_velocity'])
    ground_winddir = np.array(flight_data['ground_wind_direction'])
    apparent_windspeed = np.array(flight_data['kite_apparent_windspeed'])
    kite_aoa = np.array(flight_data['kite_angle_of_attack'])
    
    timestep = flight_data['time'].iloc[1]-flight_data['time'].iloc[0]

    # Create measurement array for the EKF
    if 'kite_pos' in measurements:
        Z.append(kite_pos[:,0])
        Z.append(kite_pos[:,1])
        Z.append(kite_pos[:,2])
    if 'kite_vel' in measurements:
        Z.append(kite_vel[:,0])
        Z.append(kite_vel[:,1])
        Z.append(kite_vel[:,2])
    if 'kite_acc' in measurements:
        Z.append(kite_acc[:,0])
        Z.append(kite_acc[:,1])
        Z.append(kite_acc[:,2])
    if 'ground_wvel' in measurements:
        Z.append(ground_windspeed)*kappa/np.log(10/z0)
    if 'apparent_wvel' in measurements:
        Z.append(apparent_windspeed)
    if 'tether_len' in measurements:
        Z.append(tether_length)      
    if 'aoa' in measurements:
        Z.append(kite_aoa)
    Z = np.array(Z).T
    
    if correct_height:
        tether_len0 = tether_length[0]
    else:
        tether_len0 = None
    x0, u0 = find_initial_state_vector(kite_pos[0], kite_vel[0], kite_acc[0], np.mean(ground_winddir[0:3000]), np.mean(ground_windspeed[0]), tether_force[0], tether_len0, n_tether_elements, kite, kcu,tether)

    # Create input classes
    ekf_input = EKF_input(Z,timestep,x0,u0)
    tether_input = tether_model_input(n_tether_elements, kite_acc, tether_force, tether_length, kcu_vel, kcu_acc)
    
    return ekf_input, tether_input

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