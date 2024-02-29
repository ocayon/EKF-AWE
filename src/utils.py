import numpy as np
import casadi as ca
from config import kappa, z0
from scipy.interpolate import splrep, splev
from dataclasses import dataclass
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

@dataclass
class EKFInput:
    kite_pos: np.array      # Kite position in ENU coordinates
    kite_vel: np.array      # Kite velocity in ENU coordinates
    kite_acc: np.array      # Kite acceleration in ENU coordinates
    tether_force: float     # Ground tether force
    apparent_windspeed: float = None # Apparent windspeed
    tether_length: float = None     # Tether length
    kite_aoa: np.array = None      # Kite angle of attack
    kcu_vel: np.array = None    # KCU velocity in ENU coordinates
    kcu_acc: np.array = None    # KCU acceleration in ENU coordinates

@dataclass
class EKFOutput:
    kite_pos: np.array    # Kite position in ENU coordinates
    kite_vel: np.array      # Kite velocity in ENU coordinates
    wind_velocity: float   # Wind velocity
    wind_direction: float # Wind direction
    roll: float           # Roll angle
    pitch: float         # Pitch angle
    yaw: float          # Yaw angle
    tether_length: float # Tether length
    tether_force: np.array # Tether force vector at the kite
    kite_aoa: float    # Kite angle of attack
    kite_sideslip: float # Kite sideslip angle
    CL : float          # Lift coefficient
    CD : float          # Drag coefficient
    CS : float          # Side force coefficient

    


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
    def __init__(self, kite_model, kcu_model, tether_material, tether_diameter, meas_stdv, model_stdv, opt_measurements):
        self.kite_model = kite_model
        self.kcu_model = kcu_model
        self.tether_material = tether_material
        self.tether_diameter = tether_diameter


        self.stdv_dynamic_model = np.array([model_stdv['x'], model_stdv['x'], model_stdv['x'], 
                   model_stdv['v'], model_stdv['v'], model_stdv['v'], 
                   model_stdv['uf'], model_stdv['wdir'], 
                   model_stdv['CL'], model_stdv['CD'], model_stdv['CS'],
                   model_stdv['bias_lt'], model_stdv['bias_aoa']])
        stdv_y = []
        for _ in range(3):
            stdv_y.append(meas_stdv['x'])
        for _ in range(3):
            stdv_y.append(meas_stdv['v'])
        for key in opt_measurements:
            if key == 'kite_acc':   
                for _ in range(3):
                    stdv_y.append(meas_stdv['a'])
            elif key == 'ground_wvel':
                stdv_y.append(meas_stdv['uf'])

            elif key == 'apparent_windspeed':
                stdv_y.append(meas_stdv['va'])
            elif key == 'tether_length':
                stdv_y.append(meas_stdv['tether_length'])
            elif key == 'aoa':
                stdv_y.append(meas_stdv['aoa'])
        stdv_y = np.array(stdv_y)
        self.stdv_measurements = stdv_y



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

def calculate_angle_2vec(vector_a, vector_b, reference_vector=None):
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
    ex_kite = dcm[:,0]      # Kite x axis
    ey_kite = dcm[:,1]      # Kite y axis perpendicular to v and tether
    ez_kite = dcm[:,2]      # Kite z axis pointing in the direction of the tension
    va = vw-v_kite
    va_proj = project_onto_plane(va, ey_kite)           # Projected apparent wind velocity onto kite y axis
    aoa = calculate_angle(ez_kite,va_proj)-90             # Angle of attack
    va_proj = project_onto_plane(va, ez_kite)           # Projected apparent wind velocity onto kite z axis
    sideslip = 90-calculate_angle(ey_kite,va_proj)         # Sideslip angle
    return aoa, sideslip

def create_input_from_KP_csv(flight_data, system_specs, kite_sensor = 0, kcu_sensor = None):
    """Create input classes and initial state vector from flight data"""
    from config import n_tether_elements
    from run_EKF import create_kite, create_kcu,create_tether
    n_intervals = len(flight_data)
    # Kite measurements
    kite_pos = np.array([flight_data['kite_'+str(kite_sensor)+'_rx'],flight_data['kite_'+str(kite_sensor)+'_ry'],flight_data['kite_'+str(kite_sensor)+'_rz']]).T
    kite_vel = np.array([flight_data['kite_'+str(kite_sensor)+'_vx'],flight_data['kite_'+str(kite_sensor)+'_vy'],flight_data['kite_'+str(kite_sensor)+'_vz']]).T
    kite_acc = np.array([flight_data['kite_'+str(kite_sensor)+'_ax'],flight_data['kite_'+str(kite_sensor)+'_ay'],flight_data['kite_'+str(kite_sensor)+'_az']]).T
    # KCU measurements
    if kcu_sensor is not None:
        kcu_vel = np.array([flight_data['kite_'+str(kcu_sensor)+'_vx'],flight_data['kite_'+str(kcu_sensor)+'_vy'],flight_data['kite_'+str(kcu_sensor)+'_vz']]).T
        kcu_acc = np.array([flight_data['kite_'+str(kcu_sensor)+'_ax'],flight_data['kite_'+str(kcu_sensor)+'_ay'],flight_data['kite_'+str(kcu_sensor)+'_az']]).T
    else:
        kcu_vel = np.zeros((n_intervals,3))
        kcu_acc = np.zeros((n_intervals,3))
    # Tether measurements
    tether_force = np.array(flight_data['ground_tether_force'])
    tether_length = np.array(flight_data['ground_tether_length'])
      
    # Airflow measurements
    ground_windspeed = np.array(flight_data['ground_wind_velocity'])
    ground_winddir = np.array(flight_data['ground_wind_direction'])
    apparent_windspeed = np.array(flight_data['kite_apparent_windspeed'])
    kite_aoa = np.array(flight_data['kite_angle_of_attack'])
    
    timestep = flight_data['time'].iloc[1]-flight_data['time'].iloc[0]
    ekf_input_list = []
    for i in range(len(flight_data)):
        ekf_input_list.append(EKFInput(kite_pos = kite_pos[i], 
                                    kite_vel = kite_vel[i], 
                                    kite_acc = kite_acc[i], 
                                    tether_force = tether_force[i],
                                    apparent_windspeed = apparent_windspeed[i], 
                                    tether_length = tether_length[i],
                                    kite_aoa = kite_aoa[i], 
                                    kcu_vel = kcu_vel[i], 
                                    kcu_acc = kcu_acc[i]))
                

    kite = create_kite(system_specs.kite_model)
    kcu = create_kcu(system_specs.kcu_model)
    tether = create_tether(system_specs.tether_material,system_specs.tether_diameter)

    x0, u0 = find_initial_state_vector(kite_pos[0], kite_vel[0], kite_acc[0], 
                                       np.mean(ground_winddir[0:3000])/180*np.pi, np.mean(ground_windspeed[0]), tether_force[0], 
                                       tether_length[i], n_tether_elements, kite, kcu,tether)

    return ekf_input_list, x0
