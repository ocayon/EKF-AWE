
import numpy as np
import pandas as pd
from config import kite_model, kcu_model, tether_diameter, tether_material, year, month, day, \
                     doIEKF, max_iterations, epsilon, measurements, stdv_x, stdv_y, n_tether_elements, z0, kappa
from utils import create_kite, create_kcu,  get_measurements, calculate_vw_loglaw, calculate_euler_from_reference_frame, calculate_airflow_angles
from kalman_filter import ExtendedKalmanFilter, DynamicModel, ObservationModel, observability_Lie_method
from tether_model import create_tether
import time

def run_EKF(Z,x0,u0,ekf,tether,meas_ft,meas_lt,meas_akite, meas_akcu,meas_vkcu):
    # Define results matrices
    n_intervals = Z.shape[0]   
    N = n_intervals
    n = x0.shape[0]
    nm = Z.shape[1]
    # allocate space to store traces and other Kalman filter params
    XX_k1_k1    = np.zeros([n, N])
    err_meas    = np.zeros([nm, N])
    z_k1_k1    = np.zeros([nm, N-1])
    PP_k1_k1    = np.zeros([n, N])
    STD_x_cor   = np.zeros([n, N])
    STD_z       = np.zeros([nm, N])
    ZZ_pred     = np.zeros([nm, N])
    IEKFitcount = np.zeros([N, 1])
    
    # arrays for tether model and other results
    euler_angles = []
    airflow_angles = []
    tether_length = []
    Ft = []             # Tether force

    # Store Initial values
    XX_k1_k1[:,0]   = x0
    ZZ_pred[:,0]    = Z[0]

    # Define Initial Matrix P
    P_k1_k1 = np.eye(n)*1**2
    
    # Initial conditions
    x_k1_k1 = x0
    u = u0
     
    start_time = time.time()
    mins = -1
    
    for k in range(n_intervals-1):
        zi = Z[k]
        # zi[2] = res_tether[-1]
        ############################################################
        # Propagate state with dynamic model
        ############################################################
        x_k1_k = dyn_model.propagate(x_k1_k1,u)

        ############################################################
        # Update state with Kalmann filter
        ############################################################
        ekf.initialize(x_k1_k,u,zi)
        # Predict next step
        ekf.predict()
        # Update next step
        ekf.update()
        x_k1_k1 = ekf.x_k1_k1
        
        ############################################################
        # Calculate Input for next step with quasi-static tether model
        ############################################################
        r_kite = x_k1_k1[:3]
        v_kite = x_k1_k1[3:6]
        vw = calculate_vw_loglaw(x_k1_k1[6], z0, x_k1_k1[2], x_k1_k1[7])
        if meas_akcu is None:
            v_kcu = None
            a_kcu = None
        else:
            v_kcu = meas_vkcu[k]
            a_kcu = meas_akcu[k]
        tether.solve_tether_shape(n_tether_elements, r_kite, v_kite, vw, kite, kcu, tension_ground = meas_ft[k], tether_length = None,
                                a_kite = meas_akite[k], a_kcu = a_kcu, v_kcu = v_kcu)
        u = tether.Ft_kite
        ############################################################
        # Store results
        ############################################################
        XX_k1_k1[:,k] = np.array(x_k1_k1).reshape(-1)
        STD_x_cor[:,k] = ekf.std_x_cor
        STD_z[:,k] = ekf.std_z
        ZZ_pred [:,k] = ekf.z_k1_k
        err_meas[:,k] = ekf.z_k1_k - zi

        # Store tether force and tether model results
        Ft.append(u)
        euler_angles.append(calculate_euler_from_reference_frame(tether.dcm_b2w))
        tether_length.append(tether.stretched_tether_length)
        airflow_angles.append(calculate_airflow_angles(tether.dcm_b2w, v_kite, vw))

        # Print progress
        if k%600==0:
            elapsed_time = time.time() - start_time
            start_time = time.time()  # Record end time
            mins +=1
            print(f"Real time: {mins} minutes.  Elapsed time: {elapsed_time:.2f} seconds")
        
    # Store results
    ti = 0
    results = np.vstack((XX_k1_k1[:,ti:k],np.array(Ft)[ti:k,:].T,np.array(euler_angles)[ti:k,:].T,np.array(airflow_angles)[ti:k,:].T,np.array(tether_length)[ti:k].T))
    column_names = ['x','y','z','vx','vy','vz','uf','wdir','CL', 'CD', 'CS', 'bias_lt','bias_aoa','Ftx','Fty','Ftz','roll', 'pitch', 'yaw', 'aoa','ss','tether_len']
    df = pd.DataFrame(data=results.T, columns=column_names)

    return df

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read and process data
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

file_name = kite_model+'_'+year+'-'+month+'-'+day
file_path = '../processed_data/flight_data/'+kite_model+'/'+ file_name+'.csv'


flight_data = pd.read_csv(file_path)
flight_data = flight_data.reset_index()

flight_data = flight_data.iloc[:600*10]

ts = flight_data['time'].iloc[1]-flight_data['time'].iloc[0] # Sample time


#%% Define system model
kite = create_kite(kite_model)
kcu = create_kcu(kcu_model)
tether = create_tether(tether_material,tether_diameter)

# Declare classes
dyn_model = DynamicModel(kite,ts)
obs_model = ObservationModel(dyn_model.x,dyn_model.u,measurements,kite)
n           =  dyn_model.x.shape[0]                                 # state dimension
nm          =  obs_model.hx.shape[0]                                 # number of measurements
m           =  dyn_model.u.shape[0]                                 # number of inputs
#%% Get measurement array
meas_dict,Z = get_measurements(flight_data,measurements,False)

#%% Define inputs tether model

meas_ft = np.array(flight_data['ground_tether_force'])
meas_lt = np.array(flight_data['ground_tether_length'])
if kite.model_name == 'v3':
    meas_akite = np.vstack((np.array(flight_data['kite_0_ax']),np.array(flight_data['kite_0_ay']),np.array(flight_data['kite_0_az']))).T
    meas_akcu = None
    meas_vkcu = None
else:
    meas_akite = np.vstack((np.array(flight_data['kite_0_ax']),np.array(flight_data['kite_0_ay']),np.array(flight_data['kite_0_az']))).T
    meas_akcu = np.vstack((np.array(flight_data['kite_1_ax']),np.array(flight_data['kite_1_ay']),np.array(flight_data['kite_1_az']))).T
    meas_vkcu = np.vstack((np.array(flight_data['kite_1_vx']),np.array(flight_data['kite_1_vy']),np.array(flight_data['kite_1_vz']))).T

#%% Initial state vector 
ground_wdir0 = np.mean(flight_data['ground_wind_direction'].iloc[0:3000])/180*np.pi # Initial wind direction
ground_wvel0 = np.mean(flight_data['ground_wind_velocity'].iloc[0:3000]) # Initial wind velocity
uf0 = ground_wvel0*kappa/np.log(10/z0)
wvel0 = uf0/kappa*np.log(flight_data['kite_0_rz'].iloc[0]/z0)
if np.isnan(wvel0):
    wvel0 = 9
    ground_wdir0 = 180/180*np.pi
vw = [wvel0*np.cos(ground_wdir0),wvel0*np.sin(ground_wdir0),0] # Initial wind velocity
row = flight_data.iloc[0] # Initial row of flight data
kite_pos = np.array([row['kite_0_rx'],row['kite_0_ry'],row['kite_0_rz']]) # Initial kite position
kite_vel = np.array([row['kite_0_vx'],row['kite_0_vy'],row['kite_0_vz']]) # Initial kite velocity
if meas_akcu is None:
        v_kcu = None
        a_kcu = None
else:
    v_kcu = meas_vkcu[0]
    a_kcu = meas_akcu[0]
a_kite = meas_akite[0]
# tether.opt_guess = list(calculate_polar_coordinates(np.array(kite_pos)))
tether.solve_tether_shape(n_tether_elements, kite_pos, kite_vel, vw, kite, kcu, tension_ground = meas_ft[0], tether_length = None,
                            a_kite = a_kite, a_kcu = a_kcu, v_kcu = v_kcu)
x0 = np.vstack((kite_pos,kite_vel))
x0 = np.append(x0,[0.6,ground_wdir0,tether.CL,tether.CD,tether.CS,0,0])     # Initial state vector (Last two elements are bias, used if needed)
u0 = tether.Ft_kite
#%%
########################################################################
## Initialize Extended Kalman filter
########################################################################

# Initialize EKF
ekf = ExtendedKalmanFilter(stdv_x, stdv_y, ts, doIEKF, epsilon, max_iterations)
ekf.calc_Fx = dyn_model.get_fx_jac_fun()
ekf.calc_Hx = obs_model.get_hx_jac_fun()
ekf.calc_hx = obs_model.get_hx_fun()

# Check observability matrix
check_obs = False
if check_obs == True:
    observability_Lie_method(dyn_model.fx,obs_model.hx,dyn_model.x, dyn_model.u, x0,u0)


#%% Main loop
df = run_EKF(Z,x0,u0,ekf,tether,meas_ft,meas_lt,meas_akite, meas_akcu,meas_vkcu)

#%% Save results


addition = ''
path = '../results/'+kite_model+'/'
# Save the DataFrame to a CSV file
csv_filename = file_name+'_res_GPS'+addition+'.csv'
df.to_csv(path+csv_filename, index=False)
# Save the DataFrame to a CSV file
csv_filename = file_name+'_fd'+addition+'.csv'
flight_data.to_csv(path+csv_filename, index=False)

