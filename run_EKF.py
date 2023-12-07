
import numpy as np
import pandas as pd
from config import kite_model, kcu_model, tether_diameter, tether_material, year, month, day, \
                     doIEKF, max_iterations, epsilon, measurements, stdv_x, stdv_y
from utils import create_kite, create_kcu, create_tether, DynamicModel, ObservationModel, ExtendedKalmanFilter, \
                initialize_state, get_measurements,  \
                 observability_Lie_method, calculate_quasi_static_tether

import time

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read and process data
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

file_name = kite_model+'_'+year+'-'+month+'-'+day
file_path = './data/'+ file_name+'.csv'


flight_data = pd.read_csv(file_path)
flight_data = flight_data.reset_index()


ts = flight_data['time'].iloc[1]-flight_data['time'].iloc[0] # Sample time

#%% Define system model
kite = create_kite(kite_model)
kcu = create_kcu(kcu_model)
tether = create_tether(tether_material,tether_diameter)

# Declare classes
dyn_model = DynamicModel(kite,ts)
obs_model = ObservationModel(dyn_model.x,dyn_model.u,measurements)
n           =  dyn_model.x.shape[0]                                 # state dimension
nm          =  obs_model.hx.shape[0]                                 # number of measurements
m           =  dyn_model.u.shape[0]                                 # number of inputs

#%% Define measurement noise matrix 
meas_dict,Z = get_measurements(flight_data,measurements,False)


#%% Initial state vector 
x0,u0,opt_guess = initialize_state(flight_data,kite,kcu,tether)
x0[-3] = 0.8
x0[-2] = 0.1
x0[-1] = 0

#%%
########################################################################
## Initialize Extended Kalman filter
########################################################################
n_intervals = flight_data.shape[0] - 1
N = n_intervals

# allocate space to store traces
XX_k1_k1    = np.zeros([n, N])
err_meas    = np.zeros([nm, N])
z_k1_k1    = np.zeros([nm, N-1])
PP_k1_k1    = np.zeros([n, N])
STD_x_cor   = np.zeros([n, N])
STD_z       = np.zeros([nm, N])
ZZ_pred     = np.zeros([nm, N])
IEKFitcount = np.zeros([N, 1])

# Store Initial values
XX_k1_k1[:,0]   = x0
ZZ_pred[:,0]    = Z[0]

# Define Initial Matrix P
P_k1_k1 = np.eye(n)*1**2

x_k1_k1 = x0
u = u0
t = np.array(flight_data['time'])
Ft = []             # Tether force
res_williams = []   # Williams model results

# Initialize EKF
ekf = ExtendedKalmanFilter(stdv_x, stdv_y, ts, doIEKF, epsilon, max_iterations)
ekf.calc_Fx = dyn_model.get_fx_jac_fun()
ekf.calc_Hx = obs_model.get_hx_jac_fun()
ekf.calc_hx = obs_model.get_hx_fun()

# Check observability matrix
check_obs = True
if check_obs == True:
    observability_Lie_method(dyn_model.fx,obs_model.hx,dyn_model.x, dyn_model.u, x0,u0)


start_time = time.time()
mins = -1
for k in range(n_intervals):
    
    row = flight_data.iloc[k]
    zi = Z[k]
    
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
    u, res_tether, opt_guess = calculate_quasi_static_tether(x_k1_k1,row,opt_guess, kite, kcu, tether)
    
    ############################################################
    # Store results
    ############################################################
    XX_k1_k1[:,k]   = np.array(x_k1_k1).reshape(-1)
    STD_x_cor[:,k]  = ekf.std_x_cor
    STD_z[:,k]      = ekf.std_z
    ZZ_pred [:,k]    = ekf.z_k1_k
    err_meas[:,k] = ekf.z_k1_k - zi
    res_williams.append(res_tether)
    Ft.append(u)

    # Print progress
    if k%600==0:
        elapsed_time = time.time() - start_time
        start_time = time.time()  # Record end time
        mins +=1
        print(f"Real time: {mins} minutes.  Elapsed time: {elapsed_time:.2f} seconds")
    

#%% Save results
ti = 0
results = np.vstack((XX_k1_k1[:,ti:k],np.array(Ft)[ti:k,:].T,np.array(res_williams)[ti:k,:].T))
column_names = ['x','y','z','vx','vy','vz','uf','wdir','CL', 'CD', 'CS', 'Ftx','Fty','Ftz','roll', 'pitch', 'yaw', 'aoa', 'sideslip', 'CLw', 'CDw', 'CSw', 'cd_kcu', 'tether_len']
df = pd.DataFrame(data=results.T, columns=column_names)

flight_data = flight_data.iloc[ti:k]

path = './results/'+kite_model+'/'
# Save the DataFrame to a CSV file
csv_filename = file_name+'_res_GPS.csv'
df.to_csv(path+csv_filename, index=False)
# Save the DataFrame to a CSV file
csv_filename = file_name+'_fd.csv'
flight_data.to_csv(path+csv_filename, index=False)

