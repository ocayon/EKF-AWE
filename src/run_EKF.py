
import numpy as np
import pandas as pd
from config import kite_model, kcu_model, tether_diameter, tether_material, year, month, day, \
                     doIEKF, max_iterations, epsilon, measurements, stdv_x, stdv_y, n_tether_elements, z0, kappa
from utils import create_kite, create_kcu,  get_measurements, calculate_vw_loglaw, calculate_euler_from_reference_frame, calculate_airflow_angles, create_input_from_KP_csv
from kalman_filter import ExtendedKalmanFilter, DynamicModel, ObservationModel, observability_Lie_method
from tether_model import create_tether
import time
### NEED to improve inputs and outputs
def run_EKF(efk,tether,kite, kcu,ekf_input,tether_input):
    # Define results matrices
    n_intervals = ekf_input.Z.shape[0]   
    N = n_intervals
    n = ekf_input.x0.shape[0]
    nm = ekf_input.Z.shape[1]
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
    XX_k1_k1[:,0]   = ekf_input.x0
    ZZ_pred[:,0]    = ekf_input.Z[0]

    # Define Initial Matrix P
    P_k1_k1 = np.eye(n)*1**2
    
    # Initial conditions
    x_k1_k1 = ekf_input.x0
    u = ekf_input.u0
     
    start_time = time.time()
    mins = -1
    
    for k in range(n_intervals):
        zi = ekf_input.current_z(k)
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
        
        kite_acc, fti, lti, kcu_acc, kcu_vel = tether_input.current_state(k)
        tether.solve_tether_shape(n_tether_elements, r_kite, v_kite, vw, kite, kcu, tension_ground = fti, tether_length = lti,
                                a_kite = kite_acc, a_kcu = kcu_acc, v_kcu = kcu_vel)
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
    k +=1
    results = np.vstack((XX_k1_k1[:,ti:k],np.array(Ft)[ti:k,:].T,np.array(euler_angles)[ti:k,:].T,np.array(airflow_angles)[ti:k,:].T,np.array(tether_length)[ti:k].T))
    column_names = ['x','y','z','vx','vy','vz','uf','wdir','CL', 'CD', 'CS', 'bias_lt','bias_aoa','Ftx','Fty','Ftz','roll', 'pitch', 'yaw', 'aoa','ss','tether_len']
    df = pd.DataFrame(data=results.T, columns=column_names)

    return df

#%% Read and process data 
if __name__ == "__main__":
    # File path
    file_name = kite_model+'_'+year+'-'+month+'-'+day
    file_path = '../processed_data/flight_data/'+kite_model+'/'+ file_name+'.csv'
    flight_data = pd.read_csv(file_path)
    flight_data = flight_data.reset_index()
    # flight_data = flight_data.iloc[:18000]

    # Define system model
    kite = create_kite(kite_model)
    kcu = create_kcu(kcu_model)
    tether = create_tether(tether_material,tether_diameter)

    # Create input classes
    ekf_input, tether_input = create_input_from_KP_csv(flight_data, measurements, kite, kcu, tether, kite_sensor = 0, kcu_sensor = None)
    # Alternatively, you can use the following code to create the input classes
    # ekf_input = EKF_input(kite_pos,kite_vel,timestep,x0,u0, apparent_windspeed = va/None,...)
    # tether_input = tether_model_input(n_tether_elements, kite_acc, tether_force, tether_length, kcu_vel, kcu_acc)

    # Create dynamic model and observation model
    dyn_model = DynamicModel(kite,ekf_input.ts)
    obs_model = ObservationModel(dyn_model.x,dyn_model.u,ekf_input.measurements,kite)

    # Initialize EKF
    ekf = ExtendedKalmanFilter(stdv_x, stdv_y, ekf_input.ts,dyn_model,obs_model, doIEKF, epsilon, max_iterations)

    # Check observability matrix
    check_obs = False
    if check_obs == True:
        observability_Lie_method(dyn_model.fx,obs_model.hx,dyn_model.x, dyn_model.u, ekf_input.x0,ekf_input.u0)
        
    #%% Main loop
    ekf_output = run_EKF(ekf,tether,kite, kcu,ekf_input,tether_input)

    save_results = True
    if save_results == True:
        #%% Save results
        addition = ''
        path = '../results/'+kite_model+'/'
        # Save the DataFrame to a CSV file
        csv_filename = file_name+'_res_GPS'+addition+'.csv'
        ekf_output.to_csv(path+csv_filename, index=False)

        # Save the DataFrame to a CSV file
        csv_filename = file_name+'_fd.csv'
        flight_data.to_csv(path+csv_filename, index=False)

    