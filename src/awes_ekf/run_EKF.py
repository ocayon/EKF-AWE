
import numpy as np
import pandas as pd
from config import z0, kappa, model_stdv,meas_stdv
from utils import calculate_vw_loglaw, calculate_euler_from_reference_frame, calculate_airflow_angles, ModelSpecs, SystemSpecs
from utils import  convert_ekf_output_to_df, get_measurement_vector, tether_input,EKFOutput,  get_input_vector, calculate_reference_frame_euler
from tether import Tether
from ekf.kalman_filter import ExtendedKalmanFilter, DynamicModel, ObservationModel, observability_Lie_method
from kite import Kite
from kcu import KCU
import time
from pathlib import Path
from load_data.load_kp_csv import create_input_from_KP_csv

def create_ekf_output(x, u, kite, tether,kcu, model_specs):
    """Store results in a list of instances of the class EKFOutput"""
    # Store tether force and tether model results
    kite_pos = x[0:3]
    kite_vel = x[3:6]
    if model_specs.log_profile:
        wind_vel  = x[6]/kappa*np.log(x[2]/z0)
        wind_dir = x[7]
        z_wind = x[8]
        vw = np.array([wind_vel*np.cos(wind_dir), wind_vel*np.sin(wind_dir), z_wind])
    else:
        vw = x[6:9]
        wind_vel = np.linalg.norm(vw)
        wind_dir = np.arctan2(vw[1],vw[0])
        z_wind = vw[2]
    tension_ground = u[1]
    tether_length = x[12]
    elevation_0 = x[13]
    azimuth_0 = x[14]

    if kcu.data_available:
        kcu_acc = u[2:5]
        kcu_vel = u[5:8]
        kite_acc = None
    else:
        kcu_acc = None
        kcu_vel = None
        kite_acc = u[2:5]

    args = (model_specs.n_tether_elements, kite_pos, kite_vel, vw, kite, kcu, tension_ground )
    opt_guess = [elevation_0, azimuth_0, tether_length]
    res = tether.calculate_tether_shape(opt_guess, *args, a_kite = kite_acc, a_kcu = kcu_acc, v_kcu = kcu_vel, return_values=True)
    dcm_b2w = res[2]
    dcm_t2w = res[3]
    euler_angles = calculate_euler_from_reference_frame(dcm_b2w)
    euler_angles1 = calculate_euler_from_reference_frame(dcm_t2w)
    cd_kcu = res[-5]
    cd_tether = res[-4]
    if model_specs.model_yaw:
        ex, ey, ez = calculate_reference_frame_euler( euler_angles[0], 
                                                     euler_angles[1], 
                                                     x[15]*180/np.pi, 
                                                     bodyFrame='NED')
        dcm = np.vstack(([ex], [ey], [ez])).T
        airflow_angles = calculate_airflow_angles(dcm, kite_vel, vw)
    else:
        airflow_angles = calculate_airflow_angles(dcm_b2w, kite_vel, vw)
    


    ekf_output = EKFOutput(kite_pos = kite_pos,
                                kite_vel = kite_vel,
                                wind_velocity = wind_vel,
                                wind_direction = wind_dir,
                                tether_force= tension_ground,
                                roll = euler_angles[0],
                                pitch = euler_angles[1],
                                yaw = euler_angles[2],
                                kite_aoa = airflow_angles[0],
                                kite_sideslip = airflow_angles[1],
                                tether_length = x[12],
                                CL = x[9],
                                CD = x[10],
                                CS = x[11],
                                elevation_first_element = x[13],
                                azimuth_first_element = x[14], 
                                cd_kcu = cd_kcu,
                                cd_tether = cd_tether,
                                z_wind = z_wind, 
                                roll_tether = euler_angles1[0])
    
    if model_specs.model_yaw:
        ekf_output.k_steering_law = x[16]
        ekf_output.yaw = x[15]
                            
    return ekf_output

def initialize_ekf(ekf_input, model_specs, system_specs):
    """Initialize the Extended Kalman Filter"""
    kite = Kite(system_specs.kite_model)
    if ekf_input.kcu_acc is not None:
        kcu = KCU(system_specs.kcu_model, data_available=True)
    else:
        kcu = KCU(system_specs.kcu_model, data_available=False)
        
    tether = Tether(system_specs.tether_material,system_specs.tether_diameter,model_specs.n_tether_elements)
    
    
    
    
    # Create dynamic model and observation model
    dyn_model = DynamicModel(kite,tether,kcu,model_specs)
    obs_model = ObservationModel(dyn_model.x,dyn_model.u,model_specs,kite,tether,kcu)

    if model_specs.tether_offset:    
        system_specs.stdv_dynamic_model = np.append(system_specs.stdv_dynamic_model, 1e-6)
        
        
    # Initialize EKF
    ekf = ExtendedKalmanFilter(system_specs.stdv_dynamic_model, system_specs.stdv_measurements, model_specs.ts,dyn_model,obs_model, kite, tether, kcu, model_specs.doIEKF, model_specs.epsilon, model_specs.max_iterations)
    
    return ekf, dyn_model,kite, kcu, tether

def update_tether(x,ekf_input, model_specs, tether, kite, kcu):
    kite_acc, fti, lti, kcu_acc, kcu_vel = tether_input(ekf_input,model_specs)
    tether.solve_tether_shape(model_specs.n_tether_elements, x[0:3], x[3:6], calculate_vw_loglaw(x[6], z0, x[2], x[7]), kite, kcu, tension_ground = fti, tether_length = lti,
                                a_kite = kite_acc, a_kcu = kcu_acc, v_kcu = kcu_vel)
    return tether

def update_ekf(ekf, dyn_model, u, z, kite, tether, kcu,ts):
    ############################################################
    # Propagate state with dynamic model
    ############################################################
    
    x_k1_k = ekf.x_k1_k 
    
    
    ############################################################
    # Update state with Kalmann filter
    ############################################################
    ekf.initialize(x_k1_k,u,z)
    # Predict next step
    ekf.predict()
    # Update next step
    ekf.update()
    
    return ekf

def update_state_ekf_tether(ekf, tether, kite, kcu, dyn_model, ekf_input, model_specs):
    """Update the state of the Extended Kalman Filter and the tether model"""


    zi = get_measurement_vector(ekf_input,model_specs)

    ############################################################
    # Update EKF
    ############################################################
    u = get_input_vector(ekf_input,kcu)

    ekf = update_ekf(ekf, dyn_model, u, zi, kite, tether, kcu,ekf_input.ts)
    
    if np.isnan(ekf.x_k1_k1).any():
        ekf.x_k1_k1 = ekf.x_k1_k
        print('EKF update returns Nan values, integration of current step ommited')
            
    ekf_output = create_ekf_output(ekf.x_k1_k1, u, kite, tether, kcu, model_specs)

    return ekf, tether, ekf_output

def run_EKF(ekf_input_list, model_specs, system_specs,x0):
    """Run the Extended Kalman Filter
    Args:
        ekf_input_list: list of EKFInput classes
        model_specs: ModelSpecs class
        system_specs: SystemSpecs class
        x0: initial state vector
    Returns:
        df: DataFrame with the results
    """
    # Initialize EKF
    ekf, dyn_model,kite, kcu, tether = initialize_ekf(ekf_input_list[0], model_specs, system_specs)
    
    # Initial measurement vector
    tether = update_tether(x0, ekf_input_list[0], model_specs, tether, kite, kcu)
        
    # Define results matrices
    n_intervals = len(ekf_input_list)

    ekf_output_list = []    # List of instances of EKFOutput

    if model_specs.tether_offset:
        x0 = np.append(x0,0)
    # Initial conditions
    ekf.x_k1_k1 = x0
     
    start_time = time.time()
    mins = -1
    
    for k in range(1,n_intervals):
        # Prediction step
        ekf_input = ekf_input_list[k-1]
        u = get_input_vector(ekf_input,kcu)
        ekf.x_k1_k = dyn_model.propagate(ekf.x_k1_k1,u, kite, tether, kcu, ekf_input.ts)

        ## Update step
        ekf_input = ekf_input_list[k]
        ekf, tether, ekf_ouput = update_state_ekf_tether(ekf, tether, kite, kcu, dyn_model, ekf_input, model_specs)
        # Store results
        ekf_output_list.append(ekf_ouput)

        # Print progress
        if k%600==0:
            elapsed_time = time.time() - start_time
            start_time = time.time()  # Record end time
            mins +=1
            print(f"Real time: {mins} minutes.  Elapsed time: {elapsed_time:.2f} seconds")
        
    return ekf_output_list

#%% Read and process data 
if __name__ == "__main__":
    #%% Choose flight data
    year = '2019'
    month = '10'
    day = '08'
    kite_model = 'v3'                   # Kite model name, if Costum, change the kite parameters next
    kcu_model = 'KP1'                   # KCU model name
    tether_diameter = 0.01            # Tether diameter [m]
    n_tether_elements = 5
    opt_measurements = []
    tether_material = 'Dyneema-SK78'    # Tether material
    # File path
    file_name = f"{kite_model}_{year}-{month}-{day}"
    file_path = Path('../../processed_data/flight_data') / kite_model / f'{file_name}.csv'
    flight_data = pd.read_csv(file_path)
    flight_data = flight_data.reset_index()
    flight_data = flight_data.iloc[:15000]
    timestep = flight_data['time'].iloc[1] - flight_data['time'].iloc[0]

    model_specs = ModelSpecs(timestep, n_tether_elements, opt_measurements=opt_measurements)
    system_specs = SystemSpecs(kite_model, kcu_model, tether_material, tether_diameter, meas_stdv, model_stdv, model_specs)
    # Create input classes
    ekf_input_list,x0 = create_input_from_KP_csv(flight_data, system_specs, model_specs,kite_sensor = 0, kcu_sensor = 1)

    # Check observability matrix
    # check_obs = False
    # if check_obs == True:
    #     observability_Lie_method(dyn_model.fx,obs_model.hx,dyn_model.x, dyn_model.u, ekf_input.x0,ekf_input.u0)

    #%% Main loop
    ekf_output_list = run_EKF(ekf_input_list, model_specs, system_specs,x0)

    #%% Store results
    save_results = True
    if save_results == True:
        ekf_output_df = convert_ekf_output_to_df(ekf_output_list)
        addition = ''
        path = '../results/'+kite_model+'/'
        # Save the DataFrame to a CSV file
        csv_filename = file_name+'_res_GPS'+addition+'.csv'
        ekf_output_df.to_csv(path+csv_filename, index=False)

        # Save the DataFrame to a CSV file
        csv_filename = file_name+'_fd.csv'
        flight_data.to_csv(path+csv_filename, index=False)

    