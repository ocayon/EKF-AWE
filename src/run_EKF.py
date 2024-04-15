
import numpy as np
import pandas as pd
from config import kite_model, kcu_model, tether_diameter, tether_material, \
                     doIEKF, max_iterations, epsilon, opt_measurements, meas_stdv,model_stdv, n_tether_elements, z0, kappa
from model_definitions import kite_models, kcu_cylinders, tether_materials
from utils import calculate_vw_loglaw, calculate_euler_from_reference_frame, calculate_airflow_angles, ModelSpecs, SystemSpecs
from utils import  KiteModel, KCUModel, EKFInput, convert_ekf_output_to_df, get_measurement_vector, tether_input,EKFOutput, create_input_from_KP_csv, get_input_vector
from tether_model import TetherModel
from kalman_filter import ExtendedKalmanFilter, DynamicModel, ObservationModel, observability_Lie_method
import time
from pathlib import Path

def create_ekf_output(x, u, kite, tether,kcu):
    """Store results in a list of instances of the class EKFOutput"""
    # Store tether force and tether model results
    kite_pos = x[0:3]
    kite_vel = x[3:6]
    wind_vel = calculate_vw_loglaw(x[6], z0, x[2], x[7])
    tension_ground = u[1]
    tether_length = x[11]
    elevation_0 = x[12]
    azimuth_0 = x[13]

    if kcu.data_available:
        kcu_acc = u[2:5]
        kcu_vel = u[5:8]
        kite_acc = None
    else:
        kcu_acc = None
        kcu_vel = None
        kite_acc = u[2:5]

    args = (n_tether_elements, kite_pos, kite_vel, wind_vel, kite, kcu, tension_ground )
    opt_guess = [elevation_0, azimuth_0, tether_length]
    res = tether.calculate_tether_shape(opt_guess, *args, a_kite = kite_acc, a_kcu = kcu_acc, v_kcu = kcu_vel, return_values=True)
    dcm_b2w = res[2]
    euler_angles = calculate_euler_from_reference_frame(dcm_b2w)
    airflow_angles = calculate_airflow_angles(dcm_b2w, kite_vel, wind_vel)
    cd_kcu = res[-5]
    cd_tether = res[-4]
    

    wind_vel  = x[6]/kappa*np.log(x[2]/z0)
    wind_dir = x[7]
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
                                tether_length = x[11],
                                CL = x[8],
                                CD = x[9],
                                CS = x[10],
                                elevation_first_element = x[12],
                                azimuth_first_element = x[13], 
                                cd_kcu = cd_kcu,
                                cd_tether = cd_tether)
                            
    return ekf_output


def create_kite(model_name):
    """"Create kite model class from model name and model dictionary"""
    if model_name in kite_models:
        model_params = kite_models[model_name]
        return KiteModel(model_name, model_params["mass"], model_params["area"], model_params["distance_kcu_kite"],
                     model_params["total_length_bridle_lines"], model_params["diameter_bridle_lines"],model_params['KCU'], model_params["span"])
    else:
        raise ValueError("Invalid kite model")
    
def create_kcu(model_name, data_available = False):
    """"Create KCU model class from model name and model dictionary"""
    if model_name in kcu_cylinders:
        model_params = kcu_cylinders[model_name]
        return KCUModel(model_params["length"], model_params["diameter"], model_params["mass"], data_available)
    else:
        raise ValueError("Invalid KCU model")
        
def create_tether(material_name,diameter,n_tether_elements):
    """"Create tether model class from material name and diameter"""
    if material_name in tether_materials:
        material_params = tether_materials[material_name]
        return TetherModel(material_name,diameter,material_params["density"],material_params["cd"],material_params["Youngs_modulus"],n_tether_elements)
    else:
        raise ValueError("Invalid tether material")

def initialize_ekf(ekf_input, model_specs, system_specs):
    """Initialize the Extended Kalman Filter"""
    kite = create_kite(system_specs.kite_model)
    if ekf_input.kcu_acc is not None:
        kcu = create_kcu(system_specs.kcu_model, data_available=True)
    else:
        kcu = create_kcu(system_specs.kcu_model, data_available=False)
    tether = create_tether(system_specs.tether_material,system_specs.tether_diameter,n_tether_elements)
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
    tether.solve_tether_shape(n_tether_elements, x[0:3], x[3:6], calculate_vw_loglaw(x[6], z0, x[2], x[7]), kite, kcu, tension_ground = fti, tether_length = lti,
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


    zi = get_measurement_vector(ekf_input,model_specs.opt_measurements)

    ############################################################
    # Update EKF
    ############################################################
    u = get_input_vector(ekf_input,kcu)

    ekf = update_ekf(ekf, dyn_model, u, zi, kite, tether, kcu,ekf_input.ts)
    
    if np.isnan(ekf.x_k1_k1).any():
        ekf.x_k1_k1 = ekf.x_k1_k
        print('EKF update returns Nan values, integration of current step ommited')
            
    ekf_output = create_ekf_output(ekf.x_k1_k1, u, kite, tether, kcu)

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
        
    print(x0)
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
    # File path
    file_name = f"{kite_model}_{year}-{month}-{day}"
    file_path = Path('../processed_data/flight_data') / kite_model / (file_name + '.csv')
    flight_data = pd.read_csv(file_path)
    flight_data = flight_data.reset_index()
    flight_data = flight_data.iloc[:15000]
    timestep = flight_data['time'].iloc[1] - flight_data['time'].iloc[0]

    model_specs = ModelSpecs(timestep, n_tether_elements, opt_measurements=opt_measurements)
    system_specs = SystemSpecs(kite_model, kcu_model, tether_material, tether_diameter, meas_stdv, model_stdv, opt_measurements)
    # Create input classes
    ekf_input_list,x0 = create_input_from_KP_csv(flight_data, system_specs, kite_sensor = 0, kcu_sensor = 1)

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

    