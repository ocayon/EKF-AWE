
import numpy as np
import pandas as pd
from config import kite_model, kcu_model, tether_diameter, tether_material, \
                     doIEKF, max_iterations, epsilon, opt_measurements, meas_stdv,model_stdv, n_tether_elements, z0, kappa
from model_definitions import kite_models, kcu_cylinders, tether_materials
from utils import calculate_vw_loglaw, calculate_euler_from_reference_frame, calculate_airflow_angles, ModelSpecs, SystemSpecs
from utils import  KiteModel, KCUModel, EKFInput, convert_ekf_output_to_df, get_measurement_vector, tether_input,EKFOutput, create_input_from_KP_csv
from tether_model import TetherModel
from kalman_filter import ExtendedKalmanFilter, DynamicModel, ObservationModel, observability_Lie_method
import time
from pathlib import Path

def create_ekf_output(ekf, tether):
    """Store results in a list of instances of the class EKFOutput"""
    # Store tether force and tether model results
    euler_angles = calculate_euler_from_reference_frame(tether.dcm_b2w)
    airflow_angles = calculate_airflow_angles(tether.dcm_b2w, ekf.x_k1_k1[3:6], calculate_vw_loglaw(ekf.x_k1_k1[6], z0, ekf.x_k1_k1[2], ekf.x_k1_k1[7]))
    x = ekf.x_k1_k1

    wind_vel  = x[6]/kappa*np.log(x[2]/z0)
    ekf_output = EKFOutput(kite_pos = x[0:3],
                                kite_vel = x[3:6],
                                wind_velocity = wind_vel,
                                wind_direction = x[7],
                                tether_force=tether.Ft_kite,
                                roll = euler_angles[0],
                                pitch = euler_angles[1],
                                yaw = euler_angles[2],
                                kite_aoa = airflow_angles[0],
                                kite_sideslip = airflow_angles[1],
                                tether_length = tether.stretched_tether_length,
                                CL = x[8],
                                CD = x[9],
                                CS = x[10])
                            
    return ekf_output


def create_kite(model_name):
    """"Create kite model class from model name and model dictionary"""
    if model_name in kite_models:
        model_params = kite_models[model_name]
        return KiteModel(model_name, model_params["mass"], model_params["area"], model_params["distance_kcu_kite"],
                     model_params["total_length_bridle_lines"], model_params["diameter_bridle_lines"],model_params['KCU'], model_params["span"])
    else:
        raise ValueError("Invalid kite model")
    
def create_kcu(model_name):
    """"Create KCU model class from model name and model dictionary"""
    if model_name in kcu_cylinders:
        model_params = kcu_cylinders[model_name]
        return KCUModel(model_params["length"], model_params["diameter"], model_params["mass"])
    else:
        raise ValueError("Invalid KCU model")
        
def create_tether(material_name,diameter):
    """"Create tether model class from material name and diameter"""
    if material_name in tether_materials:
        material_params = tether_materials[material_name]
        return TetherModel(material_name,diameter,material_params["density"],material_params["cd"],material_params["Youngs_modulus"])
    else:
        raise ValueError("Invalid tether material")

def initialize_ekf(ekf_input, model_specs, system_specs):
    """Initialize the Extended Kalman Filter"""
    kite = create_kite(system_specs.kite_model)
    kcu = create_kcu(system_specs.kcu_model)
    tether = create_tether(system_specs.tether_material,system_specs.tether_diameter)
    # Create dynamic model and observation model
    dyn_model = DynamicModel(kite,model_specs.ts)
    obs_model = ObservationModel(dyn_model.x,dyn_model.u,model_specs.opt_measurements,kite)
    # Initialize EKF
    ekf = ExtendedKalmanFilter(system_specs.stdv_dynamic_model, system_specs.stdv_measurements, model_specs.ts,dyn_model,obs_model, model_specs.doIEKF, model_specs.epsilon, model_specs.max_iterations)
    return ekf, dyn_model,kite, kcu, tether

def update_tether(x,ekf_input, model_specs, tether, kite, kcu):
    kite_acc, fti, lti, kcu_acc, kcu_vel = tether_input(ekf_input,model_specs)
    tether.solve_tether_shape(n_tether_elements, x[0:3], x[3:6], calculate_vw_loglaw(x[6], z0, x[2], x[7]), kite, kcu, tension_ground = fti, tether_length = lti,
                                a_kite = kite_acc, a_kcu = kcu_acc, v_kcu = kcu_vel)
    return tether

def update_ekf(ekf, dyn_model, u, z):
    ############################################################
    # Propagate state with dynamic model
    ############################################################
    
    x_k1_k = dyn_model.propagate(ekf.x_k1_k1,u)

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
    if model_specs.correct_height:
        zi[2] = tether.kite_pos[2]
    ############################################################
    # Update EKF
    ############################################################
    ekf = update_ekf(ekf, dyn_model, tether.Ft_kite, zi)
    ############################################################
    # Calculate Input for next step with quasi-static tether model
    ############################################################
    tether = update_tether(ekf.x_k1_k1,ekf_input, model_specs, tether, kite, kcu)

    ekf_output = create_ekf_output(ekf, tether)

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
    if model_specs.correct_height:
        x0[2] = tether.kite_pos[2]

    # Define results matrices
    n_intervals = len(ekf_input_list)

    ekf_output_list = []    # List of instances of EKFOutput

    # Initial conditions
    ekf.x_k1_k1 = x0
     
    start_time = time.time()
    mins = -1
    
    for k in range(n_intervals):
        ekf_input = ekf_input_list[k]
        
        ## Update step
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

    timestep = flight_data['time'].iloc[1] - flight_data['time'].iloc[0]

    model_specs = ModelSpecs(timestep, n_tether_elements, opt_measurements=opt_measurements, correct_height=False)
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

    