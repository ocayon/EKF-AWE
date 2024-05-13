
import numpy as np
import pandas as pd
from config import model_stdv,meas_stdv
from utils import SimulationConfig, SystemParameters
from utils import  get_measurement_vector
from tether import Tether
from ekf.kalman_filter import ExtendedKalmanFilter, DynamicModel, ObservationModel, observability_Lie_method
from kite import Kite
from kcu import KCU
import time
from pathlib import Path
from load_data.load_kp_csv import create_input_from_KP_csv
from ekf.ekf_output import create_ekf_output, convert_ekf_output_to_df


def initialize_ekf(ekf_input, model_specs, system_specs):
    """
    Initialize the Extended Kalman Filter with system components and models.

    Args:
        ekf_input (EKFInput): Input parameters for the EKF.
        model_specs (SimulationConfig): Configuration settings for the simulation models.
        system_specs (SystemParameters): Specifications of the system components.

    Returns:
        tuple: Returns a tuple containing initialized components of the EKF including the filter itself,
               dynamic model, kite, KCU (Kite Control Unit), and tether.
    """
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
    # Initialize input vector
    ekf.get_input_vector(ekf_input,kcu)
    return ekf, dyn_model,kite, kcu, tether


def update_state_ekf_tether(ekf, tether, kite, kcu, dyn_model, ekf_input, model_specs):
    """
    Update the state of the Extended Kalman Filter (EKF) and the tether model based on new measurements.

    Args:
        ekf (ExtendedKalmanFilter): The EKF instance to be updated.
        tether (Tether): The tether model to update.
        kite (Kite): The kite model involved in the EKF process.
        kcu (KCU): The kite control unit.
        dyn_model (DynamicModel): The dynamic model used in the EKF.
        ekf_input (EKFInput): New input measurements for the EKF.
        model_specs (SimulationConfig): Configuration settings for the simulation models.

    Returns:
        tuple: Returns updated EKF instance, tether model, and an output structure with updated state.
    """


    zi = get_measurement_vector(ekf_input,model_specs)

    ############################################################
    # Update EKF
    ############################################################
    ekf.get_input_vector(ekf_input,kcu)
    
    ############################################################
    # Update state with Kalmann filter
    ############################################################
    ekf.initialize(ekf.x_k1_k,ekf.u,zi)
    # Predict next step
    ekf.predict()
    # Update next step
    ekf.update()

    if np.isnan(ekf.x_k1_k1).any():
        ekf.x_k1_k1 = ekf.x_k1_k
        print('EKF update returns Nan values, integration of current step ommited')
            
    ekf_output = create_ekf_output(ekf.x_k1_k1, ekf.u, kite, tether, kcu, model_specs)

    return ekf, tether, ekf_output

def run_EKF(ekf_input_list, model_specs, system_specs,x0):
    """
   Execute the Extended Kalman Filter process over a series of input data.

   Args:
       ekf_input_list (list of EKFInput): A list of EKFInput instances to be processed.
       model_specs (SimulationConfig): Configuration settings for the simulation models.
       system_specs (SystemParameters): Specifications of the system components.
       x0 (ndarray): The initial state vector for the EKF.

   Returns:
       list: A list of DataFrames each representing the EKF results for each input in the ekf_input_list.
   """
    # Initialize EKF
    ekf, dyn_model,kite, kcu, tether = initialize_ekf(ekf_input_list[0], model_specs, system_specs)
        
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
        ekf_input = ekf_input_list[k]
        ekf.x_k1_k = dyn_model.propagate(ekf.x_k1_k1,ekf.u, kite, tether, kcu, ekf_input.ts)

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
    n_tether_elements = 5
    opt_measurements = []
    tether_material = 'Dyneema-SK78'    # Tether material
    # File path
    file_name = f"{kite_model}_{year}-{month}-{day}"
    file_path = Path('../../processed_data/flight_data') / kite_model / f'{file_name}.csv'
    flight_data = pd.read_csv(file_path)
    flight_data = flight_data.reset_index()
    flight_data = flight_data.iloc[:36000]
    timestep = flight_data['time'].iloc[1] - flight_data['time'].iloc[0]

    model_specs = SimulationConfig(timestep, n_tether_elements, opt_measurements=opt_measurements)
    system_specs = SystemParameters(kite_model, kcu_model, tether_material, tether_diameter, meas_stdv, model_stdv, model_specs)
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
        path = '../../results/'+kite_model+'/'
        # Save the DataFrame to a CSV file
        csv_filename = file_name+'_res_GPS'+addition+'.csv'
        filepath = path+csv_filename

        ekf_output_df.to_csv(path+csv_filename, index=False)

        # Save the DataFrame to a CSV file
        csv_filename = file_name+'_fd.csv'
        flight_data.to_csv(path+csv_filename, index=False)

    