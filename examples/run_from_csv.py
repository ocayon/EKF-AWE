
import time as time
from awes_ekf.run_EKF import initialize_ekf, propagate_state_EKF
from awes_ekf.load_data.read_data import read_processed_flight_data
from awes_ekf.load_data.load_kp_csv import create_input_from_KP_csv
from awes_ekf.setup.settings import load_config, SimulationConfig, SystemParameters
from awes_ekf.load_data.save_data import save_results

if __name__ == '__main__':
    #%% Load flight data and configuration settings
    year = '2024'
    month = '02'
    day = '16'
    kite_model = 'v9'
    flight_data = read_processed_flight_data(year,month,day,kite_model)
    flight_data = flight_data.iloc[:10000]
    config_data = load_config('examples/v9_config.yaml')
    #%% Initialize EKF
    simConfig = SimulationConfig(**config_data['simulation_parameters'])
    systemParams = SystemParameters(config_data['system_parameters'], simConfig)
    # Create input classes
    ekf_input_list,x0 = create_input_from_KP_csv(flight_data, systemParams, simConfig,kite_sensor = 0, kcu_sensor = 1)

    ekf, dyn_model,kite, kcu, tether = initialize_ekf(ekf_input_list[0], simConfig, systemParams,x0)
    
    #%% Main loop
    ekf_output_list = []    # List of instances of EKFOutput    
    start_time = time.time()
    mins = -1
    for k,ekf_input in enumerate(ekf_input_list):

        # Propagate state EKF
        ekf, ekf_ouput = propagate_state_EKF(ekf, dyn_model, ekf_input, simConfig, tether, kite, kcu)
        
        # Store results
        ekf_output_list.append(ekf_ouput)

        # Print progress
        if k%600==0:
            elapsed_time = time.time() - start_time
            start_time = time.time()  # Record end time
            mins +=1
            print(f"Real time: {mins} minutes.  Elapsed time: {elapsed_time:.2f} seconds")

    #%% Store results
    save_results(ekf_output_list, flight_data, kite_model, year, month, day)