import time as time
from awes_ekf.ekf.initialize_and_update_ekf import initialize_ekf, propagate_state_EKF
from awes_ekf.load_data.read_data import read_processed_flight_data
from awes_ekf.load_data.create_input_from_csv import create_input_from_csv
from awes_ekf.setup.settings import load_config, SimulationConfig, TuningParameters
from awes_ekf.load_data.save_data import save_results
from awes_ekf.setup.kite import Kite
from awes_ekf.setup.tether import Tether
from awes_ekf.setup.kcu import KCU
from awes_ekf.ekf.ekf_output import convert_ekf_output_to_df
from awes_ekf.postprocess.postprocessing import postprocess_results

if __name__ == '__main__':
    #%% Load flight data and configuration settings
    year = '2019'
    month = '10'
    day = '08'
    kite_model = 'v3'
    flight_data = read_processed_flight_data(year,month,day,kite_model)
    flight_data = flight_data.iloc[:10000]
    config_data = load_config('examples/v3_config.yaml')
    # Posprocessing settings
    remove_IMU_offsets = True  # Remove IMU offsets, only for soft wing with KCU
    correct_IMU_deformation = True # Correct IMU deformation, only for soft wing with KCU
    remove_vane_offsets = True # Remove vane offsets, only for soft wing with KCU
    estimate_kite_angle = False # Estimate kite angle, only for soft wing with KCU
    #%% Initialize EKF
    simConfig = SimulationConfig(**config_data['simulation_parameters'])

    kite = Kite(**config_data['kite'])
    if config_data['kcu'] is not None:
        kcu = KCU(**config_data['kcu'])
    else:
        kcu = None
    tether = Tether(**config_data['tether'])

    tuningParams = TuningParameters(config_data['tuning_parameters'], simConfig)

    # Create input classes
    ekf_input_list,x0 = create_input_from_csv(flight_data, kite,kcu,tether, simConfig, kite_sensor = 0)

    ekf, dyn_model = initialize_ekf(ekf_input_list[0], simConfig, tuningParams,x0,kite,kcu,tether)
    
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

    #%% Postprocess results
    ekf_output_df = convert_ekf_output_to_df(ekf_output_list)
    results, flight_data = postprocess_results(
        ekf_output_df,
        flight_data,
        kite,
        kcu,
        imus=[],
        remove_IMU_offsets=remove_IMU_offsets,
        correct_IMU_deformation=correct_IMU_deformation,
        remove_vane_offsets=remove_vane_offsets,
        estimate_kite_angle=estimate_kite_angle,
    )
    # %% Store results
    save_results(ekf_output_df, flight_data, kite_model, year, month, day)