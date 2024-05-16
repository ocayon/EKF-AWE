
import pandas as pd
# from utils import SimulationConfig, SystemParameters

from run_EKF import run_EKF
from pathlib import Path
from load_data.load_kp_csv import create_input_from_KP_csv
from ekf.ekf_output import convert_ekf_output_to_df
from setup.settings import load_config, SimulationConfig, SystemParameters

if __name__ == '__main__':
    #%% Choose flight data
    year = '2024'
    month = '02'
    day = '16'
    
    # Load configuration settings
    config_data = load_config('v9_config.yaml')
    simConfig = SimulationConfig(**config_data['simulation_parameters'])
    systemParams = SystemParameters(config_data['system_parameters'], simConfig)
    
    kite_model = systemParams.kite_model
    
    # File path
    file_name = f"{kite_model}_{year}-{month}-{day}"
    file_path = Path('../../processed_data/flight_data') / kite_model / (file_name + '.csv')
    flight_data = pd.read_csv(file_path)
    flight_data = flight_data.reset_index()
    flight_data = flight_data.iloc[:10000]

    timestep = flight_data['time'].iloc[1] - flight_data['time'].iloc[0]

    
    # Create input classes
    ekf_input_list,x0 = create_input_from_KP_csv(flight_data, systemParams, simConfig,kite_sensor = 0, kcu_sensor = 1)

    # Check observability matrix
    # check_obs = False
    # if check_obs == True:
    #     observability_Lie_method(dyn_model.fx,obs_model.hx,dyn_model.x, dyn_model.u, ekf_input.x0,ekf_input.u0)

    #%% Main loop
    ekf_output_list = run_EKF(ekf_input_list, simConfig, systemParams,x0)

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
