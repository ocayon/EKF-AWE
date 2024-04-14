from run_EKF import run_EKF
from config import kite_model, n_tether_elements, opt_measurements, kcu_model, tether_material, tether_diameter, meas_stdv, model_stdv, doIEKF
from pathlib import Path
from utils import ModelSpecs, SystemSpecs, create_input_from_KP_csv, convert_ekf_output_to_df
import pandas as pd

if __name__ == '__main__':
    #%% Choose flight data
    year = '2023'
    month = '11'
    day = '27'

    # File path
    file_name = f"{kite_model}_{year}-{month}-{day}"
    file_path = Path('../processed_data/flight_data') / kite_model / (file_name + '.csv')
    flight_data = pd.read_csv(file_path)
    flight_data = flight_data.reset_index()
    flight_data = flight_data.iloc[:15000]

    timestep = flight_data['time'].iloc[1] - flight_data['time'].iloc[0]

    model_specs = ModelSpecs(timestep, n_tether_elements, opt_measurements=opt_measurements, correct_height=False, doIEKF=doIEKF)
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