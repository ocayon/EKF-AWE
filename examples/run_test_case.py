import time as time
from awes_ekf.ekf.initialize_and_update_ekf import initialize_ekf, propagate_state_EKF
from awes_ekf.load_data.read_data import read_processed_flight_data
from awes_ekf.load_data.create_input_from_csv import create_input_from_csv, find_initial_state_vector
from awes_ekf.setup.settings import load_config, SimulationConfig, TuningParameters
from awes_ekf.load_data.save_data import save_results
from awes_ekf.setup.kite import Kite
from awes_ekf.setup.tether import Tether
from awes_ekf.setup.kcu import KCU
from awes_ekf.ekf.ekf_output import convert_ekf_output_to_df, EKFOutput
from awes_ekf.postprocess.postprocessing import postprocess_results


config_file_name = "v3_config.yaml"

if __name__ == "__main__":
    
    config_data = load_config("examples/"+config_file_name)
    #%% Load data
    year = str(config_data["year"])
    month = str(config_data["month"])
    day = str(config_data["day"])
    kite_model = config_data["kite"]["model_name"]
    remove_IMU_offsets = config_data['postprocess']["remove_IMU_offsets"]
    correct_IMU_deformation = config_data['postprocess']["correct_IMU_deformation"]
    remove_vane_offsets = config_data['postprocess']["remove_vane_offsets"]
    estimate_kite_angle = config_data['postprocess']["estimate_kite_angle"]
    
    flight_data = read_processed_flight_data(year, month, day, kite_model)

    # flight_data = flight_data.iloc[:100]
    # flight_data.reset_index(drop=True, inplace=True)

    # %% Initialize EKF
    simConfig = SimulationConfig(**config_data["simulation_parameters"])

    kite = Kite(**config_data["kite"])
    if config_data["kcu"]:
        kcu = KCU(**config_data["kcu"])
    else:
        kcu = None
    tether = Tether(kite,kcu,simConfig.obsData,**config_data["tether"])

    tuningParams = TuningParameters(config_data["tuning_parameters"], simConfig)

    # Create input classes
    ekf_input_list = create_input_from_csv(
        flight_data, kite, kcu, tether, simConfig, kite_sensor=0
    )
    x0 = find_initial_state_vector(tether,ekf_input_list[0],simConfig)
    print(x0)
    ekf, dyn_model = initialize_ekf(
        ekf_input_list[0], simConfig, tuningParams, x0, kite, kcu, tether
    )

    # %% Main loop
    ekf_output_list = []  # List of instances of EKFOutput
    start_time = time.time()
    mins = -1
    for k, ekf_input in enumerate(ekf_input_list):

        # Propagate state EKF
        try :
            ekf, ekf_ouput = propagate_state_EKF(
                ekf, dyn_model, ekf_input, simConfig, tether, kite, kcu
            )
            # Store results
            ekf_output_list.append(ekf_ouput)
        except:  
            try:
                print('Integration error at iteration: ', k)
                x0 = find_initial_state_vector(tether, ekf_input, simConfig)
            except:
                print('Tether model error at iteration: ', k)
                continue    
            ekf, dyn_model = initialize_ekf(
                ekf_input, simConfig, tuningParams, x0, kite, kcu, tether
            )
            flight_data.drop(k, inplace=True)  
            # continue                      

        # Print progress
        if k % 600 == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()  # Record end time
            mins += 1
            print(
                f"Real time: {mins} minutes.  Elapsed time: {elapsed_time:.2f} seconds"
            )

    
    # Postprocess results
    ekf_output_df = convert_ekf_output_to_df(ekf_output_list)
    ekf_output_df.dropna(subset=["kite_pos_x"], inplace=True)
    ekf_output_df.reset_index(drop=True, inplace=True)
    rows_to_keep = ekf_output_df.index
    flight_data = flight_data.iloc[rows_to_keep]
    flight_data.reset_index(drop=True, inplace=True)
    indices = ekf_output_df.index
    flight_data = flight_data.iloc[indices]
    results, flight_data = postprocess_results(
        ekf_output_df,
        flight_data,
        kite,
        kcu,
        imus=[0],
        remove_IMU_offsets=remove_IMU_offsets,
        correct_IMU_deformation=correct_IMU_deformation,
        remove_vane_offsets=remove_vane_offsets,
        estimate_kite_angle=estimate_kite_angle,
    )
    # %% Store results
    save_results(ekf_output_df, flight_data, kite_model, year, month, day)
