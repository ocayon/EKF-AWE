import enum
import os.path
import time as time
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

# Todo: this python path update is added because of the import from data.process_KP_data. We could move process_KP_data
#  to the examples dir to avoid this.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from awes_ekf.ekf.initialize_and_update_ekf import initialize_ekf, propagate_state_EKF
from awes_ekf.load_data.read_data import read_processed_flight_data
from awes_ekf.load_data.create_input_from_csv import (
    create_input_from_csv,
    find_initial_state_vector,
)
from awes_ekf.setup.settings import load_config, SimulationConfig, TuningParameters
from awes_ekf.load_data.save_data import save_results
from awes_ekf.setup.kite import PointMassEKF
from awes_ekf.setup.tether import Tether
from awes_ekf.setup.kcu import KCU
from awes_ekf.ekf.ekf_output import convert_ekf_output_to_df, EKFOutput
from awes_ekf.postprocess.postprocessing import postprocess_results
from data.process_KP_data import ProcessKpData
from plot_wind_results import plot_wind_results
from plot_kite_trajectory import plot_kite_trajectories
from plot_kite_aero import plot_kite_aero
from plot_kite_orientation import plot_kite_orientation

def list_available_flights(log_directory: Path, file_extension: str = ".csv") -> list:
    """
    Lists all available flight log files in the specified directory.
    
    :param log_directory: Path to the directory containing flight logs.
    :param file_extension: Extension of the log files (default is ".csv").
    :return: List of available flight log files.
    """
    flight_logs = list(log_directory.glob(f"*{file_extension}"))
    
    if not flight_logs:
        print(f"No flight logs found in {log_directory}.")
        return []
    
    print(f"Available flight logs in {log_directory}:")
    [print(f"{i + 1}: {log_file.name}") for i, log_file in enumerate(flight_logs)]
    
    return flight_logs

class LogProvider(enum.Enum):
    Kitepower = 1
    Kitekraft = 2


@dataclass
class AWESModel:
    model_name: str
    log_provider: LogProvider


class AnalyzeAweFromCsvLog:
    class AnalysisMode(enum.Enum):
        Analyze   = 1
        PlotWind  = 2
        PlotAero  = 3
        PlotOther = 4

    def __init__(self, awes_model: AWESModel,
                 date: datetime,
                 analysis_mode: AnalysisMode,
                 log_directory: Path):
        self.config_data = load_config("examples/" + self.get_config_file_name(awes_model)) # Todo: In this function we should have a check if the config has all required data.
        self.update_configuration_for_execution(date.strftime("%Y-%m-%d"))

        self.log_provider = awes_model.log_provider

        self.analysis_mode = analysis_mode
        self.log_directory = log_directory

        self.run()

    @staticmethod
    def get_config_file_name(awes_model: AWESModel) -> str:
        config_file_mapping = {
            (LogProvider.Kitepower, 'v3'): 'v3_config.yaml',
            (LogProvider.Kitepower, 'v9'): 'v9_config.yaml',
            (LogProvider.Kitekraft, 'vX'): 'kft_config.yaml'
        }
        try:
            return config_file_mapping[(awes_model.log_provider, awes_model.model_name)]
        except KeyError:
            raise ValueError(
                f"No configuration file found for {awes_model.log_provider} with model {awes_model.model_name}")

    def update_configuration_for_execution(self, date: str) -> None:
        self.config_data['year'], self.config_data['month'], self.config_data['day'] = date.split('-')

    def pre_process_log(self) -> None:
        if self.log_provider == LogProvider.Kitepower:
            ProcessKpData(config_data=self.config_data, log_directory=self.log_directory)
        elif self.log_provider == LogProvider.Kitekraft:
            # Could add pre processing script here.
            print("Pre-processing for Kitekraft not implemented yet.")  
            pass
    def filter_by_time(self, start_minute: int, end_minute: int) -> None:
        """Filter the flight data by the given start and end minute."""
        #TODO: Frequency of the data is 10Hz, hardcoded
        if start_minute > 0 or end_minute > 0:
            if end_minute > 0 and end_minute <= len(self.flight_data/600):
                self.flight_data = self.flight_data.iloc[start_minute * 600 : end_minute * 600].reset_index(drop=True)
            else:
                self.flight_data = self.flight_data.iloc[start_minute * 600 :].reset_index(drop=True)
            print(f"Filtered data from minute {start_minute} to {end_minute}.")
        else:
            print("No time filtering applied.")

    def run_analysis(self) -> None:
        # Todo: This function could be split into multiple to mserve as a clearer documentation of how the ekf and
        #  simulation have to be used
        year = str(self.config_data["year"])
        month = str(self.config_data["month"])
        day = str(self.config_data["day"])
        kite_model = self.config_data["kite"]["model_name"]


        flight_data = read_processed_flight_data(year, month, day, kite_model)
        start_minute = int(input("Enter the start minute for analysis or skip: ").strip() or 0)
        end_minute = int(input("Enter the end minute for analysis or skip: ").strip() or 0)
        dt = flight_data["time"].diff().mean()
        
        if start_minute > 0 or end_minute > 0:
            if end_minute > 0 and end_minute <= len(flight_data)/60*dt:
                flight_data = flight_data.iloc[int(start_minute * 60/dt) : int(end_minute * 60/dt)].reset_index(drop=True)
            else:
                flight_data = flight_data.iloc[int(start_minute * 60/dt) :].reset_index(drop=True)
            print(f"Filtered data from minute {start_minute} to {end_minute}.")
        else:
            print("No time filtering applied.")


        # %% Initialize EKF
        simConfig = SimulationConfig(**self.config_data["simulation_parameters"])

        # Create system components
        kite = PointMassEKF(simConfig, **self.config_data["kite"])
        if self.config_data["kcu"]:
            kcu = KCU(**self.config_data["kcu"])
        else:
            kcu = None
        tether = Tether(kite, kcu, simConfig.obsData, **self.config_data["tether"])
        kite.calc_fx = kite.get_fx_fun(tether)
        
        tuningParams = TuningParameters(self.config_data["tuning_parameters"], simConfig)

       # Create input classes
        ekf_input_list = create_input_from_csv(
            flight_data, kite, kcu, tether, simConfig, kite_sensor=0
        )

        # Find initial state vector
        x0 = find_initial_state_vector(tether, ekf_input_list[0], simConfig)
        
        ekf, ekf_input_list = initialize_ekf(
            ekf_input_list, simConfig, tuningParams, x0, kite, kcu, tether
        )

        # %% Main loop
        ekf_output_list = []  # List of instances of EKFOutput
        start_time = time.time()
        mins = -1
        k_nis = 1000
        import numpy as np
        # TODO: Add a timeseries class
        for k, ekf_input in enumerate(ekf_input_list):
            # if k > 4000 and k< 5000:
            #     ekf_input.kite_position = np.array([np.nan, np.nan, np.nan])
            #     ekf_input.kite_velocity = np.array([np.nan, np.nan, np.nan])
            try:
                ekf, ekf_ouput = propagate_state_EKF(
                    ekf, ekf_input, simConfig, tether, kite, kcu
                )
                # Store results
                ekf_output_list.append(ekf_ouput)
            except Exception as e:
                print(e)
                try:
                    print("Integration error at iteration: ", k)
                    x0 = find_initial_state_vector(tether, ekf_input, simConfig)
                except:
                    print("Tether model error at iteration: ", k)
                    x0 = ekf.x_k1_k1
                    # continue
                ekf, ekf_input_list[k::] = initialize_ekf(
                    ekf_input_list[k::],
                    simConfig,
                    tuningParams,
                    x0,
                    kite,
                    kcu,
                    tether,
                    find_offsets=False,
                )
                flight_data.drop(k, inplace=True)
                continue

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
        ekf_output_df.dropna(subset=["kite_pitch"], inplace=True)
        ekf_output_df.reset_index(drop=True, inplace=True)
        rows_to_keep = ekf_output_df.index
        print(rows_to_keep)
        flight_data = flight_data.iloc[rows_to_keep]
        flight_data.reset_index(drop=True, inplace=True)
        indices = ekf_output_df.index
        flight_data = flight_data.iloc[indices]
        ekf_output_df, flight_data = postprocess_results(
            ekf_output_df,
            flight_data,
            kite,
            kcu,
            self.config_data,
        )
        # %% Store results
        save_results(ekf_output_df, flight_data, kite_model, year, month, day, self.config_data, addition="")

    def run(self):
        if self.analysis_mode == AnalyzeAweFromCsvLog.AnalysisMode.Analyze:
            self.pre_process_log()
            self.run_analysis()
        elif self.analysis_mode == AnalyzeAweFromCsvLog.AnalysisMode.PlotWind:
            plot_wind_results(self.config_data)
            plt.show()
        elif self.analysis_mode == AnalyzeAweFromCsvLog.AnalysisMode.PlotAero:
            plot_kite_aero(self.config_data) #Todo: doesn't seem to work for all v9 files
            plt.show()
        elif self.analysis_mode == AnalyzeAweFromCsvLog.AnalysisMode.PlotOther:
            plot_kite_orientation(self.config_data)
            plot_kite_trajectories(self.config_data)  # Todo: doesn't seem to work for all v9 files
            plt.show()


def get_awes_model_from_string(awes_model_str: str) -> AWESModel:
    if awes_model_str.startswith('kp'):
        log_provider = LogProvider.Kitepower
    elif awes_model_str.startswith('kft'):
        log_provider = LogProvider.Kitekraft
    else:
        raise ValueError(f"Invalid awes_model_str: {awes_model_str}")

    parts = awes_model_str.split('-')
    if len(parts) < 2:
        raise ValueError(f"Invalid awes_model_str format: {awes_model_str}")

    model_name = parts[1]
    return AWESModel(model_name=model_name, log_provider=log_provider)


def main() -> None:
    default_model_str = 'kp-v9'
    default_date = datetime.strptime('2023-11-27', '%Y-%m-%d').date()
    default_run_option = 'analyze'
    default_log_dir = Path('./data/v9/')

    log_dir = Path(
        input(f"Enter the directory with the flight logs [default: {default_log_dir}]: ").strip() or default_log_dir)
    
    # List available flights and select one
    available_flights = list_available_flights(log_dir)
    if not available_flights:
        sys.exit("No flights available to analyze.")
    
    selected_flight_index = int(input(f"Select a flight to analyze (1-{len(available_flights)}): ")) - 1
    selected_flight = available_flights[selected_flight_index]
    print(f"Selected flight: {selected_flight.name}")

    # Extract the date part from the filename
    filename_parts = selected_flight.stem.split('_')
    date_str = filename_parts[0]  # "2021-10-07"
    time_str = filename_parts[1]  # "19-38-15"

    # Convert the date string to a datetime object
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Query for the system model
    valid_model_str = ['kp-v3', 'kp-v9', 'kft']
    awes_model_str = input(
        f"Enter the system model (options: {', '.join(valid_model_str)}) [default: {default_model_str}]: ").strip()
    if not awes_model_str:
        awes_model_str = default_model_str
    if awes_model_str not in valid_model_str:
        print(f"Error: Invalid system model. Valid options are: {', '.join(valid_model_str)}")
        sys.exit(1)
    awes_model = get_awes_model_from_string(awes_model_str)

    # Query for the run option
    valid_options = ['analyze', 'plot-wind', 'plot-aero', 'plot-other']
    run_option = input(
        f"Enter the run option (options: {', '.join(valid_options)}) [default: {default_run_option}]: ").strip()
    if not run_option:
        run_option = default_run_option
    if run_option not in valid_options:
        print(f"Error: Invalid run option. Valid options are: {', '.join(valid_options)}")
        sys.exit(1)

    if run_option == 'analyze':
        analysis_mode = AnalyzeAweFromCsvLog.AnalysisMode.Analyze
    elif run_option == 'plot-wind':
        analysis_mode = AnalyzeAweFromCsvLog.AnalysisMode.PlotWind
    elif run_option == 'plot-aero':
        analysis_mode = AnalyzeAweFromCsvLog.AnalysisMode.PlotAero
    elif run_option == 'plot-other':
        analysis_mode = AnalyzeAweFromCsvLog.AnalysisMode.PlotOther

    

    print("Starting analysis with:")
    print(f"System Model: {awes_model_str}")
    print(f"Date: {date_str}, Time: {time_str}")
    print(f"Run Option: {run_option}")
    print(f"Log directory: {log_dir}")
    
    AnalyzeAweFromCsvLog(awes_model=awes_model,
                         date=date,
                         analysis_mode=analysis_mode,
                         log_directory=log_dir)


if __name__ == "__main__":
    main()
