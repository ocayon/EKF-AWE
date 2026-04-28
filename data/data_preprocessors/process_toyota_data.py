import numpy as np
import pandas as pd
import os
from pathlib import Path
from awes_ekf.utils import llh_to_enu


def load_log_file(log_directory: Path, log_date: str) -> pd.DataFrame:
    all_logs = os.listdir(log_directory)
    log_path = ""
    for log in all_logs:
        if log.startswith(log_date):
            log_path = f"{log_directory}/{log}"
            break
    delimiter = detect_delimiter(log_path)
    # print(f"Loading log file: {log_path}")
    log = pd.read_csv(log_path, delimiter=delimiter, low_memory=False)
    # log = log[log["kite_height"] > 50]  # Select indexes where kite is flying
    return log


def detect_delimiter(file_path: str) -> str:
    with open(file_path, "r") as file:
        first_line = file.readline()
        if "," in first_line:
            return ","
        elif " " in first_line:
            return " "
        else:
            return ","  # Default to comma if neither is detected


def save_flight_data(
    flight_data: pd.DataFrame, config_data: dict, log_date: str
) -> None:
    model = config_data["kite"]["model_name"]
    csv_filepath = f"./processed_data/flight_data/{model}/"
    os.makedirs(csv_filepath, exist_ok=True)
    csv_filename = f"{model}_{log_date}.csv"
    flight_data.to_csv(os.path.join(csv_filepath, csv_filename), index=False)


def process_data(config_data: dict, log_directory: Path) -> pd.DataFrame:
    log_date = f'{config_data["year"]}-{config_data["month"]}-{config_data["day"]}'
    log = load_log_file(log_directory, log_date)
    window_size = 20
    dt = log["time"].iloc[1] - log["time"].iloc[0]
    log = log.reset_index()
    log.loc[:, log.select_dtypes(include=[float, int]).columns] = log.select_dtypes(
        include=[float, int]
    ).interpolate()

    flight_data = log

    save_flight_data(flight_data, config_data, log_date)
