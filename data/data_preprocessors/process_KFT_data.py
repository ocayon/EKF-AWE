import numpy as np
import pandas as pd
from pathlib import Path
import os
from awes_ekf.utils import Rx, Ry, Rz

def load_and_filter_data(log_directory: Path, log_date: str) -> pd.DataFrame:
    # Search for a file with matching date in the directory
    all_logs = os.listdir(log_directory)
    log_path = ""
    for log in all_logs:
        if log.startswith(log_date):
            log_path = log_directory / log
            break
    df = pd.read_csv(log_path, delimiter=',')
    
    # Filter rows based on kite position threshold
    df = df[df["positionKiteIdentified_2"] > 35]
    df = df.reset_index(drop=True)
    return df

def calculate_thrust_force(df: pd.DataFrame) -> np.ndarray:
    thrust_force_vector = []
    for i in range(len(df['time'])):
        R = Rz(df['eulerAnglesKiteIdentified_2'][i]) @ \
            Ry(df['eulerAnglesKiteIdentified_1'][i]) @ \
            Rx(df['eulerAnglesKiteIdentified_0'][i]) @ \
            Ry(np.deg2rad(90))
        ex = R[:, 0]
        thrust_force_vector.append(-ex * df['thrustSet'][i])
    return np.array(thrust_force_vector)

def save_flight_data(flight_data: pd.DataFrame, model: str, log_date: str) -> None:
    csv_filepath = f"./processed_data/flight_data/{model}/"
    os.makedirs(csv_filepath, exist_ok=True)
    csv_filename = f"{model}_{log_date}.csv"
    flight_data.to_csv(os.path.join(csv_filepath, csv_filename), index=False)

def process_data(config_data: dict, log_directory: Path) -> pd.DataFrame:
    log_date = f'{config_data["year"]}-{config_data["month"]}-{config_data["day"]}'
    model = config_data["kite"]["model_name"]

    # Load and filter data
    df = load_and_filter_data(log_directory, log_date)
    flight_data = pd.DataFrame()

    # Add kite position data
    flight_data['kite_position_x'] = df['positionKiteIdentified_0']
    flight_data['kite_position_y'] = df['positionKiteIdentified_1']
    flight_data['kite_position_z'] = df['positionKiteIdentified_2']

    # Velocity data
    flight_data['kite_velocity_x'] = df['velocityKiteIdentifiedByXsens_0']
    flight_data['kite_velocity_y'] = df['velocityKiteIdentifiedByXsens_1']
    flight_data['kite_velocity_z'] = df['velocityKiteIdentifiedByXsens_2']

    # Acceleration data
    flight_data['kite_acceleration_x'] = df['accelerationMeasuredInHoverFrame_0']
    flight_data['kite_acceleration_y'] = df['accelerationMeasuredInHoverFrame_1']
    flight_data['kite_acceleration_z'] = df['accelerationMeasuredInHoverFrame_2']

    # Thrust force data
    thrust_force_vector = calculate_thrust_force(df)
    flight_data['thrust_force_x'], flight_data['thrust_force_y'], flight_data['thrust_force_z'] = thrust_force_vector.T

    # Ground station data
    flight_data['ground_tether_force'] = df['forceTetherMeasured']
    flight_data['tether_length'] = df['lengthTetherRolledOutIdentified']
    flight_data['tether_reelout_speed'] = df['speedWinchIdentified']
    flight_data['ground_wind_velocity'] = df['velocityWindMagnitudeAtGroundMeasuredFused']
    flight_data['ground_wind_direction'] = df['velocityWindAzimuthAtGroundMeasuredFused']

    # Time data
    flight_data['time'] = df['time']

    # Save the processed data
    save_flight_data(flight_data, model, log_date)

# Example main function to run the process_data function
def main() -> None:
    config = {
        "year": "2024",
        "month": "06",
        "day": "07",
        "kite": {"model_name": "kitekraft", "sensor_ids": [0]},
        "kcu": {"sensor_ids": []}
    }
    
    log_dir = Path(f'data/{config["kite"]["model_name"]}/')
    process_data(config_data=config, log_directory=log_dir)

if __name__ == "__main__":
    main()
