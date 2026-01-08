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
    log = pd.read_csv(log_path, delimiter=delimiter, low_memory=False)
    log = log[log["kite_height"] > 80]  # Select indexes where kite is flying
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


def fuse_sensor_data(
    log: pd.DataFrame, sensors: list, prefix: str, dt: float, window_size: int
) -> dict:
    fused_data = {}
    if not sensors:
        return fused_data

    fused_velocity_x, fused_velocity_y, fused_velocity_z = 0, 0, 0
    valid_sensors = [sensor for sensor in sensors if sensor is not None]

    for sensor in valid_sensors:
        if f"kite_{sensor}_vy" in log.columns:
            fused_velocity_x += log[f"kite_{sensor}_vy"]
            fused_velocity_y += log[f"kite_{sensor}_vx"]
            fused_velocity_z += -log[f"kite_{sensor}_vz"]

    if valid_sensors:
        fused_velocity_x /= len(valid_sensors)
        fused_velocity_y /= len(valid_sensors)
        fused_velocity_z /= len(valid_sensors)

    fused_data.update(
        {
            f"{prefix}_velocity_x": fused_velocity_x,
            f"{prefix}_velocity_y": fused_velocity_y,
            f"{prefix}_velocity_z": fused_velocity_z,
        }
    )

    # Calculate acceleration if available; otherwise, use velocity gradients
    accelerations_x, accelerations_y, accelerations_z = [], [], []
    for sensor in valid_sensors:
        if (
            f"kite_{sensor}_ay" in log.columns
            and not log[f"kite_{sensor}_ay"].isnull().all()
        ):
            accelerations_x.append(log[f"kite_{sensor}_ay"])
            accelerations_y.append(log[f"kite_{sensor}_ax"])
            accelerations_z.append(-log[f"kite_{sensor}_az"])

    if accelerations_x:
        fused_data.update(
            {
                f"{prefix}_acceleration_x": sum(accelerations_x) / len(accelerations_x),
                f"{prefix}_acceleration_y": sum(accelerations_y) / len(accelerations_y),
                f"{prefix}_acceleration_z": sum(accelerations_z) / len(accelerations_z),
            }
        )
    else:
        # Smooth gradients to approximate acceleration
        ax, ay, az = (
            np.gradient(fused_velocity_x, dt),
            np.gradient(fused_velocity_y, dt),
            np.gradient(fused_velocity_z, dt),
        )
        # Replace NaN and Inf values with 0
        ax = np.nan_to_num(ax, nan=0.0, posinf=0.0, neginf=0.0)
        ay = np.nan_to_num(ay, nan=0.0, posinf=0.0, neginf=0.0)
        az = np.nan_to_num(az, nan=0.0, posinf=0.0, neginf=0.0)

        fused_data.update(
            {
                f"{prefix}_acceleration_x": np.convolve(
                    ax, np.ones(window_size) / window_size, mode="same"
                ),
                f"{prefix}_acceleration_y": np.convolve(
                    ay, np.ones(window_size) / window_size, mode="same"
                ),
                f"{prefix}_acceleration_z": np.convolve(
                    az, np.ones(window_size) / window_size, mode="same"
                ),
            }
        )

    return fused_data


def add_orientation_data(
    log: pd.DataFrame, sensors: list, prefix: str, flight_data: pd.DataFrame
):
    valid_sensors = [sensor for sensor in sensors if sensor is not None]
    if not valid_sensors:
        return

    for sensor in valid_sensors:
        flight_data[f"{prefix}_pitch_{sensor}"] = np.deg2rad(
            log[f"kite_{sensor}_pitch"]
        )
        flight_data[f"{prefix}_roll_{sensor}"] = np.deg2rad(log[f"kite_{sensor}_roll"])
        flight_data[f"{prefix}_yaw_{sensor}"] = np.deg2rad(log[f"kite_{sensor}_yaw"])

        # Compute or copy roll, pitch, and yaw rates if available
        roll_rate_col = f"kite_{sensor}_roll_rate"
        pitch_rate_col = f"kite_{sensor}_pitch_rate"
        yaw_rate_col = f"kite_{sensor}_yaw_rate"

        if all(
            col in log.columns for col in [roll_rate_col, pitch_rate_col, yaw_rate_col]
        ):
            if log[yaw_rate_col].isnull().all():
                dt = log["time"].iloc[1] - log["time"].iloc[0]
                roll_rate = np.gradient(flight_data[f"{prefix}_roll_{sensor}"], dt)
                pitch_rate = np.gradient(flight_data[f"{prefix}_pitch_{sensor}"], dt)
                yaw_rate = np.gradient(flight_data[f"{prefix}_yaw_{sensor}"], dt)
                # Replace NaN and Inf values with 0
                roll_rate = np.nan_to_num(roll_rate, nan=0.0, posinf=0.0, neginf=0.0)
                pitch_rate = np.nan_to_num(pitch_rate, nan=0.0, posinf=0.0, neginf=0.0)
                yaw_rate = np.nan_to_num(yaw_rate, nan=0.0, posinf=0.0, neginf=0.0)
                window_size = 20
                flight_data[f"{prefix}_roll_rate_{sensor}"] = np.convolve(
                    roll_rate, np.ones(window_size) / window_size, mode="same"
                )
                flight_data[f"{prefix}_pitch_rate_{sensor}"] = np.convolve(
                    pitch_rate, np.ones(window_size) / window_size, mode="same"
                )
                flight_data[f"{prefix}_yaw_rate_{sensor}"] = np.convolve(
                    yaw_rate, np.ones(window_size) / window_size, mode="same"
                )
            else:
                flight_data[f"{prefix}_roll_rate_{sensor}"] = log[roll_rate_col]
                flight_data[f"{prefix}_pitch_rate_{sensor}"] = log[pitch_rate_col]
                flight_data[f"{prefix}_yaw_rate_{sensor}"] = log[yaw_rate_col]
        else:
            # If any of the rate columns are missing, compute from orientation
            dt = log["time"].iloc[1] - log["time"].iloc[0]
            roll_rate = np.gradient(flight_data[f"{prefix}_roll_{sensor}"], dt)
            pitch_rate = np.gradient(flight_data[f"{prefix}_pitch_{sensor}"], dt)
            yaw_rate = np.gradient(flight_data[f"{prefix}_yaw_{sensor}"], dt)
            # Replace NaN and Inf values with 0
            roll_rate = np.nan_to_num(roll_rate, nan=0.0, posinf=0.0, neginf=0.0)
            pitch_rate = np.nan_to_num(pitch_rate, nan=0.0, posinf=0.0, neginf=0.0)
            yaw_rate = np.nan_to_num(yaw_rate, nan=0.0, posinf=0.0, neginf=0.0)
            window_size = 20
            flight_data[f"{prefix}_roll_rate_{sensor}"] = np.convolve(
                roll_rate, np.ones(window_size) / window_size, mode="same"
            )
            flight_data[f"{prefix}_pitch_rate_{sensor}"] = np.convolve(
                pitch_rate, np.ones(window_size) / window_size, mode="same"
            )
            flight_data[f"{prefix}_yaw_rate_{sensor}"] = np.convolve(
                yaw_rate, np.ones(window_size) / window_size, mode="same"
            )


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

    flight_data = pd.DataFrame()
    try:
        ref_lat, ref_lon, ref_alt = 54.126469, -9.781307, 12.7
        lat, lon, alt = (
            log["gps_log_lat"] / 1e7,
            log["gps_log_lon"] / 1e7,
            log["gps_log_alt"] / 1000,
        )
        east, north, up = llh_to_enu(ref_lat, ref_lon, ref_alt, lat, lon, alt)
        (
            flight_data["kite_position_x"],
            flight_data["kite_position_y"],
            flight_data["kite_position_z"],
        ) = (east, north, up)
        raise Exception
    except:
        flight_data["kite_position_x"] = log["kite_pos_east"]
        flight_data["kite_position_y"] = log["kite_pos_north"]
        flight_data["kite_position_z"] = log["kite_height"]

    # Kite velocity and acceleration
    kite_sensors = config_data["kite"]["sensor_ids"]
    try:
        (
            flight_data["kite_velocity_x"],
            flight_data["kite_velocity_y"],
            flight_data["kite_velocity_z"],
        ) = (
            log["gps_log_vel_e_m_s"],
            log["gps_log_vel_n_m_s"],
            -log["gps_log_vel_d_m_s"],
        )
    except:
        fused_kite_data = fuse_sensor_data(log, kite_sensors, "kite", dt, window_size)
        flight_data = flight_data.assign(**fused_kite_data)

    add_orientation_data(log, kite_sensors, "kite", flight_data)

    kcu_sensors = config_data["kcu"].get("sensor_ids", [])
    if kcu_sensors:
        add_orientation_data(log, kcu_sensors, "kcu", flight_data)

    # Ground station data
    flight_data["ground_tether_force"] = log["ground_tether_force"] * 9.81
    flight_data["ground_wind_speed"] = log["ground_wind_velocity"]
    flight_data["ground_wind_direction"] = 360 - 90 - log["ground_upwind_direction"]
    flight_data["tether_length"] = log["ground_tether_length"] + 15.45
    flight_data["tether_reelout_speed"] = log["ground_tether_reelout_speed"]

    # KCU control data
    try:
        flight_data["kcu_set_depower"] = log["kite_set_depower"]
        flight_data["kcu_set_steering"] = log["kite_set_steering"]
    except KeyError:
        flight_data["kcu_set_depower"] = np.zeros(len(log))
        flight_data["kcu_set_steering"] = np.zeros(len(log))

    flight_data["kcu_actual_steering"] = log["kite_actual_steering"]
    flight_data["kcu_actual_depower"] = log["kite_actual_depower"]

    # Airspeed and kite-specific data
    coeff_a = 1.0638463337431572
    coeff_b = -0.41490555652632466
    try:
        flight_data["kite_apparent_windspeed"] = (
            log["airspeed_apparent_windspeed"] * coeff_a + coeff_b
        )
    except KeyError:
        flight_data["kite_apparent_windspeed"] = np.zeros(len(log))
    try:
        flight_data["bridle_angle_of_attack"] = log["airspeed_angle_of_attack"]
    except KeyError:
        flight_data["bridle_angle_of_attack"] = np.zeros(len(log))
    if config_data["kite"]["model_name"] == "v9":
        flight_data["bridle_sideslip_angle"] = log["airspeed_sideslip_angle"]

    try:
        flight_data["kite_airspeed_temperature"] = log["airspeed_temperature"]
    except KeyError:
        flight_data["kite_airspeed_temperature"] = np.zeros(len(log))

    flight_data["kite_heading"] = log["kite_heading"]
    flight_data["kite_azimuth"] = -log["kite_azimuth"]

    # Time and date data
    flight_data["time"] = log["time"] - log["time"].iloc[0]
    flight_data["time_of_day"] = log["time_of_day"]
    flight_data["unix_time"] = log["time"]
    flight_data["date"] = log["date"]

    # import matplotlib.pyplot as plt
    # elevation_from_position = np.arctan2(flight_data["kite_position_z"], np.sqrt(flight_data["kite_position_x"]**2 + flight_data["kite_position_y"]**2))
    # plt.plot(log["time"], elevation_from_position*180/np.pi, label="Elevation from position")
    # plt.show()

    # Tether azimuth and elevation
    try:
        flight_data["tether_azimuth_ground"] = -np.unwrap(
            log["ground_tether_fleet_angle_rad"]
        )
        flight_data["tether_elevation_ground"] = log[
            "ground_tether_elevation_angle_rad"
        ]
    except:
        flight_data["tether_azimuth_ground"] = log["kite_azimuth"]
        flight_data["tether_elevation_ground"] = log["kite_elevation"]

    try:
        flight_data["sensor_tether_force"] = log["ground_tlb_net_weight"] * 9.81
    except:
        flight_data["sensor_tether_force"] = np.zeros(len(log))

    # Additional fields
    columns_to_copy = [
        col
        for col in log.columns
        if "Wind " in col or "Z-wind" in col or "load_cell" in col
    ]
    if columns_to_copy:
        flight_data = pd.concat([flight_data, log[columns_to_copy]], axis=1)

    try:
        flight_data["kite_course"] = log["kite_course"]
    except:
        pass

    try:
        flight_data["kite_elevation"] = log["kite_elevation"]
    except:
        pass

    try:
        flight_data["tether_elevation_ground"] = (
            (log["elevation_encoder_value"]) * 2 * np.pi / 8192
        )
        print("Tether elevation from encoder")
    except:
        flight_data["tether_elevation_ground"] = np.zeros(len(log))

    try:
        flight_data["flight_phase_index"] = log["flight_phase_index"]
    except KeyError:
        flight_data["flight_phase_index"] = np.zeros(len(log))

    # from awes_ekf.utils import raw_force_to_tether_force
    # tether_force = raw_force_to_tether_force(flight_data["sensor_tether_force"],elevation_from_encoder)S
    import matplotlib.pyplot as plt

    plt.plot(
        log["time"],
        flight_data["tether_elevation_ground"] * 180 / np.pi,
        label="Tether elevation from encoder",
    )
    # plt.show()

    # plt.figure()
    # plt.plot(log["time"])
    # plt.show()

    print(flight_data.columns)
    print("----------------------------")
    print(log.columns)
    # plt.plot(log["time"], elevation_from_encoder)
    # plt.plot(log["time"], flight_data["tether_elevation_ground"])
    # plt.show()
    save_flight_data(flight_data, config_data, log_date)


if __name__ == "__main__":
    from awes_ekf.setup import SimulationConfig, load_config

    config = load_config()
    simConfig = SimulationConfig(**config["simulation_parameters"])
    log_dir = Path(f'data/{config["kite"]["model_name"]}/')
    process_data(config_data=config, log_directory=log_dir)
