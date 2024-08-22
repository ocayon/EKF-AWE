import numpy as np
import pandas as pd
import os
from pathlib import Path


class ProcessKpData:
    def __init__(self, config_data: dict, log_directory: Path):
        self.config_data = config_data
        self.log_date = f'{self.config_data["year"]}-{self.config_data["month"]}-{self.config_data["day"]}'

        log = self.load_log_file(log_directory)
        processed_data = self.process_KP_data(log)
        self.save_flight_data(processed_data)

    def load_log_file(self, log_directory: Path) -> pd.DataFrame:
        all_logs = os.listdir(log_directory)
        log_path = ""
        for log in all_logs:
            if log.startswith(self.log_date):
                log_path = f"{log_directory}/{log}"
                break

        log = pd.read_csv(log_path, delimiter=" ", low_memory=False)
        log = log[
            log["kite_height"] > 80
        ]  # Select the indexes where the kite is flying
        return log

    def fuse_sensor_data(self, log: pd.DataFrame, sensors: list, prefix: str, dt: float, window_size: int) -> dict:
        fused_data = {}

        if not sensors:  # Check if the sensor list is empty or None
            return fused_data

        # Initialize sums for fused velocity and acceleration
        fused_velocity_x = 0
        fused_velocity_y = 0
        fused_velocity_z = 0

        valid_sensors = [sensor for sensor in sensors if sensor is not None]

        for sensor in valid_sensors:
            if f"kite_{sensor}_vy" in log.columns:
                # Velocity data
                fused_velocity_x += log[f"kite_{sensor}_vy"]
                fused_velocity_y += log[f"kite_{sensor}_vx"]
                fused_velocity_z += -log[f"kite_{sensor}_vz"]

        if not valid_sensors:
            return fused_data

        # Average the fused velocity components
        num_sensors = len(valid_sensors)
        fused_velocity_x /= num_sensors
        fused_velocity_y /= num_sensors
        fused_velocity_z /= num_sensors

        # Save the fused velocity data
        fused_data[f"{prefix}_velocity_x"] = fused_velocity_x
        fused_data[f"{prefix}_velocity_y"] = fused_velocity_y
        fused_data[f"{prefix}_velocity_z"] = fused_velocity_z

        # Calculate and fuse acceleration
        accelerations_x = []
        accelerations_y = []
        accelerations_z = []

        for sensor in valid_sensors:
            if f"kite_{sensor}_ay" in log.columns and not log[f"kite_{sensor}_ay"].isnull().all():
                accelerations_x.append(log[f"kite_{sensor}_ay"])
                accelerations_y.append(log[f"kite_{sensor}_ax"])
                accelerations_z.append(-log[f"kite_{sensor}_az"])

        if accelerations_x:
            # If there are valid accelerations, average them
            fused_acceleration_x = sum(accelerations_x) / len(accelerations_x)
            fused_acceleration_y = sum(accelerations_y) / len(accelerations_y)
            fused_acceleration_z = sum(accelerations_z) / len(accelerations_z)
        else:
            # If no valid acceleration data, differentiate the velocity
            ax = np.gradient(fused_velocity_x, dt)
            ay = np.gradient(fused_velocity_y, dt)
            az = np.gradient(fused_velocity_z, dt)
            
            # Smooth the acceleration data
            fused_acceleration_x = np.convolve(ax, np.ones(window_size) / window_size, mode="same")
            fused_acceleration_y = np.convolve(ay, np.ones(window_size) / window_size, mode="same")
            fused_acceleration_z = np.convolve(az, np.ones(window_size) / window_size, mode="same")

        # Save the fused acceleration data
        fused_data[f"{prefix}_acceleration_x"] = fused_acceleration_x
        fused_data[f"{prefix}_acceleration_y"] = fused_acceleration_y
        fused_data[f"{prefix}_acceleration_z"] = fused_acceleration_z

        return fused_data


    def add_orientation_data(self, log: pd.DataFrame, sensors: list, prefix: str, flight_data: pd.DataFrame):
        valid_sensors = [sensor for sensor in sensors if sensor is not None]

        if not valid_sensors:  # Check if the sensor list is empty or None
            return

        for sensor in valid_sensors:
            # Euler angles data
            flight_data[f"{prefix}_pitch_{sensor}"] = np.deg2rad(log[f"kite_{sensor}_pitch"])
            flight_data[f"{prefix}_roll_{sensor}"] = np.deg2rad(log[f"kite_{sensor}_roll"])
            flight_data[f"{prefix}_yaw_{sensor}"] = np.deg2rad(log[f"kite_{sensor}_yaw"])

            # Angular velocity data
            if log[f"kite_{sensor}_yaw_rate"].isnull().all():
                dt = log["time"].iloc[1] - log["time"].iloc[0]
                roll_rate = np.gradient(flight_data[f"{prefix}_roll_{sensor}"], dt)
                pitch_rate = np.gradient(flight_data[f"{prefix}_pitch_{sensor}"], dt)
                yaw_rate = np.gradient(flight_data[f"{prefix}_yaw_{sensor}"], dt)

                # Smooth the angular velocity data
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
                flight_data[f"{prefix}_roll_rate_{sensor}"] = log[f"kite_{sensor}_roll_rate"]
                flight_data[f"{prefix}_pitch_rate_{sensor}"] = log[f"kite_{sensor}_pitch_rate"]
                flight_data[f"{prefix}_yaw_rate_{sensor}"] = log[f"kite_{sensor}_yaw_rate"]

    def process_KP_data(self, log: pd.DataFrame) -> pd.DataFrame:
        # Smooth radius
        window_size = 20
        dt = log["time"].iloc[1] - log["time"].iloc[0]  # Time step
        log = log.reset_index()  # Reset the index
        log = log.interpolate()  # Interpolate the missing data

        flight_data = pd.DataFrame()  # Create a new dataframe for the flight data

        # Add position data
        flight_data["kite_position_x"] = log["kite_pos_east"]
        flight_data["kite_position_y"] = log["kite_pos_north"]
        flight_data["kite_position_z"] = log["kite_height"]

        # Fuse and add kite sensor data
        kite_sensors = self.config_data["kite"]["sensor_ids"]
        fused_kite_data = self.fuse_sensor_data(log, kite_sensors, "kite", dt, window_size)
        flight_data = flight_data.assign(**fused_kite_data)

        # Add kite orientation data (roll, pitch, yaw and their rates)
        self.add_orientation_data(log, kite_sensors, "kite", flight_data)

        # Fuse and add KCU sensor data (only if sensors are defined)
        kcu_sensors = self.config_data["kcu"].get("sensor_ids", [])
        fused_kcu_data = self.fuse_sensor_data(log, kcu_sensors, "kcu", dt, window_size)
        flight_data = flight_data.assign(**fused_kcu_data)

        # Add KCU orientation data (roll, pitch, yaw and their rates, only if sensors are defined)
        self.add_orientation_data(log, kcu_sensors, "kcu", flight_data)

        # Add the ground station data
        flight_data["ground_tether_force"] = (
            log["ground_tether_force"] * 9.81
        )  # Convert to N
        flight_data["ground_wind_speed"] = log["ground_wind_velocity"]
        flight_data["ground_wind_direction"] = (
            360 - 90 - log["ground_upwind_direction"]
        )  # Convert from NED clockwise to ENU counter-clockwise
        flight_data["tether_length"] = log[
            "ground_tether_length"
        ]  # Tether length
        flight_data["tether_reelout_speed"] = log[
            "ground_tether_reelout_speed"
        ]  # Tether reelout speed

        # Add the KCU control data
        flight_data["kcu_set_depower"] = log["kite_set_depower"]
        flight_data["kcu_set_steering"] = log["kite_set_steering"]
        flight_data["kcu_actual_steering"] = log["kite_actual_steering"]
        flight_data["kcu_actual_depower"] = log["kite_actual_depower"]

        # Add the airspeed data
        flight_data["kite_apparent_windspeed"] = log["airspeed_apparent_windspeed"]
        flight_data["bridle_angle_of_attack"] = log["airspeed_angle_of_attack"]
        if self.config_data["kite"]["model_name"] == "v9":
            flight_data["bridle_sideslip_angle"] = log["airspeed_sideslip_angle"]
        flight_data["kite_airspeed_temperature"] = log["airspeed_temperature"]

        kite_radius = np.linalg.norm(
            flight_data[
                ["kite_position_x", "kite_position_y", "kite_position_z"]
            ],
            axis=1,
        )

        print(
            "Date: ",
            log["date"].iloc[0],
            "Flight length: ",
            round(len(flight_data) / 10 / 60, 1),
            "min",
        )

        # Add the time data
        flight_data["time"] = log["time"] - log["time"].iloc[0]
        flight_data["time_of_day"] = log["time_of_day"]
        flight_data["unix_time"] = log["time"]
        flight_data["date"] = log["date"]

        # Add azimuth and elevation
        flight_data["kite_azimuth"] = log["kite_azimuth"]
        flight_data["kite_elevation"] = log["kite_elevation"]

        # Add time of day
        flight_data["time_of_day"] = log["time_of_day"]  # Seems double with line 159

        columns_to_copy = []
        flight_data_add = False
        for column in log.columns:
            if "Wind " in column:
                columns_to_copy.append(column)
            if "Z-wind" in column:
                columns_to_copy.append(column)
            if "load_cell" in column:
                columns_to_copy.append(column)
        if len(columns_to_copy) > 0:
            flight_data_add = pd.concat([log[col] for col in columns_to_copy], axis=1)
            flight_data = pd.concat([flight_data, flight_data_add], axis=1)

        return flight_data

    def save_flight_data(self, flight_data: pd.DataFrame) -> None:
        # Todo make this a function in an abstract parent class. Only there we need to know teh details of how to save
        #  the data.

        model = self.config_data["kite"]["model_name"]

        # Create the directory if it doesn't exist
        csv_filepath = "./processed_data/flight_data/" + model + "/"
        if not os.path.exists(csv_filepath):
            os.makedirs(csv_filepath)

        csv_filename = model + "_" + self.log_date + ".csv"
        flight_data.to_csv(csv_filepath + csv_filename, index=False)


def main() -> None:
    config = {
        "year": "2019",
        "month": "10",
        "day": "08",
        "kite": {"model_name": "v3", "sensor_id": [0, 1]},
        "kcu": {"sensor_id": []},  # No KCU sensors in this case
    }

    log_dir = Path(f'data/{config["kite"]["model_name"]}/')

    ProcessKpData(config_data=config, log_directory=log_dir)


if __name__ == "__main__":
    main()
