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
        log_path = ''
        for log in all_logs:
            if log.startswith(self.log_date):
                log_path = f'{log_directory}/{log}'
                break
                    
        log = pd.read_csv(log_path, delimiter=" ", low_memory=False)
        log = log[log["kite_height"] > 80]  # Select the indexes where the kite is flying
        return log

    def process_KP_data(self, log: pd.DataFrame) -> pd.DataFrame:
        # Smooth radius
        window_size = 20
        dt = log["time"].iloc[1] - log["time"].iloc[0]  # Time step
        log = log.reset_index()  # Reset the index
        log = log.interpolate()  # Interpolate the missing data

        flight_data = pd.DataFrame()  # Create a new dataframe for the flight data

        sensors = [0, 1]
        # %% Add the data to the flight data dataframe
        # Add position data
        flight_data["kite_position_east"] = log["kite_pos_east"]
        flight_data["kite_position_north"] = log["kite_pos_north"]
        flight_data["kite_position_up"] = log["kite_height"]

        for sensor in sensors:
            # Velocity data
            flight_data["kite_velocity_east_s" + str(sensor)] = log[
                "kite_" + str(sensor) + "_vy"
            ]
            flight_data["kite_velocity_north_s" + str(sensor)] = log[
                "kite_" + str(sensor) + "_vx"
            ]
            flight_data["kite_velocity_up_s" + str(sensor)] = -log[
                "kite_" + str(sensor) + "_vz"
            ]
            # Euler angles data
            flight_data["kite_pitch_s" + str(sensor)] = np.deg2rad(
                log["kite_" + str(sensor) + "_pitch"]
            )
            flight_data["kite_roll_s" + str(sensor)] = np.deg2rad(
                log["kite_" + str(sensor) + "_roll"]
            )
            flight_data["kite_yaw_s" + str(sensor)] = np.deg2rad(
                log["kite_" + str(sensor) + "_yaw"]
            )
            # Acceleration data
            if log["kite_" + str(sensor) + "_ax"].isnull().all():
                # Differentiate velocity to get acceleration
                ax = np.diff(flight_data["kite_velocity_east_s" + str(sensor)]) / dt
                ay = np.diff(flight_data["kite_velocity_north_s" + str(sensor)]) / dt
                az = np.diff(flight_data["kite_velocity_up_s" + str(sensor)]) / dt
                # Add the last value as 0 to keep the same length
                flight_data["kite_acceleration_east_s" + str(sensor)] = np.concatenate(
                    (ax, [0])
                )
                flight_data["kite_acceleration_north_s" + str(sensor)] = np.concatenate(
                    (ay, [0])
                )
                flight_data["kite_acceleration_up_s" + str(sensor)] = np.concatenate(
                    (az, [0])
                )
                # Smooth the acceleration data
                flight_data["kite_acceleration_east_s" + str(sensor)] = np.convolve(
                    flight_data["kite_acceleration_east_s" + str(sensor)],
                    np.ones(window_size) / window_size,
                    mode="same",
                )
                flight_data["kite_acceleration_north_s" + str(sensor)] = np.convolve(
                    flight_data["kite_acceleration_north_s" + str(sensor)],
                    np.ones(window_size) / window_size,
                    mode="same",
                )
                flight_data["kite_acceleration_up_s" + str(sensor)] = np.convolve(
                    flight_data["kite_acceleration_up_s" + str(sensor)],
                    np.ones(window_size) / window_size,
                    mode="same",
                )
            else:
                flight_data["kite_acceleration_east_s" + str(sensor)] = log["kite_0_ay"]
                flight_data["kite_acceleration_north_s" + str(sensor)] = log["kite_0_ax"]
                flight_data["kite_acceleration_up_s" + str(sensor)] = -log["kite_0_az"]

            # Add angular velocity data
            if log["kite_" + str(sensor) + "_yaw_rate"].isnull().all():
                # Differentiate orientation to get angular velocity
                roll_rate = np.diff(flight_data["kite_roll_s" + str(sensor)]) / dt
                pitch_rate = np.diff(flight_data["kite_pitch_s" + str(sensor)]) / dt
                yaw_rate = np.diff(flight_data["kite_yaw_s" + str(sensor)]) / dt
                # Add the last value as 0 to keep the same length
                flight_data["kite_roll_rate_s" + str(sensor)] = np.concatenate(
                    (roll_rate, [0])
                )
                flight_data["kite_pitch_rate_s" + str(sensor)] = np.concatenate(
                    (pitch_rate, [0])
                )
                flight_data["kite_yaw_rate_s" + str(sensor)] = np.concatenate(
                    (yaw_rate, [0])
                )
                # Smooth the yaw rate data
                flight_data["kite_roll_rate_s" + str(sensor)] = np.convolve(
                    flight_data["kite_roll_rate_s" + str(sensor)],
                    np.ones(window_size) / window_size,
                    mode="same",
                )
                flight_data["kite_pitch_rate_s" + str(sensor)] = np.convolve(
                    flight_data["kite_pitch_rate_s" + str(sensor)],
                    np.ones(window_size) / window_size,
                    mode="same",
                )
                flight_data["kite_yaw_rate_s" + str(sensor)] = np.convolve(
                    flight_data["kite_yaw_rate_s" + str(sensor)],
                    np.ones(window_size) / window_size,
                    mode="same",
                )

        # Add the ground station data
        flight_data["ground_tether_force"] = (
            log["ground_tether_force"] * 9.81
        )  # Convert to N
        flight_data["ground_wind_velocity"] = log["ground_wind_velocity"]
        flight_data["ground_wind_direction"] = (
            360 - 90 - log["ground_upwind_direction"]
        )  # Convert from NED clockwise to ENU counter-clockwise
        flight_data["ground_tether_length"] = log["ground_tether_length"]  # Tether length
        flight_data["ground_tether_reelout_speed"] = log[
            "ground_tether_reelout_speed"
        ]  # Tether reelout speed

        # Add the KCU data
        flight_data["kcu_set_depower"] = log["kite_set_depower"]
        flight_data["kcu_set_steering"] = log["kite_set_steering"]
        flight_data["kcu_actual_steering"] = log["kite_actual_steering"]
        flight_data["kcu_actual_depower"] = log["kite_actual_depower"]

        # Add the airspeed data
        flight_data["kite_apparent_windspeed"] = log["airspeed_apparent_windspeed"]
        flight_data["kite_angle_of_attack"] = log["airspeed_angle_of_attack"]
        if self.config_data['kite']['model_name'] == 'v9':
            flight_data["kite_sideslip_angle"] = log["airspeed_sideslip_angle"]
        flight_data["kite_airspeed_temperature"] = log["airspeed_temperature"]

        kite_radius = np.linalg.norm(
            flight_data[["kite_position_east", "kite_position_north", "kite_position_up"]],
            axis=1,
        )
        offset_tether_length = np.mean(kite_radius - flight_data["ground_tether_length"])
        # flight_data["ground_tether_length"] = (
        #     flight_data["ground_tether_length"] + offset_tether_length
        # )
        print(
            "Date: ",
            log["date"].iloc[0],
            "Flight length: ",
            round(len(flight_data) / 10 / 60, 1),
            "min",
        )
        print("Offset tether length: ", offset_tether_length)
        # Add the time data
        flight_data["time"] = log["time"] - log["time"].iloc[0]
        flight_data["time_of_day"] = log["time_of_day"]
        flight_data["unix_time"] = log["time"]
        flight_data['date'] = log['date']

        # Add heading and course
        flight_data["kite_course"] = log["kite_course"]
        flight_data["kite_heading"] = log["kite_heading"]

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
        
        model = self.config_data['kite']['model_name']
        
        # Create the directory if it doesn't exist
        csv_filepath = "./processed_data/flight_data/" + model + "/"
        if not os.path.exists(csv_filepath):
            os.makedirs(csv_filepath)

        csv_filename = model + "_" + self.log_date + ".csv"
        flight_data.to_csv(csv_filepath + csv_filename, index=False)


def main() -> None:    
    config = {'year': '2019',
              'month': '10',
              'day': '08',
              'kite': {'model_name': 'v3'}}

    log_dir = Path(f'data/{config["kite"]["model_name"]}/')

    ProcessKpData(config_data=config, log_directory=log_dir)
    
    
if __name__ == "__main__":
    main()
