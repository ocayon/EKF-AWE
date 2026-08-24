import yaml
import numpy as np
from dataclasses import dataclass
import os

# %% Define atmospheric parameters
rho = 1.225  # Air density [kg/m^3]
kappa = 0.4  # Von Karman constant [-]
g = 9.81  # Gravity acceleration [m/s^2]
z0 = 0.01  # Surface roughness [m]


# Load the configuration file
def load_config():
    # Set the directory path where configuration files are stored
    config_directory = "data/config/"

    # List all files in the config directory
    config_files = os.listdir(config_directory)

    # Prompt user to select a file from the list
    print("Available configuration files:")
    for index, filename in enumerate(config_files, start=1):
        print(f"{index}: {filename}")

    # Get user selection
    selection = int(input("Select a configuration file by number: ")) - 1

    # Ensure selection is valid
    if 0 <= selection < len(config_files):
        selected_file = config_files[selection]
        config_path = os.path.join(config_directory, selected_file)

        # Load the configuration file
        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)

        # Optional: Check if the config has all required data
        if not validate_config(config_data):
            raise ValueError("Configuration file is missing required data.")

        print(f"Configuration loaded from: {selected_file}")
        return config_data
    else:
        raise ValueError("Invalid selection. Please choose a valid file number.")


def validate_config(config_data):
    # Placeholder validation function to ensure required fields are present
    required_keys = [
        "simulation_parameters",
        "tuning_parameters",
        "kite",
        "kcu",
        "tether",
    ]  # Example keys
    return all(key in config_data for key in required_keys)


class SimulationConfig:
    def __init__(self, **kwargs):
        self.ts = kwargs.get("timestep")
        self.opt_measurements = kwargs.get("opt_measurements", [])
        self.doIEKF = kwargs.get("doIEKF", True)
        self.epsilon = float(kwargs.get("epsilon", 1e-6))
        self.max_iterations = kwargs.get("max_iterations", 200)
        self.log_profile = kwargs.get("log_profile", False)
        self.initial_wind_velocity = np.array(
            kwargs.get("initial_wind_velocity", [1e-3, 8, 0]), dtype=float
        )
        self.tether_offset = kwargs.get("tether_offset", True)
        # Sensors calibrated in a pre-run before the main filter loop. Keys match
        # the measurement names in ObservationData; a sensor is only calibrated if
        # the corresponding measurement is also enabled. The apparent wind speed
        # gets a pitot calibration (scale factor on the dynamic pressure), the
        # angle of attack an additive offset (vane misalignment).
        self.calibrate_sensor = {
            "kite_apparent_windspeed": kwargs.get(
                "calibrate_apparent_windspeed", True
            ),
            "bridle_angle_of_attack": kwargs.get(
                "find_offset_angle_of_attack", True
            ),
        }
        # Also fit the pitot transducer zero on top of the calibration
        # coefficient. Only useful over a wide apparent wind speed range.
        self.pitot_calibration_fit_zero = kwargs.get(
            "pitot_calibration_fit_zero", False
        )
        # Window the sensor calibrations are fitted on, in minutes of flight.
        # It starts after the filter transient, lasts as long as asked for, and
        # always stops short of the end of the flight, where the data is often
        # unreliable. A longer window averages out more turbulence.
        self.calibration_start_minutes = float(
            kwargs.get("calibration_start_minutes", 5)
        )
        self.calibration_duration_minutes = float(
            kwargs.get("calibration_duration_minutes", 15)
        )
        self.calibration_end_margin_minutes = float(
            kwargs.get("calibration_end_margin_minutes", 3)
        )
        self.enforce_vertical_wind_to_0 = kwargs.get(
            "enforce_vertical_wind_to_0", False
        )
        self.model_yaw = kwargs.get("model_yaw", False)
        # Deterministic dependence of the aero coefficients on the measured
        # steering input, with the constants estimated as near-zero-process-
        # noise states (the same mechanism as the depower constants k_cl_up/
        # k_cd_up). The steering is loop-phase-locked, so leaving it to the
        # random walks makes them chase a loop-frequency sinusoid with lag.
        # Stage 1: CS = CL * tan(k_phi_us * us) plus the CS state as residual.
        self.steering_dependent_cs = kwargs.get("steering_dependent_cs", False)
        # Stage 2: k_cl_us * |us| on CL and k_cd_us * us^2 on CD.
        self.steering_dependent_clcd = kwargs.get("steering_dependent_clcd", False)
        # Left/right asymmetry: signed k_cl_us_odd * us on CL, on top of the
        # even stage-2 terms. CL only: a signed CD term was tried on the
        # 2019-10-08 flight and made the CD pattern-lock worse.
        self.steering_dependent_cl_asym = kwargs.get(
            "steering_dependent_cl_asym", False
        )
        # First-order lag [s] applied to the steering input before it enters
        # the model: the aero response lags the measured actuation (identified
        # at ~0.3 s on the 2019-10-08 flight). 0 disables the filter.
        self.steering_input_lag = float(kwargs.get("steering_input_lag", 0.0))
        self.thrust_force = kwargs.get("thrust_force", False)
        self.debug = kwargs.get("debug", False)
        measurements = kwargs.get("measurements", {})
        self.obsData = ObservationData(**measurements)


@dataclass
class ObservationData:
    tether_length: bool = True
    tether_elevation: bool = True
    tether_azimuth: bool = True
    kite_position: bool = True
    kite_velocity: bool = True
    tether_force: bool = True
    kite_acceleration: bool = False
    kcu_position: bool = False
    kcu_acceleration: bool = False
    kcu_velocity: bool = False
    kite_apparent_windspeed: bool = False
    bridle_angle_of_attack: bool = False
    bridle_angle_of_sideslip: bool = False
    kite_yaw_angle: bool = False
    kite_thrust_force: bool = False
    dynamic_depower: bool = False


class TuningParameters:
    def __init__(self, config, simConfig):
        self.dict_model_stdv = config["model_stdv"]
        self.dict_meas_stdv = config["meas_stdv"]

        if simConfig.log_profile:
            indices = [
                "x",
                "x",
                "x",
                "v",
                "v",
                "v",
                "uf",
                "wdir",
                "vwz",
                "CL",
                "CD",
                "CS",
                "tether_length",
                "tether_elevation",
                "tether_azimuth",
            ]
        else:
            indices = [
                "x",
                "x",
                "x",
                "v",
                "v",
                "v",
                "vw",
                "vw",
                "vwz",
                "CL",
                "CD",
                "CS",
                "tether_length",
                "tether_elevation",
                "tether_azimuth",
            ]

        self.stdv_dynamic_model = np.array(
            [float(self.dict_model_stdv[key]) for key in indices]
        )
        if simConfig.model_yaw:
            self.stdv_dynamic_model = np.append(
                self.stdv_dynamic_model,
                [self.dict_model_stdv["yaw"], 1e-6],  # Yaw  and yaw offset
            )
        if simConfig.obsData.tether_length:
            self.stdv_dynamic_model = np.append(
                self.stdv_dynamic_model, 1e-6
            )  # Tether length offset
        if simConfig.obsData.tether_elevation:
            self.stdv_dynamic_model = np.append(self.stdv_dynamic_model, 1e-6)
        if simConfig.obsData.tether_azimuth:
            self.stdv_dynamic_model = np.append(self.stdv_dynamic_model, 1e-6)
        if simConfig.obsData.dynamic_depower:
            self.stdv_dynamic_model = np.append(
                self.stdv_dynamic_model, 1e-6
            )  # Depower constant
            self.stdv_dynamic_model = np.append(
                self.stdv_dynamic_model, 1e-6
            )  # Depower constant
        if simConfig.steering_dependent_cs:
            self.stdv_dynamic_model = np.append(
                self.stdv_dynamic_model, 1e-6
            )  # Steering-to-sideforce constant k_phi_us
        if simConfig.steering_dependent_clcd:
            self.stdv_dynamic_model = np.append(
                self.stdv_dynamic_model, 1e-6
            )  # Steering constant k_cl_us
            self.stdv_dynamic_model = np.append(
                self.stdv_dynamic_model, 1e-6
            )  # Steering constant k_cd_us
        if simConfig.steering_dependent_cl_asym:
            self.stdv_dynamic_model = np.append(
                self.stdv_dynamic_model, 1e-6
            )  # Steering asymmetry constant k_cl_us_odd
        self.indices_measurements = [
            "x",
            "x",
            "x",
            "v",
            "v",
            "v",
            "least_squares",
            "least_squares",
            "least_squares",
        ]

        self.update_observation_vector(simConfig)

    def update_observation_vector(self, simConfig):
        stdv_y = []
        if simConfig.obsData.kite_position:
            for _ in range(3):
                stdv_y.append(self.dict_meas_stdv["x"])
        if simConfig.obsData.kite_velocity:
            for _ in range(3):
                stdv_y.append(self.dict_meas_stdv["v"])
        for _ in range(3):
            stdv_y.append(float(self.dict_meas_stdv["least_squares"]))
        if simConfig.model_yaw:
            stdv_y.append(self.dict_meas_stdv["yaw"])
        if simConfig.obsData.tether_length:
            stdv_y.append(self.dict_meas_stdv["tether_length"])
        if simConfig.obsData.tether_elevation:
            stdv_y.append(self.dict_meas_stdv["tether_elevation"])
        if simConfig.obsData.tether_azimuth:
            stdv_y.append(self.dict_meas_stdv["tether_azimuth"])
        if simConfig.enforce_vertical_wind_to_0:
            stdv_y.append(self.dict_meas_stdv["z_wind"])
        if simConfig.obsData.kite_apparent_windspeed:
            stdv_y.append(self.dict_meas_stdv["va"])
        if simConfig.obsData.bridle_angle_of_attack:
            stdv_y.append(self.dict_meas_stdv["aoa"])

        self.stdv_measurements = np.array(stdv_y)
