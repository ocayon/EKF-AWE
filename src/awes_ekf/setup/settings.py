import yaml
import numpy as np
from dataclasses import dataclass

# %% Define atmospheric parameters
rho = 1.225  # Air density [kg/m^3]
kappa = 0.4  # Von Karman constant [-]
g = 9.81  # Gravity acceleration [m/s^2]
z0 = 0.1  # Surface roughness [m]


# Load the configuration file
def load_config(filepath):
    with open(filepath, "r") as file:
        return yaml.safe_load(file)


class SimulationConfig:
    def __init__(self, **kwargs):
        self.ts = kwargs.get("timestep")
        self.opt_measurements = kwargs.get("opt_measurements", [])
        self.doIEKF = kwargs.get("doIEKF", True)
        self.epsilon = float(kwargs.get("epsilon", 1e-6))
        self.max_iterations = kwargs.get("max_iterations", 200)
        self.log_profile = kwargs.get("log_profile", False)
        self.tether_offset = kwargs.get("tether_offset", True)
        self.enforce_vertical_wind_to_0 = kwargs.get("enforce_vertical_wind_to_0", False)
        self.model_yaw = kwargs.get("model_yaw", False)
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

        self.stdv_dynamic_model = np.array([float(self.dict_model_stdv[key]) for key in indices])
        if simConfig.model_yaw:
            self.stdv_dynamic_model = np.append(
                self.stdv_dynamic_model, [self.dict_model_stdv["yaw"], 1e-6]  # Yaw  and yaw offset
            )
        if simConfig.obsData.tether_length:
            self.stdv_dynamic_model = np.append(self.stdv_dynamic_model, 1e-6) # Tether length offset
        if simConfig.obsData.tether_elevation:
            self.stdv_dynamic_model = np.append(self.stdv_dynamic_model, 1e-6)
        if simConfig.obsData.tether_azimuth:
            self.stdv_dynamic_model = np.append(self.stdv_dynamic_model, 1e-6)
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
        
        
