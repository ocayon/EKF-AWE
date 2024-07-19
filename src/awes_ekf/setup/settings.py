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
        self.enforce_z_wind = kwargs.get("enforce_z_wind", False)
        self.model_yaw = kwargs.get("model_yaw", False)
        self.thrust_force = kwargs.get("thrust_force", False)
        measurements = kwargs.get("measurements", {})
        self.obsData = ObservationData(**measurements)


@dataclass
class ObservationData:
    tether_length: bool = True
    tether_elevation: bool = True
    tether_azimuth: bool = True
    kite_pos: bool = True
    kite_vel: bool = True
    tether_force: bool = True
    kite_acc: bool = False
    kcu_pos: bool = False
    kcu_acc: bool = False
    kcu_vel: bool = False
    apparent_windspeed: bool = False
    angle_of_attack: bool = False
    angle_of_sideslip: bool = False
    yaw_angle: bool = False
    thrust_force: bool = False


class TuningParameters:
    def __init__(self, config, simConfig):
        model_stdv = config["model_stdv"]
        meas_stdv = config["meas_stdv"]

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

        self.stdv_dynamic_model = np.array([float(model_stdv[key]) for key in indices])
        if simConfig.model_yaw:
            self.stdv_dynamic_model = np.append(
                self.stdv_dynamic_model, [model_stdv["yaw"], 1e-6]  # Yaw  and yaw offset
            )
        if simConfig.obsData.tether_length:
            self.stdv_dynamic_model = np.append(self.stdv_dynamic_model, 1e-6) # Tether length offset
        indices = [
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
        stdv_y = [float(meas_stdv[key]) for key in indices]
        if simConfig.model_yaw:
            stdv_y.append(meas_stdv["yaw"])
        if simConfig.obsData.tether_length:
            stdv_y.append(meas_stdv["tether_length"])
        if simConfig.obsData.tether_elevation:
            stdv_y.append(meas_stdv["tether_elevation"])
        if simConfig.obsData.tether_azimuth:
            stdv_y.append(meas_stdv["tether_azimuth"])
        if simConfig.enforce_z_wind:
            stdv_y.append(meas_stdv["z_wind"])
        if simConfig.obsData.apparent_windspeed:
            stdv_y.append(meas_stdv["va"])
        if simConfig.obsData.angle_of_attack:
            stdv_y.append(meas_stdv["aoa"])

        self.stdv_measurements = np.array(stdv_y)
