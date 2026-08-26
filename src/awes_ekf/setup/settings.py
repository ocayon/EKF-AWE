import yaml
import numpy as np
from dataclasses import dataclass
import os
from pathlib import Path

# %% Define atmospheric parameters
rho = 1.225  # Air density [kg/m^3]
kappa = 0.4  # Von Karman constant [-]
g = 9.81  # Gravity acceleration [m/s^2]
z0 = 0.01  # Surface roughness [m]

_EKF_CONFIG_NAME = "ekf_config.yaml"


def get_kite(system_config):
    """Primary kite component dict of an awesIO system config.

    Handles the current awesIO layout (``components.kites`` array), the
    legacy singular ``components.kite``, and an inline-flat layout (wing /
    control_system directly under ``components``).
    """
    components = system_config.get("components", {})
    kites = components.get("kites")
    if isinstance(kites, list) and kites:
        return kites[0]
    return components.get("kite", components)


def get_tether(system_config):
    """Primary tether component dict of an awesIO system config."""
    components = system_config.get("components", {})
    tethers = components.get("tethers")
    if isinstance(tethers, list) and tethers:
        return tethers[0]
    return components.get("tether", {})


def _distance_kcu_kite(wing_struct, control_sys_struct):
    """KCU-to-wing distance [m]: the wing CG height above the bridle point.

    bridle_point_node is the body-frame ORIGIN in awesIO (always [0,0,0]), so
    the distance is the wing CG z. Reading the node itself gave 0 and collapsed
    the bridle segment in the tether model (bridle drag lost, KCU drag split
    corrupted). A system yaml without a wing CG (e.g. the LEI V9) may instead
    carry the measured distance explicitly on the control_system structure.
    """
    wing_cg = wing_struct.get("center_of_mass")
    if wing_cg is not None and wing_cg[2]:
        return wing_cg[2]
    return control_sys_struct.get("distance_kcu_kite") or 0.0


def _prompt_path(message):
    try:
        from prompt_toolkit import prompt
        from prompt_toolkit.completion import PathCompleter

        return prompt(message, completer=PathCompleter(expanduser=True)).strip()
    except ImportError:
        return input(message).strip()


# Load the configuration folder
def load_config(config_folder=None):
    """Load and merge a kite configuration from a config folder.

    AWETrim convention: the folder (``data/<KITE-NAME>/``) holds

    - ``ekf_config.yaml`` — simulation_parameters and tuning_parameters, and
    - one or more awesIO-validated ``system*.yaml`` files with the physical
      properties of the hardware. A kite can have several system variants
      depending on what it was flown with (e.g. ``system_flown_2019.yaml``
      vs ``system_flown_2025.yaml`` differ in KCU and tether); with more
      than one candidate the user picks which hardware the EKF assumes.

    The ``kite`` / ``kcu`` / ``tether`` blocks the models consume are
    extracted from the chosen system yaml and merged into the returned
    config dict, with ``system_yaml_used`` recording the variant.
    """
    if config_folder is None:
        raw = _prompt_path(
            "Enter the config folder (with system*.yaml and ekf_config.yaml), "
            "or leave empty to list data/: "
        )
        if raw:
            config_folder = raw
        else:
            candidates = sorted(
                path.parent for path in Path("data").glob(f"*/{_EKF_CONFIG_NAME}")
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No data/*/{_EKF_CONFIG_NAME} found; pass the folder path"
                )
            print("Available config folders:")
            for index, path in enumerate(candidates, start=1):
                print(f"{index}: {path.name}")
            selection = (
                input(f"Select a config folder (1-{len(candidates)}) [default: 1]: ").strip()
                or "1"
            )
            config_folder = candidates[int(selection) - 1]

    folder = Path(config_folder).expanduser()
    if not folder.exists() and (Path("data") / folder).exists():
        folder = Path("data") / folder  # a bare kite name resolves to data/<name>
    folder = folder.resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    ekf_config_path = folder / _EKF_CONFIG_NAME
    if not ekf_config_path.exists():
        raise FileNotFoundError(f"{_EKF_CONFIG_NAME} not found in {folder}")

    # Every system variant in the folder is a candidate; the physical
    # properties the EKF assumes -- KCU mass and tether above all -- depend
    # on which one is chosen.
    candidates = sorted(
        set(folder.glob("system*.yaml")) | set(folder.glob("system*.yml"))
    )
    if not candidates:
        raise FileNotFoundError(f"No system*.yaml / system*.yml in {folder}")
    if len(candidates) == 1:
        system_yaml_path = candidates[0]
    else:
        print("Available system configuration files:")
        for index, path in enumerate(candidates, start=1):
            print(f"{index}: {path.name}")
        selection = (
            input(f"Select a system file (1-{len(candidates)}) [default: 1]: ").strip()
            or "1"
        )
        system_yaml_path = candidates[int(selection) - 1]
    print(f"Using system config: {system_yaml_path.name}")

    with open(ekf_config_path, "r", encoding="utf-8") as file:
        config_data = yaml.safe_load(file)
    with open(system_yaml_path, "r", encoding="utf-8") as file:
        system_config = yaml.safe_load(file)

    kite_node = get_kite(system_config)
    wing_struct = kite_node.get("wing", {}).get("structure", {})
    bridle_struct = kite_node.get("bridle", {}).get("structure", {})
    control_sys_struct = kite_node.get("control_system", {}).get("structure", {})
    tether_struct = get_tether(system_config).get("structure", {})

    config_data["kite"] = {
        "model_name": kite_node.get("name", "unknown"),
        "mass": wing_struct.get("mass", 0.0),
        "area": wing_struct.get(
            "projected_surface_area", wing_struct.get("planform_surface_area", 0.0)
        ),
        "span": wing_struct.get("span", 0.0),
        "sensor_ids": [0, 1],
    }
    config_data["kcu"] = {
        "length": control_sys_struct.get("length", 1.0),
        "diameter": control_sys_struct.get("diameter", 0.48),
        "mass": control_sys_struct.get("mass", 0.0),
        "distance_kcu_kite": _distance_kcu_kite(wing_struct, control_sys_struct),
        "total_length_bridle_lines": bridle_struct.get(
            "total_nominal_line_length", 0.0
        ),
        "diameter_bridle_lines": bridle_struct.get("avg_line_diameter", 0.0),
    }
    config_data["tether"] = {
        "material_name": tether_struct.get("material", {}).get("type", "Dyneema-SK78"),
        "diameter": tether_struct.get("diameter", 0.01),
        "n_elements": 30,
    }
    # Record WHICH system variant the EKF assumed; it ends up in the results
    # config, so a flown-2019 vs flown-2025 run stays identifiable.
    config_data["system_yaml_used"] = system_yaml_path.name

    if not validate_config(config_data):
        raise ValueError("Configuration is missing required data.")

    print(f"EKF config loaded from: {ekf_config_path}")
    print(f"Kite model: {config_data['kite']['model_name']}")
    return config_data


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
        # Drag polar: CD = CD0 + k_cd_cl2 * CL_eff^2 (+ the steering term),
        # with k_cd_cl2 a near-constant state and the CD state the parasitic
        # residual CD0. Couples CD to CL the way the physics does, so
        # lift/drag axis-misattribution noise can no longer walk the wing CD
        # to unphysical near-zero values independently of CL.
        self.drag_polar = kwargs.get("drag_polar", False)
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
        if simConfig.drag_polar:
            self.stdv_dynamic_model = np.append(
                self.stdv_dynamic_model, 1e-6
            )  # Induced-drag constant k_cd_cl2
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
