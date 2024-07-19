from dataclasses import dataclass
import numpy as np


@dataclass
class EKFInput:
    kite_pos: np.ndarray  # Kite position in ENU coordinates
    kite_vel: np.ndarray  # Kite velocity in ENU coordinates
    kite_acc: np.ndarray  # Kite acceleration in ENU coordinates
    ts: float  # Timestep (s)
    tether_force: float  # Ground tether force (N)
    apparent_windspeed: float = None  # Apparent windspeed
    tether_length: float # Tether length, (offsets are calculated automatically)
    kite_aoa: np.ndarray = None  # Kite angle of attack (between last tether element and wind incident)
    kcu_vel: np.ndarray = None  # KCU velocity in ENU coordinates
    kcu_acc: np.ndarray = None  # KCU acceleration in ENU coordinates
    reelout_speed: float # Reelout speed (positive values reel out) # Todo: this is needed
    elevation: float = None  # Elevation angle (angle of tether element at GS exit with respect to the horizont) (rad)
    azimuth: float = None  # Azimuth angle (With respect to east in EN plane) (rad)
    kite_yaw: float = None  # Yaw angle (don't use)
    steering_input: float = None  # Steering input (don't use, define units later)
    thrust_force: np.ndarray = None  # Thrust force (kitekraft value, don't use)
    up: float = None  # Depower input (don't use, define units later)
    wind_vel: np.ndarray = np.array([1e-3, 1e-3, 0])  # Wind velocity pointing downwind (Initial estimation of wind velocity on the ground, ENU)

    def __post_init__(self):
        if np.linalg.norm(self.kite_acc) < 1e-5:
            self.kite_acc = np.full_like(self.kite_acc, 1e-5)
