from dataclasses import dataclass
import numpy as np

@dataclass
class EKFInput:
    kite_pos: np.ndarray      # Kite position in ENU coordinates
    kite_vel: np.ndarray      # Kite velocity in ENU coordinates
    kite_acc: np.ndarray      # Kite acceleration in ENU coordinates
    ts: float              # Timestep (s)
    tether_force: float     # Ground tether force
    apparent_windspeed: float = None # Apparent windspeed
    tether_length: float = None     # Tether length
    kite_aoa: np.ndarray = None      # Kite angle of attack
    kcu_vel: np.ndarray = None    # KCU velocity in ENU coordinates
    kcu_acc: np.ndarray = None    # KCU acceleration in ENU coordinates
    reelout_speed: float = None  # Reelout speed
    elevation: float = None      # Elevation angle
    azimuth: float = None        # Azimuth angle
    kite_yaw: float = None            # Yaw angle
    steering_input: float = None # Steering input
    thrust_force: np.ndarray = None # Thrust force