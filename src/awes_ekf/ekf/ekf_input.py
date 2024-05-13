from dataclasses import dataclass
import numpy as np

@dataclass
class EKFInput:
    kite_pos: np.array      # Kite position in ENU coordinates
    kite_vel: np.array      # Kite velocity in ENU coordinates
    kite_acc: np.array      # Kite acceleration in ENU coordinates
    ts: float              # Timestep (s)
    tether_force: float     # Ground tether force
    apparent_windspeed: float = None # Apparent windspeed
    tether_length: float = None     # Tether length
    kite_aoa: np.array = None      # Kite angle of attack
    kcu_vel: np.array = None    # KCU velocity in ENU coordinates
    kcu_acc: np.array = None    # KCU acceleration in ENU coordinates
    reelout_speed: float = None  # Reelout speed
    elevation: float = None      # Elevation angle
    azimuth: float = None        # Azimuth angle
    kite_yaw: float = None            # Yaw angle
    steering_input: float = None # Steering input