from dataclasses import dataclass, field
import numpy as np

@dataclass
class EKFInput:
    """Dataclass for storing input data for the Extended Kalman Filter"""
    # Required attributes
    kite_pos: np.ndarray  # Kite position in ENU coordinates
    kite_vel: np.ndarray  # Kite velocity in ENU coordinates
    kite_acc: np.ndarray  # Kite acceleration in ENU coordinates
    ts: float  # Timestep (s)
    tether_force: float  # Ground tether force (N)
    tether_length: float  # Tether length (offsets are calculated automatically)
    reelout_speed: float  # Reelout speed (positive values reel out)

    # Optional attributes
    apparent_windspeed: float = None  # Apparent windspeed
    kite_aoa: float = None  # Kite angle of attack (between last tether element and wind incident)
    kcu_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # KCU velocity in ENU coordinates
    kcu_acc: np.ndarray = field(default_factory=lambda: np.zeros(3))  # KCU acceleration in ENU coordinates
    tether_elevation: float = None  # Elevation angle (angle of tether element at GS exit with respect to the horizon) (rad)
    tether_azimuth: float = None  # Azimuth angle (With respect to east in EN plane) (rad)
    kite_yaw: float = None  # Yaw angle (don't use) 
    steering_input: float = None  # Steering input (don't use, define units later)
    thrust_force: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Thrust force (fly-gen kites value)
    depower_input: float = None  # Depower input (Units are irrelevant, as constant is automatically calculated)

    def __post_init__(self):
        # Ensure all array attributes have a length of 3
        for attr_name in ['kite_pos', 'kite_vel', 'kite_acc', 'kcu_vel', 'kcu_acc', 'thrust_force']:
            attr = getattr(self, attr_name)
            if not isinstance(attr, np.ndarray) or attr.shape != (3,):
                raise ValueError(f"{attr_name} must be a numpy array of shape (3,)")
        
        # Ensure kite_acc is not zero
        if np.linalg.norm(self.kite_acc) < 1e-5:
            self.kite_acc = np.full_like(self.kite_acc, 1e-5)
