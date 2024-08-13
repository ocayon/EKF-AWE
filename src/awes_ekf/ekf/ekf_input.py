from dataclasses import dataclass, field
import numpy as np

@dataclass
class EKFInput:
    """Dataclass for storing input data for the Extended Kalman Filter"""
    # Required attributes
    kite_position: np.ndarray  # Kite position in ENU coordinates (m)
    kite_velocity: np.ndarray  # Kite velocity in ENU coordinates (m/s)
    kite_acceleration: np.ndarray  # Kite acceleration in ENU coordinates (m/s^2)
    timestep: float  # Timestep (s)
    tether_force: float  # Ground tether force (N)
    tether_length: float  # Tether length (offsets are calculated automatically) (m)
    tether_reelout_speed: float  # Reelout speed (positive values reel out) (m/s)

    # Optional attributes
    kite_apparent_windspeed: float = None  # Apparent windspeed (m/s)
    kite_angle_of_attack: float = None  # Kite angle of attack (between last tether element and wind incident) (rad)
    kcu_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # KCU velocity in ENU coordinates (m/s)
    kcu_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))  # KCU acceleration in ENU coordinates (m/s^2)
    tether_elevation_ground: float = None  # Elevation angle (angle of tether element at GS exit with respect to the horizon) (rad)
    tether_azimuth_ground: float = None  # Azimuth angle (With respect to east in EN plane) (rad)
    kite_yaw: float = None  # Yaw angle (don't use) (rad)
    steering_input: float = None  # Steering input (don't use, define units later) (from -1 to 1)
    kite_thrust_force: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Thrust force (fly-gen kites value) (N)
    depower_input: float = None  # Depower input (Units are irrelevant, as constant is automatically calculated) (from 0 to 1)

    def __post_init__(self):
        # Ensure all array attributes have a length of 3
        for attr_name in ['kite_position', 'kite_velocity', 'kite_acceleration', 'kcu_velocity', 'kcu_acceleration', 'kite_thrust_force']:
            attr = getattr(self, attr_name)
            if not isinstance(attr, np.ndarray) or attr.shape != (3,):
                raise ValueError(f"{attr_name} must be a numpy array of shape (3,)")

        # Ensure kite_acceleration is not zero
        if np.linalg.norm(self.kite_acceleration) < 1e-5:
            self.kite_acceleration = np.full_like(self.kite_acceleration, 1e-5)

