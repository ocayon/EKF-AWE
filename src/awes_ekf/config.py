######################################################################################################
# Description: This file contains the configuration parameters for the initialisation of the Kalman filter.
######################################################################################################

#%% Import libraries
import numpy as np

#%% Define system parameters
# Define system parameters
kite_model = 'v9'                   # Kite model name, if Costum, change the kite parameters next
kcu_model = 'KP2'                   # KCU model name
tether_diameter = 0.014            # Tether diameter [m]
tether_material = 'Dyneema-SK78'    # Tether material

#%% Define atmospheric parameters
rho = 1.225                         # Air density [kg/m^3]
kappa = 0.4                         # Von Karman constant [-]
g = 9.81                            # Gravity acceleration [m/s^2]
z0 = 0.1                            # Surface roughness [m]

#%% Define simulation parameters
log_profile = True                 # Model wind speed as logarithmic with height
tether_offset = True                # Use tether offset in the measurements
enforce_z_wind = False              # Enforce the z wind speed to be zero
model_yaw = False                   # Model the yaw angle in the state vector

# Tether model
n_tether_elements = 5              # Number of tether elements

# Kalman filter parameters
doIEKF = True                       # Use the iterated extended Kalman filter
max_iterations = 200                 # Maximum number of iterations for the IEKF
epsilon = 1e-6                      # Tolerance for the IEKF

# Measurements
opt_measurements = []            # List of measurements to be used in the Kalman filter
# opt_measurements = ['apparent_windspeed']            # List of measurements to be used in the Kalman filter

# Measurement standard deviations

meas_stdv = {
    'x': 2.5,                  # Position
    'v': 1,                    # Velocity       
    'a': 10,                   # Acceleration
    'uf': 0.1,                 # Friction velocity
    'va': 0.5,                 # Apparent velocity
    'wdir': (10/180 * np.pi),  # Wind direction
    'tether_length': 1,
    'aoa':      4,     # Angle of attack
    'relout_speed': 0.01,       # Reelout speed
    'least_squares': 1e-5,       # Least squares
    'elevation': 0.2,
    'azimuth':0.2,
    'z_wind': 0.1,
    'yaw': 5/180*np.pi
}

model_stdv = {
    'v3': {
        'x': 2.5,                  # Position
        'v': 1,                  # Velocity       
        'uf': 2e-3,               # Friction velocity
        'wdir': (0.2/180 * np.pi),   # Wind direction
        'vw': 1e-1,
        'vwz': 1e-2,                # Vertical windspeed
        'CL': 1e-2,                 # Lift coefficient
        'CD': 1e-2,                 # Drag coefficient
        'CS': 1e-2,                  # Side force coefficient
        'elevation' : 0.3,    # Elevation angle
        'azimuth' : 0.3,     # Azimuth angle
        'tether_length' : 0.1, # Tether length
        'yaw': 5/180*np.pi
    },
    'v9': {
        'x': 2.5,                  # Position
        'v': 1,                  # Velocity       
        'uf': 5e-3,               # Friction velocity
        'wdir': (0.2/180 * np.pi),   # Wind direction
        'vw': 6e-2,                 # Wind speed
        'vwz': 6e-2,                # Vertical windspeed
        'CL': 1e-2,                 # Lift coefficient
        'CD': 3e-3,                 # Drag coefficient
        'CS': 1e-2,                  # Side force coefficient
        'elevation' : 0.3,    # Elevation angle
        'azimuth' : 0.3,     # Azimuth angle
        'tether_length' : 0.1, # Tether length
        'yaw': 5/180*np.pi
    }
}



