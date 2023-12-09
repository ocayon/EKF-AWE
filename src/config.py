######################################################################################################
# Description: This file contains the configuration parameters for the initialisation of the Kalman filter.
######################################################################################################

#%% Import libraries
import numpy as np

#%% Choose flight data

year = '2023'
month = '10'
day = '26'

#%% Define system parameters
# Define system parameters
kite_model = 'v9'                   # Kite model name, if Costum, change the kite parameters next
kcu_model = 'KP2'                   # KCU model name
tether_diameter = 0.014             # Tether diameter [m]
tether_material = 'Dyneema-SK78'    # Tether material

#%% Define atmospheric parameters
rho = 1.225                         # Air density [kg/m^3]
kappa = 0.4                         # Von Karman constant [-]
g = 9.81                            # Gravity acceleration [m/s^2]
z0 = 0.1                            # Surface roughness [m]

#%% Define simulation parameters
# Tether model
n_tether_elements = 5              # Number of tether elements
# Kalman filter parameters
doIEKF = True                       # Use the iterated extended Kalman filter
max_iterations = 10                 # Maximum number of iterations for the IEKF
epsilon = 1e-6                      # Tolerance for the IEKF

# Measurements
measurements = ['GPS_pos', 'GPS_vel']

# Measurement standard deviations

meas_stdv = {
    'x': 2.5,                  # Position
    'v': 1,                    # Velocity       
    'a': 10,                   # Acceleration
    'uf': 0.1,                 # Friction velocity
    'va': 0.5,                 # Apparent velocity
    'wdir': (10/180 * np.pi),  # Wind direction
}

# Model standard deviations
model_stdv = {
    'x': 2.5,                  # Position
    'v': 1,                  # Velocity       
    'uf': 1e-3,               # Friction velocity
    'wdir': (0.1/180 * np.pi),   # Wind direction
    'CL': 1e-2,                 # Lift coefficient
    'CD': 1e-2,                 # Drag coefficient
    'CS': 1e-2                  # Side force coefficient
}

# # Model standard deviations for v3
# model_stdv = {
#     'x': 2.5,                  # Position
#     'v': 1,                  # Velocity       
#     'uf': 5e-4,               # Friction velocity
#     'wdir': (0.01/180 * np.pi),   # Wind direction
#     'CL': 1e-2,                 # Lift coefficient
#     'CD': 1e-3,                 # Drag coefficient
#     'CS': 1e-2                  # Side force coefficient
# }

# Get standard deviation vectors
stdv_x = np.array([model_stdv['x'], model_stdv['x'], model_stdv['x'], 
                   model_stdv['v'], model_stdv['v'], model_stdv['v'], 
                   model_stdv['uf'], model_stdv['wdir'], 
                   model_stdv['CL'], model_stdv['CD'], model_stdv['CS']])

stdv_y = []
for key in measurements:
    if key == 'GPS_pos':
        for _ in range(3):
            stdv_y.append(meas_stdv['x'])
    elif key == 'GPS_vel':
        for _ in range(3):
            stdv_y.append(meas_stdv['v'])
    elif key == 'GPS_acc':   
        for _ in range(3):
            stdv_y.append(meas_stdv['a'])
    elif key == 'ground_wvel':
        stdv_y.append(meas_stdv['uf'])

    elif key == 'apparent_wvel':
        stdv_y.append(meas_stdv['va'])
stdv_y = np.array(stdv_y)


#%% Dictionary of existing kite models
# Configuration dictionary for kite models
kite_models = {
    "v3": {
        "KCU": True,
        "mass": 15,
        "area": 19.75,
        "distance_kcu_kite": 11.5,
        "total_length_bridle_lines": 96,
        "diameter_bridle_lines": 2.5e-3,
    },
    "v9": {
        "KCU": True,
        "mass": 62,
        "area": 46.854,
        "distance_kcu_kite": 15.45,
        "total_length_bridle_lines": 300,
        "diameter_bridle_lines": 4e-3,
    },
    "custom": {
        "KCU": True,
        "mass": 25,
        "area": 30,
        "distance_kcu_kite": 10,
        "total_length_bridle_lines": 120,
        "diameter_bridle_lines": 3e-3,
    },
}
tether_materials = {
    "Dyneema-SK78": {
        "density": 970,
        "cd": 1.1,
        "Youngs_modulus": 110e9,
    },
}
kcu_cylinders = {
    "KP1": {
        "length": 1,
        "diameter": 0.48,
        "mass": 18 + 1.6 + 8,
    },
    "KP2": {
        "length": 1.2,
        "diameter": 0.62,
        "mass": 18 + 1.6 + 12,
    },
    "custom": {
        "length": 1.2,
        "diameter": 0.62,
        "mass": 18 + 1.6 + 12,
    },
}