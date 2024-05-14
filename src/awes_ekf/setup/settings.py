import yaml
import numpy as np

#%% Define atmospheric parameters
rho = 1.225                         # Air density [kg/m^3]
kappa = 0.4                         # Von Karman constant [-]
g = 9.81                            # Gravity acceleration [m/s^2]
z0 = 0.1                            # Surface roughness [m]

# Load the configuration file
def load_config(filepath):
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)
    
class SimulationConfig:
    def __init__(self, **kwargs):
        self.ts = kwargs.get('timestep')
        self.n_tether_elements = kwargs.get('n_tether_elements')
        self.opt_measurements = kwargs.get('opt_measurements', [])
        self.kcu_data = kwargs.get('kcu_data', False)
        self.doIEKF = kwargs.get('doIEKF', True)
        self.epsilon = float(kwargs.get('epsilon', 1e-6))
        self.max_iterations = kwargs.get('max_iterations', 200)
        self.log_profile = kwargs.get('log_profile', False)
        self.tether_offset = kwargs.get('tether_offset', True)
        self.enforce_z_wind = kwargs.get('enforce_z_wind', False)
        self.model_yaw = kwargs.get('model_yaw', False)

class SystemParameters:
    def __init__(self, config, simConfig):
        self.kite_model = config['kite_model']
        self.kcu_model = config['kcu_model']
        self.tether_material = config['tether_material']
        self.tether_diameter = config['tether_diameter']
        model_stdv = config['model_stdv']
        meas_stdv = config['meas_stdv']

        if simConfig.log_profile:
            indices = ['x', 'x', 'x', 'v', 'v', 'v', 'uf', 'wdir', 'vwz', 'CL', 'CD', 'CS', 'tether_length', 'elevation', 'azimuth']
        else:
            indices = ['x', 'x', 'x', 'v', 'v', 'v', 'vw', 'vw', 'vwz', 'CL', 'CD', 'CS', 'tether_length', 'elevation', 'azimuth']

        self.stdv_dynamic_model = np.array([float(model_stdv[key]) for key in indices])
        if simConfig.model_yaw:
            self.stdv_dynamic_model = np.append(self.stdv_dynamic_model, [model_stdv['yaw'], 1e-6])
        
        indices = ['x', 'x', 'x', 'v', 'v', 'v', 'tether_length', 'elevation', 'azimuth', 'least_squares', 'least_squares', 'least_squares']
        stdv_y = [float(meas_stdv[key]) for key in indices]
        if simConfig.model_yaw:
            stdv_y.append(meas_stdv['yaw'])
        if simConfig.enforce_z_wind:
            stdv_y.append(meas_stdv['z_wind'])
        
        for key in simConfig.opt_measurements:
            if key == 'apparent_windspeed':
                stdv_y.append(meas_stdv['va'])

        self.stdv_measurements = np.array(stdv_y)



