import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
# Example usage
plt.close('all')
config_file_name = "v3_config.yaml"
config = load_config("examples/" + config_file_name)

# Load results and flight data and plot kite reference frame
results, flight_data = read_results(str(config['year']), str(config['month']), str(config['day']), config['kite']['model_name'])

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
pu.plot_time_series(flight_data,results['mahalanobis_distance'],ax, plot_phase=True)

plt.legend(['NIS','Mahalanobis distance','Norm of normalized residuals'])
plt.show()
#%%