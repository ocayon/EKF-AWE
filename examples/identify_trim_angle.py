import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
import pandas as pd

# Example usage
plt.close("all")
config_file_name = "v9_config.yaml"
config = load_config("examples/" + config_file_name)

# Load results and flight data and plot kite reference frame
cut = 10000
results, flight_data = read_results(
    str(config["year"]),
    str(config["month"]),
    str(config["day"]),
    config["kite"]["model_name"],
)
# results1, flight_data1 = read_results('2023','10','26','v9')
results = results.iloc[cut:-cut]
# results1 = results1.iloc[cut:-cut]

# # concatenate dataframe pandas
# results = pd.concat([results,results1])

flight_data = flight_data.iloc[cut:-cut]
# flight_data1 = flight_data1.iloc[cut:-cut]
# flight_data = pd.concat([flight_data,flight_data1])

# results = results[flight_data['kite_angle_of_attack']<14]
# flight_data = flight_data[flight_data['kite_angle_of_attack']<14]

results = results.reset_index(drop=True)
flight_data = flight_data.reset_index(drop=True)


# %% AERO COEFFICIENTS IDENTIFICATION
def calculate_weighted_least_squares(y, A, W):
    x_hat = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ y
    return x_hat
# def calculate_ls_estimation(alphas, us, up, coeffs):


### Create file with results
alpha = np.array(flight_data["kite_angle_of_attack"])
# alpha = np.array(results['aoa_IMU_0'])
plt.figure()
plt.plot(alpha)
# alpha = np.array(results['aoa_IMU_0'])
# alpha = results['aoa_IMU_0']
mean_alpha = np.mean(alpha[flight_data['powered'] == 'powered'])
std_alpha = np.std(alpha[flight_data['powered'] == 'powered'])
print(f"Mean alpha powered: {mean_alpha}")
print(f"Std alpha depowered: {std_alpha}")

plt.plot(alpha)
us = np.array(flight_data["us"])
us = np.concatenate((np.zeros(12), us[:-12]))
up = np.array(flight_data["up"])
up = np.concatenate((np.zeros(12), up[:-12]))
p = np.array(results["omega_p"])
q = np.array(results["omega_q"])
r = np.array(results["omega_r"])
# Create least squares matrix
# x: [1, alpha, alpha^2, us, us^2, up, up^2]
A = np.vstack([np.ones_like(alpha), abs(up), abs(up)**2]).T
W = np.eye(len(alpha))
mask = (alpha < 0) & (alpha > 15)
W[mask, mask] = 0.1
# Solve for coefficients
coeffs = calculate_weighted_least_squares(alpha, A, W)

print(f"Coefficients: {coeffs}")
# Calculate estimated CL
aoa_est = A @ coeffs

# calculate the mean squared error
mse = np.mean((alpha - aoa_est)**2)
print(f"MSE: {mse}")
print(f"MRSE: {np.sqrt(mse)}")

# Plot results
plt.figure()
plt.plot(alpha, label="alpha")
plt.plot(aoa_est, label="CL_est")
plt.legend()
plt.show()

# Plot 

# %% TRIM ANGLE IDENTIFICATION
# Create file with results
