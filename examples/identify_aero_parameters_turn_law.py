import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
import pandas as pd

# Example usage
plt.close("all")
config_file_name = "v3_config.yaml"
config = load_config("examples/" + config_file_name)

# Load results and flight data and plot kite reference frame
cut = 80000
results, flight_data,_ = read_results(
    str(config["year"]),
    str(config["month"]),
    str(config["day"]),
    config["kite"]["model_name"],
)


# %% AERO COEFFICIENTS IDENTIFICATION
def calculate_weighted_least_squares(y, A, W):
    x_hat = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ y
    return x_hat

def construct_A_matrix_model_1(alpha):
    # C_L = a_0 + a_1 * alpha
    A = np.vstack([np.ones_like(alpha), alpha]).T
    return A

def construct_A_matrix_model_2(alpha):
    # C_L = a_0 + a_1 * alpha + a_2 * alpha^2
    A = np.vstack([np.ones_like(alpha), alpha, alpha**2]).T
    return A

def construct_A_matrix_model_3(alpha, us):
    # C_L = a_0 + a_1 * alpha + a_2 * alpha^2 + a_3 * us
    A = np.vstack([np.ones_like(alpha), alpha, alpha**2, us]).T
    return A

def construct_A_matrix_model_4(alpha, up,us):
    # C_L = a_0 + a_1 * alpha + a_2 * alpha^2 + a_3 * up + a_4 * us
    A = np.vstack([np.ones_like(alpha), alpha, up, us]).T
    return A

def construct_A_matrix_model_5(alpha, up, us):
    # C_L = a_0 + a_1 * alpha + a_2 * alpha^2 + a_3 * up + a_4 * us + a_5 * alpha * us + a_6 * alpha * up + a_7 * up * us 
    A = np.vstack([np.ones_like(alpha), alpha, alpha**2,up, us, alpha*us, alpha*up, up*us]).T
    return A

def construct_A_matrix_model_6(up,us):
    # C_L = a_0 + a_1 * up + a_2 * us + a_3 * up^2 + a_4 * us^2 + a_5 * up * us
    A = np.vstack([np.ones_like(up), up, us, up**2, us**2, up*us]).T
    return A

mask = flight_data["cycle"].isin([65,66])
# 
us = abs(np.array(flight_data["us"]))
up = np.array(flight_data["up"])
alpha = np.deg2rad(np.array(results["wing_angle_of_attack_bridle"])-5*us)
ss = np.array(results["wing_sideslip_angle_bridle"])

mean_alpha = np.mean(alpha[flight_data['powered'] == 'powered'])
std_alpha = np.std(alpha[flight_data['powered'] == 'powered'])
print(f"Mean alpha powered: {mean_alpha}", f"Std alpha powered: {std_alpha}")

mean_alpha = np.mean(alpha[flight_data['powered'] == 'depowered'])
std_alpha = np.std(alpha[flight_data['powered'] == 'depowered'])
print(f"Mean alpha depowered: {mean_alpha}", f"Std alpha depowered: {std_alpha}")

yaw_rate = np.array(results["yaw_rate"])
mass = config['kite']['mass']
yaw = flight_data['kite_yaw_0']
elevation = flight_data['kite_elevation']
models = {
    "Model 1": construct_A_matrix_model_1(alpha[mask]),
    "Model 2": construct_A_matrix_model_2(alpha[mask]),
    "Model 3": construct_A_matrix_model_3(alpha[mask], us[mask]),
    "Model 4": construct_A_matrix_model_4(alpha[mask], up[mask], us[mask]),
    "Model 5": construct_A_matrix_model_5(alpha[mask],up[mask],us[mask]),
    "Model 6": construct_A_matrix_model_6(up[mask],us[mask]),
}

W = np.eye(len(alpha[mask]))

results_dict = {}
for model_name, A in models.items():
    coeffs = calculate_weighted_least_squares(results[mask]["wing_lift_coefficient"], A, W)
    results_dict[model_name] = {"coeffs": coeffs}
mask = (flight_data["cycle"]>30)&(flight_data["cycle"]<70)
# mask = np.bool_(np.ones_like(flight_data["cycle"]))
models = {
    "Model 1": construct_A_matrix_model_1(alpha[mask]),
    "Model 2": construct_A_matrix_model_2(alpha[mask]),
    "Model 3": construct_A_matrix_model_3(alpha[mask], us[mask]),
    "Model 4": construct_A_matrix_model_4(alpha[mask], up[mask], us[mask]),
    "Model 5": construct_A_matrix_model_5(alpha[mask],up[mask],us[mask]),
    "Model 6": construct_A_matrix_model_6(up[mask],us[mask]),
}

for model_name, A in models.items(): 
    CL_est = A @ results_dict[model_name]["coeffs"]
    mse = np.mean((results[mask]["wing_lift_coefficient"] - CL_est) ** 2)
    print(f"{model_name} - MSE: {mse}")
    print(f"Coefficients for {model_name}: {results_dict[model_name]['coeffs']}")
    results_dict[model_name]["CL_est"] = CL_est
    results_dict[model_name]["MSE"] = mse

plt.figure()
for model_name, result in results_dict.items():
    if results_dict[model_name]["MSE"] < 0.002:
        plt.plot(
            flight_data[mask]["time"],
            result["CL_est"],
            label=f"{model_name} (MSE: {result['MSE']:.3f})"
        )

plt.plot(flight_data["time"], results["wing_lift_coefficient"], label="Measured CL", color="black", alpha=0.5)
plt.xlabel("Time [s]")
plt.ylabel("Lift coefficient")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.scatter(alpha[mask], results[mask]["wing_lift_coefficient"], label="Measured CL", color="black", alpha=0.2)

for model_name, result in results_dict.items():
    if results_dict[model_name]["MSE"] < 0.002:
        plt.scatter(
            alpha[mask],
            result["CL_est"],
            label=f"{model_name} (MSE: {result['MSE']:.3f})",
            alpha=0.2
        )

plt.xlabel("Time [s]")
plt.ylabel("Lift coefficient")
plt.legend()
plt.grid(True)
plt.show()




aoa_plot = np.deg2rad(np.linspace(-10, 20, 100))

models = {
    "Model 1": construct_A_matrix_model_1(aoa_plot),
    "Model 2": construct_A_matrix_model_2(aoa_plot),
    "Model 3": construct_A_matrix_model_3(aoa_plot, np.ones_like(aoa_plot)),
    "Model 4": construct_A_matrix_model_4(aoa_plot, np.ones_like(aoa_plot), np.zeros_like(aoa_plot)),
    "Model 5": construct_A_matrix_model_5(aoa_plot, np.ones_like(aoa_plot), np.zeros_like(aoa_plot)),
}

plt.figure()
plt.plot(alpha, results["wing_lift_coefficient"], 'o', label="Measured CL", color="black", alpha=0.1)
for model_name, A in models.items():
    CL_est = A @ results_dict[model_name]["coeffs"]
    plt.plot(aoa_plot, CL_est, label=model_name)

plt.xlabel("Angle of attack [deg]")
plt.ylabel("Lift coefficient")
plt.legend()
plt.grid(True)
plt.show()

