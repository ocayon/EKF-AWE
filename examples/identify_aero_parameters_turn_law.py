import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
from awes_ekf.utils import calculate_weighted_least_squares

# Example usage
plt.close("all")
config_file_name = "v3_config.yaml"
config = load_config()

# Load results and flight data and plot kite reference frame
cut = 80000
results, flight_data,_ = read_results(
    str(config["year"]),
    str(config["month"]),
    str(config["day"]),
    config["kite"]["model_name"],
    addition="_lt",
)

def construct_A_matrix(dependencies, **kwargs):
    """
    Constructs the A matrix based on the dependencies provided.
    
    Parameters:
        dependencies (list): List of strings representing dependencies for the model.
                             Each string should be a valid Python expression involving the inputs.
        kwargs (dict): Dictionary of inputs where keys match variable names in dependencies.
    
    Returns:
        np.array: The A matrix for the regression.
    """
    A = []
    global_scope = {"np": np}  # Include np in global scope for eval
    global_scope.update(kwargs)  # Add input variables to the scope
    
    for dep in dependencies:
        term = eval(dep, global_scope)
        A.append(term)
    return np.vstack(A).T

def fit_and_evaluate_model(data, dependencies, **kwargs):
    """
    Fits a model using weighted least squares and prints mean squared error.
    
    Parameters:
        data (np.array): The dependent variable data (e.g., CL, CD).
        dependencies (list): List of dependencies in string format for model construction.
        weights (np.array): Weight matrix for the weighted least squares calculation.
        kwargs (dict): Dictionary of inputs like alpha, up, us, etc. required by the dependencies.
        
    Returns:
        dict: Coefficients and Mean Squared Error (MSE).
    """
    # Construct A matrix with the specified dependencies
    A = construct_A_matrix(dependencies, **kwargs)
    # Calculate coefficients using weighted least squares
    coeffs = calculate_weighted_least_squares(data, A)
    # Calculate estimated values
    data_est = A @ coeffs
    # Mean Squared Error
    mse = np.mean((data - data_est) ** 2)
    # Print results
    print(f"Coefficients: {coeffs}")
    print(f"Mean Squared Error: {mse}")
    return {"coeffs": coeffs, "MSE": mse, "data_est": data_est}

# Define the dependencies and data inputs
dependencies = [
    "np.ones_like(alpha)",  # a_0
    "alpha",                # a_1 * alpha
    "alpha**2",             # a_2 * alpha^2
    "up",                   # a_3 * up
    "us",                   # a_4 * us
    "alpha * us",           # a_5 * alpha * us
    "alpha * up",           # a_6 * alpha * up
    "up * us"               # a_7 * up * us
]

# Mask and input data (as per example)
mask = flight_data["cycle"].isin([65, 66])
alpha = np.deg2rad(np.array(results["wing_angle_of_attack_bridle"]))[mask]
up = np.array(flight_data["up"])[mask]
us = abs(np.array(flight_data["us"]))[mask]
data = results[mask]["wing_lift_coefficient"]  # or wing_drag_coefficient or other target data

# Call the function to fit the model and evaluate
results = fit_and_evaluate_model(
    data,
    dependencies,
    alpha=alpha,
    up=up,
    us=us
)

# Plot results (optional)
plt.figure()
plt.plot(flight_data["time"][mask], data, label="Measured Data", color="black", alpha=0.5)
plt.plot(flight_data["time"][mask], results["data_est"], label="Model Estimation")
plt.xlabel("Time [s]")
plt.ylabel("Coefficient")
plt.legend()
plt.grid(True)
plt.show()