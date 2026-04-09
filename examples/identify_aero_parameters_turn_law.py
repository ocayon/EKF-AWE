import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
from awes_ekf.utils import (
    calculate_weighted_least_squares,
    calculate_turn_rate_law,
    fit_rational_law,
)

# Example usage
plt.close("all")
# config = load_config()

# Load results and flight data and plot kite reference frame
cut = 20000
results, flight_data, _ = read_results(
    "2019",
    "10",
    "08",
    "v3",
    addition="_va",
)
results = results[cut:-cut]
flight_data = flight_data[cut:-cut]
results = results[
    flight_data["up"] < 0.1
]  # Filter for up < 0.5 m/s to focus on low lift conditions
flight_data = flight_data[
    flight_data["up"] < 0.1
]  # Filter for up < 0.5 m/s to focus on low lift conditions

flight_data = flight_data.reset_index(drop=True)
results = results.reset_index(drop=True)


if "kite_yaw_rate" not in flight_data.columns:
    flight_data["kite_yaw_rate"] = np.convolve(
        np.gradient(flight_data["kite_yaw_0"], flight_data["time"]),
        np.ones(20) / 20,
        mode="same",
    )

plt.plot(flight_data["time"], results["wing_lift_coefficient"], label="CL")
plt.xlabel("Time [s]")
plt.ylabel("CL")
# plt.show()


plt.figure()
plt.plot(
    flight_data["time"],
    (flight_data["kite_yaw_1"] + np.pi / 2) % (2 * np.pi),
    label="Measured Yaw",
    color="black",
)
plt.plot(flight_data["time"], flight_data["kite_course"], label="Course Angle")
plt.xlabel("Time [s]")
plt.ylabel("Angle [rad]")
plt.legend()
plt.grid(True)
plt.show()
# yaw_rate_meas = np.convolve(
#     flight_data["kite_yaw_rate_1"], np.ones(20) / 20, mode="same"
# )
# flight_data["kite_yaw_rate"] = yaw_rate_meas
yaw_rate_meas = flight_data["kite_yaw_rate"]
yaw_rate, coeffs = calculate_turn_rate_law(
    results, flight_data, model="simple", steering_offset=False
)
print(f"Coefficients: {coeffs}")

plt.figure()
plt.plot(flight_data["time"], yaw_rate_meas, label="Measured Yaw Rate", color="black")
plt.plot(flight_data["time"], yaw_rate, label="Estimated Yaw Rate")

mse = np.mean((yaw_rate - yaw_rate_meas) ** 2)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}, MSE: {mse:.4f}")
yaw_rate, coeffs = calculate_turn_rate_law(
    results, flight_data, model="simple_weight", steering_offset=False
)
plt.plot(flight_data["time"], yaw_rate, label="Estimated Yaw Rate")

mse = np.mean((yaw_rate - yaw_rate_meas) ** 2)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}, MSE: {mse:.4f}")
yaw_rate, coeffs = calculate_turn_rate_law(
    results, flight_data, model="full", steering_offset=False
)
plt.plot(flight_data["time"], yaw_rate, label="Estimated Yaw Rate")

mse = np.mean((yaw_rate - yaw_rate_meas) ** 2)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}, MSE: {mse:.4f}")


us = flight_data["kcu_actual_steering"] / 100
va = results["kite_apparent_windspeed"]
vk = np.sqrt(
    results["kite_velocity_x"] ** 2
    + results["kite_velocity_y"] ** 2
    + results["kite_velocity_z"] ** 2
)
yaw = flight_data["kite_yaw_0"]
course = flight_data["kite_course"]
beta = flight_data["kite_elevation"]
mass = 60


coeffs, res = fit_rational_law(
    yaw_rate_meas,
    us,
    va,
    vk,
    mass,
    beta,
    course,
)

print(f"Coefficients: {coeffs}")
print(f"Res: {res}")


plt.xlabel("Time [s]")
plt.ylabel("Yaw Rate [rad/s]")
plt.legend()
plt.grid(True)
plt.show()


def identify_course_rate_least_squares(
    flight_data,
    results,
    smooth_window=21,
    v_tau_epsilon=1e-6,
):
    """Estimate course rate coefficients k1*, k2* with weighted least squares."""

    time = flight_data["time"].to_numpy()
    course = np.unwrap(flight_data["kite_course"].to_numpy())
    course_rate = np.gradient(course, time)

    if smooth_window and smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        course_rate = np.convolve(course_rate, kernel, mode="same")

    us = flight_data["kcu_actual_steering"].to_numpy() / 100
    va = results["kite_apparent_windspeed"].to_numpy()
    v_tau = np.sqrt(
        results["kite_velocity_x"] ** 2
        + results["kite_velocity_y"] ** 2
        + results["kite_velocity_z"] ** 2
    ).to_numpy()
    v_tau = np.maximum(v_tau, v_tau_epsilon)

    psi = (
        flight_data["kite_yaw_0"].to_numpy() if "kite_yaw_0" in flight_data else course
    )
    beta = (
        results["kite_elevation"].to_numpy()
        if "kite_elevation" in results
        else flight_data["kite_elevation"].to_numpy()
    )
    varphi = (
        results["tether_roll"].to_numpy()
        if "tether_roll" in results
        else np.zeros_like(beta)
    )

    term1 = va**2 * us / v_tau
    term2 = np.sin(psi) * np.cos(beta + varphi) / v_tau
    A = np.vstack([term1, term2]).T

    valid_mask = np.isfinite(A).all(axis=1) & np.isfinite(course_rate)
    course_rate = course_rate[valid_mask]
    A = A[valid_mask]
    time = time[valid_mask]
    us = us[valid_mask]
    va = va[valid_mask]

    coeffs = calculate_weighted_least_squares(course_rate, A)
    course_rate_est = A @ coeffs

    mse = np.mean((course_rate - course_rate_est) ** 2)
    rmse = np.sqrt(mse)

    return {
        "coeffs": coeffs,
        "course_rate_meas": course_rate,
        "course_rate_est": course_rate_est,
        "time": time,
        "us_va": us * va,
        "us_va2": us * va**2 / v_tau,
        "mse": mse,
        "rmse": rmse,
    }


course_fit = identify_course_rate_least_squares(flight_data, results)
print(
    f"Course-rate coeffs k1*={course_fit['coeffs'][0]:.4f}, k2*={course_fit['coeffs'][1]:.4f}, "
    f"RMSE={course_fit['rmse']:.4f}, MSE={course_fit['mse']:.4f}"
)

plt.figure()
plt.plot(
    course_fit["time"],
    course_fit["course_rate_meas"],
    label="Measured Course Rate",
    color="black",
)
plt.plot(
    course_fit["time"], course_fit["course_rate_est"], label="Estimated Course Rate"
)
plt.xlabel("Time [s]")
plt.ylabel("Course Rate [rad/s]")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.scatter(course_fit["us_va"], course_fit["course_rate_meas"], s=8, alpha=0.6)
plt.scatter(course_fit["us_va"], course_fit["course_rate_est"], s=8, alpha=0.6)
plt.xlabel("u_s v_a")
plt.ylabel("Course Rate [rad/s]")
plt.grid(True)

plt.figure()
plt.scatter(course_fit["us_va2"], course_fit["course_rate_meas"], s=8, alpha=0.6)
plt.scatter(course_fit["us_va2"], course_fit["course_rate_est"], s=8, alpha=0.6)
plt.xlabel("(u_s v_a)^2")
plt.ylabel("Course Rate [rad/s]")
plt.grid(True)
plt.show()


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
    "alpha",  # a_1 * alpha
    "alpha**2",  # a_2 * alpha^2
    "up",  # a_3 * up
    "us",  # a_4 * us
    # "alpha * us",           # a_5 * alpha * us
    # "alpha * up",           # a_6 * alpha * up
    # "up * us"               # a_7 * up * us
]

# Mask and input data (as per example)
mask = flight_data["cycle"].isin([65, 66])
alpha = np.deg2rad(np.array(results["wing_angle_of_attack_bridle"]))[mask]
up = np.array(flight_data["up"])[mask]
us = abs(np.array(flight_data["us"]))[mask]
data = results[mask][
    "wing_lift_coefficient"
]  # or wing_drag_coefficient or other target data

# Call the function to fit the model and evaluate
fit = fit_and_evaluate_model(data, dependencies, alpha=alpha, up=up, us=us)

# Plot results (optional)
plt.figure()
plt.plot(
    flight_data["time"][mask], data, label="Measured Data", color="black", alpha=0.5
)
plt.plot(flight_data["time"][mask], fit["data_est"], label="Model Estimation")
plt.xlabel("Time [s]")
plt.ylabel("Coefficient")
plt.legend()
plt.grid(True)
plt.show()
