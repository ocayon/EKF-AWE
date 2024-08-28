import numpy as np

def calculate_weighted_least_squares(y, A, W):
    x_hat = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ y
    return x_hat

def find_turn_rate_law(yaw_rate, apparent_windspeed, steering_input, elevation, yaw, kite,stdv_meas, coordinated_turn = True, va_eq_vk = True, mass_effects = False):
    """
    Find the turn rate law of the kite.
    """

    if coordinated_turn:
        if va_eq_vk:
            if mass_effects:
                # construct the A matrix
                A = np.vstack([apparent_windspeed*steering_input, np.sin(yaw)*np.cos(elevation)/apparent_windspeed]).T
            else:
                # construct the A matrix
                A = np.vstack([apparent_windspeed*steering_input]).T


    coeffs = calculate_weighted_least_squares(yaw_rate, A, np.diag(stdv_meas**2))

    return coeffs
