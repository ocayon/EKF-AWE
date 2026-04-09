import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.load_data.read_data import read_results
from awes_ekf.utils import calculate_weighted_least_squares, calculate_turn_rate_law


plt.close("all")


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
        "us_va2": (us * va) ** 2,
        "mse": mse,
        "rmse": rmse,
    }


def identify_yaw_rate_simple(flight_data, results, model="simple"):
    yaw_rate_meas = flight_data["kite_yaw_rate"]
    yaw_rate_est, coeffs = calculate_turn_rate_law(
        results, flight_data, model=model, steering_offset=False
    )
    mse = np.mean((yaw_rate_est - yaw_rate_meas) ** 2)
    rmse = np.sqrt(mse)

    us = flight_data["kcu_actual_steering"].to_numpy() / 100
    va = results["kite_apparent_windspeed"].to_numpy()

    return {
        "coeffs": coeffs,
        "yaw_rate_meas": yaw_rate_meas,
        "yaw_rate_est": yaw_rate_est,
        "time": flight_data["time"].to_numpy(),
        "us_va": us * va,
        "mse": mse,
        "rmse": rmse,
    }


def main():
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

    # Filter: focus on low up to keep comparable operating point
    mask = flight_data["up"] < 0.1
    mask = mask & (flight_data.cycle == 65)
    flight_data = flight_data[mask].reset_index(drop=True)
    results = results[mask].reset_index(drop=True)

    if "kite_yaw_rate" not in flight_data.columns:
        flight_data["kite_yaw_rate"] = np.convolve(
            np.gradient(flight_data["kite_yaw_0"], flight_data["time"]),
            np.ones(20) / 20,
            mode="same",
        )

    course_fit = identify_course_rate_least_squares(flight_data, results)
    yaw_fit = identify_yaw_rate_simple(flight_data, results, model="simple_weight")

    print(
        f"Course-rate coeffs k1*={course_fit['coeffs'][0]:.4f}, k2*={course_fit['coeffs'][1]:.4f}, "
        f"RMSE={course_fit['rmse']:.4f}, MSE={course_fit['mse']:.4f}"
    )
    print(
        f"Yaw-rate coeffs={yaw_fit['coeffs']}, RMSE={yaw_fit['rmse']:.4f}, MSE={yaw_fit['mse']:.4f}"
    )

    # Time series comparison
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

    plt.figure()
    plt.plot(
        yaw_fit["time"],
        yaw_fit["yaw_rate_meas"],
        label="Measured Yaw Rate",
        color="black",
    )
    plt.plot(yaw_fit["time"], yaw_fit["yaw_rate_est"], label="Estimated Yaw Rate")
    plt.xlabel("Time [s]")
    plt.ylabel("Yaw Rate [rad/s]")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.plot(
        course_fit["time"],
        course_fit["course_rate_meas"],
        label="Measured Course Rate",
        color="tab:blue",
    )
    plt.plot(
        yaw_fit["time"],
        yaw_fit["yaw_rate_meas"],
        label="Measured Yaw Rate",
        color="tab:orange",
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Rate [rad/s]")
    plt.legend()
    plt.grid(True)

    plt.figure()
    course_angle = np.mod(np.degrees(flight_data["kite_course"].to_numpy()), 360)
    yaw_angle = np.mod(np.degrees(flight_data["kite_yaw_0"].to_numpy()) + 90, 360)
    plt.plot(flight_data["time"], course_angle, label="Course", color="tab:blue")
    plt.plot(flight_data["time"], yaw_angle, label="Yaw", color="tab:orange")
    plt.xlabel("Time [s]")
    plt.ylabel("Angle [deg]")
    plt.legend()
    plt.grid(True)

    # Scatter diagnostics with transparency
    plt.figure()
    plt.scatter(
        course_fit["us_va"],
        course_fit["course_rate_meas"],
        s=10,
        alpha=0.35,
        label="Measured",
    )
    plt.scatter(
        course_fit["us_va"],
        course_fit["course_rate_est"],
        s=10,
        alpha=0.35,
        label="Estimated",
    )
    plt.xlabel("u_s v_a")
    plt.ylabel("Course Rate [rad/s]")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.scatter(
        course_fit["us_va2"],
        course_fit["course_rate_meas"],
        s=10,
        alpha=0.35,
        label="Measured",
    )
    plt.scatter(
        course_fit["us_va2"],
        course_fit["course_rate_est"],
        s=10,
        alpha=0.35,
        label="Estimated",
    )
    plt.xlabel("(u_s v_a)^2")
    plt.ylabel("Course Rate [rad/s]")
    plt.legend()
    plt.grid(True)

    plt.figure()
    plt.scatter(
        yaw_fit["us_va"], yaw_fit["yaw_rate_meas"], s=10, alpha=0.35, label="Measured"
    )
    plt.scatter(
        yaw_fit["us_va"], yaw_fit["yaw_rate_est"], s=10, alpha=0.35, label="Estimated"
    )
    plt.xlabel("u_s v_a")
    plt.ylabel("Yaw Rate [rad/s]")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
