import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import least_squares as _scipy_least_squares
from awes_ekf.plotting.plot_utils import plot_time_series
from awes_ekf.plotting.color_palette import set_plot_style_no_latex, get_color_list
import scipy.stats as stats
from scipy.stats import linregress
from awes_ekf.utils import (
    calculate_turn_rate_law,
    find_time_delay,
    calculate_weighted_least_squares,
)

colors = get_color_list()


def plot_aerodynamics(results, flight_data, config_data):
    set_plot_style_no_latex()

    plot_aerodynamic_coefficients(flight_data, results, config_data)

    date = config_data["year"] + "-" + config_data["month"] + "-" + config_data["day"]
    plot_polars(
        results, flight_data, date=date, model=config_data["kite"]["model_name"]
    )

    if "kcu_actual_steering" in flight_data.columns:
        plot_identify_turn_dynamics(results, flight_data, config_data)
    try:
        CL_mean_powered = results["wing_lift_coefficient"][
            flight_data["powered"] == "powered"
        ].mean()
        CD_mean_powered = results["wing_drag_coefficient"][
            flight_data["powered"] == "powered"
        ].mean()
        print(f"Mean CL powered: {CL_mean_powered:.2f}")
        print(f"Mean CD powered: {CD_mean_powered:.2f}")

        for col in results.columns:
            if "drag_coefficient" in col:
                print(
                    f"Mean {col} powered: {results[col][flight_data['powered'] == 'powered'].mean():.2f}"
                )
        CD_mean_powered_sys = np.mean(
            results["wing_drag_coefficient"][flight_data["powered"] == "depowered"]
            + results["kcu_drag_coefficient"][flight_data["powered"] == "depowered"]
            + results["bridles_drag_coefficient"][flight_data["powered"] == "depowered"]
        )
        print(f"Mean CD powered system: {CD_mean_powered_sys:.2f}")
        print(f"Mean CL/CD powered: {CL_mean_powered/CD_mean_powered:.2f}")

        std_CL_powered = results["wing_lift_coefficient"][
            flight_data["powered"] == "powered"
        ].std()
        std_CD_powered = results["wing_drag_coefficient"][
            flight_data["powered"] == "powered"
        ].std()
        print(f"Std CL powered: {std_CL_powered:.2f}")
        print(f"Std CD powered: {std_CD_powered:.2f}")

        CL_mean_depowered = results["wing_lift_coefficient"][
            flight_data["powered"] == "depowered"
        ].mean()
        CD_mean_depowered = results["wing_drag_coefficient"][
            flight_data["powered"] == "depowered"
        ].mean()
        print(f"Mean CL depowered: {CL_mean_depowered:.2f}")
        print(f"Mean CD depowered: {CD_mean_depowered:.2f}")
        print(f"Mean CL/CD depowered: {CL_mean_depowered/CD_mean_depowered:.2f}")

        mean_aoa_powered = results["wing_angle_of_attack_bridle"][
            flight_data["powered"] == "powered"
        ].mean()
        std_aoa_powered = results["wing_angle_of_attack_bridle"][
            flight_data["powered"] == "powered"
        ].std()
        mean_aoa_depowered = results["wing_angle_of_attack_bridle"][
            flight_data["powered"] == "depowered"
        ].mean()
        std_aoa_depowered = results["wing_angle_of_attack_bridle"][
            flight_data["powered"] == "depowered"
        ].std()
        print(f"Mean AoA powered: {mean_aoa_powered:.2f} ± {std_aoa_powered:.2f}")
        print(f"Mean AoA depowered: {mean_aoa_depowered:.2f} ± {std_aoa_depowered:.2f}")

        print(
            "Max aerodynamic roll (powered): {:.2f} deg".format(
                max(
                    np.degrees(
                        np.arctan2(
                            results["wing_sideforce_coefficient"][
                                flight_data["powered"] == "powered"
                            ],
                            results["wing_lift_coefficient"][
                                flight_data["powered"] == "powered"
                            ],
                        )
                    )
                )
            )
        )
        print(
            "Max aerodynamic roll (depowered): {:.2f} deg".format(
                max(
                    np.degrees(
                        np.arctan2(
                            results["wing_sideforce_coefficient"][
                                flight_data["powered"] == "depowered"
                            ],
                            results["wing_lift_coefficient"][
                                flight_data["powered"] == "depowered"
                            ],
                        )
                    )
                )
            )
        )
        print(
            "Min aerodynamic roll (powered): {:.2f} deg".format(
                min(
                    np.degrees(
                        np.arctan2(
                            results["wing_sideforce_coefficient"][
                                flight_data["powered"] == "powered"
                            ],
                            results["wing_lift_coefficient"][
                                flight_data["powered"] == "powered"
                            ],
                        )
                    )
                )
            )
        )
        print(
            "Min aerodynamic roll (depowered): {:.2f} deg".format(
                min(
                    np.degrees(
                        np.arctan2(
                            results["wing_sideforce_coefficient"][
                                flight_data["powered"] == "depowered"
                            ],
                            results["wing_lift_coefficient"][
                                flight_data["powered"] == "depowered"
                            ],
                        )
                    )
                )
            )
        )

    except KeyError as e:
        print(f"KeyError: {e}. Some aerodynamic coefficients may not be available.")

    plt.show()


def plot_identify_turn_dynamics(results, flight_data, config_data):
    # Turn rate law
    ts = np.mean(np.diff(flight_data["time"]))

    if "kite_yaw_rate" not in flight_data.columns:
        if "kite_yaw_rate_1" in flight_data.columns:
            flight_data["kite_yaw_rate"] = np.unwrap(flight_data["kite_yaw_rate_1"])
        else:
            yaw_rate = np.convolve(
                np.gradient(flight_data["kite_yaw_0"], ts),
                np.ones(10) / 10,
                mode="same",
            )
            flight_data["kite_yaw_rate"] = np.unwrap(yaw_rate)

    # Calculate time delay between yaw rate and steering input
    signal_delay, _ = find_time_delay(
        flight_data["kite_yaw_rate"], flight_data["kcu_actual_steering"]
    )
    time_delay = signal_delay * ts
    if abs(time_delay) > 3:
        print(
            "Warning: Time delay between yaw rate and steering input is very high. Inversing steering input."
        )
        flight_data["kcu_actual_steering"] = -flight_data["kcu_actual_steering"]
        flight_data["kcu_set_steering"] = -flight_data["kcu_set_steering"]
        signal_delay, _ = find_time_delay(
            flight_data["kite_yaw_rate"], flight_data["kcu_actual_steering"]
        )
        time_delay = signal_delay * ts
    print("Time delay turn rate:", time_delay)

    # Calculate time delay between aero roll and steering input
    signal_delay, _ = find_time_delay(
        np.arctan2(
            results["wing_sideforce_coefficient"], results["wing_lift_coefficient"]
        ),
        flight_data["kcu_actual_steering"],
    )
    flight_data["kcu_actual_steering_delay"] = np.roll(
        flight_data["kcu_actual_steering"], int(signal_delay)
    )
    time_delay = signal_delay * ts
    print("Time delay steering force:", time_delay)

    # Calculate yaw-rate fits for the two models used in turn-law identification.
    # two_fit is the no-inertial-forces model; full includes inertial effects.
    yaw_rate_two_fit, coeffs_two_fit = calculate_turn_rate_law(
        results, flight_data, model="simple_weight", steering_offset=True
    )
    yaw_rate_full, coeffs_full = calculate_turn_rate_law(
        results, flight_data, model="full", steering_offset=True
    )

    print("Yaw rate coefficients (two_fit, no inertial forces):", coeffs_two_fit)
    print("Yaw rate coefficients (full):", coeffs_full)

    # Calculate mean errors
    error_two_fit = abs(
        np.degrees(yaw_rate_two_fit) - np.degrees(flight_data["kite_yaw_rate"])
    )
    error_full = abs(
        np.degrees(yaw_rate_full) - np.degrees(flight_data["kite_yaw_rate"])
    )
    mean_error_two_fit = np.mean(error_two_fit)
    mean_error_full = np.mean(error_full)

    # Prepare data for the main scatter plot (yaw rate in deg/s)
    x = flight_data["kcu_actual_steering"] / 100 * results["kite_apparent_windspeed"]
    y = np.degrees(flight_data["kite_yaw_rate"])

    # Scatter and line plot
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.2, color=colors[2], label="Measured")
    plt.plot(
        x,
        yaw_rate_two_fit,
        label=f"two_fit (no inertial forces) (Mean Error: {mean_error_two_fit:.2f} deg/s)",
        linestyle="--",
    )

    plt.plot(
        x,
        yaw_rate_full,
        label=f"full (Mean Error: {mean_error_full:.2f} deg/s)",
        linestyle="-.",
    )

    plt.xlabel(r"$u_\mathrm{s} \cdot v_\mathrm{a}$ [m/s]")
    plt.ylabel("Kite Yaw Rate [rad/s]")
    plt.legend()
    plt.tight_layout()

    # Define least squares function
    def least_squares(x, y, A_matrix):
        m = len(y)
        n = A_matrix.shape[1]
        x_hat = np.linalg.inv(A_matrix.T @ A_matrix) @ A_matrix.T @ y
        residuals = y - np.dot(A_matrix, x_hat).reshape(-1)
        sigma_squared = (residuals.T @ residuals) / (m - n)
        measurement_error = np.sqrt(sigma_squared)
        print("Measurement error:", measurement_error)
        Qx = sigma_squared * np.linalg.inv(A_matrix.T @ A_matrix)
        return x_hat, Qx, measurement_error

    # Define a function for fitting and plotting with filled bounds
    def plot_fit(flight_data, results, condition):
        # Filter data based on condition
        mask = (
            flight_data["kcu_actual_steering_delay"] < 0
            if condition == "<0"
            else flight_data["kcu_actual_steering_delay"] > 0
        )
        x = np.array(flight_data[mask]["kcu_actual_steering_delay"]) / 100
        y = np.array(
            np.arctan2(
                results[mask]["wing_sideforce_coefficient"],
                results[mask]["wing_lift_coefficient"],
            )
        )

        # Construct A_matrix and fit parameters
        A_matrix = np.vstack([x, np.ones(len(x))]).T
        x_hat, Qx, error = least_squares(x, y, A_matrix)

        # Set x_range for each condition
        x_range = (
            np.linspace(min(x), 0, 100)
            if condition == "<0"
            else np.linspace(0, max(x), 100)
        )
        fit_line = x_hat[0] * x_range + x_hat[1]
        upper_bound = fit_line + error
        lower_bound = fit_line - error

        # Plotting
        plt.scatter(
            flight_data[mask]["kcu_actual_steering"] / 100,
            y,
            alpha=0.2,
            color=colors[2],
        )
        plt.scatter(
            flight_data[mask]["kcu_actual_steering_delay"] / 100,
            y,
            alpha=0.2,
            color=colors[3],
        )

        # Linear fit line
        plt.plot(x_range, fit_line, linestyle="--", color=colors[0])

        # Fill between upper and lower bounds
        plt.fill_between(x_range, lower_bound, upper_bound, alpha=0.6, color=colors[0])

        fit_us0 = fit_line[-1] if condition == "<0" else fit_line[0]
        # Add thicker gridlines at y=0 and x=0
        plt.axhline(fit_us0, linewidth=1.5, linestyle="--", color=colors[5])
        plt.axhline(0, color="black", linewidth=0.5)

        # Labels and legend
        plt.xlabel(r"$u_\mathrm{s}$")
        plt.ylabel(r"$\phi_\mathrm{aero}$")
        plt.legend()
        plt.tight_layout()

    plt.figure(figsize=(6, 4))
    plt.title(title)
    # Call the function for each condition
    plot_fit(flight_data, results, condition="<0")
    plot_fit(flight_data, results, condition=">0")
    plt.legend(["Linear Fit", "EKF", "EKF Delay Corrected"])

    # Print mean errors and standard deviations
    print("Mean error yaw rate (two_fit, no inertial forces):", mean_error_two_fit)
    print("Mean error yaw rate (full):", mean_error_full)
    print("Std error yaw rate (two_fit, no inertial forces):", np.std(error_two_fit))
    print("Std error yaw rate (full):", np.std(error_full))

    # Sideforce time series plot
    # fit_sideforce = slope * flight_data["kcu_actual_steering_delay"] / 100 + intercept
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    plot_time_series(
        flight_data,
        results["wing_sideforce_coefficient"],
        axs[0],
        ylabel="$C_{S}$",
        plot_phase=False,
        label="EKF 0",
    )
    # axs[0].plot(flight_data["time"], slope * flight_data["kcu_actual_steering"] / 100 + intercept, label="Linear Fit", linestyle="--")
    # axs[0].plot(flight_data["time"], fit_sideforce, label="Linear Fit - Delay Corrected", linestyle="--")
    axs[0].legend(frameon=True)

    # Yaw Rate Comparison
    axs[1].plot(
        flight_data["time"],
        np.degrees(flight_data["kite_yaw_rate"]),
        label="Measured Yaw Rate",
    )
    axs[1].plot(
        flight_data["time"],
        np.degrees(yaw_rate_two_fit),
        label="two_fit (no inertial forces)",
        linestyle="--",
    )
    axs[1].plot(
        flight_data["time"],
        np.degrees(yaw_rate_full),
        label="full",
        linestyle="-.",
    )
    axs[1].legend(frameon=True)
    axs[1].set_ylabel("Yaw Rate [deg/s]")

    # Steering Input
    plot_time_series(
        flight_data,
        -flight_data["kcu_actual_steering"] / 100,
        axs[2],
        ylabel="$u_s$",
        plot_phase=False,
        label="Actual steering",
    )
    plot_time_series(
        flight_data,
        -flight_data["kcu_set_steering"] / 100,
        axs[2],
        ylabel="$u_s$",
        plot_phase=False,
        label="Set steering",
    )
    axs[2].legend(frameon=True)
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylim([-0.4, 0.4])
    plt.tight_layout()


def plot_aerodynamic_coefficients(flight_data, results, config_data):
    # Smooth AoA if present
    if "bridle_angle_of_attack" in flight_data.columns:
        flight_data["bridle_angle_of_attack"] = np.convolve(
            flight_data["bridle_angle_of_attack"], np.ones(10) / 10, mode="same"
        )

    aoa_imu = None
    if "sensor_ids" in config_data["kite"] and all(
        f"wing_angle_of_attack_imu_{i}" in results.columns
        for i in config_data["kite"]["sensor_ids"]
    ):
        # Calculate mean AoA from IMU sensors if multiple IDs are provided
        aoa_imu = np.mean(
            [
                results[f"wing_angle_of_attack_imu_{i}"]
                for i in config_data["kite"]["sensor_ids"]
            ],
            axis=0,
        )

    # Prepare the plot
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Plot Lift Coefficient
    plot_time_series(
        flight_data,
        results["wing_lift_coefficient"],
        axs[0],
        ylabel="$C_L$",
        plot_phase=True,
    )

    # Plot Drag Coefficients
    plot_time_series(
        flight_data, results["wing_drag_coefficient"], axs[1], label="$C_\mathrm{D}$"
    )
    if "kcu_drag_coefficient" in results.columns:
        plot_time_series(
            flight_data,
            results["kcu_drag_coefficient"],
            axs[1],
            label="$C_\mathrm{D,kcu}$",
        )
    if "bridles_drag_coefficient" in results.columns:
        plot_time_series(
            flight_data,
            results["bridles_drag_coefficient"],
            axs[1],
            label="$C_\mathrm{D,bridles}$",
        )
    if "tether_drag_coefficient" in results.columns:
        plot_time_series(
            flight_data,
            results["tether_drag_coefficient"],
            axs[1],
            label="$C_\mathrm{D,t}$",
            plot_phase=True,
            ylabel="$C_D$",
        )
    axs[1].legend()

    # Plot Angle of Attack
    if "bridle_angle_of_attack" in flight_data.columns:
        plot_time_series(
            flight_data,
            flight_data["bridle_angle_of_attack"],
            axs[2],
            label=r"$\alpha_\mathrm{b}$ measured",
        )
    if aoa_imu is not None:
        plot_time_series(
            flight_data, aoa_imu, axs[2], label=r"$\alpha_\mathrm{w}$ from IMU"
        )
    if "wing_angle_of_attack_bridle" in results.columns:
        plot_time_series(
            flight_data,
            results["wing_angle_of_attack_bridle"],
            axs[2],
            label=r"$\mathbf{\alpha_w}$ from bridle",
        )
    if "wing_angle_of_attack" in results.columns:
        plot_time_series(
            flight_data,
            results["wing_angle_of_attack"],
            axs[2],
            label=r"$\alpha_\mathrm{w}$ from EKF",
        )
    if "kite_angle_of_attack" in results.columns:
        plot_time_series(
            flight_data,
            results["kite_angle_of_attack"],
            axs[2],
            label=r"$\alpha_\mathrm{k}$ from EKF",
            plot_phase=True,
            ylabel=r"$\alpha$ [$^\circ$]",
        )

    # Finalize the angle of attack plot
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel(r"$\alpha$ [$^\circ$]")
    axs[2].legend(frameon=True)

    # Plot Aerodynamic Roll (arctan(CS/CL))
    if "wing_sideforce_coefficient" in results.columns:
        aerodynamic_roll = np.degrees(
            np.arctan2(
                results["wing_sideforce_coefficient"], results["wing_lift_coefficient"]
            )
        )
        plot_time_series(
            flight_data,
            aerodynamic_roll,
            axs[3],
            ylabel=r"Roll [$^\circ$]",
            plot_phase=True,
        )
        axs[3].set_xlabel("Time [s]")
        axs[3].set_ylabel(r"Aerodynamic Roll [$^\circ$]")
        axs[3].legend(frameon=True)
    else:
        axs[3].text(
            0.5,
            0.5,
            "Sideforce coefficient not available",
            ha="center",
            va="center",
            transform=axs[3].transAxes,
        )
        axs[3].set_ylabel(r"Aerodynamic Roll [$^\circ$]")

    # Adjust layout
    plt.tight_layout()


def plot_polars(results, flight_data, date="", model="", label="Wing"):
    # Attempt to find angle of attack data in preferred order and notify which is used
    if (
        "wing_angle_of_attack_bridle" in results.columns
        and results["wing_angle_of_attack_bridle"].notna().any()
    ):
        alpha = results["wing_angle_of_attack_bridle"]
        aoa_label = "Wing AoA (Bridle)"
    elif (
        "wing_angle_of_attack" in results.columns
        and results["wing_angle_of_attack"].notna().any()
    ):
        alpha = results["wing_angle_of_attack"]
        aoa_label = "Wing AoA"
    elif "kite_angle_of_attack" in results.columns:
        alpha = results["kite_angle_of_attack"]
        aoa_label = "Kite AoA"
    else:
        raise ValueError("No suitable angle of attack data found in results.")

    print(f"Using angle of attack data: {aoa_label}")

    # Calculate cl and cd for the wing
    cl_wing = np.sqrt(
        results["wing_lift_coefficient"] ** 2
        + results["wing_sideforce_coefficient"] ** 2
    )
    cd_wing = results["wing_drag_coefficient"]

    # Define binning for angle of attack data
    step = 1
    bins = np.arange(int(alpha.min()) - step / 2, int(alpha.max()) + step / 2, step)
    bin_indices = np.digitize(alpha, bins)
    num_bins = len(bins)

    # Calculate mean CL, CD, and derived quantities for each bin
    cl_wing_means = np.array(
        [cl_wing[bin_indices == i].mean() for i in range(1, num_bins)]
    )
    cd_wing_means = np.array(
        [cd_wing[bin_indices == i].mean() for i in range(1, num_bins)]
    )
    cl_cd_ratio = cl_wing_means / cd_wing_means
    cl_cubed_cd_squared_ratio = (cl_wing_means**3) / (cd_wing_means**2)

    # Calculate SEM (Standard Error of the Mean) and 99% confidence intervals
    cl_wing_sems = np.array(
        [
            cl_wing[bin_indices == i].std() / np.sqrt(np.sum(bin_indices == i))
            for i in range(1, num_bins)
        ]
    )
    cd_wing_sems = np.array(
        [
            cd_wing[bin_indices == i].std() / np.sqrt(np.sum(bin_indices == i))
            for i in range(1, num_bins)
        ]
    )

    confidence_level = 0.99
    z = stats.norm.ppf(0.5 + confidence_level / 2)  # 99% confidence level
    cl_wing_cis = z * cl_wing_sems
    cd_wing_cis = z * cd_wing_sems

    # Calculate bin centers for plotting
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Create a figure with four subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), sharex=True)
    fig.suptitle(
        f"{label} Polar Analysis - Mean and 99% Confidence Intervals\nUsing {aoa_label}",
        fontsize=16,
    )

    # CL plot
    axs[0, 0].plot(
        bin_centers, cl_wing_means, "o-", markersize=8, linewidth=1.5, label=label
    )
    axs[0, 0].fill_between(
        bin_centers, cl_wing_means - cl_wing_cis, cl_wing_means + cl_wing_cis, alpha=0.2
    )
    axs[0, 0].set_title("$C_L$ with 99% Confidence Interval", fontsize=12)
    axs[0, 0].set_ylabel("$C_L$", fontsize=14)
    axs[0, 0].set_xlabel(r"$\alpha$ [$^\circ$]", fontsize=14)
    axs[0, 0].grid(True)

    # CD plot
    axs[0, 1].plot(
        bin_centers, cd_wing_means, "o-", markersize=8, linewidth=1.5, label=label
    )
    axs[0, 1].fill_between(
        bin_centers, cd_wing_means - cd_wing_cis, cd_wing_means + cd_wing_cis, alpha=0.2
    )
    axs[0, 1].set_title("$C_D$ with 99% Confidence Interval", fontsize=12)
    axs[0, 1].set_ylabel("$C_D$", fontsize=14)
    axs[0, 1].set_xlabel(r"$\alpha$ [$^\circ$]", fontsize=14)
    axs[0, 1].grid(True)

    # CL/CD plot
    axs[1, 0].plot(
        bin_centers,
        cl_cd_ratio,
        "o-",
        markersize=8,
        linewidth=1.5,
        label=f"{label} $C_L / C_D$",
    )
    axs[1, 0].set_title(r"$\frac{C_L}{C_D}$ with 99% Confidence Interval", fontsize=12)
    axs[1, 0].set_ylabel(r"$\frac{C_L}{C_D}$", fontsize=14)
    axs[1, 0].set_xlabel(r"$\alpha$ [$^\circ$]", fontsize=14)
    axs[1, 0].grid(True)

    # CL^3/CD^2 plot
    axs[1, 1].plot(
        bin_centers,
        cl_cubed_cd_squared_ratio,
        "o-",
        markersize=8,
        linewidth=1.5,
        label=f"{label} $C_L^3 / C_D^2$",
    )
    axs[1, 1].set_title(
        r"$\frac{C_L^3}{C_D^2}$ with 99% Confidence Interval", fontsize=12
    )
    axs[1, 1].set_ylabel(r"$\frac{C_L^3}{C_D^2}$", fontsize=14)
    axs[1, 1].set_xlabel(r"$\alpha$ [$^\circ$]", fontsize=14)
    axs[1, 1].grid(True)

    # Mean angle of attack lines (if available)
    if "powered" in flight_data.columns:
        mean_aoa_pow = np.mean(alpha[flight_data["powered"] == "powered"])
        mean_aoa_dep = np.mean(alpha[flight_data["powered"] == "depowered"])
        for ax in axs.flat:
            ax.axvline(x=mean_aoa_pow, linestyle="--", label="Mean reel-out AoA")
            ax.axvline(x=mean_aoa_dep, linestyle="--", label="Mean reel-in AoA")

    # for i in range(2):
    #     for j in range(2):
    #         axs[i, j].set_xlim([mean_aoa_dep -2, mean_aoa_pow + 2])

    # Add legend to the first subplot
    axs[0, 0].legend(loc="lower right")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle

    # Save polars as csv
    polars = np.vstack([bin_centers, cl_wing_means, cd_wing_means]).T
    np.savetxt(
        "results/polars/polar_" + model + "_" + date + ".csv",
        polars,
        delimiter=",",
        header="aoa,cl,cd",
        comments="",
    )


def plot_steering_vs_aero_roll(results, flight_data):
    """Plot scatter of steering input vs sideforce coefficient."""
    if "kcu_actual_steering" not in flight_data.columns:
        print("kcu_actual_steering not found in flight_data")
        return
    if "wing_sideforce_coefficient" not in results.columns:
        print("wing_sideforce_coefficient not found in results")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(
        flight_data["kcu_actual_steering"] / 100,
        np.arctan2(
            results["wing_sideforce_coefficient"], results["wing_lift_coefficient"]
        ),
        alpha=0.3,
        color=colors[2],
        s=20,
    )
    plt.xlabel(r"Steering Input $u_s$ [normalized]")
    plt.ylabel(r"Sideforce Coefficient $C_S$")
    plt.title(r"Steering Input vs Sideforce Coefficient")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


# ── Turn-rate law identification helpers ──────────────────────────────────────


def _rmse_turn(y_true, y_pred):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    return np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2))


def _fit_simple_turn(chi_dot, us, va, asym_mode="fit", k_asym_fixed=0.0):
    """Eq. (41): chi_dot_b = gk * va * (us - k_asym)"""
    term1 = va * us
    if asym_mode == "off":
        A = term1.reshape(-1, 1)
    elif asym_mode == "fixed":
        A = (va * (us - k_asym_fixed)).reshape(-1, 1)
    else:
        A = np.vstack([term1, va]).T
    valid = np.isfinite(A).all(axis=1) & np.isfinite(chi_dot)
    coeffs = calculate_weighted_least_squares(chi_dot[valid], A[valid])
    if asym_mode == "off":
        gk, k_asym = coeffs[0], 0.0
    elif asym_mode == "fixed":
        gk, k_asym = coeffs[0], k_asym_fixed
    else:
        c1, c2 = coeffs[0], coeffs[1]
        gk = c1
        k_asym = -c2 / c1 if abs(c1) > 1e-10 else 0.0
    return (gk, k_asym), A @ coeffs


def _fit_two_term_turn(chi_dot, us, va, chi, beta, asym_mode="fit", k_asym_fixed=0.0):
    """Eq. (40): chi_dot_b = c1*(va*(us-k_asym)) + c2*(sin(chi)*cos(beta)/va)"""
    term1 = va * us
    term_grav = np.sin(chi) * np.cos(beta) / np.maximum(va, 1e-6)
    if asym_mode == "off":
        A = np.vstack([term1, term_grav]).T
    elif asym_mode == "fixed":
        A = np.vstack([va * (us - k_asym_fixed), term_grav]).T
    else:
        A = np.vstack([term1, va, term_grav]).T
    valid = np.isfinite(A).all(axis=1) & np.isfinite(chi_dot)
    coeffs = calculate_weighted_least_squares(chi_dot[valid], A[valid])
    if asym_mode == "off":
        c1, c2, k_asym = coeffs[0], coeffs[1], 0.0
    elif asym_mode == "fixed":
        c1, c2, k_asym = coeffs[0], coeffs[1], k_asym_fixed
    else:
        c1 = coeffs[0]
        c2 = coeffs[2]
        k_asym = -coeffs[1] / c1 if abs(c1) > 1e-10 else 0.0
    return (c1, c2, k_asym), A @ coeffs


def _fit_full_rational_turn(
    chi_dot,
    us,
    va,
    v_tau,
    chi,
    beta,
    mass=50.0,
    g=9.81,
    x0=None,
    asym_mode="fit",
    k_asym_fixed=0.0,
    gravity_mode="fit",
):
    """Eq. (38): nonlinear rational model with (k1, k2, k3, k4, k_asym)."""
    with_asym = asym_mode == "fit"
    with_gravity = gravity_mode == "fit"

    x0_arr = list(x0) if x0 is not None else []
    x0_arr += [0.0] * max(0, 5 - len(x0_arr))
    k1_0, k2_0, k3_0, k4_0, ka_0 = x0_arr[:5]
    if x0 is None:
        k1_0, k2_0, k3_0, k4_0, ka_0 = 1.0, 8.0, 5.0, 8.0, 0.0

    if with_gravity and with_asym:
        x0_int = [k1_0, k2_0, k3_0, k4_0, ka_0]
        bounds = ([-1e2, -1e2, -1e2, -1e2, -0.1], [1e2, 1e2, 1e2, 1e2, 0.1])
    elif with_gravity and not with_asym:
        x0_int = [k1_0, k2_0, k3_0, k4_0]
        bounds = ([-1e2, -1e2, -1e2, -1e2], [1e2, 1e2, 1e2, 1e2])
    elif not with_gravity and with_asym:
        x0_int = [k1_0, k2_0, k4_0, ka_0]
        bounds = ([-1e2, -1e2, -1e2, -0.1], [1e2, 1e2, 1e2, 0.1])
    else:
        x0_int = [k1_0, k2_0, k4_0]
        bounds = ([-1e2, -1e2, -1e2], [1e2, 1e2, 1e2])

    def predict(k1, k2, k3, k4, k_asym):
        grav = k3 * mass * g * np.sin(chi) * np.cos(beta) if with_gravity else 0.0
        num = k1 * va**2 * (us - k_asym) + grav
        den = np.maximum(k4 * mass * v_tau + k2 * va, 1e-6)
        return -num / den

    def residuals(x):
        if with_gravity and with_asym:
            return predict(*x) - chi_dot
        elif with_gravity and not with_asym:
            k1, k2, k3, k4 = x
            return predict(k1, k2, k3, k4, k_asym_fixed) - chi_dot
        elif not with_gravity and with_asym:
            k1, k2, k4, k_asym = x
            return predict(k1, k2, 0.0, k4, k_asym) - chi_dot
        else:
            k1, k2, k4 = x
            return predict(k1, k2, 0.0, k4, k_asym_fixed) - chi_dot

    valid = (
        np.isfinite(chi_dot)
        & np.isfinite(va)
        & np.isfinite(v_tau)
        & np.isfinite(us)
        & np.isfinite(chi)
        & np.isfinite(beta)
    )
    res = _scipy_least_squares(
        lambda x: residuals(x)[valid],
        x0=np.array(x0_int, dtype=float),
        bounds=bounds,
    )
    if with_gravity and with_asym:
        k1, k2, k3, k4, k_asym = res.x
    elif with_gravity and not with_asym:
        k1, k2, k3, k4 = res.x
        k_asym = k_asym_fixed
    elif not with_gravity and with_asym:
        k1, k2, k4, k_asym = res.x
        k3 = 0.0
    else:
        k1, k2, k4 = res.x
        k3 = 0.0
        k_asym = k_asym_fixed

    return (k1, k2, k3, k4, k_asym), predict(k1, k2, k3, k4, k_asym)


def plot_turn_rate_fit_results(
    phase_results,
    time,
    chi_dot_meas,
    flight_data,
    chi,
    beta,
    v_tau,
    us,
    va,
    yaw_dot_flight=None,
    chi_dot_flight=None,
    turn_rate_source="chi_dot",
    turn_rate_label="chi_dot",
    selected_turn_rate_from_gradient=False,
    yaw_used_for_derivative=None,
    yaw_used_for_derivative_label=None,
    selected_turn_rate_source=None,
    plot_cycle=None,
    phase_name=None,
    palette=None,
    plot_dir=None,
    mass=50.0,
    g=9.81,
):
    """
    Create and save the three turn-rate law identification figures.

    Parameters
    ----------
    phase_results : dict
        Output of the per-phase fitting loop (keys are phase identifiers).
    time, chi_dot_meas, chi, beta, v_tau, us, va : np.ndarray
        Full-length signal arrays.
    flight_data : pd.DataFrame
        Must contain a phase/cycle column and optionally 'cycle'.
    plot_dir : Path or str, optional
        Directory where figures are saved.  Created if it does not exist.
    """
    if phase_name is None:
        phase_name = {1: "reel-out", 3: "reel-in"}
    if palette is None:
        palette = get_color_list()
    if plot_dir is None:
        plot_dir = Path("results") / "plots_paper"
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    phase_keys = sorted(
        phase_results,
        key=lambda p: int(p) if str(p).isdigit() else str(p),
    )
    phase_labels = [
        phase_name.get(int(p), str(p)) if str(p).isdigit() else str(p)
        for p in phase_keys
    ]

    meas_color = palette[0]
    simple_color = palette[1]
    two_color = palette[2]
    full_color = palette[5]
    symmetry_color = palette[6]

    # Figure 1: RMSE comparison bar chart
    rmse_simple = [phase_results[p]["simple"]["RMSE"] for p in phase_keys]
    rmse_simple_sym = [phase_results[p]["simple_symmetric"]["RMSE"] for p in phase_keys]
    rmse_two = [phase_results[p]["two_term"]["RMSE"] for p in phase_keys]
    rmse_full = [phase_results[p]["full"]["RMSE"] for p in phase_keys]
    x = np.arange(len(phase_keys))
    width = 0.2

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(
        x - 1.5 * width,
        rmse_simple,
        width,
        label="simple (no weight)",
        color=simple_color,
    )
    ax1.bar(
        x - 0.5 * width,
        rmse_simple_sym,
        width,
        label="simple (symmetric)",
        color=symmetry_color,
    )
    ax1.bar(
        x + 0.5 * width,
        rmse_two,
        width,
        label="two_fit (no inertial forces)",
        color=two_color,
    )
    ax1.bar(x + 1.5 * width, rmse_full, width, label="full (k1-k4)", color=full_color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(phase_labels, rotation=30, ha="right")
    ax1.set_ylabel("RMSE [rad/s]")
    ax1.legend()
    ax1.grid(True, axis="y")
    fig1.tight_layout()
    fig1.savefig(plot_dir / "turn_rate_rmse.png", dpi=300, bbox_inches="tight")

    # Figure 2: scatter plots per phase
    n_cols = 2
    n_rows = max(1, (len(phase_keys) + 1) // 2)
    fig2, axes2 = plt.subplots(
        n_rows, n_cols, figsize=(12, 4 * n_rows), sharex=False, sharey=False
    )
    axes2 = np.array(axes2).reshape(-1)

    for ax, phase in zip(axes2, phase_keys):
        pr = phase_results[phase]
        xdata = pr["us_va"]
        meas = pr["simple"]["meas"]

        ax.scatter(
            xdata, meas, s=8, alpha=0.35, color=meas_color, label=f"{turn_rate_label}"
        )
        if turn_rate_source != "yaw_dot" and pr.get("yaw_dot") is not None:
            ax.scatter(
                xdata, pr["yaw_dot"], s=8, alpha=0.25, color=palette[3], label="yaw_dot"
            )
        if turn_rate_source != "chi_dot" and pr.get("chi_dot") is not None:
            ax.scatter(
                xdata, pr["chi_dot"], s=8, alpha=0.25, color=palette[4], label="chi_dot"
            )
        ax.scatter(
            xdata,
            pr["simple"]["est"],
            s=8,
            alpha=0.35,
            color=simple_color,
            label="simple (no weight)",
        )
        ax.scatter(
            xdata,
            pr["two_term"]["est"],
            s=8,
            alpha=0.35,
            color=two_color,
            label="two_fit (no inertial forces)",
        )
        ax.scatter(
            xdata,
            pr["full"]["est"],
            s=8,
            alpha=0.35,
            color=full_color,
            label="full (k1-k4)",
        )

        x_line = np.linspace(np.nanmin(xdata), np.nanmax(xdata), 200)
        ax.plot(
            x_line,
            pr["simple"]["gk"] * x_line,
            color=symmetry_color,
            lw=1.3,
            label="symmetric (gk)",
        )

        p_label = (
            phase_name.get(int(phase), str(phase))
            if str(phase).isdigit()
            else str(phase)
        )
        ax.set_title(p_label)
        ax.set_xlabel(r"$u_s \cdot v_a$  [m/s]")
        ax.set_ylabel(r"$\dot{\chi}$  [rad/s]")
        ax.legend(fontsize=7)
        ax.grid(True)

    for ax in axes2[len(phase_keys) :]:
        ax.set_visible(False)

    fig2.tight_layout()
    fig2.savefig(
        plot_dir / "turn_rate_scatter_phases.png", dpi=300, bbox_inches="tight"
    )

    # Figure 3: one-cycle time-series with all three model estimates
    phase_col = (
        "flight_phase_index" if "flight_phase_index" in flight_data.columns else "cycle"
    )
    cycle_col = "cycle" if "cycle" in flight_data.columns else phase_col

    if plot_cycle is None and cycle_col in flight_data.columns:
        available = flight_data[cycle_col].unique()
        plot_cycle = int(available[len(available) // 2]) if len(available) else None

    if plot_cycle is not None and cycle_col in flight_data.columns:
        cyc_mask = (flight_data[cycle_col] == plot_cycle).to_numpy()
        if cyc_mask.sum() > 0:
            t_cyc = time[cyc_mask]
            meas_cyc = chi_dot_meas[cyc_mask]
            u_cyc = us[cyc_mask]
            v_cyc = va[cyc_mask]
            ph_cyc = flight_data.loc[cyc_mask, phase_col].to_numpy()

            est41_cyc = np.full(cyc_mask.sum(), np.nan)
            est40_cyc = np.full(cyc_mask.sum(), np.nan)
            estfull_cyc = np.full(cyc_mask.sum(), np.nan)

            for ph, pr in phase_results.items():
                idx = ph_cyc == ph
                if not idx.any():
                    continue
                est41_cyc[idx] = pr["simple"]["gk"] * v_cyc[idx] * u_cyc[idx]
                est40_cyc[idx] = pr["two_term"]["c1"] * v_cyc[idx] * u_cyc[idx] + pr[
                    "two_term"
                ]["c2"] * np.sin(chi[cyc_mask][idx]) * np.cos(
                    beta[cyc_mask][idx]
                ) / np.maximum(
                    v_cyc[idx], 1e-6
                )
                k1_ph, k2_ph = pr["full"]["k1"], pr["full"]["k2"]
                k3_ph, k4_ph = pr["full"]["k3"], pr["full"]["k4"]
                k_asym_ph = pr["full"]["k_asymmetry"]
                num = k1_ph * v_cyc[idx] ** 2 * (
                    u_cyc[idx] - k_asym_ph
                ) + k3_ph * mass * g * np.sin(chi[cyc_mask][idx]) * np.cos(
                    beta[cyc_mask][idx]
                )
                den = np.maximum(
                    k4_ph * mass * v_tau[cyc_mask][idx] + k2_ph * v_cyc[idx], 1e-6
                )
                estfull_cyc[idx] = -num / den

            fig3, ax3 = plt.subplots(figsize=(14, 4))
            ax3.plot(
                t_cyc, meas_cyc, color=meas_color, lw=1.0, label=f"{turn_rate_label}"
            )
            if yaw_dot_flight is not None and turn_rate_source != "yaw_dot":
                ax3.plot(
                    t_cyc,
                    yaw_dot_flight[cyc_mask],
                    color=palette[3],
                    lw=1.0,
                    alpha=0.85,
                    label="yaw_dot",
                )
            if chi_dot_flight is not None and turn_rate_source != "chi_dot":
                ax3.plot(
                    t_cyc,
                    chi_dot_flight[cyc_mask],
                    color=palette[4],
                    lw=1.0,
                    alpha=0.85,
                    label="chi_dot",
                )
            ax3.plot(
                t_cyc,
                est41_cyc,
                color=simple_color,
                lw=1.3,
                ls="--",
                label="simple (no weight)",
            )
            ax3.plot(
                t_cyc,
                est40_cyc,
                color=two_color,
                lw=1.3,
                ls="-.",
                label="two_fit (no inertial forces)",
            )
            ax3.plot(
                t_cyc,
                estfull_cyc,
                color=full_color,
                lw=1.3,
                ls="-",
                label="full (k1-k4)",
            )

            phase_numeric = ph_cyc.astype(int) if ph_cyc.dtype.kind in "iuf" else ph_cyc
            cmap = plt.cm.Set3
            unique_phases = np.unique(phase_numeric)
            for i, ph in enumerate(unique_phases):
                idx = phase_numeric == ph
                if not np.any(idx):
                    continue
                t_seg = t_cyc[idx]
                ph_lbl = phase_name.get(int(ph), ph) if str(ph).isdigit() else str(ph)
                ax3.axvspan(
                    t_seg[0],
                    t_seg[-1],
                    alpha=0.15,
                    color=cmap(i / max(len(unique_phases) - 1, 1)),
                    label=ph_lbl,
                )

            meas_valid = meas_cyc[np.isfinite(meas_cyc)]
            if len(meas_valid):
                ym = 0.2 * (meas_valid.max() - meas_valid.min())
                ax3.set_ylim(meas_valid.min() - ym, meas_valid.max() + ym)
            ax3.set_xlabel("Time [s]")
            ax3.set_ylabel(r"$\dot{\chi}$ [rad/s]")
            ax3.legend(fontsize=8, loc="upper right")
            ax3.grid(True)
            fig3.tight_layout()
            fig3.savefig(
                plot_dir / f"turn_rate_cycle_{plot_cycle}.png",
                dpi=300,
                bbox_inches="tight",
            )

            # Figure 4: yaw angle + derived rate debug plot (only when applicable)
            if selected_turn_rate_from_gradient and yaw_used_for_derivative is not None:
                fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
                ax4a.plot(
                    t_cyc,
                    yaw_used_for_derivative[cyc_mask],
                    color=palette[7],
                    lw=1.1,
                    label=f"yaw used: {yaw_used_for_derivative_label}",
                )
                ax4a.set_ylabel("Yaw [rad]")
                ax4a.legend(fontsize=8)
                ax4a.grid(True)
                ax4b.plot(
                    t_cyc,
                    meas_cyc,
                    color=palette[3],
                    lw=1.1,
                    label=f"yaw_rate used for fit: {selected_turn_rate_source}",
                )
                ax4b.set_xlabel("Time [s]")
                ax4b.set_ylabel("Yaw rate [rad/s]")
                ax4b.legend(fontsize=8)
                ax4b.grid(True)
                fig4.tight_layout()
                fig4.savefig(
                    plot_dir / f"yaw_used_to_derive_yaw_rate_cycle_{plot_cycle}.png",
                    dpi=300,
                    bbox_inches="tight",
                )

    plt.show()


def plot_turn_rate_identification(
    results,
    flight_data,
    config_data,
    mass=50.0,
    g=9.81,
    cut=0,
    cycles=None,
    plot_cycle=None,
    phases_to_fit=None,
    asym_mode="fit",
    turn_rate_source="chi_dot",
    yaw_rate_sensor_id=1,
):
    """
    Full turn-rate law identification pipeline: signal extraction, per-phase
    fitting (simple / two-term / full rational), and figure generation.

    Figures are saved to  results/plots_paper/<date>/
    where <date> is taken from config_data['year'/'month'/'day'].

    Parameters
    ----------
    results, flight_data : pd.DataFrame
        EKF output and flight-data DataFrames (already time-filtered if desired).
    config_data : dict
        Must contain 'year', 'month', 'day' keys for the output directory.
    mass : float
        Kite + lines mass [kg].
    g : float
        Gravitational acceleration [m/s²].
    cut : int
        Number of samples to trim from each end before fitting.
    cycles : list or None
        Cycle numbers to include.  None → all cycles present.
    plot_cycle : int or None
        Cycle to use for the time-series figure.  None → median cycle.
    phases_to_fit : list or None
        Flight-phase indices to fit.  None → [1, 3].
    asym_mode : str
        Asymmetry handling: 'fit', 'off', or 'fixed'.
    turn_rate_source : str
        'chi_dot' or 'yaw_dot'.
    yaw_rate_sensor_id : int
        Sensor index for kite_yaw_rate_<id> column.
    """
    set_plot_style_no_latex()
    palette = get_color_list()

    if phases_to_fit is None:
        phases_to_fit = [1, 2, 3, 4]  # Default to all main flight phases
    phase_name = {
        1: "reel-out",
        2: "rori",
        3: "reel-in",
        4: "riro",
        0: "single-phase",
    }  # Add default names for phases

    # Build date string and output directory
    try:
        date_str = f"{config_data['year']}-{config_data['month']}-{config_data['day']}"
    except (KeyError, TypeError):
        date_str = "unknown"
    plot_dir = Path("results") / "plots_paper" / date_str
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Trim edges
    res = results.iloc[cut : len(results) - cut if cut else None].reset_index(drop=True)
    fd = flight_data.iloc[cut : len(flight_data) - cut if cut else None].reset_index(
        drop=True
    )

    # Optionally restrict to selected cycles
    if cycles is not None:
        cycle_col = "cycle" if "cycle" in fd.columns else None
        if cycle_col:
            mask = fd[cycle_col].isin(cycles)
            res = res[mask].reset_index(drop=True)
            fd = fd[mask].reset_index(drop=True)

    time = fd["time"].to_numpy()
    us = -fd["kcu_actual_steering"].to_numpy() / 100
    va = res["kite_apparent_windspeed"].to_numpy()

    position = np.array(
        [res["kite_position_x"], res["kite_position_y"], res["kite_position_z"]]
    )
    r_norm = np.linalg.norm(position, axis=0)
    beta = np.arctan2(position[2], np.sqrt(position[0] ** 2 + position[1] ** 2))
    phi = np.arctan2(position[1], position[0])

    dbeta_dt = np.gradient(beta, time)
    dphi_dt = np.gradient(phi, time)
    v_beta = r_norm * dbeta_dt
    v_phi = r_norm * np.cos(beta) * dphi_dt
    v_tau = np.sqrt(v_beta**2 + v_phi**2)
    chi = np.unwrap(np.arctan2(v_phi, v_beta))

    v_beta_dot = np.gradient(v_beta, time)
    v_phi_dot = np.gradient(v_phi, time)
    chi_dot_kinematic = np.convolve(
        (v_beta * v_phi_dot - v_phi * v_beta_dot) / np.maximum(v_tau**2, 1e-6),
        np.ones(21) / 21,
        mode="same",
    )

    # Select measurement signal
    chi_dot_flight = yaw_dot_flight = None
    if "chi_dot" in fd.columns:
        chi_dot_flight = np.convolve(
            fd["chi_dot"].to_numpy(), np.ones(21) / 21, mode="same"
        )
    yaw_rate_col = f"kite_yaw_rate_{yaw_rate_sensor_id}"
    if yaw_rate_col in fd.columns:
        yaw_dot_flight = np.convolve(
            fd[yaw_rate_col].to_numpy(), np.ones(21) / 21, mode="same"
        )
    elif "kite_yaw_rate" in fd.columns:
        yaw_dot_flight = np.convolve(
            fd["kite_yaw_rate"].to_numpy(), np.ones(21) / 21, mode="same"
        )

    if turn_rate_source == "chi_dot":
        chi_dot_meas = (
            chi_dot_flight if chi_dot_flight is not None else chi_dot_kinematic
        )
        turn_rate_label = "chi_dot"
    else:
        chi_dot_meas = (
            yaw_dot_flight if yaw_dot_flight is not None else chi_dot_kinematic
        )
        turn_rate_label = (
            yaw_rate_col if yaw_dot_flight is not None else "chi_dot_kinematic"
        )

    # Per-phase fitting (split by flight phase, not cycles)
    if "flight_phase_index" not in fd.columns:
        print("Warning: No 'flight_phase_index' column found. Cannot split by phases.")
        return

    phase_col = "flight_phase_index"

    # Filter for the specified phases [1, 2, 3, 4]
    phases = [p for p in fd[phase_col].unique() if p in phases_to_fit]

    if not phases:
        print(f"Warning: None of the target phases {phases_to_fit} found in data.")
        print(f"Available phases: {sorted(fd[phase_col].unique())}")
        return

    # Update phase_name dictionary with any phases not yet defined
    for p in phases:
        if p not in phase_name:
            phase_name[p] = f"phase-{p}"

    print(f"Fitting turn-rate laws for {len(phases)} phase(s): {sorted(phases)}")
    print(f"Phase names: {[phase_name.get(p, f'phase-{p}') for p in sorted(phases)]}")

    phase_results = {}
    x0_full = None
    for phase in sorted(phases, key=str):
        mask = (fd[phase_col] == phase).to_numpy()
        if mask.sum() < 50:
            continue
        fd_p = chi_dot_meas[mask]
        u, v, vt = us[mask], va[mask], v_tau[mask]
        c, b = chi[mask], beta[mask]

        (gk, ka41), est_simple = _fit_simple_turn(fd_p, u, v, asym_mode=asym_mode)
        (gk_sym, _), est_simple_sym = _fit_simple_turn(fd_p, u, v, asym_mode="off")
        (c1, c2, ka40), est_two = _fit_two_term_turn(
            fd_p, u, v, c, b, asym_mode=asym_mode
        )

        (k1_ng, k2_ng, k3_ng, k4_ng, ka_ng), est_full_ng = _fit_full_rational_turn(
            fd_p,
            u,
            v,
            vt,
            c,
            b,
            mass=mass,
            g=g,
            x0=x0_full,
            asym_mode=asym_mode,
            gravity_mode="off",
        )
        x0_full = [k1_ng, k2_ng, 5.0, k4_ng, ka_ng]

        (k1, k2, k3, k4, ka38), est_full = _fit_full_rational_turn(
            fd_p,
            u,
            v,
            vt,
            c,
            b,
            mass=mass,
            g=g,
            x0=x0_full,
            asym_mode=asym_mode,
            gravity_mode="fit",
        )

        phase_results[phase] = {
            "simple": {
                "gk": gk,
                "k_asymmetry": ka41,
                "RMSE": _rmse_turn(fd_p, est_simple),
                "est": est_simple,
                "meas": fd_p,
            },
            "simple_symmetric": {
                "gk": gk_sym,
                "k_asymmetry": 0.0,
                "RMSE": _rmse_turn(fd_p, est_simple_sym),
                "est": est_simple_sym,
            },
            "two_term": {
                "c1": c1,
                "c2": c2,
                "k_asymmetry": ka40,
                "RMSE": _rmse_turn(fd_p, est_two),
                "est": est_two,
            },
            "full": {
                "k1": k1,
                "k2": k2,
                "k3": k3,
                "k4": k4,
                "k_asymmetry": ka38,
                "RMSE": _rmse_turn(fd_p, est_full),
                "est": est_full,
            },
            "full_no_gravity": {
                "k1": k1_ng,
                "k2": k2_ng,
                "k3": k3_ng,
                "k4": k4_ng,
                "k_asymmetry": ka_ng,
                "RMSE": _rmse_turn(fd_p, est_full_ng),
                "est": est_full_ng,
            },
            "yaw_dot": yaw_dot_flight[mask] if yaw_dot_flight is not None else None,
            "chi_dot": chi_dot_flight[mask] if chi_dot_flight is not None else None,
            "signals": {"u": u, "v": v, "vt": vt, "c": c, "b": b},
            "us_va": u * v,
            "time": time[mask],
            "n": mask.sum(),
        }

    if not phase_results:
        print("No phases with enough data for turn-rate identification.")
        return

    plot_turn_rate_fit_results(
        phase_results=phase_results,
        time=time,
        chi_dot_meas=chi_dot_meas,
        flight_data=fd,
        chi=chi,
        beta=beta,
        v_tau=v_tau,
        us=us,
        va=va,
        yaw_dot_flight=yaw_dot_flight,
        chi_dot_flight=chi_dot_flight,
        turn_rate_source=turn_rate_source,
        turn_rate_label=turn_rate_label,
        plot_cycle=plot_cycle,
        phase_name=phase_name,
        palette=palette,
        plot_dir=plot_dir,
        mass=mass,
        g=g,
    )
