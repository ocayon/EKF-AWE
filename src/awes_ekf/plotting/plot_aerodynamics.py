import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.plotting.plot_utils import plot_time_series
from awes_ekf.plotting.color_palette import set_plot_style_no_latex, get_color_list
import scipy.stats as stats
from scipy.stats import linregress
from awes_ekf.utils import calculate_turn_rate_law, find_time_delay

colors = get_color_list()

def plot_aerodynamics(results, flight_data, config_data):
    set_plot_style_no_latex()
    
    plot_aerodynamic_coefficients(flight_data, results, config_data)

    date = config_data["year"] + "-" + config_data["month"] + "-" + config_data["day"]
    plot_polars(results, flight_data, date=date, model=config_data["kite"]["model_name"])

    # if "kcu_actual_steering" in flight_data.columns:
    #     plot_identify_turn_dynamics(results, flight_data, config_data)

    CL_mean_powered = results["wing_lift_coefficient"][flight_data["powered"] == "powered"].mean()
    CD_mean_powered = results["wing_drag_coefficient"][flight_data["powered"] == "powered"].mean()  
    print(f"Mean CL powered: {CL_mean_powered:.2f}")
    print(f"Mean CD powered: {CD_mean_powered:.2f}")
    print(f"Mean CL/CD powered: {CL_mean_powered/CD_mean_powered:.2f}")

    std_CL_powered = results["wing_lift_coefficient"][flight_data["powered"] == "powered"].std()
    std_CD_powered = results["wing_drag_coefficient"][flight_data["powered"] == "powered"].std()
    print(f"Std CL powered: {std_CL_powered:.2f}")
    print(f"Std CD powered: {std_CD_powered:.2f}")


    CL_mean_depowered = results["wing_lift_coefficient"][flight_data["powered"] == "depowered"].mean()
    CD_mean_depowered = results["wing_drag_coefficient"][flight_data["powered"] == "depowered"].mean()
    print(f"Mean CL depowered: {CL_mean_depowered:.2f}")
    print(f"Mean CD depowered: {CD_mean_depowered:.2f}")
    print(f"Mean CL/CD depowered: {CL_mean_depowered/CD_mean_depowered:.2f}")


    plt.show()

def plot_identify_turn_dynamics(results, flight_data, config_data):
    # Turn rate law
    ts = np.mean(np.diff(flight_data["time"]))

    if "kite_yaw_rate" not in flight_data.columns:
        if "kite_yaw_rate_1" in flight_data.columns:
            flight_data["kite_yaw_rate"] = np.unwrap(flight_data["kite_yaw_rate_1"])
        else:
            yaw_rate = np.convolve(np.gradient(flight_data["kite_yaw_0"], ts), np.ones(10)/10, mode="same")
            flight_data["kite_yaw_rate"] = np.unwrap(yaw_rate)


    # Calculate time delay between yaw rate and steering input
    signal_delay, _ = find_time_delay(flight_data["kite_yaw_rate"], flight_data["kcu_actual_steering"])
    time_delay = signal_delay * ts
    if abs(time_delay) > 3:
        print("Warning: Time delay between yaw rate and steering input is very high. Inversing steering input.")
        flight_data["kcu_actual_steering"] = -flight_data["kcu_actual_steering"]
        flight_data["kcu_set_steering"] = -flight_data["kcu_set_steering"]
        signal_delay, _ = find_time_delay(flight_data["kite_yaw_rate"], flight_data["kcu_actual_steering"])
        time_delay = signal_delay * ts
    print("Time delay turn rate:", time_delay)

    # Calculate time delay between sideforce and steering input
    signal_delay, _ = find_time_delay(results["wing_sideforce_coefficient"], flight_data["kcu_actual_steering"])
    flight_data["kcu_actual_steering_delay"] = np.roll(flight_data["kcu_actual_steering"], int(signal_delay))
    time_delay = signal_delay * ts
    print("Time delay steering force:", time_delay)

    # Calculate yaw rates and coefficients with and without offset
    yaw_rate_standard, coeffs_standard = calculate_turn_rate_law(results, flight_data, model="simple", steering_offset=False)
    yaw_rate_offset_corrected, coeffs_offset = calculate_turn_rate_law(results, flight_data, model="simple", steering_offset=True)

    print("Yaw rate coefficients (standard):", coeffs_standard)
    print("Yaw rate coefficients (offset-corrected):", coeffs_offset)

    # Calculate mean errors
    error_standard = abs(np.degrees(yaw_rate_standard) - np.degrees(flight_data["kite_yaw_rate"]))
    error_offset = abs(np.degrees(yaw_rate_offset_corrected) - np.degrees(flight_data["kite_yaw_rate"]))
    mean_error_standard = np.mean(error_standard)
    mean_error_offset = np.mean(error_offset)

    # Prepare data for the main scatter plot
    x = flight_data["kcu_actual_steering"] / 100 * results["kite_apparent_windspeed"]
    y = flight_data["kite_yaw_rate"]

    # Scatter and line plot
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y, alpha=0.2, color = colors[2], label='Measured')
    A = np.vstack([x]).T
    y_identified_standard = A @ coeffs_standard
    plt.plot(
        x, 
        y_identified_standard,
        label=f'Identified Yaw Rate (Mean Error: {mean_error_standard:.2f} deg/s)', 
        linestyle="--"
    )

    # Plot offset-corrected yaw rate line
    A = np.vstack([x, results["kite_apparent_windspeed"]]).T
    y_identified_offset = A @ coeffs_offset
    plt.plot(
        x,
        y_identified_offset,
        label=f'Offset-Corrected Yaw Rate (Mean Error: {mean_error_offset:.2f} deg/s)',
        linestyle='-.'
    )

    plt.xlabel(r'$u_\mathrm{s} \cdot v_\mathrm{a}$ [m/s]')
    plt.ylabel('Kite Yaw Rate [rad/s]')
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
        mask = flight_data["kcu_actual_steering_delay"] < 0 if condition == "<0" else flight_data["kcu_actual_steering_delay"] > 0
        x = np.array(flight_data[mask]["kcu_actual_steering_delay"]) / 100
        y = np.array(results[mask]["wing_sideforce_coefficient"])
        
        # Construct A_matrix and fit parameters
        A_matrix = np.vstack([x, np.ones(len(x))]).T
        x_hat, Qx, error = least_squares(x, y, A_matrix)
        
        # Set x_range for each condition
        x_range = np.linspace(-0.4, 0, 100) if condition == "<0" else np.linspace(0, 0.4, 100)
        fit_line = x_hat[0] * x_range + x_hat[1] 
        upper_bound = fit_line + error
        lower_bound = fit_line - error

        

        # Plotting
        plt.scatter(flight_data[mask]["kcu_actual_steering"] / 100, y, alpha=0.2, color=colors[2])
        plt.scatter(flight_data[mask]["kcu_actual_steering_delay"] / 100, y, alpha=0.2, color=colors[3])

        # Linear fit line
        plt.plot(x_range, fit_line, linestyle="--", color=colors[0])

        # Fill between upper and lower bounds
        plt.fill_between(x_range, lower_bound, upper_bound, alpha=0.6, color=colors[0])

        fit_us0 = fit_line[-1] if condition == "<0" else fit_line[0]
        # Add thicker gridlines at y=0 and x=0
        plt.axhline(fit_us0, linewidth=1.5, linestyle="--", color = colors[5])
        plt.axhline(0, color="black", linewidth=0.5)
        

        # Labels and legend
        plt.xlabel(r'$u_\mathrm{s}$')
        plt.ylabel(r'$C_S$')
        plt.legend()
        plt.tight_layout()


    plt.figure(figsize=(6, 4))
    # Call the function for each condition
    plot_fit(flight_data, results, condition="<0")
    plot_fit(flight_data, results, condition=">0")
    plt.legend(["Linear Fit", "EKF", "EKF Delay Corrected"])

    # Print mean errors and standard deviations
    print("Mean error yaw rate (standard):", mean_error_standard)
    print("Mean error yaw rate (offset-corrected):", mean_error_offset)
    print("Std error yaw rate (standard):", np.std(error_standard))
    print("Std error yaw rate (offset-corrected):", np.std(error_offset))

    # Sideforce time series plot
    # fit_sideforce = slope * flight_data["kcu_actual_steering_delay"] / 100 + intercept
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    plot_time_series(flight_data, results["wing_sideforce_coefficient"], axs[0], ylabel="$C_{S}$", plot_phase=False, label="EKF 0")
    # axs[0].plot(flight_data["time"], slope * flight_data["kcu_actual_steering"] / 100 + intercept, label="Linear Fit", linestyle="--")
    # axs[0].plot(flight_data["time"], fit_sideforce, label="Linear Fit - Delay Corrected", linestyle="--")
    axs[0].legend(frameon=True)

    # Yaw Rate Comparison
    axs[1].plot(flight_data["time"], np.degrees(flight_data["kite_yaw_rate"]), label='Measured Yaw Rate')
    axs[1].plot(flight_data["time"], np.degrees(yaw_rate_standard), label='Identified Yaw Rate', linestyle='--')
    axs[1].plot(flight_data["time"], np.degrees(yaw_rate_offset_corrected), label='Offset-Corrected Yaw Rate', linestyle='-.')
    axs[1].legend(frameon=True)
    axs[1].set_ylabel("Yaw Rate [deg/s]")

    # Steering Input
    plot_time_series(flight_data, -flight_data["kcu_actual_steering"] / 100, axs[2], ylabel="$u_s$", plot_phase=False, label="Actual steering")
    plot_time_series(flight_data, -flight_data["kcu_set_steering"] / 100, axs[2], ylabel="$u_s$", plot_phase=False, label="Set steering")
    axs[2].legend(frameon=True)
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylim([-0.4, 0.4])
    plt.tight_layout()




def plot_aerodynamic_coefficients(flight_data, results, config_data):
    # Smooth AoA if present
    if "bridle_angle_of_attack" in flight_data.columns:
        flight_data["bridle_angle_of_attack"] = np.convolve(flight_data["bridle_angle_of_attack"], np.ones(10)/10, mode="same")

    aoa_imu = None
    if "sensor_ids" in config_data["kite"] and all(f"wing_angle_of_attack_imu_{i}" in results.columns for i in config_data["kite"]["sensor_ids"]):
        # Calculate mean AoA from IMU sensors if multiple IDs are provided
        aoa_imu = np.mean([results[f"wing_angle_of_attack_imu_{i}"] for i in config_data["kite"]["sensor_ids"]], axis=0)

    # Prepare the plot
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot Lift Coefficient
    plot_time_series(flight_data, results["wing_lift_coefficient"], axs[0], ylabel="$C_L$", plot_phase=True)

    # Plot Drag Coefficients
    plot_time_series(flight_data, results["wing_drag_coefficient"], axs[1], label="$C_\mathrm{D}$")
    if "kcu_drag_coefficient" in results.columns:
        plot_time_series(flight_data, results["kcu_drag_coefficient"], axs[1], label="$C_\mathrm{D,kcu}$")
    if "bridles_drag_coefficient" in results.columns:
        plot_time_series(flight_data, results["bridles_drag_coefficient"], axs[1], label="$C_\mathrm{D,bridles}$")
    if "tether_drag_coefficient" in results.columns:
        plot_time_series(flight_data, results["tether_drag_coefficient"], axs[1], label="$C_\mathrm{D,t}$", plot_phase=True, ylabel="$C_D$")
    axs[1].legend()

    # Plot Angle of Attack
    if "bridle_angle_of_attack" in flight_data.columns:
        plot_time_series(flight_data, flight_data["bridle_angle_of_attack"], axs[2], label=r"$\alpha_\mathrm{b}$ measured")
    if aoa_imu is not None:
        plot_time_series(flight_data, aoa_imu, axs[2], label=r"$\alpha_\mathrm{w}$ from IMU")
    if "wing_angle_of_attack_bridle" in results.columns:
        plot_time_series(flight_data, results["wing_angle_of_attack_bridle"], axs[2], label=r"$\mathbf{\alpha_w}$ from bridle")
    if "wing_angle_of_attack" in results.columns:
        plot_time_series(flight_data, results["wing_angle_of_attack"], axs[2], label=r"$\alpha_\mathrm{w}$ from EKF")
    if "kite_angle_of_attack" in results.columns:
        plot_time_series(flight_data, results["kite_angle_of_attack"], axs[2], label=r"$\alpha_\mathrm{k}$ from EKF", plot_phase=True, ylabel=r"$\alpha$ [$^\circ$]")


    # Finalize the angle of attack plot
    axs[2].set_xlabel("Time [s]")
    axs[2].set_ylabel(r"$\alpha$ [$^\circ$]")
    axs[2].legend(frameon=True)

    # Adjust layout
    plt.tight_layout()

def plot_polars(results, flight_data, date="", model = "",label="Wing"):
    # Attempt to find angle of attack data in preferred order and notify which is used
    if "wing_angle_of_attack_bridle" in results.columns and results["wing_angle_of_attack_bridle"].notna().any():
        alpha = results["wing_angle_of_attack_bridle"]
        aoa_label = "Wing AoA (Bridle)"
    elif "wing_angle_of_attack" in results.columns and results["wing_angle_of_attack"].notna().any():
        alpha = results["wing_angle_of_attack"]
        aoa_label = "Wing AoA"
    elif "kite_angle_of_attack" in results.columns:
        alpha = results["kite_angle_of_attack"]
        aoa_label = "Kite AoA"
    else:
        raise ValueError("No suitable angle of attack data found in results.")
    
    print(f"Using angle of attack data: {aoa_label}")

    # Calculate cl and cd for the wing
    cl_wing = np.sqrt(results["wing_lift_coefficient"] ** 2 + results["wing_sideforce_coefficient"] ** 2)
    cd_wing = results["wing_drag_coefficient"]

    # Define binning for angle of attack data
    step = 1
    bins = np.arange(int(alpha.min()) - step / 2, int(alpha.max()) + step / 2, step)
    bin_indices = np.digitize(alpha, bins)
    num_bins = len(bins)

    # Calculate mean CL, CD, and derived quantities for each bin
    cl_wing_means = np.array([cl_wing[bin_indices == i].mean() for i in range(1, num_bins)])
    cd_wing_means = np.array([cd_wing[bin_indices == i].mean() for i in range(1, num_bins)])
    cl_cd_ratio = cl_wing_means / cd_wing_means
    cl_cubed_cd_squared_ratio = (cl_wing_means ** 3) / (cd_wing_means ** 2)

    # Calculate SEM (Standard Error of the Mean) and 99% confidence intervals
    cl_wing_sems = np.array([cl_wing[bin_indices == i].std() / np.sqrt(np.sum(bin_indices == i)) for i in range(1, num_bins)])
    cd_wing_sems = np.array([cd_wing[bin_indices == i].std() / np.sqrt(np.sum(bin_indices == i)) for i in range(1, num_bins)])

    confidence_level = 0.99
    z = stats.norm.ppf(0.5 + confidence_level / 2)  # 99% confidence level
    cl_wing_cis = z * cl_wing_sems
    cd_wing_cis = z * cd_wing_sems

    # Calculate bin centers for plotting
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Create a figure with four subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), sharex=True)
    fig.suptitle(f"{label} Polar Analysis - Mean and 99% Confidence Intervals\nUsing {aoa_label}", fontsize=16)

    # CL plot
    axs[0, 0].plot(bin_centers, cl_wing_means, "o-", markersize=8, linewidth=1.5, label=label)
    axs[0, 0].fill_between(
        bin_centers, cl_wing_means - cl_wing_cis, cl_wing_means + cl_wing_cis, alpha=0.2
    )
    axs[0, 0].set_title("$C_L$ with 99% Confidence Interval", fontsize=12)
    axs[0, 0].set_ylabel("$C_L$", fontsize=14)
    axs[0, 0].set_xlabel(r"$\alpha$ [$^\circ$]", fontsize=14)
    axs[0, 0].grid(True)

    # CD plot
    axs[0, 1].plot(bin_centers, cd_wing_means, "o-", markersize=8, linewidth=1.5, label=label)
    axs[0, 1].fill_between(
        bin_centers, cd_wing_means - cd_wing_cis, cd_wing_means + cd_wing_cis, alpha=0.2
    )
    axs[0, 1].set_title("$C_D$ with 99% Confidence Interval", fontsize=12)
    axs[0, 1].set_ylabel("$C_D$", fontsize=14)
    axs[0, 1].set_xlabel(r"$\alpha$ [$^\circ$]", fontsize=14)
    axs[0, 1].grid(True)

    # CL/CD plot
    axs[1, 0].plot(bin_centers, cl_cd_ratio, "o-", markersize=8, linewidth=1.5, label=f"{label} $C_L / C_D$")
    axs[1, 0].set_title(r"$\frac{C_L}{C_D}$ with 99% Confidence Interval", fontsize=12)
    axs[1, 0].set_ylabel(r"$\frac{C_L}{C_D}$", fontsize=14)
    axs[1, 0].set_xlabel(r"$\alpha$ [$^\circ$]", fontsize=14)
    axs[1, 0].grid(True)

    # CL^3/CD^2 plot
    axs[1, 1].plot(bin_centers, cl_cubed_cd_squared_ratio, "o-", markersize=8, linewidth=1.5, label=f"{label} $C_L^3 / C_D^2$")
    axs[1, 1].set_title(r"$\frac{C_L^3}{C_D^2}$ with 99% Confidence Interval", fontsize=12)
    axs[1, 1].set_ylabel(r"$\frac{C_L^3}{C_D^2}$", fontsize=14)
    axs[1, 1].set_xlabel(r"$\alpha$ [$^\circ$]", fontsize=14)
    axs[1, 1].grid(True)

    # Mean angle of attack lines (if available)
    if "powered" in flight_data.columns:
        mean_aoa_pow = np.mean(alpha[flight_data["powered"] == "powered"])
        mean_aoa_dep = np.mean(alpha[flight_data["powered"] == "depowered"])
        for ax in axs.flat:
            ax.axvline(x=mean_aoa_pow, linestyle='--', label='Mean reel-out AoA')
            ax.axvline(x=mean_aoa_dep, linestyle='--', label='Mean reel-in AoA')

    
    for i in range(2):
        for j in range(2):
            axs[i, j].set_xlim([mean_aoa_dep -2, mean_aoa_pow + 2])
    
    # Add legend to the first subplot
    axs[0, 0].legend(loc="lower right")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle

    # Save polars as csv
    polars = np.vstack([bin_centers, cl_wing_means, cd_wing_means]).T
    np.savetxt("results/polars/polar_"+model+"_"+date+".csv", polars, delimiter=",", header="aoa,cl,cd", comments="")