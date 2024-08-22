import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
import seaborn as sns
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style, get_color
from awes_ekf.postprocess.postprocessing import (
    remove_offsets_IMU_data,
    calculate_offset_pitch_depower_turn,
)

# set_plot_style()


def plot_kite_orientation(config_data: dict) -> None:
    # Load results and flight data and plot kite reference frame
    results, flight_data = read_results(
        str(config_data["year"]),
        str(config_data["month"]),
        str(config_data["day"]),
        config_data["kite"]["model_name"],
    )

    imus = config_data["kite"]["sensor_ids"]

    for column in results.columns:
        if "pitch" in column or "roll" in column or "yaw" in column:
            results[column] = np.degrees(results[column])

    for column in flight_data.columns:
        if "pitch" in column or "roll" in column or "yaw" in column:
            flight_data[column] = np.degrees(flight_data[column])

    # Calculate errors
    pitch_error = abs(flight_data["kite_pitch_0"] - results["kite_pitch"])
    roll_error = abs(flight_data["kite_roll_0"] - results["kite_roll"])
    yaw_error = abs(flight_data["kite_yaw_0"] - results["kite_yaw"])

    mean_pitch_error = np.mean(pitch_error)
    mean_roll_error = np.mean(roll_error)
    mean_yaw_error = np.mean(yaw_error)

    std_pitch_error = np.std(pitch_error)
    std_roll_error = np.std(roll_error)
    std_yaw_error = np.std(yaw_error)

    print(
        f"Mean pitch error: {mean_pitch_error:.2f} deg, std: {std_pitch_error:.2f} deg"
    )
    print(f"Mean roll error: {mean_roll_error:.2f} deg, std: {std_roll_error:.2f} deg")
    print(f"Mean yaw error: {mean_yaw_error:.2f} deg, std: {std_yaw_error:.2f} deg")

    # %%
    fig, ax = plt.subplots()
    for imu in imus:
        pu.plot_time_series(
            flight_data,
            flight_data["kite_pitch_" + str(imu)],
            ax,
            label="Measured",
            plot_phase=False,
        )
    pu.plot_time_series(
        flight_data, results["kite_pitch"], ax, label="Estimated", plot_phase=False
    )
    ax.grid()
    ax.legend()

    fig, ax = plt.subplots()
    for imu in imus:
        pu.plot_time_series(
            flight_data,
            flight_data["kite_roll_" + str(imu)],
            ax,
            label="Measured",
            plot_phase=False,
        )
    pu.plot_time_series(
        flight_data, results["kite_roll"], ax, label="Estimated", plot_phase=False
    )
    ax.grid()
    ax.legend()

    fig, ax = plt.subplots()
    for imu in imus:
        pu.plot_time_series(
            flight_data,
            flight_data["kite_yaw_" + str(imu)],
            ax,
            label="Measured",
            plot_phase=False,
        )
    pu.plot_time_series(
        flight_data, results["kite_yaw"], ax, label="Estimated", plot_phase=False
    )
    ax.grid()
    ax.legend()

    fig, ax = plt.subplots()
    pu.plot_time_series(
        flight_data,
        results["kite_roll"] - results["tether_roll"],
        ax,
        label="Roll kite-tether",
        plot_phase=False,
    )

    pu.plot_time_series(
        flight_data,
        results["kite_pitch"] - results["tether_pitch"],
        ax,
        label="Pitch kite-tether",
        plot_phase=True,
    )

    ax.legend()


if __name__ == "__main__":
    # Example usage
    plt.close("all")
    config_file_name = "v3_config.yaml"
    config = load_config("examples/" + config_file_name)
    plot_kite_orientation(config)
    plt.show()
