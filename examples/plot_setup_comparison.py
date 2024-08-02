import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
from awes_ekf.plotting.color_palette import set_plot_style, get_color

set_plot_style()


config_file_name = "v3_config.yaml"
config = load_config("examples/" + config_file_name)
# Initialize EKF
# Load results and flight data and plot kite reference frame
results, flight_data = read_results(
    str(config["year"]),
    str(config["month"]),
    str(config["day"]),
    config["kite"]["model_name"],
    addition="",
)

results_va, _ = read_results(
    str(config["year"]),
    str(config["month"]),
    str(config["day"]),
    config["kite"]["model_name"],
    addition="_va",
)

flight_data = flight_data.loc[flight_data.index > 600]
results = results.loc[results.index > 600]
results_va = results_va.loc[results_va.index > 600]


fig, axs = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
pu.plot_time_series(
    flight_data,
    results["wind_velocity"],
    axs[0],
    label="EKF minimum",
    color=get_color("Blue"),
)
pu.plot_time_series(
    flight_data, results_va["wind_velocity"], axs[0], label="EKF va", color=get_color("Green")
)
pu.plot_time_series(
    flight_data,
    flight_data["ground_wind_velocity"],
    axs[0],
    label="Ground",
    color=get_color("Light Gray", alpha=0.5),
)
axs[0].legend()

results["wind_direction"][results["wind_direction"] > np.pi] -= 2*np.pi
results_va["wind_direction"][results_va["wind_direction"] > np.pi] -= 2*np.pi

pu.plot_time_series(
    flight_data,
    np.degrees(results["wind_direction"]),
    axs[1],
    label="EKF minimum",
    color=get_color("Blue"),
)
pu.plot_time_series(
    flight_data,
    np.degrees(results_va["wind_direction"]),
    axs[1],
    label="EKF va",
    color=get_color("Green"),
)
pu.plot_time_series(
    flight_data,
    flight_data["ground_wind_direction"],
    axs[1],
    label="Ground",
    color=get_color("Light Gray", alpha=0.5),
)
axs[1].legend()
pu.plot_time_series(
    flight_data,
    results["z_wind"],
    axs[2],
    label="EKF minimum",
    color=get_color("Blue"),
)
pu.plot_time_series(
    flight_data,
    results_va["z_wind"],
    axs[2],
    label="EKF va",
    color=get_color("Green"),
)
plt.show()

