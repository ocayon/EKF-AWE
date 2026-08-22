"""Compare apparent wind speed calibration methods on a single flight.

Runs the EKF three times over the same flight and compares the estimated wind
velocity:

    none    the pitot reading is fed to the filter as logged
    offset  a constant additive offset on the speed, the correction the filter
            used before the pitot calibration replaced it
    pitot   a calibration coefficient on the dynamic pressure, the way a pitot
            tube is actually calibrated

Both corrections are fitted on the same pre-run, so the comparison isolates the
model and not the data it was fitted on. Only the pitot calibration lives in the
EKF itself; the offset is reconstructed here for the comparison.

The flight, the configuration, the time window and the calibration window are
selected from the terminal. The pre-processed flight data is read directly from
processed_data/, so the log has to be pre-processed already (run run_analysis.py
once if it is not).
"""

import os.path
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from awes_ekf.ekf.initialize_and_update_ekf import (
    initialize_ekf,
    propagate_state_EKF,
    calibration_series,
    calibration_window,
    mean_timestep,
)
from awes_ekf.load_data.read_data import read_processed_flight_data
from awes_ekf.load_data.create_input_from_csv import (
    create_input_from_csv,
    find_initial_state_vector,
)
from awes_ekf.setup.settings import load_config, SimulationConfig, TuningParameters
from awes_ekf.setup.kite import PointMassEKF
from awes_ekf.setup.tether import Tether
from awes_ekf.setup.kcu import KCU
from awes_ekf.postprocess.postprocessing import find_offset, find_pitot_calibration
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style_no_latex


VARIANTS = [("none", "No calibration"), ("offset", "Offset (old)"), ("pitot", "Pitot calibration (new)")]

# The filter needs some time to converge from its initial state. That transient
# is plotted but excluded from the axis scaling and from the summary statistics.
SETTLE_MINUTES = 0.5

# Analysis windows stop this far before the end of a flight by default: the data
# of the last minutes is often unreliable.
END_MARGIN_MINUTES = 3


def ask(prompt, default):
    """Prompt for a number, falling back to the default on an empty answer."""
    answer = input(f"{prompt} [default: {default}]: ").strip()
    return type(default)(answer) if answer else default


def select_flight(kite_model):
    """List the pre-processed flights of a kite model and let the user pick one."""
    flight_dir = Path("processed_data/flight_data") / kite_model
    flights = sorted(flight_dir.glob("*.csv"))
    if not flights:
        sys.exit(f"No pre-processed flight data found in {flight_dir}")

    print(f"\nAvailable pre-processed flights for {kite_model}:")
    for index, flight in enumerate(flights, start=1):
        print(f"{index}: {flight.name}")

    selection = int(input(f"Select a flight (1-{len(flights)}): ")) - 1
    if not 0 <= selection < len(flights):
        sys.exit("Invalid selection.")

    # File name is <kite_model>_<yyyy>-<mm>-<dd>.csv
    date_str = flights[selection].stem.split("_")[-1]
    year, month, day = date_str.split("-")
    print(f"Selected flight: {flights[selection].name}")
    return year, month, day


def select_time_window(flight_data):
    """Ask for the stretch of flight to simulate."""
    dt = flight_data["time"].diff().mean()
    duration = (flight_data["time"].max() - flight_data["time"].min()) / 60
    print(f"\nDuration of the flight: {duration:.2f} minutes.")
    print("Note: three full EKF runs are simulated over this window.")

    start_minute = ask("Start minute of the analysis", 0)
    # Stay away from the end of the flight, the filter does odd things there
    end_minute = ask("End minute of the analysis", round(duration - END_MARGIN_MINUTES))

    flight_data = flight_data.iloc[
        int(start_minute * 60 / dt) : int(end_minute * 60 / dt)
    ].reset_index(drop=True)
    print(f"Analysing minutes {start_minute} to {end_minute} of the flight.")
    return flight_data


def select_calibration_window(config_data):
    """Ask how much of the flight the calibration should be fitted on."""
    parameters = config_data["simulation_parameters"]
    print("\nCalibration window, relative to the start of the analysis window.")
    parameters["calibration_start_minutes"] = ask(
        "Calibration start minute", float(parameters.get("calibration_start_minutes", 5))
    )
    parameters["calibration_duration_minutes"] = ask(
        "Calibration duration in minutes",
        float(parameters.get("calibration_duration_minutes", 15)),
    )
    parameters["calibration_end_margin_minutes"] = ask(
        "Keep the calibration this many minutes clear of the end",
        float(parameters.get("calibration_end_margin_minutes", 3)),
    )
    return config_data


def build_system(config_data, flight_data):
    """Build the filter components and inputs shared by all variants."""
    parameters = dict(config_data["simulation_parameters"])
    # The script applies the corrections itself, so the filter must not calibrate
    parameters["calibrate_apparent_windspeed"] = False
    parameters["find_offset_angle_of_attack"] = False
    simConfig = SimulationConfig(**parameters)

    kite = PointMassEKF(simConfig, **config_data["kite"])
    kcu = KCU(**config_data["kcu"]) if config_data["kcu"] else None
    tether = Tether(kite, kcu, simConfig.obsData, **config_data["tether"])
    kite.calc_fx = kite.get_fx_fun(tether)
    tuningParams = TuningParameters(config_data["tuning_parameters"], simConfig)

    ekf_input_list = create_input_from_csv(
        flight_data, kite, kcu, tether, simConfig, kite_sensor=0
    )
    x0 = find_initial_state_vector(
        tether,
        ekf_input_list[0],
        simConfig,
        wind_velocity=simConfig.initial_wind_velocity,
    )
    return simConfig, kite, kcu, tether, tuningParams, ekf_input_list, x0


def fit_corrections(system, config_data, flight_data):
    """Fit both corrections on one pre-run, so they see identical data."""
    simConfig, kite, kcu, tether, tuningParams, ekf_input_list, x0 = system

    estimated, measured = calibration_series(
        ekf_input_list,
        simConfig,
        tuningParams,
        x0,
        kite,
        kcu,
        tether,
        "kite_apparent_windspeed",
    )
    if len(estimated) == 0:
        sys.exit("The calibration pre-run produced no usable samples.")

    offset = find_offset(estimated, measured, offset_range=[-15, 15])
    calibration = find_pitot_calibration(
        estimated, measured, fit_zero=simConfig.pitot_calibration_fit_zero
    )

    fit_start, fit_end = calibration_window(ekf_input_list, simConfig)
    timestep = mean_timestep(ekf_input_list)
    print(
        f"\nFitted on {len(estimated)} samples "
        f"({(fit_end - fit_start) * timestep / 60:.1f} min of flight)"
    )
    print(f"  Offset:            {offset:+.3f} m/s")
    print(
        f"  Pitot calibration: k = {calibration.k:.4f}, zero = {calibration.b:.3f} m2/s2"
        f", speed scale {calibration.speed_scale:.4f}"
    )

    return {
        "none": lambda va: va,
        "offset": lambda va: va + offset,
        "pitot": lambda va: float(calibration.apply(va)),
    }, offset, calibration


def run_variant(config_data, flight_data, correction):
    """Run one full EKF simulation with the given correction applied to the pitot.

    The outputs are aligned with flight_data: a timestep the filter could not
    integrate is stored as None, so all variants share the same time axis.
    """
    simConfig, kite, kcu, tether, tuningParams, ekf_input_list, x0 = build_system(
        config_data, flight_data
    )
    for ekf_input in ekf_input_list:
        ekf_input.kite_apparent_windspeed = correction(
            ekf_input.kite_apparent_windspeed
        )

    ekf, ekf_input_list = initialize_ekf(
        ekf_input_list,
        simConfig,
        tuningParams,
        x0,
        kite,
        kcu,
        tether,
        find_offsets=False,
    )

    ekf_output_list = []
    start_time = time.time()
    timestep = mean_timestep(ekf_input_list)
    for k, ekf_input in enumerate(ekf_input_list):
        try:
            ekf, ekf_output = propagate_state_EKF(
                ekf, ekf_input, simConfig, tether, kite, kcu
            )
            ekf_output_list.append(ekf_output)
        except Exception as e:
            print(f"Integration error at iteration {k}: {e}")
            try:
                x0 = find_initial_state_vector(tether, ekf_input, simConfig)
            except Exception:
                x0 = ekf.x_k1_k1
            ekf, ekf_input_list[k::] = initialize_ekf(
                ekf_input_list[k::],
                simConfig,
                tuningParams,
                x0,
                kite,
                kcu,
                tether,
                find_offsets=False,
            )
            ekf_output_list.append(None)
            continue

        if k > 0 and k % 6000 == 0:
            print(
                f"  Minute {k * timestep / 60:.0f}, "
                f"{time.time() - start_time:.0f} s elapsed"
            )

    calibrated_va = np.array(
        [ekf_input.kite_apparent_windspeed for ekf_input in ekf_input_list], dtype=float
    )
    return ekf_output_list, calibrated_va


def output_field(ekf_output_list, field):
    """Collect one field of the EKF outputs, with NaN where the filter failed."""
    return np.array(
        [
            np.nan if output is None else getattr(output, field)
            for output in ekf_output_list
        ],
        dtype=float,
    )


def wrap_direction(wind_direction):
    """Wind direction in degrees, wrapped into a window centred on its median.

    Unwrapping accumulates drift when the direction is noisy, and a plain modulo
    jumps by 360 whenever the direction happens to sit near the wrap. Centring
    the wrap on the median of the flight keeps the curve readable and leaves a
    genuine divergence of the filter visible as a genuine excursion.
    """
    degrees = np.rad2deg(wind_direction) % 360
    centre = np.nanmedian(degrees)
    if not np.isfinite(centre):
        return degrees
    return ((degrees - centre + 180) % 360) - 180 + centre


def converged_mask(t):
    """Samples after the initial filter transient, t in minutes."""
    return t > t[0] + SETTLE_MINUTES


def set_converged_ylim(ax, series, t):
    """Scale the y axis on the converged part, so the transient does not squash it."""
    mask = converged_mask(t)
    values = np.concatenate([s[mask][np.isfinite(s[mask])] for s in series])
    if values.size:
        low, high = np.percentile(values, [0.5, 99.5])
        margin = 0.1 * (high - low) if high > low else 1.0
        ax.set_ylim(low - margin, high + margin)


def lidar_columns(flight_data):
    """Return {height: column} for the lidar horizontal wind speed columns."""
    columns = {}
    for column in flight_data.columns:
        if "Wind Speed (m/s)" in column and "min" not in column and "max" not in column:
            digits = "".join(filter(str.isdigit, column.split("m")[0]))
            if digits:
                columns[int(digits)] = column
    return dict(sorted(columns.items()))


def lidar_wind_at_kite(flight_data):
    """Interpolate the lidar wind profile at the height of the kite."""
    columns = lidar_columns(flight_data)
    if len(columns) < 2:
        return None
    heights = np.array(list(columns.keys()), dtype=float)
    profile = np.column_stack(
        [flight_data[c].to_numpy(dtype=float) for c in columns.values()]
    )
    kite_height = flight_data["kite_position_z"].to_numpy(dtype=float)

    reference = np.full(len(flight_data), np.nan)
    for i in range(len(flight_data)):
        valid = np.isfinite(profile[i])
        if valid.sum() >= 2 and np.isfinite(kite_height[i]):
            reference[i] = np.interp(kite_height[i], heights[valid], profile[i][valid])
    return reference


def plot_comparison(results, flight_data, kite_model, date, savepath):
    set_plot_style_no_latex()
    palette = get_color_list()
    colors = {"none": palette[0], "offset": palette[5], "pitot": palette[3]}

    t = flight_data["time"].to_numpy(dtype=float)
    t = (t - t[0]) / 60  # minutes since the start of the window

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    lidar = lidar_wind_at_kite(flight_data)
    if lidar is not None:
        axs[0, 0].plot(
            t,
            lidar,
            color="grey",
            linestyle="--",
            alpha=0.8,
            label="Lidar at kite height",
        )

    horizontal, direction, vertical = [], [], []
    for key, label in VARIANTS:
        vw_horizontal = output_field(results[key]["outputs"], "wind_speed_horizontal")
        wind_direction = wrap_direction(
            output_field(results[key]["outputs"], "wind_direction")
        )
        vw_vertical = output_field(results[key]["outputs"], "wind_speed_vertical")
        horizontal.append(vw_horizontal)
        direction.append(wind_direction)
        vertical.append(vw_vertical)

        axs[0, 0].plot(t, vw_horizontal, color=colors[key], alpha=0.85, label=label)
        axs[0, 1].plot(t, wind_direction, color=colors[key], alpha=0.85, label=label)
        axs[1, 0].plot(t, vw_vertical, color=colors[key], alpha=0.85, label=label)
        axs[1, 1].plot(
            results["measured_va"],
            results[key]["calibrated_va"] - results["measured_va"],
            ".",
            markersize=1,
            color=colors[key],
            alpha=0.5,
            label=label,
        )

    if lidar is not None:
        horizontal.append(lidar)
    set_converged_ylim(axs[0, 0], horizontal, t)
    set_converged_ylim(axs[0, 1], direction, t)
    set_converged_ylim(axs[1, 0], vertical, t)

    axs[0, 0].set_ylabel("Horizontal wind speed [m/s]")
    axs[0, 1].set_ylabel("Wind direction [deg]")
    axs[1, 0].set_ylabel("Vertical wind speed [m/s]")
    axs[1, 1].set_ylabel("Applied correction [m/s]")
    axs[1, 1].set_xlabel("Measured apparent wind speed [m/s]")
    for ax in (axs[0, 0], axs[0, 1], axs[1, 0]):
        ax.set_xlabel("Time [min]")
    for ax in axs.flatten():
        ax.grid(alpha=0.3)
        ax.legend(loc="best", markerscale=8)

    fig.suptitle(f"Apparent wind speed calibration comparison - {kite_model} {date}")
    fig.tight_layout()
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=300)
    print(f"\nFigure saved to {savepath}")
    plt.show()


def print_summary(results, flight_data):
    lidar = lidar_wind_at_kite(flight_data)
    t = flight_data["time"].to_numpy(dtype=float)
    converged = converged_mask((t - t[0]) / 60)

    print("\n" + "=" * 74)
    print(f"{'Variant':<26}{'va corr':>10}{'vw mean':>10}{'vw std':>10}{'vs lidar':>12}")
    print("-" * 74)
    for key, label in VARIANTS:
        vw = output_field(results[key]["outputs"], "wind_speed_horizontal")[converged]
        correction = (results[key]["calibrated_va"] - results["measured_va"])[converged]
        if lidar is not None:
            valid = np.isfinite(vw) & np.isfinite(lidar[converged])
            bias = (
                np.mean(vw[valid] - lidar[converged][valid]) if valid.any() else np.nan
            )
            bias_str = f"{bias:+.3f}"
        else:
            bias_str = "n/a"
        print(
            f"{label:<26}{np.nanmean(correction):>+10.3f}"
            f"{np.nanmean(vw):>10.3f}{np.nanstd(vw):>10.3f}{bias_str:>12}"
        )
    print("=" * 74)
    print(f"Statistics exclude the first {SETTLE_MINUTES} min of filter transient.")
    print("va corr:  mean correction applied to the pitot reading [m/s]")
    print("vs lidar: mean difference against the lidar profile interpolated at the")
    print("          height of the kite, when lidar data is available [m/s]")


def main():
    config_data = load_config()
    kite_model = config_data["kite"]["model_name"]

    if not config_data["simulation_parameters"]["measurements"].get(
        "kite_apparent_windspeed", False
    ):
        sys.exit(
            "kite_apparent_windspeed is not enabled in the selected config, so there "
            "is no pitot measurement to calibrate."
        )

    year, month, day = select_flight(kite_model)
    config_data["year"], config_data["month"], config_data["day"] = year, month, day

    flight_data = read_processed_flight_data(year, month, day, kite_model)
    flight_data = select_time_window(flight_data)
    config_data = select_calibration_window(config_data)

    print("\n" + "=" * 74)
    print("Fitting both corrections on a single pre-run")
    print("=" * 74)
    system = build_system(config_data, flight_data)
    corrections, _, _ = fit_corrections(system, config_data, flight_data)

    results = {
        "measured_va": flight_data["kite_apparent_windspeed"].to_numpy(dtype=float)
    }
    for key, label in VARIANTS:
        print("\n" + "=" * 74)
        print(f"Running variant: {label}")
        print("=" * 74)
        outputs, calibrated_va = run_variant(
            config_data, flight_data.copy(), corrections[key]
        )
        results[key] = {"outputs": outputs, "calibrated_va": calibrated_va}

    print_summary(results, flight_data)

    date = f"{year}-{month}-{day}"
    savepath = (
        Path("results/plots") / f"va_calibration_comparison_{kite_model}_{date}.png"
    )
    plot_comparison(results, flight_data, kite_model, date, savepath)


if __name__ == "__main__":
    main()
