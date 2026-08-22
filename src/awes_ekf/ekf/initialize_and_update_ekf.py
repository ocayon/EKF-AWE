import numpy as np
from awes_ekf.setup.tether import Tether
from awes_ekf.ekf import ExtendedKalmanFilter, DynamicModel, ObservationModel
from awes_ekf.setup.kite import Kite
from awes_ekf.setup.kcu import KCU
from awes_ekf.ekf.ekf_output import create_ekf_output
from awes_ekf.postprocess.postprocessing import find_offset, find_pitot_calibration
import time
import copy


def mean_timestep(ekf_input_list, n_samples=100):
    """Mean timestep of the first samples, used to convert minutes to samples."""
    return float(np.mean([i.timestep for i in ekf_input_list[:n_samples]]))


def calibration_window(ekf_input_list, simConfig):
    """First and last sample of the window a sensor calibration is fitted on.

    The window starts after the filter transient, lasts as long as the
    configuration asks for, and always stops short of the end of the flight,
    where the data is often unreliable.
    """
    samples_per_minute = 60.0 / mean_timestep(ekf_input_list)
    n_samples = len(ekf_input_list)

    start = int(simConfig.calibration_start_minutes * samples_per_minute)
    duration = int(simConfig.calibration_duration_minutes * samples_per_minute)
    margin = int(simConfig.calibration_end_margin_minutes * samples_per_minute)

    end = min(start + duration, max(n_samples - margin, 0))
    if end <= start:
        # The flight is too short for the configured window, use its middle half
        start, end = n_samples // 4, (3 * n_samples) // 4
        print(
            "Warning: flight too short for the configured calibration window, "
            f"falling back to samples {start} to {end}"
        )
    return start, end


def calibration_series(
    ekf_input_list, simConfig, tuningParams, x0, kite, kcu, tether, variable
):
    """Estimate a sensor with the filter while holding it out of the measurements.

    Returns the estimated and the measured series over the calibration window,
    which is what the sensor calibration is then fitted on.
    """
    input_variable = variable
    output_variable = (
        "kite_angle_of_attack" if variable == "bridle_angle_of_attack" else variable
    )

    fit_start, fit_end = calibration_window(ekf_input_list, simConfig)
    timestep = mean_timestep(ekf_input_list)
    print(
        f"Calibrating {variable} on minutes {fit_start * timestep / 60:.1f} to "
        f"{fit_end * timestep / 60:.1f} of the flight"
    )

    simConfig_prerun = copy.deepcopy(simConfig)
    simConfig_prerun.obsData.__dict__[input_variable] = False
    simConfig_prerun.enforce_vertical_wind_to_0 = True

    dyn_model = DynamicModel(kite, tether, simConfig_prerun)
    obs_model = ObservationModel(
        dyn_model.x, dyn_model.u, simConfig_prerun, kite, tether, kcu
    )
    tuningParams.update_observation_vector(simConfig_prerun)
    try:
        ekf = ExtendedKalmanFilter(
            tuningParams.stdv_dynamic_model,
            tuningParams.stdv_measurements,
            dyn_model,
            obs_model,
            kite,
            tether,
            kcu,
            simConfig_prerun,
        )
        ekf.update_input_vector(ekf_input_list[0])
        ekf.x_k1_k1 = x0

        ekf_output_list = []
        start_time = time.time()
        for k in range(fit_end):
            try:
                ekf, ekf_output = propagate_state_EKF(
                    ekf, ekf_input_list[k], simConfig_prerun, tether, kite, kcu
                )
            except Exception as e:
                print(f"Error at timestep {k}: {e}")
                break
            ekf_output_list.append(ekf_output)
            if k > 0 and k % 3000 == 0:
                print(
                    f"  Pre-run at minute {k * timestep / 60:.1f}, "
                    f"{time.time() - start_time:.1f} s elapsed"
                )
    finally:
        # Leave the tuning parameters as the caller passed them: they describe
        # the real filter, not the one used for this pre-run
        tuningParams.update_observation_vector(simConfig)

    # The pre-run can stop early, so only keep the samples both series have
    n_samples = min(len(ekf_output_list), fit_end)
    estimated = np.array(
        [output.__dict__[output_variable] for output in ekf_output_list[:n_samples]]
    )
    measured = np.array(
        [ekf_input.__dict__[input_variable] for ekf_input in ekf_input_list[:n_samples]]
    )
    return estimated[fit_start:], measured[fit_start:]


def initialize_ekf(
    ekf_input_list, simConfig, tuningParams, x0, kite, kcu, tether, find_offsets=True
):
    """
    Initialize the Extended Kalman Filter with system components and models.

    Args:
        ekf_input (EKFInput): Input parameters for the EKF.
        simConfig (SimulationConfig): Configuration settings for the simulation models.
        tuningParams (SystemParameters): Specifications of the system components.

    Returns:
        tuple: Returns a tuple containing initialized components of the EKF including the filter itself,
               dynamic model, kite, KCU (Kite Control Unit), and tether.
    """

    # Create dynamic model and observation model
    dyn_model = DynamicModel(kite, tether, simConfig)
    obs_model = ObservationModel(dyn_model.x, dyn_model.u, simConfig, kite, tether, kcu)

    # Initialize EKF
    ekf = ExtendedKalmanFilter(
        tuningParams.stdv_dynamic_model,
        tuningParams.stdv_measurements,
        dyn_model,
        obs_model,
        kite,
        tether,
        kcu,
        simConfig,
    )
    # Initialize input vector
    ekf.update_input_vector(ekf_input_list[0])
    # Initialize state vector
    ekf.x_k1_k1 = x0

    if find_offsets:
        calibration_variables = [
            variable
            for variable, enabled in simConfig.calibrate_sensor.items()
            if enabled
        ]
        # Calibrate the enabled sensors
        for variable in simConfig.obsData.__dict__.keys():
            if (
                variable not in calibration_variables
                or not simConfig.obsData.__dict__[variable]
            ):
                continue

            estimated, measured = calibration_series(
                ekf_input_list,
                simConfig,
                tuningParams,
                x0,
                kite,
                kcu,
                tether,
                variable,
            )
            if len(estimated) == 0:
                print(f"Not enough samples to calibrate {variable}, skipping")
                continue

            if variable == "kite_apparent_windspeed":
                # A pitot tube is calibrated in dynamic pressure, not with an
                # additive offset on the speed
                calibration = find_pitot_calibration(
                    estimated, measured, fit_zero=simConfig.pitot_calibration_fit_zero
                )
                print(
                    f"Pitot calibration for {variable}: k = {calibration.k:.4f}, "
                    f"zero = {calibration.b:.4f} m2/s2 "
                    f"(speed scale {calibration.speed_scale:.4f})"
                )
                for ekf_input in ekf_input_list:
                    ekf_input.kite_apparent_windspeed = float(
                        calibration.apply(ekf_input.kite_apparent_windspeed)
                    )
            else:
                offset = find_offset(estimated, measured, offset_range=[-15, 15])
                print(f"Offset for {variable}: {offset}")

                # Update offset
                for ekf_input in ekf_input_list:
                    ekf_input.__dict__[variable] += offset

    return ekf, ekf_input_list


def update_state_ekf_tether(ekf, tether, kite, kcu, ekf_input, simConfig):
    """
    Update the state of the Extended Kalman Filter (EKF) and the tether model based on new measurements.

    Args:
        ekf (ExtendedKalmanFilter): The EKF instance to be updated.
        tether (Tether): The tether model to update.
        kite (Kite): The kite model involved in the EKF process.
        kcu (KCU): The kite control unit.
        dyn_model (DynamicModel): The dynamic model used in the EKF.
        ekf_input (EKFInput): New input measurements for the EKF.
        simConfig (SimulationConfig): Configuration settings for the simulation models.

    Returns:
        tuple: Returns updated EKF instance, tether model, and an output structure with updated state.
    """

    ############################################################
    # Update EKF
    ############################################################
    ekf.update_input_vector(ekf_input)
    ekf.update_measurement_vector(ekf_input, simConfig)
    ############################################################
    # Update state with Kalmann filter
    ############################################################
    # Predict next step
    ekf.predict(ekf_input.timestep)
    # Update next step
    ekf.update()

    ekf_output = create_ekf_output(
        ekf.x_k1_k1, ekf.u, ekf_input, tether, kite, simConfig
    )

    for key, value in ekf.debug_info.items():
        ekf_output.__dict__[key] = value

    return ekf, ekf_output


def propagate_state_EKF(ekf, ekf_input, simConfig, tether, kite, kcu):
    # Predict step
    ekf.x_k1_k = kite.propagate(ekf.x_k1_k1, ekf.u, ekf_input.timestep)

    ## Update step
    ekf, ekf_ouput = update_state_ekf_tether(
        ekf, tether, kite, kcu, ekf_input, simConfig
    )

    return ekf, ekf_ouput
