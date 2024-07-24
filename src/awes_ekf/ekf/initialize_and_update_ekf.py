import numpy as np
from awes_ekf.setup.tether import Tether
from awes_ekf.ekf import ExtendedKalmanFilter, DynamicModel, ObservationModel
from awes_ekf.setup.kite import Kite
from awes_ekf.setup.kcu import KCU
from awes_ekf.ekf.ekf_output import create_ekf_output


def initialize_ekf(ekf_input, simConfig, tuningParams, x0, kite, kcu, tether):
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
        simConfig.ts,
        dyn_model,
        obs_model,
        kite,
        tether,
        kcu,
        simConfig,
    )
    # Initialize input vector
    ekf.update_input_vector(ekf_input)
    # Initialize state vector
    ekf.x_k1_k1 = x0

    return ekf


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
    ekf.predict()
    # Update next step
    ekf.update()

    ekf_output = create_ekf_output(
        ekf.x_k1_k1, ekf.u, ekf_input, tether, kite, simConfig
    )
    if simConfig.debug:
        for key, value in ekf.debug_info.items():
            ekf_output.__dict__[key] = value


    return ekf, ekf_output


def propagate_state_EKF(ekf, ekf_input, simConfig, tether, kite, kcu):
    # Predict step
    ekf.x_k1_k = kite.propagate(ekf.x_k1_k1, ekf.u, ekf_input.ts)

    ## Update step
    ekf, ekf_ouput = update_state_ekf_tether(
        ekf, tether, kite, kcu, ekf_input, simConfig
    )

    return ekf, ekf_ouput
