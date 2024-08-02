import numpy as np
from awes_ekf.setup.tether import Tether
from awes_ekf.ekf import ExtendedKalmanFilter, DynamicModel, ObservationModel
from awes_ekf.setup.kite import Kite
from awes_ekf.setup.kcu import KCU
from awes_ekf.ekf.ekf_output import create_ekf_output
from awes_ekf.postprocess.postprocessing import find_offset
import time
import copy

def initialize_ekf(ekf_input_list, simConfig, tuningParams, x0, kite, kcu, tether, find_offsets=True):
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
    ekf.update_input_vector(ekf_input_list[0])
    # Initialize state vector
    ekf.x_k1_k1 = x0

    #Copy ekf using deepcopy
    ekf_copy = copy.deepcopy(ekf)
    simConfig_copy = copy.deepcopy(simConfig)
    if find_offsets:
        offset_variables = ["apparent_windspeed", "angle_of_attack"]
        # Find offsets
        for variable in simConfig_copy.obsData.__dict__.keys():
            if variable in offset_variables and simConfig_copy.obsData.__dict__[variable]:
                print(f"Finding offset for {variable}")
                ekf_output_list = []
                offset_sim_length = int(10*60/simConfig_copy.ts)
                ekf_output_list = []
                simConfig_copy.obsData.__dict__[variable] = False
                simConfig_copy.enforce_z_wind = True
                obs_model = ObservationModel(dyn_model.x, dyn_model.u, simConfig_copy, kite, tether, kcu)
                tuningParams.update_observation_vector(simConfig_copy)
                ekf_copy.stdv_measurements = tuningParams.stdv_measurements
                ekf_copy.obs_model = obs_model
                start_time = time.time()
                mins = 0
                for k in range(offset_sim_length):
                    ekf_copy, ekf_ouput = propagate_state_EKF(
                        ekf_copy, ekf_input_list[k], simConfig_copy, tether, kite, kcu
                    )
                    # Store results
                    ekf_output_list.append(ekf_ouput)
                    # Print progress
                    if k % 600 == 0:
                        elapsed_time = time.time() - start_time
                        start_time = time.time()  # Record end time
                        mins += 1
                        print(
                            f"Real time: {mins} minutes.  Elapsed time: {elapsed_time:.2f} seconds"
                        )   

                # Find offset
                #TODO: Define universal namings and create timeseries class
                if variable == "angle_of_attack":
                    variable = "kite_aoa"
                converged_idx = int(5*60/simConfig_copy.ts)
                estimated_variable = np.array([ekf_output.__dict__[variable] for ekf_output in ekf_output_list])
                measured_variable = np.array([ekf_input.__dict__[variable] for ekf_input in ekf_input_list[:offset_sim_length]])
                offset = find_offset(estimated_variable[converged_idx::], measured_variable[converged_idx::], offset_range=[-15,15])
                print(f"Offset for {variable}: {offset}")

                # Update offset
                for i in range(len(ekf_input_list)):
                    ekf_input_list[i].__dict__[variable] += offset
                    


                


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
    ekf.predict()
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
    ekf.x_k1_k = kite.propagate(ekf.x_k1_k1, ekf.u, ekf_input.ts)

    ## Update step
    ekf, ekf_ouput = update_state_ekf_tether(
        ekf, tether, kite, kcu, ekf_input, simConfig
    )

    return ekf, ekf_ouput
