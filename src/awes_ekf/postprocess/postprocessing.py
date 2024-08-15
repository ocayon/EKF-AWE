import numpy as np
from awes_ekf.utils import calculate_reference_frame_euler, calculate_airflow_angles


def calculate_offset_pitch_depower_turn(flight_data, results, sensor_index=0, prefix="kite"):
    """
    Calculate the offset pitch for the depower phase on EKF mean pitch during depower.

    Parameters:
    flight_data (dict): Dictionary containing flight data with keys 'us', 'up', and the pitch column based on the prefix and sensor_index.
    results (dict): Dictionary containing EKF results with the pitch key based on the prefix.
    sensor_index (int): Index of the sensor (0 or 1).
    prefix (str): Prefix for the data type (e.g., "kite", "kcu").

    Returns:
    tuple: Offset pitch for depower and turn phases.
    """
    mask_turn = abs(flight_data["us"]) > 0.8
    mask_dep = flight_data["up"] > 0.6
    pitch_IMU = np.array(flight_data[f"{prefix}_pitch_{sensor_index}"])
    
    if prefix == "kcu":
        prefix = "tether"

    pitch_EKF = np.array(results[f"{prefix}_pitch"])
    

    offset_dep = np.mean(pitch_EKF[mask_dep] - pitch_IMU[mask_dep])
    offset_turn = np.mean(pitch_EKF[mask_turn] - pitch_IMU[mask_turn])

    print(f"Offset pitch depower: {offset_dep}, Offset pitch turn: {offset_turn}")

    return offset_dep, offset_turn


def unwrap_degrees(signal):
    for i in range(1, len(signal)):
        if abs(signal[i] - signal[i - 1]) > 180:
            signal[i::] += signal[i - 1] - signal[i]

    return signal


def normalize_angles(signal):
    """
    Normalizes an array of angles so that all angles are within the range 0 to 360 degrees.

    Parameters:
    signal (numpy.ndarray): The input array of angles in degrees.

    Returns:
    numpy.ndarray: The normalized array of angles within the range 0 to 360 degrees.
    """
    normalized_signal = np.mod(signal, 360)
    normalized_signal[normalized_signal < 0] += 360
    return normalized_signal


def compute_mse(sig1, sig2, offset):
    shifted_sig2 = sig2 + offset
    mse = np.mean((sig1 - shifted_sig2) ** 2)
    return mse


def find_offset(signal1, signal2, offset_range=[-2 * np.pi, 2 * np.pi]):
    """
    Find the offset between two signals
    :param signal1: first signal
    :param signal2: second signal
    :return: offset
    """
    offsets = np.linspace(offset_range[0], offset_range[1], 1000)
    mse_values = [compute_mse(signal1, signal2, offset) for offset in offsets]
    offset = offsets[np.argmin(mse_values)]
    return offset


def construct_transformation_matrix(e_x_b, e_y_b, e_z_b):
    # Construct the matrix by arranging the unit vectors as columns
    R = np.array([e_x_b, e_y_b, e_z_b]).T
    return R


def remove_offsets_IMU_data(results, flight_data, sensor=0, prefix="kite"):
    """Remove offsets of IMU euler angles based on EKF results"""

    if prefix == "kcu":
        prefix_results = "tether"
    else:
        prefix_results = prefix
    # Roll
    roll_column = f"{prefix}_roll_{sensor}"
    unwrapped_angles = np.unwrap(flight_data[roll_column])
    flight_data[roll_column] = unwrapped_angles
    roll_offset = find_offset(
        results[f"{prefix_results}_roll"], flight_data[roll_column]
    )
    flight_data[roll_column] = flight_data[roll_column] + roll_offset
    print("Roll offset: ", np.rad2deg(roll_offset))

    # Pitch
    pitch_column = f"{prefix}_pitch_{sensor}"
    mask_pitch = (flight_data["powered"] == "powered") & (abs(flight_data["us"]) < 0.4)
    pitch_offset = find_offset(
        results[mask_pitch][f"{prefix_results}_pitch"],
        flight_data[mask_pitch][pitch_column],
    )
    flight_data[pitch_column] = flight_data[pitch_column] + pitch_offset
    print("Pitch offset: ", np.rad2deg(pitch_offset))

    # Yaw
    yaw_column = f"{prefix}_yaw_{sensor}"
    unwrapped_angles = np.unwrap(flight_data[yaw_column])
    flight_data[yaw_column] = unwrapped_angles
    unwrapped_angles = np.unwrap(results[f"{prefix_results}_yaw"])
    results[f"{prefix_results}_yaw"] = unwrapped_angles
    yaw_offset = find_offset(
        results[f"{prefix_results}_yaw"], flight_data[yaw_column]
    )
    flight_data[yaw_column] = flight_data[yaw_column] + yaw_offset
    print("Yaw offset: ", np.rad2deg(yaw_offset))

    return flight_data



def normalize_kcu_steering_inputs(flight_data):
    """Normalize the KCU steering inputs to the range [-1, 1]"""
    min_depower = min(flight_data["kcu_actual_depower"])
    max_depower = max(flight_data["kcu_actual_depower"])
    flight_data["us"] = (flight_data["kcu_actual_steering"]) / max(
        abs(flight_data["kcu_actual_steering"])
    )
    flight_data["up"] = (flight_data["kcu_actual_depower"] - min_depower) / (
        max_depower - min_depower
    )

    return flight_data


def postprocess_results(
    results,
    flight_data,
    kite,
    kcu,
    config_data,
    correct_IMU_deformation=False,
):
    """
    Calculate angle of attack and sideslip based on kite and KCU IMU data
    :param results: results from the simulation
    :param kite: kite object
    :param IMU_0: IMU data from the kite
    :param IMU_1: IMU data from the KCU
    :param EKF_tether: EKF data from the tether orientation and IMU yaw
    :return: results with aoa and ss va radius omega and slack
    """
    if kcu is not None:
        flight_data = normalize_kcu_steering_inputs(flight_data)
        flight_data["powered"] = flight_data.apply(determine_powered_depowered, axis=1)
    # Identify turn - straight and left - right
    flight_data["turn_straight"] = flight_data.apply(determine_turn_straight, axis=1)
    flight_data["right_left"] = flight_data.apply(determine_turn_straight, axis=1)

    results["wind_direction"] = results["wind_direction"] % (2 * np.pi)

    kite_sensors = config_data["kite"]["sensor_ids"]
    kcu_sensors = config_data["kcu"].get("sensor_ids", [])

    
    for imu in kite_sensors:
        flight_data = remove_offsets_IMU_data(results, flight_data, sensor=imu, prefix="kite")
    for imu in kcu_sensors:
        flight_data = remove_offsets_IMU_data(results, flight_data, sensor=imu, prefix="kcu")

    # Calculate apparent speed based on EKF results
    wvel = results["wind_speed_horizontal"]
    vw = np.vstack(
        (
            wvel * np.cos(results["wind_direction"]),
            wvel * np.sin(results["wind_direction"]),
            np.zeros(len(results)),
        )
    ).T
    r_kite = np.vstack(
        (
            np.array(results["kite_position_x"]),
            np.array(results["kite_position_y"]),
            np.array(results["kite_position_z"]),
        )
    ).T
    v_kite = np.vstack(
        (
            np.array(results["kite_velocity_x"]),
            np.array(results["kite_velocity_y"]),
            np.array(results["kite_velocity_z"]),
        )
    ).T
    # Calculate a_kite with diff of v_kite
    dt = flight_data["time"].iloc[1] - flight_data["time"].iloc[0]
    a_kite = np.vstack(
        (
            np.concatenate((np.diff(v_kite[:, 0]) / dt, [0])),
            np.concatenate((np.diff(v_kite[:, 1]) / dt, [0])),
            np.concatenate((np.diff(v_kite[:, 2]) / dt, [0])),
        )
    ).T

    if "bridle_sideslip_angle" not in flight_data.columns:
        flight_data["bridle_sideslip_angle"] = np.zeros(len(flight_data))

    results["time"] = flight_data["time"]
    # Smooth a_kite
    window_size = 10
    a_kite[:, 0] = np.convolve(
        a_kite[:, 0], np.ones(window_size) / window_size, mode="same"
    )
    a_kite[:, 1] = np.convolve(
        a_kite[:, 1], np.ones(window_size) / window_size, mode="same"
    )
    a_kite[:, 2] = np.convolve(
        a_kite[:, 2], np.ones(window_size) / window_size, mode="same"
    )

    va_kite = vw - v_kite
    results["kite_apparent_windspeed"] = np.linalg.norm(va_kite, axis=1)

    for imu in kite_sensors:
        results["wing_angle_of_attack_imu_" + str(imu)] = np.zeros(len(results))
        results["wing_sideslip_angle_imu_" + str(imu)] = np.zeros(len(results))

    
    for imu in kite_sensors:
        offset_dep, offset_turn = calculate_offset_pitch_depower_turn(
            flight_data, results, sensor_index=imu, prefix="kite"
        )

        flight_data["offset_pitch_" + str(imu)] = (
            offset_dep * flight_data["up"]
        )  # + offset_turn * flight_data["us"]
    for imu in kcu_sensors:
        offset_dep, offset_turn = calculate_offset_pitch_depower_turn(
            flight_data, results, sensor_index=imu, prefix="kcu"
        )

        flight_data["offset_pitch_" + str(imu)] = (
            offset_dep * flight_data["up"]
        )  # + offset_turn * flight_data["us"]
            

    flight_data["cycle"] = np.zeros(len(flight_data))
    cycle_count = 0
    in_cycle = False
    ip = 0
    radius_turn = []
    omega = []

    if kcu is not None:
        slack = (
            results["tether_length"]
            + kcu.distance_kcu_kite
            - np.linalg.norm(r_kite, axis=1)
        )
    else:
        slack = results["tether_length"] - np.linalg.norm(r_kite, axis=1)

    omega_p = []
    omega_q = []
    omega_r = []
    kite_elevation = []
    for i in range(len(results)):
        res = results.iloc[i]
        fd = flight_data.iloc[i]
        # Calculate tether orientation based on euler angles
        q = 0.5 * 1.225 * kite.area * res["kite_apparent_windspeed"] ** 2
        for imu in kite_sensors:
            dcm = calculate_reference_frame_euler(
                flight_data["kite_roll_" + str(imu)].iloc[i],
                flight_data["kite_pitch_" + str(imu)].iloc[i],
                flight_data["kite_yaw_" + str(imu)].iloc[i],
                eulerFrame="NED",
                outputFrame="ENU",
            )
            # Calculate wind velocity based on KCU orientation and wind speed and direction
            airflow_angles = calculate_airflow_angles(dcm, va_kite[i])
            results.loc[i, "wing_angle_of_attack_imu_" + str(imu)] = airflow_angles[0]  # Angle of attack
            results.loc[i, "wing_sideslip_angle_imu_" + str(imu)] = airflow_angles[1]  # Sideslip angle
        ez_kite = dcm[:, 2]
        kite_elevation.append(np.arcsin(-ez_kite[2]))
        at = (
            np.dot(a_kite[i], np.array(v_kite[i]) / np.linalg.norm(v_kite[i]))
            * np.array(v_kite[i])
            / np.linalg.norm(v_kite[i])
        )
        omega_kite = np.cross(a_kite[i] - at, v_kite[i]) / (
            np.linalg.norm(v_kite[i]) ** 2
        )
        ICR = np.cross(v_kite[i], omega_kite) / (np.linalg.norm(omega_kite) ** 2)
        ex = dcm[:, 0]
        det = ex[0] * ICR[1] - ex[1] * ICR[0]

        sign_radius = det / abs(det)

        radius_turn.append(sign_radius * np.linalg.norm(ICR))
        omega.append(np.linalg.norm(omega_kite))

        omega_p.append(np.dot(dcm[:, 0], omega_kite))
        omega_q.append(np.dot(dcm[:, 1], omega_kite))
        omega_r.append(np.dot(dcm[:, 2], omega_kite))

        if kcu is not None:
            if fd["powered"] == "depowered" and not in_cycle:
                flight_data.loc[ip:i, "cycle"] = cycle_count
                ip = i
                # Entering a new cycle
                cycle_count += 1
                in_cycle = True
            elif fd["powered"] == "powered" and in_cycle:
                # Exiting the current cycle
                in_cycle = False

    print("Number of cycles:", cycle_count)
    results["slack"] = slack
    flight_data["radius_turn"] = radius_turn
    results["omega"] = omega
    results["omega_p"] = omega_p
    results["omega_q"] = omega_q
    results["omega_r"] = omega_r
    results["kite_elevation"] = kite_elevation

    
    results, flight_data = correct_aoa_ss_measurements(results, flight_data)

    if correct_IMU_deformation:
        for imu in kite_sensors:
            results["kite_pitch_" + str(imu)] = flight_data[
                "kite_pitch_" + str(imu)
            ] + (flight_data["offset_pitch_" + str(imu)])

        for imu in kcu_sensors:
            results["kcu_pitch_" + str(imu)] = flight_data[
                "kcu_pitch_" + str(imu)
            ] + (flight_data["offset_pitch_" + str(imu)])

    return results, flight_data


def calculate_wind_speed_airborne_sensors(results, flight_data, imus=[0]):
    """
    Calculate wind speed based on kite and KCU IMU data
    :param flight_data: flight data
    :return: flight data with wind speed
    """
    for imu in imus:
        flight_data["vwx_IMU_" + str(imu)] = np.zeros(len(flight_data))
        flight_data["vwy_IMU_" + str(imu)] = np.zeros(len(flight_data))
        flight_data["vwz_IMU_" + str(imu)] = np.zeros(len(flight_data))

    measured_va = flight_data["kite_apparent_windspeed"]
    measured_aoa = flight_data["bridle_angle_of_attack"]
    measured_ss = -flight_data["bridle_sideslip_angle"]

    # measured_aoa = results['aoa_IMU_0']
    # measured_ss =  results['ss_IMU_0']

    measured_va = results["va_kite"]
    for i in range(len(flight_data)):
        for imu in imus:
            ex_kite, ey_kite, ez_kite = calculate_reference_frame_euler(
                flight_data["kite_roll_" + str(imu)][i],
                flight_data["kite_pitch_" + str(imu)][i],
                flight_data["kite_yaw_" + str(imu)][i],
                bodyFrame="ENU",
            )
            # Calculate apparent wind velocity based on KCU orientation and apparent wind speed and aoa and ss
            va = (
                -ex_kite
                * measured_va[i]
                * np.cos(measured_ss[i] / 180 * np.pi)
                * np.cos(measured_aoa[i] / 180 * np.pi)
                + ey_kite
                * measured_va[i]
                * np.sin(measured_ss[i] / 180 * np.pi)
                * np.cos(measured_aoa[i] / 180 * np.pi)
                + ez_kite * measured_va[i] * np.sin(measured_aoa[i] / 180 * np.pi)
            )
            # Calculate wind velocity based on KCU orientation and wind speed and direction
            flight_data.loc[i, "vwx_IMU_" + str(imu)] = va[0] + results["vx"][i]
            flight_data.loc[i, "vwy_IMU_" + str(imu)] = va[1] + results["vy"][i]
            flight_data.loc[i, "vwz_IMU_" + str(imu)] = va[2] + results["vz"][i]

    return flight_data


def determine_turn_straight(row):
    try:
        if abs(row["us"]) > 0.2:
            return "turn"
        else:
            return "straight"
    except:
        return "none"


def determine_powered_depowered(row):
    try:
        if row["up"] > 0.6:
            return "depowered"
        else:
            return "powered"
    except:
        return "none"


def determine_left_right(row):
    if row["kite_azimuth"] < 0:
        return "right"
    else:
        return "left"


def correct_aoa_ss_measurements(results, flight_data, imu=0):

    # Correct angle of attack and sideslip angle based on EKF mean angle of attack
    aoa_imu = np.array(results["wing_angle_of_attack_imu_" + str(imu)])
    aoa_ekf = np.array(results["kite_angle_of_attack"])
    ss_ekf = np.array(results["kite_sideslip_angle"])
    aoa_vane = np.array(flight_data["bridle_angle_of_attack"])
    aoa_vane = np.convolve(aoa_vane, np.ones(10) / 10, mode="same")
    ss_vane = np.array(flight_data["bridle_sideslip_angle"])
    ss_vane = np.convolve(ss_vane, np.ones(10) / 10, mode="same")

    mask_pow = (flight_data["tether_reelout_speed"] > 0) & (
        flight_data["up"] < 0.2
    )
    aoa_trim = np.mean(aoa_imu[mask_pow])

    offset_aoa_vane = aoa_trim - np.mean(flight_data["bridle_angle_of_attack"][mask_pow])
    offset_aoa_ekf = aoa_trim - np.mean(results["kite_angle_of_attack"][mask_pow])

    offset_ss_vane = np.mean(results["kite_sideslip_angle"]) - np.mean(
        flight_data["bridle_sideslip_angle"][mask_pow]
    )
    offset_ss_ekf = np.mean(results["kite_sideslip_angle"]) - np.mean(
        results["kite_sideslip_angle"][mask_pow]
    )

    print("Offset aoa vane: ", offset_aoa_vane)
    print("Offset aoa ekf: ", offset_aoa_ekf)
    print("Offset ss vane: ", offset_ss_vane)
    print("Offset ss ekf: ", offset_ss_ekf)
    # Correct angle of attack and sideslip angle based on kite deployment
    flight_data["wing_angle_of_attack"] = aoa_vane + offset_aoa_vane
    flight_data["wing_sideslip_angle"] = ss_vane + offset_ss_vane
    results["wing_angle_of_attack"] = aoa_ekf + offset_aoa_ekf
    results["wing_sideslip_angle"] = ss_ekf + offset_ss_ekf

    results["wing_angle_of_attack"] = results["wing_angle_of_attack"] - np.degrees(
        flight_data["offset_pitch_0"]
    )
    flight_data["wing_angle_of_attack"] = flight_data[
        "wing_angle_of_attack"
    ] - np.degrees(flight_data["offset_pitch_0"])

    return results, flight_data
