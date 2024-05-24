import pytest
from awes_ekf.utils import calculate_euler_from_reference_frame, calculate_reference_frame_euler, rotate_NED2ENU
from scipy.spatial.transform import Rotation as R
import numpy as np
import pytest

import numpy as np
import pytest

@pytest.mark.parametrize("dcm, expected", [
    (np.eye(3), (0, 0, 0)),
    (np.array([[0, -1, 0], [-1, 0, 0], [0, 0, 1]]), (0, 0, np.pi)),
    (np.array([[1,0, 0], [0, -1, 0], [0, 0, -1]]), (np.pi, 0, 0)),
])
def test_calculate_euler_from_reference_frame(dcm, expected):
    roll, pitch, yaw = calculate_euler_from_reference_frame(dcm)
    assert np.allclose((roll, pitch, yaw), expected)

def test_euler_to_dcm_and_back():
    # Define a set of Euler angles
    roll, pitch, yaw = np.pi/4, np.pi/4, -np.pi/4

    # Calculate the rotation matrix from these Euler angles
    rotation_matrix = calculate_reference_frame_euler(roll, pitch, yaw, 'ENU', 'ENU')

    # Calculate Euler angles back from the rotation matrix
    calculated_roll, calculated_pitch, calculated_yaw = calculate_euler_from_reference_frame(rotation_matrix)

    # Verify that the calculated Euler angles match the original ones
    assert np.isclose(calculated_roll, roll, atol=1e-2)
    assert np.isclose(calculated_pitch, pitch, atol=1e-2)
    assert np.isclose(calculated_yaw, yaw, atol=1e-2)

    # Define a set of Euler angles
    roll, pitch, yaw = np.pi/6, np.pi/3, -np.pi/6

    # Calculate the rotation matrix from these Euler angles in NED frame
    rotation_matrix_ned = calculate_reference_frame_euler(roll, pitch, yaw, 'NED', 'NED')

    # Convert this rotation matrix back to Euler angles
    calculated_roll, calculated_pitch, calculated_yaw = calculate_euler_from_reference_frame(rotation_matrix_ned)

    # Verify that the calculated Euler angles match the original ones
    assert np.isclose(calculated_roll, roll, atol=1e-2)
    assert np.isclose(calculated_pitch, pitch, atol=1e-2)
    assert np.isclose(calculated_yaw, yaw, atol=1e-2)

def test_conversion_between_frames():
    # Define a set of Euler angles in ENU frame
    roll, pitch, yaw = np.pi/4, np.pi/4, -np.pi/4

    # Convert to NED frame
    rotation_matrix_ned = calculate_reference_frame_euler(roll, pitch, yaw, 'ENU', 'NED')

    # Convert back to ENU frame
    rotation_matrix_enu = calculate_reference_frame_euler(roll, pitch, yaw, 'ENU', 'ENU')

    # Ensure the rotation matrices are correct inverses
    assert np.allclose(rotation_matrix_enu, rotate_NED2ENU(rotation_matrix_ned), atol=1e-2)

    # Convert back to Euler angles from ENU frame
    calculated_roll, calculated_pitch, calculated_yaw = calculate_euler_from_reference_frame(rotation_matrix_enu)

    # Verify that the calculated Euler angles match the original ones
    assert np.isclose(calculated_roll, roll, atol=1e-2)
    assert np.isclose(calculated_pitch, pitch, atol=1e-2)
    assert np.isclose(calculated_yaw, yaw, atol=1e-2)

    
