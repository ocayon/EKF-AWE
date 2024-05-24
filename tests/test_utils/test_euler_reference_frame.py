import pytest
from awes_ekf.utils import calculate_euler_from_reference_frame, calculate_reference_frame_euler
from scipy.spatial.transform import Rotation as R
import numpy as np
import pytest

import numpy as np
import pytest

def test_calculate_euler_from_reference_frame():
    # Identity matrix should lead to zero angles
    dcm = np.eye(3)
    roll, pitch, yaw = calculate_euler_from_reference_frame(dcm)
    assert np.isclose(roll, 0.0)
    assert np.isclose(pitch, 0.0)
    assert np.isclose(yaw, 0.0)

    # Test for a known rotation matrix
    r = R.from_euler('zyx', [np.pi/4,np.pi/4, np.pi/2])
    roll, pitch, yaw = r.as_euler('zyx')
    assert np.isclose(roll, np.pi/4, atol=1e-2)
    assert np.isclose(pitch, np.pi/4, atol=1e-2)  # Note the expected negative pitch
    assert np.isclose(yaw, np.pi/2, atol=1e-2)

def test_calculate_reference_frame_euler():
    # Zero rotation should return identity matrix columns
    ex_kite, ey_kite, ez_kite = calculate_reference_frame_euler(0, 0, 0, 'ENU', 'ENU')
    assert np.allclose(ex_kite, [1, 0, 0])
    assert np.allclose(ey_kite, [0, 1, 0])
    assert np.allclose(ez_kite, [0, 0, 1])

    # Test conversion between NED and ENU
    ex_kite, ey_kite, ez_kite = calculate_reference_frame_euler(0, 0, 0, 'NED', 'ENU')
    assert np.allclose(ex_kite, [0, 1, 0])
    assert np.allclose(ey_kite, [1, 0, 0])
    assert np.allclose(ez_kite, [0, 0, -1])

    # Testing full rotation by 90 degrees on yaw in ENU
    ex_kite, ey_kite, ez_kite = calculate_reference_frame_euler(0, 0, np.pi/2, 'ENU', 'ENU')
    assert np.allclose(ex_kite, [0, 1, 0], atol=1e-2)
    assert np.allclose(ey_kite, [-1, 0, 0], atol=1e-2)
    assert np.allclose(ez_kite, [0, 0, 1], atol=1e-2)
