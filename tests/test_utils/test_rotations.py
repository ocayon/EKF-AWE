from awes_ekf.utils import rotate_vector_around_axis, Rx, Ry, Rz, rotate_ENU2NED, rotate_NED2ENU, R_EG_Body
import numpy as np
import pytest

@pytest.mark.parametrize("vector, rotation_axis, theta, expected", [
    (np.array([1, 0, 0]), np.array([0, 0, 1]), np.pi / 2, np.array([0, 1, 0])),
    (np.array([1, 0, 0]), np.array([1, 0, 0]), np.pi / 2, np.array([1, 0, 0])),
    (np.array([1, 0, 0]), np.array([0, 1, 0]), np.pi / 2, np.array([0, 0, -1])),
])
# Test for rotate_vector_around_axis function
def test_rotate_vector_around_axis(vector, rotation_axis, theta, expected):
    # Rotate the vector
    rotated_vector = rotate_vector_around_axis(vector, rotation_axis, theta)

    # Assert that the rotated vector matches the expected result
    assert np.allclose(rotated_vector, expected, atol=1e-6)

#%%
# Test for rotation matrices, Rx, Ry, Rz
def test_identity_rotation():
    I = np.eye(3)
    assert np.allclose(Rx(0), I)
    assert np.allclose(Ry(0), I)
    assert np.allclose(Rz(0), I)

def test_known_angles():
    # Testing 90 degrees (π/2 radians) rotations
    pi_2 = np.pi / 2
    assert np.allclose(Rx(pi_2), np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))
    assert np.allclose(Ry(pi_2), np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]))
    assert np.allclose(Rz(pi_2), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))

def test_inverse_rotation():
    theta = np.pi / 3  # 60 degrees
    Rx_inv = Rx(-theta)
    Ry_inv = Ry(-theta)
    Rz_inv = Rz(-theta)
    assert np.allclose(Rx(theta) @ Rx_inv, np.eye(3))
    assert np.allclose(Ry(theta) @ Ry_inv, np.eye(3))
    assert np.allclose(Rz(theta) @ Rz_inv, np.eye(3))

def test_orthogonality():
    theta = np.random.uniform(-np.pi, np.pi)
    assert np.allclose(Rx(theta) @ Rx(theta).T, np.eye(3))
    assert np.allclose(Ry(theta) @ Ry(theta).T, np.eye(3))
    assert np.allclose(Rz(theta) @ Rz(theta).T, np.eye(3))

def test_determinant():
    theta = np.random.uniform(-np.pi, np.pi)
    assert np.isclose(np.linalg.det(Rx(theta)), 1.0)
    assert np.isclose(np.linalg.det(Ry(theta)), 1.0)
    assert np.isclose(np.linalg.det(Rz(theta)), 1.0)

#%%
# Test for ENU to NED and NED to ENU rotation matrices

def test_identity_conversion():
    vectors = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    for v in vectors:
        assert np.allclose(rotate_NED2ENU(rotate_ENU2NED(v)), v)

def test_known_vector_conversion():
    # ENU unit vectors
    east = np.array([1, 0, 0])
    north = np.array([0, 1, 0])
    up = np.array([0, 0, 1])

    # Convert ENU to NED
    assert np.allclose(rotate_ENU2NED(east), np.array([0, 1, 0]))  # East in ENU is North in NED
    assert np.allclose(rotate_ENU2NED(north), np.array([1, 0, 0]))  # North in ENU is East in NED
    assert np.allclose(rotate_ENU2NED(up), np.array([0, 0, -1]))  # Up in ENU is Down in NED

    # Convert NED to ENU
    assert np.allclose(rotate_NED2ENU(np.array([0, 1, 0])), np.array([1, 0, 0]))  # North in NED is East in ENU
    assert np.allclose(rotate_NED2ENU(np.array([1, 0, 0])), np.array([0, 1, 0]))  # East in NED is North in ENU
    assert np.allclose(rotate_NED2ENU(np.array([0, 0, -1])), np.array([0, 0, 1]))  # Down in NED is Up in ENU

def test_matrix_properties():
    R_ENU2NED = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    R_NED2ENU = R_ENU2NED.T
    assert np.allclose(R_ENU2NED @ R_NED2ENU, np.eye(3))
    assert np.allclose(R_NED2ENU @ R_ENU2NED, np.eye(3))

def test_consistency():
    vector = np.array([5, -3, 2])
    assert np.allclose(rotate_NED2ENU(rotate_ENU2NED(vector)), vector)
    assert np.allclose(rotate_ENU2NED(rotate_NED2ENU(vector)), vector)


## Test for R_EG_Body function
import numpy as np
import pytest

# Assuming the rotation matrix functions Rx, Ry, Rz, and R_EG_Body are defined correctly.

def test_zero_rotation():
    I = np.eye(3)
    assert np.allclose(R_EG_Body(0, 0, 0), I), "Zero rotation should yield identity matrix"

def test_single_axis_rotations():
    pi_2 = np.pi / 2
    # Roll 90 degrees
    assert np.allclose(R_EG_Body(pi_2, 0, 0), Rx(pi_2)), "Roll 90 degrees should match Rx(pi/2)"
    # Pitch 90 degrees
    assert np.allclose(R_EG_Body(0, pi_2, 0), Ry(pi_2)), "Pitch 90 degrees should match Ry(pi/2)"
    # Yaw 90 degrees
    assert np.allclose(R_EG_Body(0, 0, pi_2), Rz(pi_2)), "Yaw 90 degrees should match Rz(pi/2)"

@pytest.mark.parametrize("roll, pitch, yaw, expected", [
    (0, np.pi / 2, 0, Ry(np.pi / 2)),
    (np.pi / 3, 0, 0, Rx(np.pi / 3)),
    (0, 0, -np.pi / 4, Rz(-np.pi / 4)),
    (0, np.pi/2, np.pi, np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]]).T),
    (0,0,0, np.eye(3)),
])
def test_composite_rotations(roll, pitch, yaw, expected):

    assert np.allclose(R_EG_Body(roll, pitch, yaw), expected), "Composite rotation did not match expected"

def test_orthogonality_and_determinant():
    theta = np.random.uniform(-np.pi, np.pi, 3)
    R = R_EG_Body(*theta)
    assert np.allclose(R @ R.T, np.eye(3)), "Matrix should be orthogonal"
    assert np.isclose(np.linalg.det(R), 1.0), "Determinant of a rotation matrix should be 1"





