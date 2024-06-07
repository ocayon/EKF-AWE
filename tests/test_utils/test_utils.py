import numpy as np
import casadi as ca
import pytest
from awes_ekf.utils import project_onto_plane, rotate_vector_around_axis, calculate_angle, calculate_angle_2vec, rank_observability_matrix, calculate_polar_coordinates, calculate_airflow_angles, calculate_euler_from_reference_frame

# Test for project_onto_plane function
@pytest.mark.parametrize("vector, plane_normal, expected", [
    (np.array([3, 3, 3]), np.array([0, 0, 1]), np.array([3, 3, 0])),
    (ca.SX([3, 3, 3]), ca.SX([0, 0, 1]), np.array([3, 3, 0])),
])
def test_project_onto_plane(vector, plane_normal, expected):
    result = project_onto_plane(vector, plane_normal)
    if isinstance(result, ca.SX) or isinstance(result, ca.MX):
        # Evaluate using casadi
        f = ca.Function('f', [], [result])  # No input arguments, projected as output
        result_evaluated = f()  # Evaluate

        # Since result_evaluated is a dictionary, extract the DM using the appropriate key
        dm_result = result_evaluated['o0']  # Extract the DM object

        # Convert the DM object to a NumPy array
        result = np.array(dm_result.full()).flatten()

    assert np.allclose(result, expected, atol=1e-6)

# Test for calculate_angle function
@pytest.mark.parametrize("vector_a, vector_b, deg, expected", [
    (np.array([1, 0, 0]), np.array([0, 1, 0]), True, 90),
    (np.array([1, 0, 0]), np.array([1, 0, 0]), False, 0),
])
def test_calculate_angle(vector_a, vector_b, deg, expected):
    result = calculate_angle(vector_a, vector_b, deg)
    assert np.isclose(result, expected, atol=1e-6)

# Test for calculate_angle_2vec function
@pytest.mark.parametrize("vector_a, vector_b, expected", [
    (np.array([1, 0, 0]), np.array([0, 1, 0]), np.pi / 2),
    (ca.SX([1, 0, 0]), ca.SX([0, 1, 0]), ca.SX(np.pi / 2)),
])
def test_calculate_angle_2vec(vector_a, vector_b, expected):
    result = calculate_angle_2vec(vector_a, vector_b)
    assert np.isclose(float(result), float(expected), atol=1e-6)

# Test for rank_observability_matrix function
def test_rank_observability_matrix():
    A = np.array([[1, 1], [0, 1]])
    C = np.array([[1, 0]])
    expected = 2
    result = rank_observability_matrix(A, C)
    assert result == expected

# Test for calculate_polar_coordinates function
def test_calculate_polar_coordinates():
    r = np.array([1, 1, 1])
    expected = (np.arcsin(1 / np.sqrt(3)), np.arctan2(1, 1), np.sqrt(3))
    result = calculate_polar_coordinates(r)
    assert np.allclose(result, expected, atol=1e-6)

# Test for calculate_airflow_angles function
def test_calculate_airflow_angles():
    dcm = np.eye(3)
    v_kite = np.array([1, 0, 0])
    vw = np.array([0, 1, 0])
    expected = (0, 45)
    result = calculate_airflow_angles(dcm, vw-v_kite)
    assert np.allclose(result, expected, atol=1e-6)

# Test for calculate_euler_from_reference_frame function
def test_calculate_euler_from_reference_frame():
    dcm = np.eye(3)
    expected = (0, 0, 0)
    result = calculate_euler_from_reference_frame(dcm)
    assert np.allclose(result, expected, atol=1e-6)

