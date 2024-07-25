import numpy as np
import casadi as ca
from typing import Union
from awes_ekf.setup.settings import kappa, z0

# %% Function definitions

def calculate_log_wind_velocity(uf, wdir, wvel_z, z):
    wvel = uf / kappa * np.log(z / z0)
    vw = np.array([wvel * np.cos(wdir), wvel * np.sin(wdir), wvel_z])
    return vw



def project_onto_plane(
    vector: Union[ca.SX, np.ndarray], plane_normal: Union[ca.SX, np.ndarray]
) -> Union[ca.SX, np.ndarray]:
    """
    Projects a vector onto a plane defined by its normal.

    Parameters:
    vector (array-like): The vector to be projected onto the plane.
    plane_normal (array-like): The normal vector of the plane.

    Returns:
    array-like: The projected vector onto the plane.
    """
    if type(vector) == ca.SX:
        return vector - ca.dot(vector, plane_normal) * plane_normal

    return vector - np.dot(vector, plane_normal) * plane_normal


def rotate_vector_around_axis(vector, rotation_axis, theta):
    """
    Rotates a vector v around an axis u by an angle theta using Rodrigues' rotation formula.
    The function supports both NumPy arrays and CasADi symbolic expressions.

    Parameters:
    vector (np.ndarray or casadi.SX/MX): The vector to be rotated.
    rotation_axis (np.ndarray or casadi.SX/MX): The axis vector around which to rotate.
    theta (float or casadi.SX/MX): The angle of rotation in radians.

    Returns:
    np.ndarray or casadi.SX/MX: The rotated vector, type depends on the input type.
    """
    if type(vector) == ca.SX:
        # Normalize the axis vector u for CasADi expressions
        rotation_axis = rotation_axis / ca.norm_2(rotation_axis)

        # Compute the cosine and sine of the angle for CasADi
        cos_theta = ca.cos(theta)
        sin_theta = ca.sin(theta)

        # Rodrigues' rotation formula for CasADi
        v_rot = (
            vector * cos_theta
            + ca.cross(rotation_axis, vector) * sin_theta
            + rotation_axis * ca.dot(rotation_axis, vector) * (1 - cos_theta)
        )
    else:
        # Assuming vector, rotation_axis, theta are NumPy arrays or can be treated as such
        # Normalize the axis vector rotation_axis for NumPy arrays
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        # Compute the cosine and sine of the angle for NumPy
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Rodrigues' rotation formula for NumPy
        v_rot = (
            vector * cos_theta
            + np.cross(rotation_axis, vector) * sin_theta
            + rotation_axis * np.dot(rotation_axis, vector) * (1 - cos_theta)
        )

    return v_rot


def calculate_angle(vector_a, vector_b, deg=True):
    if type(vector_a) == ca.SX:
        dot_product = ca.dot(vector_a, vector_b)
        magnitude_a = ca.norm_2(vector_a)
        magnitude_b = ca.norm_2(vector_b)

        cos_theta = dot_product / (magnitude_a * magnitude_b)
        angle_rad = ca.arccos(cos_theta)

        if deg:
            return angle_rad * 180 / np.pi
        else:
            return angle_rad
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = np.arccos(cos_theta)

    # # Determine the sign of the angle
    # cross_product = np.cross(vector_a, vector_b)
    # if cross_product[2] < 0:
    #     angle_rad = -angle_rad

    angle_deg = np.degrees(angle_rad)

    if deg:
        return angle_deg
    else:
        return angle_rad


def calculate_angle_2vec(vector_a, vector_b, reference_vector=None):

    if type(vector_a) == ca.SX:
        dot_product = ca.dot(vector_a, vector_b)
        magnitude_a = ca.norm_2(vector_a)
        magnitude_b = ca.norm_2(vector_b)

        cos_theta = dot_product / (magnitude_a * magnitude_b)
        angle_rad = ca.arccos(cos_theta)

        return angle_rad

    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    # Determine the sign of the angle
    if reference_vector is not None:
        reference_cross = np.cross(reference_vector, vector_a)
        if np.dot(reference_cross, vector_b) < 0:
            angle_rad = -angle_rad

    return angle_rad


def rank_observability_matrix(A, C):
    # Construct the observability matrix O_numeric
    n = A.shape[1]  # Number of state variables
    m = C.shape[0]  # Number of measurements
    O = np.zeros((m * n, n))

    for i in range(n):
        power_of_A = np.linalg.matrix_power(A, i)
        O[i * m : (i + 1) * m, :] = C @ power_of_A

    # Compute the rank of O using NumPy
    rank_O = np.linalg.matrix_rank(O)
    return rank_O


def calculate_polar_coordinates(r):
    # Calculate azimuth and elevation angles from a vector.
    r_mod = np.linalg.norm(r)
    az = np.arctan2(r[1], r[0])
    el = np.arcsin(r[2] / r_mod)
    return el, az, r_mod


def calculate_airflow_angles(dcm, apparent_wind_speed):
    ey_kite = dcm[:, 1]  # Kite y axis perpendicular to v and tether
    ez_kite = dcm[:, 2]  # Kite z axis pointing in the direction of the tension
    va = apparent_wind_speed
    va_proj = project_onto_plane(
        va, ey_kite
    )  # Projected apparent wind velocity onto kite y axis
    aoa = calculate_angle(ez_kite, va_proj) - 90  # Angle of attack
    va_proj = project_onto_plane(
        va, ez_kite
    )  # Projected apparent wind velocity onto kite z axis
    sideslip = 90 - calculate_angle(ey_kite, va_proj)  # Sideslip angle
    return aoa, sideslip


def Rx(theta):
    """Generate a rotation matrix for a rotation about the x-axis by `theta` radians."""
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def Ry(theta):
    """Generate a rotation matrix for a rotation about the y-axis by `theta` radians."""
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def Rz(theta):
    """Generate a rotation matrix for a rotation about the z-axis by `theta` radians."""
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def rotate_ENU2NED(vector):
    """Generate a rotation matrix to convert from ENU to NED coordinate system."""
    R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    return R @ vector


def rotate_NED2ENU(vector):
    """Generate a rotation matrix to convert from NED to ENU coordinate system."""
    R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]).T
    return R @ vector


def R_EG_Body(roll, pitch, yaw):
    """Create the total rotation matrix from Earth-fixed to body reference frame in ENU coordinate system."""
    # Perform rotation about x-axis (roll), then y-axis (pitch), then z-axis (yaw)
    return Rz(yaw).dot(Ry(pitch).dot(Rx(roll)))


def calculate_euler_from_reference_frame(dcm):

    # Calculate the roll, pitch and yaw angles from a direction cosine matrix, in NED coordinates
    r = R.from_matrix(dcm)
    euler = r.as_euler("xyz")

    return euler[0], euler[1], euler[2]


from scipy.spatial.transform import Rotation as R


def calculate_reference_frame_euler(
    roll, pitch, yaw, eulerFrame="ENU", outputFrame="ENU"
):
    """
    Calculate the Earth reference frame vectors based on Euler angles using quaternions to avoid gimbal lock.

    Parameters:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.
        eulerFrame (str): Type of input frame ('NED' or 'ENU').
        outputFrame (str): Type of output frame ('NED' or 'ENU').

    Returns:
        tuple: Transformed unit vectors along the x, y, and z axes in Earth coordinates.
    """

    # Convert Euler angles to a quaternion
    if eulerFrame == "NED":
        # NED uses a ZYX rotation sequence
        quaternion = R.from_euler("xyz", [roll, pitch, yaw])
    elif eulerFrame == "ENU":
        # ENU uses a ZYX rotation sequence as well but the interpretation of angles is different
        quaternion = R.from_euler("xyz", [roll, pitch, yaw])

    # Convert quaternion to a rotation matrix
    rotation_matrix = quaternion.as_matrix()

    # If converting from one frame to another, apply the appropriate rotation
    if eulerFrame != outputFrame:
        if outputFrame == "ENU" and eulerFrame == "NED":
            # Convert NED to ENU rotation matrix
            rotation_matrix = rotate_NED2ENU(rotation_matrix)

        elif outputFrame == "NED" and eulerFrame == "ENU":
            # Convert ENU to NED rotation matrix
            rotation_matrix = rotate_ENU2NED(rotation_matrix)

    return rotation_matrix
