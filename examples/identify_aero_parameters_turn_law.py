"""
Turn rate law identification from flight data.

Implements the three formulations from:
  Cayon & Schmehl, "Quasi-Steady Mechanics of Tethered Flight"

  Eq. (41)  Simple:       chi_dot_b = gk * va * us
  Eq. (40)  Two-term:     chi_dot_b = c1*(va*us) + c2*(sin(chi)*cos(beta)/va)
  Eq. (38)  Full rational: chi_dot_b = -(k1*va^2*us + m*g*sin(chi)*cos(beta))
                                       / (m*v_tau + k2*va)

Parameters identified by least squares (Eq. 41, 40) or nonlinear fit (Eq. 38):
  gk  = -0.5*rho*S*K_us / (m + 0.25*rho*S*b*K_rhat)   [kinematic turn gain]
  c1  = gk
  c2  = -m*g / (m + 0.25*rho*S*b*K_rhat)
  k1  = 0.5*rho*S*K_us
  k2  = 0.25*rho*S*b*K_rhat

All three fits are performed for each flight phase separately.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from pathlib import Path
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style_no_latex
from awes_ekf.plotting.plot_aerodynamics import plot_turn_rate_fit_results
from awes_ekf.utils import calculate_weighted_least_squares

plt.close("all")
set_plot_style_no_latex()

# ── Configuration ─────────────────────────────────────────────────────────────
MASS = 50.0  # kite + lines mass [kg]
G = 9.81  # gravity [m/s²]
CUT = 10  # trim edges of the dataset

# Asymmetry handling per law: "fit", "off", "fixed", "from_simple", "from_two_term"
ASYM_MODE_SIMPLE = "fit"
ASYM_MODE_TWO_TERM = "fit"
ASYM_MODE_FULL = "fit"  # warm-start full rational with two-term's fitted asymmetry
K_ASYM_FIXED = 0.0
# Turn-rate signal used for fitting: "yaw_dot" or "chi_dot".
TURN_RATE_SOURCE = "chi_dot"
# Which kite_yaw_rate_<x> column to use when TURN_RATE_SOURCE == "yaw_dot".
YAW_RATE_SENSOR_ID = 0
PHASE_NAME = {1: "reel-out", 3: "reel-in"}
PHASES_TO_FIT = [1, 2, 3, 4]  # only reel-out (1) and reel-in (3)
PALETTE = get_color_list()

# ── Load data ─────────────────────────────────────────────────────────────────
results, flight_data, _ = read_results("2025", "10", "09", "v3", addition="")
results = results[CUT:-CUT].reset_index(drop=True)
flight_data = flight_data[CUT:-CUT].reset_index(drop=True)

CYCLES = range(1, 5)  # cycles to include
PLOT_CYCLE = 2  # single cycle to show in the stitched time-series plot
mask = flight_data["cycle"].isin(CYCLES)
results = results[mask].reset_index(drop=True)
flight_data = flight_data[mask].reset_index(drop=True)

time = flight_data["time"].to_numpy()

# Extract relevant signals
us = -flight_data["kcu_actual_steering"].to_numpy() / 100
va = results["kite_apparent_windspeed"].to_numpy()
position = np.array(
    [results["kite_position_x"], results["kite_position_y"], results["kite_position_z"]]
)
v_kite = np.array(
    [results["kite_velocity_x"], results["kite_velocity_y"], results["kite_velocity_z"]]
)

r = np.linalg.norm(position, axis=0)

beta = np.arctan2(
    position[2], np.sqrt(position[0] ** 2 + position[1] ** 2)
)  # elevation
phi = np.arctan2(position[1], position[0])  # azimuth in wind window coordinates

# ── Kinematics from position/velocity vectors ─────────────────────────────────
r_norm = np.linalg.norm(position, axis=0)
r_hat = position / np.maximum(r_norm, 1e-6)  # (3, N) unit radial vector

v_r = np.sum(r_hat * v_kite, axis=0)  # radial (tether) speed, signed
v_tau_vec = v_kite - v_r * r_hat  # tangential velocity vector (3, N)

# Course angle χ and χ̇ using spherical wind-frame coordinates.
# χ is measured from the direction of increasing elevation β (toward zenith,
# perpendicular to wind direction x) within the tangential plane τ.
# χ = atan2(v_φ, v_β) where:
#   v_β = r·dβ/dt  — component toward zenith (χ=0 reference)
#   v_φ = r·cos(β)·dφ/dt — component clockwise in wind window (χ=90° reference)
dbeta_dt = np.gradient(beta, time)
dphi_dt = np.gradient(phi, time)

v_beta = r * dbeta_dt  # toward zenith
v_phi = r * np.cos(beta) * dphi_dt  # clockwise in wind window

v_tau = np.sqrt(v_beta**2 + v_phi**2)  # tangential speed (always >= 0)
chi = np.arctan2(v_phi, v_beta)  # course angle χ [rad]
chi = np.unwrap(chi)  # unwrap to avoid discontinuities for plotting and fitting

# χ̇ = d/dt[atan2(v_φ, v_β)] = (v_β·dv_φ/dt − v_φ·dv_β/dt) / v_τ²
# No angle unwrapping needed — computed directly from velocity components.
v_beta_dot = np.gradient(v_beta, time)
v_phi_dot = np.gradient(v_phi, time)

phi_dot = dphi_dt
vz = results["kite_velocity_z"].to_numpy()
vz_dot = np.gradient(vz, time)

chi_dot_raw = (vz * v_phi_dot - v_phi * vz_dot) / np.maximum(
    v_tau**2, 1e-6
)  # - phi_dot * np.sin(beta)
chi_dot_kinematic = np.convolve(chi_dot_raw, np.ones(21) / 21, mode="same")


# chi = flight_data["kite_course"].to_numpy()  # override χ with course angle from data
# chi = np.unwrap(chi)  # unwrap to avoid discontinuities for plotting and fitting
chi_dot_flight = None
yaw_dot_flight = None
turn_rate_label = TURN_RATE_SOURCE
selected_turn_rate = None
yaw_used_for_derivative = None
yaw_used_for_derivative_label = None
selected_turn_rate_source = None
selected_turn_rate_from_gradient = False
try:
    if "chi_dot" in flight_data.columns:
        chi_dot_flight = np.convolve(
            flight_data["chi_dot"].to_numpy(), np.ones(21) / 21, mode="same"
        )
    yaw_rate_col = f"kite_yaw_rate_{YAW_RATE_SENSOR_ID}"
    if yaw_rate_col in flight_data.columns and TURN_RATE_SOURCE == "caca":
        yaw_dot_flight = np.convolve(
            flight_data[yaw_rate_col].to_numpy(), np.ones(21) / 21, mode="same"
        )
        print(f"Using {yaw_rate_col} from flight data for turn rate fitting.")
    elif "kite_yaw_rate" in flight_data.columns:
        yaw_dot_flight = np.convolve(
            flight_data["kite_yaw_rate"].to_numpy(), np.ones(21) / 21, mode="same"
        )
        print(f"Using {yaw_rate_col} from flight data for turn rate fitting.")
    else:
        yaw_angle_col = f"kite_yaw_{YAW_RATE_SENSOR_ID}"
        if yaw_angle_col not in flight_data.columns:
            yaw_angle_col = "kite_yaw_0"
        yaw_used_for_derivative = flight_data[yaw_angle_col].to_numpy()
        yaw_used_for_derivative_label = yaw_angle_col
        yaw_dot_flight = np.convolve(
            np.gradient(yaw_used_for_derivative, time),
            np.ones(21) / 21,
            mode="same",
        )
    if TURN_RATE_SOURCE == "chi_dot":
        if chi_dot_flight is None:
            print("No chi_dot in flight data; using kinematics-derived χ̇ for fitting.")
            selected_turn_rate = chi_dot_kinematic
            selected_turn_rate_source = "chi_dot_kinematic"
        else:
            selected_turn_rate = chi_dot_flight
            selected_turn_rate_source = "chi_dot"
        turn_rate_label = "chi_dot"
    else:
        if yaw_dot_flight is None:
            yaw_rate_col = f"kite_yaw_rate_{YAW_RATE_SENSOR_ID}"
            print(
                f"No {yaw_rate_col} in flight data; using kinematics-derived yaw angle rate for fitting."
            )
            yaw_angle_col = f"kite_yaw_{YAW_RATE_SENSOR_ID}"
            if yaw_angle_col not in flight_data.columns:
                yaw_angle_col = "kite_yaw_0"
            yaw_used_for_derivative = flight_data[yaw_angle_col].to_numpy()
            yaw_used_for_derivative_label = yaw_angle_col
            selected_turn_rate = np.convolve(
                np.gradient(yaw_used_for_derivative, time),
                np.ones(21) / 21,
                mode="same",
            )
            turn_rate_label = f"{yaw_rate_col} (fallback)"
            selected_turn_rate_source = f"gradient({yaw_angle_col})"
            selected_turn_rate_from_gradient = True
        else:
            selected_turn_rate = yaw_dot_flight
            turn_rate_label = f"kite_yaw_rate_{YAW_RATE_SENSOR_ID}"
            if yaw_used_for_derivative is not None:
                selected_turn_rate_source = f"gradient({yaw_used_for_derivative_label})"
                selected_turn_rate_from_gradient = True
            else:
                yaw_rate_col = f"kite_yaw_rate_{YAW_RATE_SENSOR_ID}"
                if yaw_rate_col in flight_data.columns:
                    selected_turn_rate_source = yaw_rate_col
                elif "kite_yaw_rate" in flight_data.columns:
                    selected_turn_rate_source = "kite_yaw_rate"
                else:
                    selected_turn_rate_source = turn_rate_label
except KeyError:
    print(
        "No turn-rate measurement column found; using kinematics-derived χ̇ for fitting."
    )
    selected_turn_rate = chi_dot_kinematic
    selected_turn_rate_source = "chi_dot_kinematic"

chi_dot_meas = selected_turn_rate

# ── Turn rate law functions ────────────────────────────────────────────────────


def fit_simple(chi_dot, us, va, asym_mode="fit", k_asym_fixed=0.0):
    """
    Eq. (41): chi_dot_b = gk * va * (us - k_asymmetry)
    Reformulated as linear: chi_dot_b = c1*(va*us) + c2*(va)
    where c1 = gk and c2 = -gk*k_asymmetry, so k_asymmetry = -c2/c1
    Returns: (gk, k_asymmetry), chi_dot_est
    """
    term1 = va * us
    if asym_mode == "off":
        A = term1.reshape(-1, 1)
    elif asym_mode == "fixed":
        A = (va * (us - k_asym_fixed)).reshape(-1, 1)
    else:
        term2 = va
        A = np.vstack([term1, term2]).T
    valid = np.isfinite(A).all(axis=1) & np.isfinite(chi_dot)
    coeffs = calculate_weighted_squares_1d(chi_dot[valid], A[valid])
    if asym_mode == "off":
        gk = coeffs[0]
        k_asymmetry = 0.0
    elif asym_mode == "fixed":
        gk = coeffs[0]
        k_asymmetry = k_asym_fixed
    else:
        c1, c2 = coeffs[0], coeffs[1]
        gk = c1
        k_asymmetry = -c2 / c1 if abs(c1) > 1e-10 else 0.0

    return (gk, k_asymmetry), A @ coeffs


def fit_two_term(chi_dot, us, va, chi, beta, asym_mode="fit", k_asym_fixed=0.0):
    """
    Eq. (40): chi_dot_b = c1*(va*(us - k_asymmetry)) + c2*(sin(chi)*cos(beta)/va)
    Reformulated as linear: chi_dot_b = coeff1*(va*us) + coeff2*(va) + coeff3*(sin(chi)*cos(beta)/va)
    where coeff1 = c1, coeff2 = -c1*k_asymmetry, coeff3 = c2
    Returns: (c1, c2, k_asymmetry), chi_dot_est
    gk = c1,  and from c2 = -m*g/(m + 0.25*rho*S*b*K_rhat)
    """
    term1 = va * us
    term_gravity = np.sin(chi) * np.cos(beta) / np.maximum(va, 1e-6)
    if asym_mode == "off":
        A = np.vstack([term1, term_gravity]).T
    elif asym_mode == "fixed":
        term1_fixed = va * (us - k_asym_fixed)
        A = np.vstack([term1_fixed, term_gravity]).T
    else:
        term2 = va
        A = np.vstack([term1, term2, term_gravity]).T
    valid = np.isfinite(A).all(axis=1) & np.isfinite(chi_dot)
    coeffs = calculate_weighted_squares_1d(chi_dot[valid], A[valid])
    if asym_mode == "off":
        c1, c2 = coeffs[0], coeffs[1]
        k_asymmetry = 0.0
    elif asym_mode == "fixed":
        c1, c2 = coeffs[0], coeffs[1]
        k_asymmetry = k_asym_fixed
    else:
        coeff1, coeff2, coeff3 = coeffs[0], coeffs[1], coeffs[2]
        c1 = coeff1
        c2 = coeff3
        k_asymmetry = -coeff2 / coeff1 if abs(coeff1) > 1e-10 else 0.0
    return (c1, c2, k_asymmetry), A @ coeffs


def fit_full_rational(
    chi_dot,
    us,
    va,
    r,
    v_r,
    v_tau,
    chi,
    beta,
    mass=MASS,
    g=G,
    x0=None,
    asym_mode="fit",
    k_asym_fixed=0.0,
    gravity_mode="fit",
):
    """
    Eq. (38): chi_dot_b = -(k1*va^2*(us - k_asymmetry) + m*g*sin(chi)*cos(beta))
                                                    / (k4*(v_r/r) + k2*va)
    Nonlinear least-squares fit for (k1, k2, k3, k_asymmetry).
      k1  ~ 0.5*rho*S*K_us        (steering aerodynamic gain)
      k2  ~ 0.25*rho*S*b*K_rhat   (yaw-damping aerodynamic gain)
            k3  ~ gravity numerator gain
            k4  ~ radial-rate denominator gain
      k_asymmetry ~ asymmetry in steering input
    gravity_mode: "fit" keeps the gravity term coefficient k3 in the numerator;
        "off" fixes k3 = 0 and fits a gravity-free model.
    x0: initial guess for the physical coefficients.
    Returns: (k1, k2, k3, k_asymmetry), chi_dot_est
    """
    if x0 is None:
        if gravity_mode == "fit":
            x0 = [1, 8, 8, 0] if asym_mode == "fit" else [1, 8, 8, 8]
        else:
            x0 = [1, 8, 8, 0] if asym_mode == "fit" else [1, 8, 8]

    with_asym = asym_mode == "fit"
    with_gravity = gravity_mode == "fit"

    def to_internal_x0(x_phys):
        x_arr = np.asarray(x_phys, dtype=float).reshape(-1)

        def get(index, default=0.0):
            return x_arr[index] if index < x_arr.size else default

        k1_0 = get(0, 1.0)
        k2_0 = get(1, 8.0)
        k3_0 = get(2, 0.0) if with_gravity else 0.0

        if with_gravity and with_asym:
            k4_0 = get(3, k2_0)
            k_asym_0 = get(4, 0.0)
        elif with_gravity and not with_asym:
            k4_0 = get(3, k2_0)
            k_asym_0 = 0.0
        elif not with_gravity and with_asym:
            if x_arr.size >= 5:
                k4_0 = get(3, k2_0)
                k_asym_0 = get(4, 0.0)
            else:
                k4_0 = get(2, k2_0)
                k_asym_0 = get(3, 0.0)
        else:
            k4_0 = get(2, k2_0)
            k_asym_0 = 0.0

        if with_gravity and with_asym:
            return np.array([k1_0, k2_0, k3_0, k4_0, k_asym_0], dtype=float)
        if with_gravity and not with_asym:
            return np.array([k1_0, k2_0, k3_0, k4_0], dtype=float)
        if not with_gravity and with_asym:
            return np.array([k1_0, k2_0, k4_0, k_asym_0], dtype=float)
        return np.array([k1_0, k2_0, k4_0], dtype=float)

    def predict(k1, k2, k3, k4, k_asymmetry):
        gravity = k3 * mass * g * np.sin(chi) * np.cos(beta) if with_gravity else 0.0
        num = k1 * va**2 * (us - k_asymmetry) + gravity
        radial_rate = v_tau * mass  # * v_r / np.maximum(r, 1e-6)
        den = np.maximum(k4 * radial_rate + k2 * va, 1e-6)
        return -num / den

    def residuals(x):
        if with_gravity and with_asym:
            k1, k2, k3, k4, k_asymmetry = x
        elif with_gravity and not with_asym:
            k1, k2, k3, k4 = x
            k_asymmetry = k_asym_fixed
        elif not with_gravity and with_asym:
            k1, k2, k4, k_asymmetry = x
            k3 = 0.0
        else:
            k1, k2, k4 = x
            k3 = 0.0
            k_asymmetry = k_asym_fixed
        return predict(k1, k2, k3, k4, k_asymmetry) - chi_dot

    valid = (
        np.isfinite(chi_dot)
        & np.isfinite(va)
        & np.isfinite(v_tau)
        & np.isfinite(us)
        & np.isfinite(chi)
        & np.isfinite(beta)
    )
    if with_gravity and with_asym:
        bounds = ([-1e2, -1e2, -1e2, -1e2, -0.1], [1e2, 1e2, 1e2, 1e2, 0.1])
    elif with_gravity and not with_asym:
        bounds = ([-1e2, -1e2, -1e2, -1e2], [1e2, 1e2, 1e2, 1e2])
    elif not with_gravity and with_asym:
        bounds = ([-1e2, -1e2, -1e2, -0.1], [1e2, 1e2, 1e2, 0.1])
    else:
        bounds = ([-1e2, -1e2, -1e2], [1e2, 1e2, 1e2])
    res = least_squares(
        lambda x: residuals(x)[valid],
        x0=to_internal_x0(x0),
        bounds=bounds,
    )
    if with_gravity and with_asym:
        k1, k2, k3, k4, k_asymmetry = res.x
    elif with_gravity and not with_asym:
        k1, k2, k3, k4 = res.x
        k_asymmetry = k_asym_fixed
    elif not with_gravity and with_asym:
        k1, k2, k4, k_asymmetry = res.x
        k3 = 0.0
    else:
        k1, k2, k4 = res.x
        k3 = 0.0
        k_asymmetry = k_asym_fixed
    return (k1, k2, k3, k4, k_asymmetry), predict(k1, k2, k3, k4, k_asymmetry)


def calculate_weighted_squares_1d(y, A):
    """Thin wrapper: unweighted least squares."""
    return calculate_weighted_least_squares(y, A)


def rmse(y_true, y_pred):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    return np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2))


# ── Per-phase fitting ─────────────────────────────────────────────────────────

phase_col = (
    "flight_phase_index" if "flight_phase_index" in flight_data.columns else "cycle"
)
phases = [p for p in flight_data[phase_col].unique() if p in PHASES_TO_FIT]

print(f"\nFitting turn rate laws per {phase_col}")
print("=" * 110)
print(
    f"{'Phase':<22} {'Model':<16} {'k1/gk/c1':>10} {'k2/c2':>10} {'k3':>10} {'k4':>10} {'k_asym':>10} {'RMSE':>8}"
)
print("-" * 110)

phase_results = {}
x0_full = None  # warm-start: updated with each cycle's fitted coefficients
for phase in sorted(phases, key=str):
    mask = (flight_data[phase_col] == phase).to_numpy()
    n = mask.sum()
    if n < 50:
        continue

    fd = chi_dot_meas[mask]
    u = us[mask]
    v = va[mask]
    vt = v_tau[mask]
    c = chi[mask]
    b = beta[mask]

    # Eq. (41) simple
    simple_mode = (
        ASYM_MODE_SIMPLE if ASYM_MODE_SIMPLE in ["fit", "off", "fixed"] else "fit"
    )
    (gk, k_asym_41), est_simple = fit_simple(
        fd,
        u,
        v,
        asym_mode=simple_mode,
        k_asym_fixed=K_ASYM_FIXED,
    )
    r_simple = rmse(fd, est_simple)

    # Eq. (41) simple symmetric reference (no asymmetry term)
    (gk_sym, _), est_simple_sym = fit_simple(
        fd,
        u,
        v,
        asym_mode="off",
        k_asym_fixed=0.0,
    )
    r_simple_sym = rmse(fd, est_simple_sym)

    # Eq. (40) two-term
    if ASYM_MODE_TWO_TERM == "from_simple":
        two_mode = "fixed"
        two_k_asym = k_asym_41
    elif ASYM_MODE_TWO_TERM == "off":
        two_mode = "off"
        two_k_asym = 0.0
    elif ASYM_MODE_TWO_TERM == "fixed":
        two_mode = "fixed"
        two_k_asym = K_ASYM_FIXED
    else:
        two_mode = "fit"
        two_k_asym = K_ASYM_FIXED
    (c1, c2, k_asym_40), est_two = fit_two_term(
        fd,
        u,
        v,
        c,
        b,
        asym_mode=two_mode,
        k_asym_fixed=two_k_asym,
    )
    r_two = rmse(fd, est_two)

    # Eq. (38) gravity-free pass used to warm-start the full model
    if ASYM_MODE_FULL == "from_simple":
        full_ng_mode = "fixed"
        full_ng_k_asym = k_asym_41
    elif ASYM_MODE_FULL == "from_two_term":
        full_ng_mode = "fixed"
        full_ng_k_asym = k_asym_40
    elif ASYM_MODE_FULL == "off":
        full_ng_mode = "fixed"
        full_ng_k_asym = 0.0
    elif ASYM_MODE_FULL == "fixed":
        full_ng_mode = "fixed"
        full_ng_k_asym = K_ASYM_FIXED
    else:
        full_ng_mode = "fit"
        full_ng_k_asym = K_ASYM_FIXED

    (k1_ng, k2_ng, k3_ng, k4_ng, k_asym_ng), est_full_ng = fit_full_rational(
        fd,
        u,
        v,
        r[mask],
        v_r[mask],
        vt,
        c,
        b,
        x0=x0_full,
        asym_mode=full_ng_mode,
        k_asym_fixed=full_ng_k_asym,
        gravity_mode="off",
    )

    x0_full = [k1_ng, k2_ng, 5, k4_ng, k_asym_ng]

    # Eq. (38) full rational — warm-started from the gravity-free solution
    if ASYM_MODE_FULL == "from_simple":
        full_mode = "fixed"
        full_k_asym = k_asym_41
    elif ASYM_MODE_FULL == "from_two_term":
        full_mode = "fixed"
        full_k_asym = k_asym_40
    elif ASYM_MODE_FULL == "off":
        full_mode = "fixed"
        full_k_asym = 0.0
    elif ASYM_MODE_FULL == "fixed":
        full_mode = "fixed"
        full_k_asym = K_ASYM_FIXED
    else:
        full_mode = "fit"
        full_k_asym = K_ASYM_FIXED

    (k1, k2, k3, k4, k_asym_38), est_full = fit_full_rational(
        fd,
        u,
        v,
        r[mask],
        v_r[mask],
        vt,
        c,
        b,
        x0=x0_full,
        asym_mode=full_mode,
        k_asym_fixed=full_k_asym,
        gravity_mode="fit",
    )
    norm_coeffs = np.sqrt(k1**2 + k2**2 + k3**2)
    if norm_coeffs < 1e-2:
        k1 = k1 / norm_coeffs
        k2 = k2 / norm_coeffs
        k3 = k3 / norm_coeffs
    r_full = rmse(fd, est_full)

    phase_results[phase] = {
        "simple": {
            "gk": gk,
            "k_asymmetry": k_asym_41,
            "RMSE": r_simple,
            "est": est_simple,
            "meas": fd,
        },
        "simple_symmetric": {
            "gk": gk_sym,
            "k_asymmetry": 0.0,
            "RMSE": r_simple_sym,
            "est": est_simple_sym,
        },
        "two_term": {
            "c1": c1,
            "c2": c2,
            "k_asymmetry": k_asym_40,
            "RMSE": r_two,
            "est": est_two,
        },
        "full": {
            "k1": k1,
            "k2": k2,
            "k3": k3,
            "k4": k4,
            "k_asymmetry": k_asym_38,
            "RMSE": r_full,
            "est": est_full,
        },
        "full_no_gravity": {
            "k1": k1_ng,
            "k2": k2_ng,
            "k3": k3_ng,
            "k4": k4_ng,
            "k_asymmetry": k_asym_ng,
            "RMSE": rmse(fd, est_full_ng),
            "est": est_full_ng,
        },
        "yaw_dot": yaw_rate_ if yaw_dot_flight is None else yaw_dot_flight[mask],
        "chi_dot": None if chi_dot_flight is None else chi_dot_flight[mask],
        "signals": {"u": u, "v": v, "vt": vt, "c": c, "b": b},
        "us_va": u * v,
        "time": time[mask],
        "n": n,
    }

    print(
        f"{str(phase):<22} {'Eq.(41)':<12} {gk:>10.4f} {'—':>10} {'—':>10} {'—':>10} {k_asym_41:>10.4f} {r_simple:>8.4f}"
    )
    print(
        f"{'':22} {'Eq.(41) sym':<12} {gk_sym:>10.4f} {'—':>10} {'—':>10} {'—':>10} {0.0:>10.4f} {r_simple_sym:>8.4f}"
    )
    print(
        f"{'':22} {'two_fit':<16} {c1:>10.4f} {c2:>10.4f} {'—':>10} {'—':>10} {k_asym_40:>10.4f} {r_two:>8.4f}"
    )
    print(
        f"{'':22} {'full':<16} {k1:>10.4f} {k2:>10.4f} {k3:>10.4f} {k4:>10.4f} {k_asym_38:>10.4f} {r_full:>8.4f}"
    )
    print(
        f"{'':22} {'Eq.(38) NG':<12} {k1_ng:>10.4f} {k2_ng:>10.4f} {k3_ng:>10.4f} {k4_ng:>10.4f} {k_asym_ng:>10.4f} {rmse(fd, est_full_ng):>8.4f}"
    )
    print()

print("=" * 110)


# ── Plots ─────────────────────────────────────────────────────────────────────

if not phase_results:
    print("No phases with enough data to plot.")
else:
    date_str = "2025-10-09"
    plot_turn_rate_fit_results(
        phase_results=phase_results,
        time=time,
        chi_dot_meas=chi_dot_meas,
        flight_data=flight_data,
        chi=chi,
        beta=beta,
        v_tau=v_tau,
        us=us,
        va=va,
        yaw_dot_flight=yaw_dot_flight,
        chi_dot_flight=chi_dot_flight,
        turn_rate_source=TURN_RATE_SOURCE,
        turn_rate_label=turn_rate_label,
        selected_turn_rate_from_gradient=selected_turn_rate_from_gradient,
        yaw_used_for_derivative=yaw_used_for_derivative,
        yaw_used_for_derivative_label=yaw_used_for_derivative_label,
        selected_turn_rate_source=selected_turn_rate_source,
        plot_cycle=PLOT_CYCLE,
        phase_name=PHASE_NAME,
        palette=PALETTE,
        plot_dir=Path("results") / "plots_paper" / date_str,
        mass=MASS,
        g=G,
    )
