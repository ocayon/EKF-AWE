import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.load_data.read_data import read_results
from awes_ekf.utils import calculate_weighted_least_squares, calculate_turn_rate_law


plt.close("all")


def identify_course_rate_least_squares(
    flight_data,
    results,
    smooth_window=21,
    v_tau_epsilon=1e-6,
):
    """Estimate course rate coefficients k1*, k2* with weighted least squares."""

    time = flight_data["time"].to_numpy()
    course = np.unwrap(flight_data["kite_course"].to_numpy())
    course_rate = np.gradient(course, time)

    if smooth_window and smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        course_rate = np.convolve(course_rate, kernel, mode="same")

    us = flight_data["kcu_actual_steering"].to_numpy() / 100
    va = results["kite_apparent_windspeed"].to_numpy()
    v_tau = np.sqrt(
        results["kite_velocity_x"] ** 2
        + results["kite_velocity_y"] ** 2
        + results["kite_velocity_z"] ** 2
    ).to_numpy()
    v_tau = np.maximum(v_tau, v_tau_epsilon)

    psi = (
        flight_data["kite_yaw_0"].to_numpy() if "kite_yaw_0" in flight_data else course
    )
    beta = (
        results["kite_elevation"].to_numpy()
        if "kite_elevation" in results
        else flight_data["kite_elevation"].to_numpy()
    )
    varphi = (
        results["tether_roll"].to_numpy()
        if "tether_roll" in results
        else np.zeros_like(beta)
    )

    term1 = va**2 * us / v_tau
    term2 = np.sin(psi) * np.cos(beta + varphi) / v_tau
    A = np.vstack([term1, term2]).T

    valid_mask = np.isfinite(A).all(axis=1) & np.isfinite(course_rate)
    course_rate = course_rate[valid_mask]
    A = A[valid_mask]
    time = time[valid_mask]
    us = us[valid_mask]
    va = va[valid_mask]

    coeffs = calculate_weighted_least_squares(course_rate, A)
    course_rate_est = A @ coeffs

    mse = np.mean((course_rate - course_rate_est) ** 2)
    rmse = np.sqrt(mse)

    return {
        "coeffs": coeffs,
        "course_rate_meas": course_rate,
        "course_rate_est": course_rate_est,
        "time": time,
        "us_va": us * va,
        "us_va2": (us * va) ** 2,
        "mse": mse,
        "rmse": rmse,
    }


def identify_yaw_rate_simple(flight_data, results, model="simple"):
    yaw_rate_meas = flight_data["kite_yaw_rate"]
    yaw_rate_est, coeffs = calculate_turn_rate_law(
        results, flight_data, model=model, steering_offset=False
    )
    mse = np.mean((yaw_rate_est - yaw_rate_meas) ** 2)
    rmse = np.sqrt(mse)

    us = flight_data["kcu_actual_steering"].to_numpy() / 100
    va = results["kite_apparent_windspeed"].to_numpy()

    return {
        "coeffs": coeffs,
        "yaw_rate_meas": yaw_rate_meas,
        "yaw_rate_est": yaw_rate_est,
        "time": flight_data["time"].to_numpy(),
        "us_va": us * va,
        "mse": mse,
        "rmse": rmse,
    }


import numpy as np
from scipy import signal


def preprocess(x, fs, detrend=True, band=None):
    """
    x: 1D array
    fs: sampling frequency [Hz]
    detrend: remove mean/linear trend
    band: tuple (f_low, f_high) in Hz for bandpass; use None to skip filtering
    """
    x = np.asarray(x).astype(float)

    # Handle NaNs: simple linear interpolation (better than dropping for spectra)
    if np.any(~np.isfinite(x)):
        t = np.arange(len(x)) / fs
        mask = np.isfinite(x)
        x = np.interp(t, t[mask], x[mask])

    if detrend:
        x = signal.detrend(x, type="linear")

    if band is not None:
        f_low, f_high = band
        if f_low <= 0:
            # lowpass
            b, a = signal.butter(4, f_high / (0.5 * fs), btype="lowpass")
        elif f_high >= 0.5 * fs:
            # highpass
            b, a = signal.butter(4, f_low / (0.5 * fs), btype="highpass")
        else:
            b, a = signal.butter(
                4, [f_low / (0.5 * fs), f_high / (0.5 * fs)], btype="bandpass"
            )
        x = signal.filtfilt(b, a, x)

    return x


def welch_psd(x, fs, nperseg, noverlap=None):
    f, Pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=False)
    return f, Pxx


def coherence_xy(x, y, fs, nperseg, noverlap=None):
    f, Cxy = signal.coherence(
        x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, detrend=False
    )
    return f, Cxy


def dominant_frequency_from_psd(f, Pxx, fmin=0.01, fmax=None):
    """
    Returns dominant frequency peak of PSD in [fmin, fmax].
    """
    if fmax is None:
        fmax = f[-1]
    mask = (f >= fmin) & (f <= fmax)
    if not np.any(mask):
        return np.nan
    i = np.argmax(Pxx[mask])
    return f[mask][i]


def cutoff_frequency_from_coherence(f, Cxy, threshold=0.6, fmin=0.01, fmax=None):
    """
    Define f_c as the *highest* frequency within [fmin, fmax] where coherence >= threshold.
    """
    if fmax is None:
        fmax = f[-1]
    mask = (f >= fmin) & (f <= fmax) & np.isfinite(Cxy)
    if not np.any(mask):
        return np.nan

    f_sel = f[mask]
    C_sel = Cxy[mask]

    idx = np.where(C_sel >= threshold)[0]
    if len(idx) == 0:
        return np.nan
    return f_sel[idx[-1]]


def characterize_timescales(
    phi,
    alpha,
    Va,
    fs,
    cycle_band=(0.02, 2.0),
    coherence_threshold=0.6,
    cycles_per_window=4,
):
    """
    phi: azimuth [rad or deg] timeseries
    alpha: AoA timeseries
    Va: apparent wind speed timeseries
    fs: sampling Hz

    Returns dict with f_slow, T_slow, f_fast, T_fast, eta, plus spectra arrays.
    """
    # --- Preprocess
    phi_p = preprocess(phi, fs, detrend=True, band=cycle_band)
    alpha_p = preprocess(alpha, fs, detrend=True, band=None)
    Va_p = preprocess(Va, fs, detrend=True, band=None)

    # --- Welch window: choose based on expected cycle frequency from phi
    # First rough PSD with a generic window to estimate cycle peak
    nperseg0 = int(min(len(phi_p), max(256, fs * 10)))  # ~10s or >=256 samples
    nperseg0 = max(128, nperseg0)
    f0, Pphi0 = welch_psd(phi_p, fs, nperseg=nperseg0, noverlap=nperseg0 // 2)

    f_slow = dominant_frequency_from_psd(
        f0, Pphi0, fmin=cycle_band[0], fmax=cycle_band[1]
    )
    T_slow = 1.0 / f_slow if np.isfinite(f_slow) and f_slow > 0 else np.nan

    # Now choose a window length that contains several cycles (for stable coherence)
    if np.isfinite(T_slow):
        win_sec = cycles_per_window * T_slow
        nperseg = int(win_sec * fs)
    else:
        nperseg = nperseg0

    # Ensure sensible bounds
    nperseg = int(np.clip(nperseg, 256, len(phi_p)))
    noverlap = nperseg // 2

    # --- Final PSD and coherence with chosen window
    f_phi, Pphi = welch_psd(phi_p, fs, nperseg=nperseg, noverlap=noverlap)
    f_slow = dominant_frequency_from_psd(
        f_phi, Pphi, fmin=cycle_band[0], fmax=cycle_band[1]
    )
    T_slow = 1.0 / f_slow if np.isfinite(f_slow) and f_slow > 0 else np.nan

    # Coherence AoA <-> Va
    f_c, C_aVa = coherence_xy(alpha_p, Va_p, fs, nperseg=nperseg, noverlap=noverlap)
    f_fast = cutoff_frequency_from_coherence(
        f_c,
        C_aVa,
        threshold=coherence_threshold,
        fmin=cycle_band[0],
        fmax=0.5 * fs * 0.95,
    )
    T_fast = (
        1.0 / (2 * np.pi * f_fast) if np.isfinite(f_fast) and f_fast > 0 else np.nan
    )

    eta = (T_fast / T_slow) if np.isfinite(T_fast) and np.isfinite(T_slow) else np.nan

    return {
        "f_slow_hz": f_slow,
        "T_slow_s": T_slow,
        "f_fast_hz": f_fast,
        "T_fast_s": T_fast,
        "eta": eta,
        "welch": {"f_phi": f_phi, "P_phi": Pphi, "f_coh": f_c, "C_alpha_Va": C_aVa},
        "params": {
            "nperseg": nperseg,
            "noverlap": noverlap,
            "coh_thresh": coherence_threshold,
        },
    }


cut = 20000
results, flight_data, _ = read_results(
    "2019",
    "10",
    "08",
    "v3",
    addition="_va",
)

results = results[cut:-cut]
flight_data = flight_data[cut:-cut]

# ---- Example usage ----
fs = 10.0  # Hz
phi = flight_data["kite_azimuth"].to_numpy()
alpha = flight_data["kite_yaw_rate_0"].to_numpy()
Va = flight_data["kite_apparent_windspeed"].to_numpy()
results = characterize_timescales(phi, alpha, Va, fs, coherence_threshold=0.4)
print(
    results["f_slow_hz"],
    results["T_slow_s"],
    results["f_fast_hz"],
    results["T_fast_s"],
    results["eta"],
)
