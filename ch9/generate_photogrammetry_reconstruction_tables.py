"""Generate LaTeX tables for the 2025 photogrammetry reconstruction cases.

Telemetry is sampled from ``results/v3/v3_2025-10-09.h5`` at times obtained
from the table-consistent video-frame mapping. Geometric span and mean
point-position error are photogrammetry outputs and are therefore recorded as
case metadata rather than recomputed from the EKF.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from awes_ekf.load_data.read_data import read_results

DEFAULT_VIDEO_FPS = 30.0
DEFAULT_ANCHOR_FRAME = 7182
DEFAULT_ANCHOR_EKF_TIME_S = 67.63333333333334
DEFAULT_VA_FILTER_ALPHA = 0.95


@dataclass(frozen=True)
class ReconstructionCase:
    case: str
    condition: str
    frame: int
    span_m: float
    mean_position_error_cm: float


CASES = (
    ReconstructionCase("PS_1", "Powered straight reel-out", 7182, 7.30, 2.91),
    ReconstructionCase("PT_1", "Powered right-turn reel-out", 7362, 7.25, 2.94),
    ReconstructionCase("PS_2", "Powered straight reel-out", 7721, np.nan, np.nan),
    ReconstructionCase("PT_2", "Powered left-turn reel-out", 7811, np.nan, np.nan),
    ReconstructionCase("PS_3", "Powered straight reel-out", 17372, 7.32, 5.95),
    ReconstructionCase("DR_1", "Depowered reel-in", 17611, 7.12, 4.14),
    ReconstructionCase("DR_2", "Depowered reel-in", 17701, np.nan, np.nan),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the three photogrammetry reconstruction tables."
    )
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--anchor-frame", type=int, default=DEFAULT_ANCHOR_FRAME)
    parser.add_argument(
        "--anchor-ekf-time-s",
        type=float,
        default=DEFAULT_ANCHOR_EKF_TIME_S,
    )
    parser.add_argument(
        "--va-filter-alpha",
        type=float,
        default=DEFAULT_VA_FILTER_ALPHA,
        help="Forward-backward exponential-filter coefficient for apparent wind speed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    return parser


def frame_to_ekf_time(
    frame: int,
    *,
    video_fps: float,
    anchor_frame: int,
    anchor_ekf_time_s: float,
) -> float:
    return anchor_ekf_time_s + (float(frame) - anchor_frame) / video_fps


def _ema_pass(values: np.ndarray, alpha: float) -> np.ndarray:
    output = np.full_like(values, np.nan, dtype=float)
    finite = np.flatnonzero(np.isfinite(values))
    if finite.size == 0:
        return output
    first = int(finite[0])
    output[first] = values[first]
    for index in range(first + 1, len(values)):
        value = values[index]
        output[index] = (
            alpha * output[index - 1] + (1.0 - alpha) * value
            if np.isfinite(value)
            else output[index - 1]
        )
    return output


def zero_phase_exponential_filter(
    series: pd.Series,
    alpha: float,
) -> np.ndarray:
    values = series.to_numpy(dtype=float)
    forward = _ema_pass(values, alpha)
    return _ema_pass(forward[::-1], alpha)[::-1]


def load_case_data(
    *,
    video_fps: float,
    anchor_frame: int,
    anchor_ekf_time_s: float,
    va_filter_alpha: float,
) -> pd.DataFrame:
    results, flight_data, _ = read_results("2025", "10", "09", "v3")
    if (
        "tether_force_kite" in results.columns
        and "tether_force_kite" not in flight_data.columns
    ):
        flight_data = flight_data.copy()
        flight_data["tether_force_kite"] = results["tether_force_kite"].to_numpy(
            dtype=float
        )
    required = {
        "time",
        "kcu_actual_depower",
        "kcu_actual_steering",
        "kite_apparent_windspeed",
        "bridle_angle_of_attack",
        "bridle_sideslip_angle",
        "tether_force_kite",
        "powered",
        "turn_straight",
        "tether_reelout_speed",
    }
    missing = sorted(required - set(flight_data.columns))
    if missing:
        raise ValueError(
            "The regenerated 2025 HDF5 is missing required fields: "
            + ", ".join(missing)
        )

    flight_data["kite_apparent_windspeed_filtered"] = zero_phase_exponential_filter(
        flight_data["kite_apparent_windspeed"],
        va_filter_alpha,
    )

    times = flight_data["time"].to_numpy(dtype=float)
    time_min = float(np.nanmin(times))
    time_max = float(np.nanmax(times))
    rows = []
    for case in CASES:
        target_time = frame_to_ekf_time(
            case.frame,
            video_fps=video_fps,
            anchor_frame=anchor_frame,
            anchor_ekf_time_s=anchor_ekf_time_s,
        )
        if not time_min <= target_time <= time_max:
            raise ValueError(
                f"{case.case} target time {target_time:.3f} s is outside the "
                f"HDF5 interval [{time_min:.3f}, {time_max:.3f}] s."
            )
        index = int(np.argmin(np.abs(times - target_time)))
        sample = flight_data.iloc[index]
        rows.append(
            {
                "case": case.case,
                "condition": case.condition,
                "frame": case.frame,
                "target_time_s": target_time,
                "sample_time_s": float(sample["time"]),
                "time_error_s": float(sample["time"] - target_time),
                "u_dp": float(sample["kcu_actual_depower"]) / 100.0,
                "u_s": float(sample["kcu_actual_steering"]) / 100.0,
                "span_m": case.span_m,
                "va_ms": float(sample["kite_apparent_windspeed_filtered"]),
                "alpha_fl_deg": float(sample["bridle_angle_of_attack"]),
                "beta_s_deg": float(sample["bridle_sideslip_angle"]),
                "force_kcu_n": float(sample["tether_force_kite"]),
                "mean_position_error_cm": case.mean_position_error_cm,
                "powered": str(sample["powered"]),
                "turn_straight": str(sample["turn_straight"]),
                "reelout_speed_ms": float(sample["tether_reelout_speed"]),
            }
        )
    return pd.DataFrame(rows).set_index("case", drop=False)


def reconstruction_cases_table(
    case_data: pd.DataFrame,
    *,
    video_fps: float,
    anchor_frame: int,
    anchor_ekf_time_s: float,
) -> str:
    rows = "\n".join(
        f"${row.case}$ & {row.condition} & {int(row.frame)} & "
        f"{row.target_time_s:.1f} \\\\"
        for row in case_data.itertuples()
    )
    return rf"""%
\begin{{table}}[htp]
    \centering
    \caption{{
    Photogrammetry reconstruction cases used in the in-flight shape analysis.
    The time $t$ is obtained from the merged stereo-video frame number using a
    video rate of ${video_fps:.0f}~\unit{{Hz}}$ and synchronization frame
    {anchor_frame} at $t={anchor_ekf_time_s:.1f}~\unit{{s}}$.
    }}
    \label{{9_2:tab:photogrammetry_reconstruction_cases}}
    \begin{{tabular}}{{llrr}}
        \toprule
        Case & Flight condition & Frame & $t$ $(\unit{{s}})$ \\
        \midrule
{rows}
        \bottomrule
    \end{{tabular}}
\end{{table}}
%
"""


def state_comparison_table(
    case_data: pd.DataFrame,
    *,
    case_names: tuple[str, str],
    row_names: tuple[str, str],
    caption: str,
    label: str,
) -> str:
    rows = []
    for case_name, row_name in zip(case_names, row_names):
        row = case_data.loc[case_name]
        rows.append(
            f"{row_name:<15} & {row.u_dp:.4f} & {row.u_s:.4f} & "
            f"{row.span_m:.2f} & {row.va_ms:.2f} & "
            f"{row.alpha_fl_deg:.1f} & {row.beta_s_deg:.0f} & "
            f"{row.force_kcu_n:.0f} & {row.mean_position_error_cm:.2f} \\\\"
        )
    body = "\n".join(rows)
    return rf"""%
\begin{{table}}[h!]
\centering
\caption{{
{caption}
}}
\label{{{label}}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lcccccccc}} \hline
State & $u_\mathrm{{dp}}$ ($\unit{{-}}$) & $u_\mathrm{{s}}$ ($\unit{{-}}$) & $b$ ($\unit{{m}}$) & $v_\mathrm{{a}}$ ($\unit{{m s^{{-1}}}}$) & $\alpha_\mathrm{{fl}}$ ($\unit{{\degree}}$) & $\beta_\mathrm{{s}}$ ($\unit{{\degree}}$) & $F_\mathrm{{T,KCU}}$ ($\unit{{N}}$) & $\epsilon_\mathrm{{p,avg}}$ ($\unit{{cm}}$) \\ \hline
{body}
\hline
\end{{tabular}}%
}}
\end{{table}}
%
"""


def main() -> None:
    args = build_parser().parse_args()
    if args.video_fps <= 0.0:
        raise ValueError("--video-fps must be positive.")
    if not 0.0 <= args.va_filter_alpha < 1.0:
        raise ValueError("--va-filter-alpha must satisfy 0 <= alpha < 1.")

    case_data = load_case_data(
        video_fps=args.video_fps,
        anchor_frame=args.anchor_frame,
        anchor_ekf_time_s=args.anchor_ekf_time_s,
        va_filter_alpha=args.va_filter_alpha,
    )

    tables = {
        "photogrammetry_reconstruction_cases.tex": reconstruction_cases_table(
            case_data,
            video_fps=args.video_fps,
            anchor_frame=args.anchor_frame,
            anchor_ekf_time_s=args.anchor_ekf_time_s,
        ),
        "depowering_states_and_changes.tex": state_comparison_table(
            case_data,
            case_names=("PS_3", "DR_1"),
            row_names=("Powered", "Depowered"),
            caption=(
                "Inputs and -- photogrammetry and EKF-derived -- estimated "
                "outputs for powered straight flight $PS_3$ and depowered "
                "straight flight $DR_1$, see "
                "\\autoref{fig:powered_depowered_camera_frame}, including the "
                "average point-position error $\\epsilon_\\text{p,avg}$ and "
                "front-line angle $\\alpha_\\mathrm{fl}$."
            ),
            label="tab:depowering_states_and_changes",
        ),
        "turning_states_and_changes.tex": state_comparison_table(
            case_data,
            case_names=("PS_1", "PT_1"),
            row_names=("Straight flight", "Right turn"),
            caption=(
                "Inputs and -- photogrammetry and EKF-derived -- estimated "
                "outputs for straight flight $PS_1$ and turning flight "
                "$PT_1$, see \\autoref{fig:straight_vs_turning_camera_frame}."
            ),
            label="tab:turning_states_and_changes",
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in tables.items():
        output_path = args.output_dir / filename
        output_path.write_text(content, encoding="utf-8")
        print(f"Saved {output_path}")

    print(
        case_data[
            [
                "case",
                "frame",
                "target_time_s",
                "sample_time_s",
                "powered",
                "turn_straight",
                "reelout_speed_ms",
                "u_dp",
                "u_s",
                "va_ms",
                "alpha_fl_deg",
                "beta_s_deg",
                "force_kcu_n",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
