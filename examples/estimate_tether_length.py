#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from awes_ekf.utils import estimate_tether_length


def main():
    p = argparse.ArgumentParser(description="Estimate tether length from a CSV log.")
    p.add_argument("csv", type=str, help="Path to CSV file")
    p.add_argument(
        "--method",
        choices=["distance", "enu", "elev"],
        default="distance",
        help="Base method: kite_distance | sqrt(E,N,H) | height/sin(elev)",
    )
    p.add_argument(
        "--tether-mass-per-m",
        type=float,
        default=None,
        help="Optional tether mass per meter (kg/m) for small-sag correction",
    )
    args = p.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")

    df = pd.read_csv(path)

    # Try columns in your dataset
    cols = df.columns.str.strip()

    # Map available columns
    kd = df["kite_distance"].to_numpy() if "kite_distance" in cols else None
    e = df["kite_pos_east"].to_numpy() if "kite_pos_east" in cols else None
    n = df["kite_pos_north"].to_numpy() if "kite_pos_north" in cols else None
    h = df["kite_height"].to_numpy() if "kite_height" in cols else None
    el = df["kite_elevation"].to_numpy() if "kite_elevation" in cols else None
    gf = df["ground_tether_force"].to_numpy() if "ground_tether_force" in cols else None

    if args.method == "distance" and kd is None:
        print("kite_distance not found; falling back to ENU if available...")
        args.method = "enu"

    if args.method == "enu" and (e is None or n is None or h is None):
        print("ENU columns missing; falling back to elevation if available...")
        args.method = "elev"

    if args.method == "elev" and (h is None or el is None):
        raise SystemExit(
            "Need kite_height and kite_elevation for elevation-based estimate, but they are missing."
        )

    # Select inputs per method
    if args.method == "distance":
        L = estimate_tether_length(
            kite_distance=kd,
            ground_tether_force=gf,
            tether_mass_per_m=args.tether_mass_per_m,
        )
    elif args.method == "enu":
        L = estimate_tether_length(
            pos_east=e,
            pos_north=n,
            height=h,
            ground_tether_force=gf,
            tether_mass_per_m=args.tether_mass_per_m,
        )
    else:  # elev
        L = estimate_tether_length(
            height=h,
            elevation=el,
            ground_tether_force=gf,
            tether_mass_per_m=args.tether_mass_per_m,
        )

    # Report summary
    L = np.asarray(L)
    valid = np.isfinite(L)
    print(
        f"Estimated tether length: count={valid.sum()}, mean={np.nanmean(L):.2f} m, "
        f"median={np.nanmedian(L):.2f} m, min={np.nanmin(L):.2f} m, max={np.nanmax(L):.2f} m"
    )

    # If ground_tether_length is present, show a quick linear calibration
    if "ground_tether_length" in cols:
        gl = df["ground_tether_length"].to_numpy()
        mask = np.isfinite(L) & np.isfinite(gl)
        if mask.sum() >= 2:
            x = L[mask]
            y = gl[mask]
            A = np.vstack([x, np.ones_like(x)]).T
            coeffs, *_ = np.linalg.lstsq(A, y, rcond=None)
            a, b = coeffs
            yhat = a * x + b
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2) if mask.sum() > 1 else np.nan
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            med_off = np.median(y - x)
            print(
                f"Calibration to ground_tether_length: y ≈ {a:.4f} * L + {b:.2f}, R^2={r2:.3f}, "
                f"median_offset={med_off:.2f} m"
            )
        else:
            print(
                "Not enough overlapping samples to calibrate against ground_tether_length."
            )


if __name__ == "__main__":
    main()
