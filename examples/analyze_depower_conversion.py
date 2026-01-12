"""
Analyze and validate the depower conversion between 2019 and 2025 systems.

This script investigates:
1. The actual ranges of depower values in both datasets
2. Whether the conversion makes physical sense
3. The validity of the empirical calibration approach
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def convert_2019_depower_to_2025_updata(
    x19_depower,
    x19_pow=22.68,
    x19_dep=0.02,
    ld_max=1.7,
    ld_affine_offset=0.2,
    ld_affine_scale=5.0,
):
    """
    EMPIRICAL CALIBRATION between 2019 and 2025 depower systems.

    WARNING: This is NOT a physics-based conversion. It's a transfer function
    between two potentially different control systems.

    Assumptions (to be validated):
    1. 2019 kite_actual_depower can be mapped to normalized power u_p ∈ [0,1]
    2. Both systems measure the same physical tape deployment (uncertain)
    3. Linear interpolation is valid (empirical, not theoretical)

    Physical chain (conceptual, not validated):
    1. Normalize 2019 depower to powered fraction u_p ∈ [0, 1]
    2. Convert u_p to tape deployment ld = (1 - u_p) * ld_max
    3. Invert 2025 affine relation: up_data_25 = (ld - offset) / scale

    Valid range: Only tested on observed 2019 range [0.02°, 22.68°]
    """
    x19_depower = np.asarray(x19_depower)

    # Step 1: Normalize to u_p ∈ [0, 1]
    u_p_19 = np.clip((x19_depower - x19_dep) / (x19_pow - x19_dep), 0, 1)

    # Step 2: Convert to tape deployment
    ld = (1.0 - u_p_19) * ld_max

    # Step 3: Invert 2025 affine relation
    up_data_25 = (ld - ld_affine_offset) / ld_affine_scale

    return up_data_25, ld, u_p_19


def main():
    print("=" * 80)
    print("DEPOWER CONVERSION VALIDATION ANALYSIS")
    print("=" * 80)

    # Load 2019 RAW data
    print("\n1. Loading 2019 raw data...")
    df_19 = pd.read_csv("./data/flight_logs/v3/2019-10-08_11.csv", sep=" ")

    # Use all data (powered column doesn't exist in raw)
    powered_19 = df_19.copy()

    print(f"   Total rows: {len(df_19)}")
    print(f"\n   2019 kite_actual_depower statistics:")
    print(f"     min:  {powered_19['kite_actual_depower'].min():.4f}°")
    print(f"     max:  {powered_19['kite_actual_depower'].max():.4f}°")
    print(f"     mean: {powered_19['kite_actual_depower'].mean():.4f}°")
    print(f"     std:  {powered_19['kite_actual_depower'].std():.4f}°")

    # Load 2025 RAW data
    print("\n2. Loading 2025 raw data...")
    df_25 = pd.read_csv("./data/flight_logs/v3/2025-10-09_58-33-00.csv")

    print(f"   Total rows: {len(df_25)}")
    print(f"\n   2025 kite_actual_depower statistics:")
    print(f"     min:  {df_25['kite_actual_depower'].min():.4f}")
    print(f"     max:  {df_25['kite_actual_depower'].max():.4f}")
    print(f"     mean: {df_25['kite_actual_depower'].mean():.4f}")
    print(f"     std:  {df_25['kite_actual_depower'].std():.4f}")

    # Convert 2019 data
    print("\n3. Converting 2019 data to 2025-equivalent...")
    powered_19["up_data_converted"], powered_19["ld_computed"], powered_19["u_p"] = (
        convert_2019_depower_to_2025_updata(powered_19["kite_actual_depower"])
    )

    print(f"\n   Converted 2019 up_data_25 range:")
    print(f"     min:  {powered_19['up_data_converted'].min():.4f}")
    print(f"     max:  {powered_19['up_data_converted'].max():.4f}")
    print(f"     mean: {powered_19['up_data_converted'].mean():.4f}")

    print(f"\n   Computed tape deployment ld (meters):")
    print(f"     min:  {powered_19['ld_computed'].min():.4f} m")
    print(f"     max:  {powered_19['ld_computed'].max():.4f} m")
    print(f"     Physical limit check: ld should be in [0, 1.7]m")

    # CRITICAL VALIDATION
    print("\n" + "=" * 80)
    print("CRITICAL VALIDATION CHECKS")
    print("=" * 80)

    # Check 1: Are we extrapolating?
    x19_min_obs = 0.02
    x19_max_obs = 22.68
    x19_actual_min = powered_19["kite_actual_depower"].min()
    x19_actual_max = powered_19["kite_actual_depower"].max()

    if x19_actual_min < x19_min_obs or x19_actual_max > x19_max_obs:
        print(f"\n⚠️  WARNING: Data outside calibration range!")
        print(f"   Calibrated on: [{x19_min_obs:.2f}, {x19_max_obs:.2f}]°")
        print(f"   Actual data:   [{x19_actual_min:.2f}, {x19_actual_max:.2f}]°")
    else:
        print(
            f"\n✓  Data within calibration range [{x19_min_obs:.2f}, {x19_max_obs:.2f}]°"
        )

    # Check 2: Physical plausibility of tape deployment
    ld_min = powered_19["ld_computed"].min()
    ld_max = powered_19["ld_computed"].max()
    if ld_min < -0.1 or ld_max > 1.8:
        print(f"\n⚠️  WARNING: Computed tape deployment outside physical range!")
        print(f"   Expected: [0, 1.7]m")
        print(f"   Computed: [{ld_min:.3f}, {ld_max:.3f}]m")
    else:
        print(f"\n✓  Tape deployment in plausible range [{ld_min:.3f}, {ld_max:.3f}]m")

    # Check 3: Comparison with 2025 actual values
    up_19_conv_min = powered_19["up_data_converted"].min()
    up_19_conv_max = powered_19["up_data_converted"].max()
    up_25_min = df_25["kite_actual_depower"].min()
    up_25_max = df_25["kite_actual_depower"].max()

    print(f"\n   Comparing 2019 converted vs 2025 actual:")
    print(f"     2019 converted range: [{up_19_conv_min:.4f}, {up_19_conv_max:.4f}]")
    print(f"     2025 actual range:    [{up_25_min:.4f}, {up_25_max:.4f}]")

    # Check overlap
    overlap = not (up_19_conv_max < up_25_min or up_19_conv_min > up_25_max)
    if overlap:
        overlap_min = max(up_19_conv_min, up_25_min)
        overlap_max = min(up_19_conv_max, up_25_max)
        print(f"   ✓  Ranges OVERLAP: [{overlap_min:.4f}, {overlap_max:.4f}]")
    else:
        print(f"   ⚠️  WARNING: Ranges DO NOT OVERLAP!")
        print(f"        This suggests the conversion may be incorrect.")

    # Check 4: Sign of up_data
    if up_19_conv_min < 0:
        print(
            f"\n   ℹ️  NOTE: Converted up_data includes negative values ({up_19_conv_min:.4f})"
        )
        print(f"        This occurs at 'fully powered' state (x19 = {x19_max_obs}°)")
        print(
            f"        Interpretation: The 0.2m offset may represent slack/minimum length"
        )

    # Histogram comparison
    print("\n4. Creating comparison plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: 2019 raw depower angle
    ax = axes[0, 0]
    ax.hist(powered_19["kite_actual_depower"], bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("2019 kite_actual_depower (degrees)")
    ax.set_ylabel("Count")
    ax.set_title("2019 Raw Depower Angle Distribution")
    ax.axvline(x19_min_obs, color="r", linestyle="--", label=f"Calibration endpoints")
    ax.axvline(x19_max_obs, color="r", linestyle="--")
    ax.legend()

    # Plot 2: 2019 converted to up_data
    ax = axes[0, 1]
    ax.hist(
        powered_19["up_data_converted"],
        bins=50,
        alpha=0.7,
        edgecolor="black",
        color="orange",
    )
    ax.set_xlabel("2019 converted to up_data (2025-equivalent)")
    ax.set_ylabel("Count")
    ax.set_title("2019 Converted to 2025 Units")
    ax.axvline(-0.04, color="r", linestyle="--", label="Expected range")
    ax.axvline(0.30, color="r", linestyle="--")
    ax.legend()

    # Plot 3: 2025 actual depower
    ax = axes[1, 0]
    ax.hist(
        df_25["kite_actual_depower"],
        bins=50,
        alpha=0.7,
        edgecolor="black",
        color="green",
    )
    ax.set_xlabel("2025 kite_actual_depower")
    ax.set_ylabel("Count")
    ax.set_title("2025 Actual Depower Distribution")

    # Plot 4: Overlap comparison
    ax = axes[1, 1]
    ax.hist(
        powered_19["up_data_converted"],
        bins=50,
        alpha=0.5,
        label="2019 converted",
        edgecolor="black",
    )
    ax.hist(
        df_25["kite_actual_depower"],
        bins=50,
        alpha=0.5,
        label="2025 actual",
        edgecolor="black",
    )
    ax.set_xlabel("Depower value (2025 units)")
    ax.set_ylabel("Count")
    ax.set_title("Overlap Check: 2019 Converted vs 2025 Actual")
    ax.legend()

    plt.tight_layout()
    output_path = "./results/plots_paper/depower_conversion_validation.pdf"
    plt.savefig(output_path, dpi=150)
    print(f"   Saved: {output_path}")

    # Summary and recommendations
    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    print(
        """
    This analysis reveals:
    
    1. DIFFERENT RANGES: The 2019 and 2025 systems have completely different
       value ranges, suggesting they measure different physical quantities.
       
    2. EMPIRICAL NATURE: The conversion is a transfer function between control
       states, NOT a physics-based transformation.
       
    3. VALIDITY: The conversion is only valid within the observed 2019 range.
       
    RECOMMENDATIONS:
    
    If ranges DO NOT OVERLAP:
      → The systems likely measure different quantities (motor angle vs tape length)
      → Conversion may be fundamentally flawed
      → Consider using both as separate control metrics
      
    If ranges DO OVERLAP:
      → Conversion can be used as empirical calibration
      → Add warnings for extrapolation outside observed range
      → Document that this is NOT physics-based
      
    Next steps:
    1. Check if 'kite_actual_depower' is the same quantity in both years
    2. Verify the 2025 affine relation ld = 0.2 + 5*up_data represents tape deployment
    3. If different quantities, consider renaming to avoid confusion
    """
    )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
