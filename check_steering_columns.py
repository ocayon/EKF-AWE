import pandas as pd
import numpy as np


def check_all_steering_columns(csv_file):
    """Check all steering-related columns and their change rates."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {csv_file.split('/')[-1]}")
    print(f"{'='*70}")

    # Read file
    try:
        df = pd.read_csv(csv_file, sep=",", low_memory=False)
        if len(df.columns) == 1:
            df = pd.read_csv(csv_file, sep=r"\s+", low_memory=False)
    except:
        df = pd.read_csv(csv_file, sep=r"\s+", low_memory=False)

    print(f"Total rows: {len(df)}, Total columns: {len(df.columns)}")

    # Find all steering-related columns
    steering_cols = [col for col in df.columns if "steer" in str(col).lower()]

    print(f"\nSteering-related columns found: {len(steering_cols)}")
    for col in steering_cols:
        print(f"  - {col}")

    # Analyze each steering column
    time_col = "time" if "time" in df.columns else df.columns[0]
    time = df[time_col].values

    for col in steering_cols:
        print(f"\n{'-'*70}")
        print(f"Column: {col}")
        print(f"{'-'*70}")

        values = df[col].values

        # Basic stats
        print(f"Min: {np.nanmin(values):.2f}, Max: {np.nanmax(values):.2f}")
        print(f"Mean: {np.nanmean(values):.2f}, Std: {np.nanstd(values):.2f}")
        print(f"Range: {np.nanmax(values) - np.nanmin(values):.2f}")

        # Calculate rate of change
        dt = np.diff(time)
        dvalues = np.diff(values)

        # Filter valid time steps
        valid_mask = (dt > 0.001) & (~np.isnan(dvalues))
        if np.sum(valid_mask) == 0:
            print("No valid time steps for rate calculation")
            continue

        rates = np.abs(dvalues[valid_mask] / dt[valid_mask])

        print(f"\nRate of change statistics:")
        print(f"  Mean rate: {np.mean(rates):.2f} units/s")
        print(f"  Median rate: {np.median(rates):.2f} units/s")
        print(f"  Max rate: {np.max(rates):.2f} units/s")
        print(f"  95th percentile rate: {np.percentile(rates, 95):.2f} units/s")

        # Count fast changes
        fast_threshold = 5.0  # units per second
        fast_changes = np.sum(rates > fast_threshold)
        print(
            f"  Fast changes (>{fast_threshold} units/s): {fast_changes} ({100*fast_changes/len(rates):.1f}%)"
        )

        # Show some sample values
        sample_indices = np.linspace(0, len(values) - 1, 10, dtype=int)
        print(f"\nSample values (10 points):")
        for i in sample_indices:
            print(f"  t={time[i]:.1f}s: {values[i]:.2f}")


# Analyze both files
file1 = (
    "/home/jellepoland/ownCloud/phd/code/EKF-AWE/data/flight_logs/v3/2019-10-08_11.csv"
)
file2 = "/home/jellepoland/ownCloud/phd/code/EKF-AWE/data/flight_logs/v3/2025-10-09_58-33-00.csv"

check_all_steering_columns(file1)
check_all_steering_columns(file2)
