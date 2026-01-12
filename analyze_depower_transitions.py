import pandas as pd
import numpy as np


def analyze_depower_transitions(csv_file):
    """Analyze the time it takes for kite_actual_depower to transition between states."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {csv_file}")
    print(f"{'='*60}")

    # Read the CSV file - handle both comma and space-separated files
    try:
        df = pd.read_csv(csv_file, sep=",")
        if len(df.columns) == 1:
            # Try space-separated
            df = pd.read_csv(csv_file, delim_whitespace=True)
    except:
        df = pd.read_csv(csv_file, delim_whitespace=True)

    # Handle different column naming conventions
    depower_col = None
    if "kite_actual_depower" in df.columns:
        depower_col = "kite_actual_depower"
    else:
        # Try to find it with spaces or different separators
        for col in df.columns:
            if "kite_actual_depower" in str(col):
                depower_col = col
                break

    if depower_col is None:
        print(f"Column 'kite_actual_depower' not found in {csv_file}")
        print(f"First few columns: {df.columns[:10].tolist()}")
        return

    print(f"Using column: '{depower_col}'")

    # Get the time and depower columns
    time_col = "time" if "time" in df.columns else df.columns[0]
    time = df[time_col].values
    depower = df[depower_col].values

    # Calculate basic statistics
    print(
        f"Min depower: {np.nanmin(depower):.2f}, Max depower: {np.nanmax(depower):.2f}"
    )
    print(f"Mean: {np.nanmean(depower):.2f}, Std: {np.nanstd(depower):.2f}")

    # Detect significant changes (use a threshold based on data range)
    depower_range = np.nanmax(depower) - np.nanmin(depower)
    threshold = max(1.0, depower_range * 0.1)  # At least 10% of range or 1.0
    print(f"Using change threshold: {threshold:.2f}")

    # Calculate derivative to find rapid changes
    depower_diff = np.diff(depower)

    # Find transition points
    transitions = []
    i = 0
    while i < len(depower_diff) - 10:
        # Look for significant change
        if abs(depower_diff[i]) > threshold / 10:  # Start of potential transition
            # Found start of change
            transition_start_idx = i
            transition_start_value = depower[i]

            # Follow the change until it stabilizes
            j = i + 1
            while j < len(depower) - 5:
                # Check if value has stabilized (next 5 values similar)
                if j + 5 < len(depower):
                    recent_std = np.std(depower[j : j + 5])
                    if recent_std < 0.5:  # Stabilized
                        # Check if we had a significant total change
                        total_change = depower[j] - transition_start_value
                        if abs(total_change) > threshold:
                            # Valid transition
                            transition_time = time[j] - time[transition_start_idx]

                            transition_type = (
                                "low-to-high" if total_change > 0 else "high-to-low"
                            )

                            transitions.append(
                                {
                                    "type": transition_type,
                                    "start_time": time[transition_start_idx],
                                    "end_time": time[j],
                                    "duration": transition_time,
                                    "start_value": transition_start_value,
                                    "end_value": depower[j],
                                    "change": total_change,
                                }
                            )

                            i = j + 5  # Skip past this transition
                            break
                j += 1
            else:
                i += 1
        else:
            i += 1

    # Analyze transitions
    if transitions:
        low_to_high = [t["duration"] for t in transitions if t["type"] == "low-to-high"]
        high_to_low = [t["duration"] for t in transitions if t["type"] == "high-to-low"]

        print(f"\nTotal transitions found: {len(transitions)}")
        print(f"  Low-to-high transitions: {len(low_to_high)}")
        print(f"  High-to-low transitions: {len(high_to_low)}")

        if low_to_high:
            print(f"\nLow-to-high transitions:")
            print(f"  Average duration: {np.mean(low_to_high):.3f} seconds")
            print(f"  Median duration: {np.median(low_to_high):.3f} seconds")
            print(f"  Std deviation: {np.std(low_to_high):.3f} seconds")
            print(
                f"  Min: {np.min(low_to_high):.3f} s, Max: {np.max(low_to_high):.3f} s"
            )

        if high_to_low:
            print(f"\nHigh-to-low transitions:")
            print(f"  Average duration: {np.mean(high_to_low):.3f} seconds")
            print(f"  Median duration: {np.median(high_to_low):.3f} seconds")
            print(f"  Std deviation: {np.std(high_to_low):.3f} seconds")
            print(
                f"  Min: {np.min(high_to_low):.3f} s, Max: {np.max(high_to_low):.3f} s"
            )

        # Show first few transitions
        print(f"\nFirst 10 transitions:")
        for i, t in enumerate(transitions[:10]):
            print(
                f"  {i+1}. {t['type']}: {t['start_value']:.2f} → {t['end_value']:.2f} "
                f"(Δ={t['change']:+.2f}) in {t['duration']:.3f}s"
            )
    else:
        print("\nNo transitions detected.")
        print(
            f"Depower values remain relatively constant around {np.mean(depower):.2f}"
        )


# Analyze both files
file1 = (
    "/home/jellepoland/ownCloud/phd/code/EKF-AWE/data/flight_logs/v3/2019-10-08_11.csv"
)
file2 = "/home/jellepoland/ownCloud/phd/code/EKF-AWE/data/flight_logs/v3/2025-10-09_58-33-00.csv"

analyze_depower_transitions(file1)
analyze_depower_transitions(file2)
