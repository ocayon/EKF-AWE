import pandas as pd
import numpy as np


def analyze_steering_transition_time(csv_file):
    """Analyze the time to transition from rest state to full asymmetric steering."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {csv_file}")
    print(f"{'='*60}")

    # Read the CSV file - handle both comma and space-separated files
    print("Reading file...")
    try:
        df = pd.read_csv(csv_file, sep=",", low_memory=False)
        if len(df.columns) == 1:
            # Try space-separated
            print("Retrying with space separator...")
            df = pd.read_csv(csv_file, sep=r"\s+", low_memory=False)
    except Exception as e:
        print(f"Error reading with comma: {e}")
        df = pd.read_csv(csv_file, sep=r"\s+", low_memory=False)

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Try to find steering column - check multiple possibilities
    steering_col = None
    possible_names = ["kcu_actual_steering", "kite_actual_steering"]

    for name in possible_names:
        if name in df.columns:
            steering_col = name
            break

    # If not found, search for any column containing "steering"
    if steering_col is None:
        for col in df.columns:
            if "actual_steering" in str(col).lower():
                steering_col = col
                break

    if steering_col is None:
        print(f"Steering column not found")
        print(
            f"Available columns with 'steering': {[c for c in df.columns if 'steering' in str(c).lower()]}"
        )
        return

    print(f"Using column: '{steering_col}'")

    # Get the time and steering columns
    time_col = "time" if "time" in df.columns else df.columns[0]
    time = df[time_col].values
    steering = df[steering_col].values

    # Calculate basic statistics
    mean_steering = np.nanmean(steering)
    std_steering = np.nanstd(steering)
    min_steering = np.nanmin(steering)
    max_steering = np.nanmax(steering)

    print(f"Min steering: {min_steering:.2f}°, Max steering: {max_steering:.2f}°")
    print(f"Mean (rest state): {mean_steering:.2f}°, Std: {std_steering:.2f}°")

    # Define rest state as values close to mean (within 1 std dev)
    rest_threshold = std_steering
    # Define full deflection as beyond 80% of the maximum deviation from mean
    left_max = max_steering
    right_max = abs(min_steering)
    full_deflection_threshold = 0.8 * max(
        left_max - mean_steering, mean_steering - min_steering
    )

    print(f"Rest state threshold: within ±{rest_threshold:.2f}° of mean")
    print(f"Full deflection threshold: {full_deflection_threshold:.2f}° from mean")

    # Find transitions from rest to full deflection
    transitions = []
    in_rest = True
    rest_start_idx = None
    transition_direction = None

    for i in range(len(steering)):
        steering_offset = steering[i] - mean_steering
        abs_offset = abs(steering_offset)

        if in_rest:
            # Check if we're in rest state
            if abs_offset <= rest_threshold:
                if rest_start_idx is None:
                    rest_start_idx = i
            else:
                # Left rest state
                if rest_start_idx is not None:
                    in_rest = False
                    transition_start_idx = rest_start_idx
                    transition_direction = "left" if steering_offset > 0 else "right"
                    transition_start_value = steering[rest_start_idx]
        else:
            # In transition, check if we reached full deflection
            if abs_offset >= full_deflection_threshold:
                # Reached full deflection
                transition_end_idx = i
                transition_time = time[transition_end_idx] - time[transition_start_idx]

                # Record this transition
                transitions.append(
                    {
                        "direction": transition_direction,
                        "start_time": time[transition_start_idx],
                        "end_time": time[transition_end_idx],
                        "duration": transition_time,
                        "start_value": transition_start_value,
                        "end_value": steering[transition_end_idx],
                        "deflection": abs(
                            steering[transition_end_idx] - transition_start_value
                        ),
                    }
                )

                # Reset to look for next rest state
                in_rest = True
                rest_start_idx = None
                transition_direction = None

    # Analyze transitions
    if transitions:
        left_transitions = [t for t in transitions if t["direction"] == "left"]
        right_transitions = [t for t in transitions if t["direction"] == "right"]
        all_durations = [t["duration"] for t in transitions]

        print(f"\nTotal rest→full transitions found: {len(transitions)}")
        print(f"  To left (positive): {len(left_transitions)}")
        print(f"  To right (negative): {len(right_transitions)}")

        print(f"\nAll transitions combined:")
        print(f"  Average duration: {np.mean(all_durations):.3f} seconds")
        print(f"  Median duration: {np.median(all_durations):.3f} seconds")
        print(f"  Std deviation: {np.std(all_durations):.3f} seconds")
        print(
            f"  Min: {np.min(all_durations):.3f} s, Max: {np.max(all_durations):.3f} s"
        )

        if left_transitions:
            left_durations = [t["duration"] for t in left_transitions]
            print(f"\nLeft transitions (rest → full left):")
            print(f"  Average duration: {np.mean(left_durations):.3f} seconds")
            print(f"  Median duration: {np.median(left_durations):.3f} seconds")
            print(
                f"  Min: {np.min(left_durations):.3f} s, Max: {np.max(left_durations):.3f} s"
            )

        if right_transitions:
            right_durations = [t["duration"] for t in right_transitions]
            print(f"\nRight transitions (rest → full right):")
            print(f"  Average duration: {np.mean(right_durations):.3f} seconds")
            print(f"  Median duration: {np.median(right_durations):.3f} seconds")
            print(
                f"  Min: {np.min(right_durations):.3f} s, Max: {np.max(right_durations):.3f} s"
            )

        # Show examples
        print(f"\nFirst 10 transitions (rest → full deflection):")
        for i, t in enumerate(transitions[:10]):
            print(
                f"  {i+1}. {t['direction']:>5}: {t['start_value']:+.2f}° → {t['end_value']:+.2f}° "
                f"(Δ={t['deflection']:.2f}°) in {t['duration']:.3f}s"
            )

        # Analyze deflection magnitudes
        deflections = [t["deflection"] for t in transitions]
        print(f"\nDeflection magnitudes:")
        print(f"  Average: {np.mean(deflections):.2f}°")
        print(f"  Median: {np.median(deflections):.2f}°")

    else:
        print("\nNo clear rest→full transitions detected.")
        print("The steering might not have distinct rest and full deflection states,")
        print("or the thresholds need adjustment.")


# Analyze both files
file1 = (
    "/home/jellepoland/ownCloud/phd/code/EKF-AWE/data/flight_logs/v3/2019-10-08_11.csv"
)
file2 = "/home/jellepoland/ownCloud/phd/code/EKF-AWE/data/flight_logs/v3/2025-10-09_58-33-00.csv"

analyze_steering_transition_time(file1)
analyze_steering_transition_time(file2)
