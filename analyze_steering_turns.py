import pandas as pd
import numpy as np


def analyze_steering_turns(csv_file):
    """Analyze the duration of turns (when steering has large asymmetrical offset)."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {csv_file}")
    print(f"{'='*60}")

    # Read the CSV file - handle both comma and space-separated files
    try:
        df = pd.read_csv(csv_file, sep=",")
        if len(df.columns) == 1:
            # Try space-separated
            df = pd.read_csv(csv_file, sep=r"\s+")
    except:
        df = pd.read_csv(csv_file, sep=r"\s+")

    # Handle different column naming conventions
    steering_col = None
    if "kite_actual_steering" in df.columns:
        steering_col = "kite_actual_steering"
    else:
        # Try to find it with spaces or different separators
        for col in df.columns:
            if "kite_actual_steering" in str(col):
                steering_col = col
                break

    if steering_col is None:
        print(f"Column 'kite_actual_steering' not found in {csv_file}")
        print(f"First few columns: {df.columns[:10].tolist()}")
        return

    print(f"Using column: '{steering_col}'")

    # Get the time and steering columns
    time_col = "time" if "time" in df.columns else df.columns[0]
    time = df[time_col].values
    steering = df[steering_col].values

    # Calculate basic statistics
    print(
        f"Min steering: {np.nanmin(steering):.2f}, Max steering: {np.nanmax(steering):.2f}"
    )
    print(f"Mean: {np.nanmean(steering):.2f}, Std: {np.nanstd(steering):.2f}")

    # Define threshold for "significant steering" (large asymmetrical offset)
    # We'll consider values beyond 2 std deviations or absolute value > 5 degrees
    mean_steering = np.nanmean(steering)
    std_steering = np.nanstd(steering)
    threshold = max(5.0, 2.0 * std_steering)  # At least 5 degrees or 2 std
    print(f"Using steering threshold: {threshold:.2f} degrees")
    print(f"  (Turns detected when |steering - mean| > {threshold:.2f})")

    # Detect turn events (when steering exceeds threshold)
    turns = []
    in_turn = False
    turn_start_idx = None
    turn_direction = None

    for i in range(len(steering)):
        steering_offset = steering[i] - mean_steering

        if not in_turn:
            # Check if we entered a turn
            if abs(steering_offset) > threshold:
                in_turn = True
                turn_start_idx = i
                turn_direction = "left" if steering_offset > 0 else "right"
                turn_start_value = steering[i]
        else:
            # We're in a turn, check if it ended
            # Turn ends when steering returns close to mean (within threshold/2)
            if abs(steering_offset) < threshold / 2:
                # Turn ended
                turn_end_idx = i
                turn_duration = time[turn_end_idx] - time[turn_start_idx]

                # Calculate average steering during turn
                turn_steering_values = steering[turn_start_idx : turn_end_idx + 1]
                avg_turn_steering = np.mean(turn_steering_values)
                max_turn_steering = np.max(np.abs(turn_steering_values - mean_steering))

                turns.append(
                    {
                        "direction": turn_direction,
                        "start_time": time[turn_start_idx],
                        "end_time": time[turn_end_idx],
                        "duration": turn_duration,
                        "start_value": turn_start_value,
                        "end_value": steering[turn_end_idx],
                        "avg_steering": avg_turn_steering,
                        "max_offset": max_turn_steering,
                    }
                )

                in_turn = False
                turn_start_idx = None

    # Analyze turns
    if turns:
        left_turns = [t["duration"] for t in turns if t["direction"] == "left"]
        right_turns = [t["duration"] for t in turns if t["direction"] == "right"]
        all_durations = [t["duration"] for t in turns]

        print(f"\nTotal turns found: {len(turns)}")
        print(f"  Left turns: {len(left_turns)}")
        print(f"  Right turns: {len(right_turns)}")

        print(f"\nAll turns combined:")
        print(f"  Average duration: {np.mean(all_durations):.3f} seconds")
        print(f"  Median duration: {np.median(all_durations):.3f} seconds")
        print(f"  Std deviation: {np.std(all_durations):.3f} seconds")
        print(
            f"  Min: {np.min(all_durations):.3f} s, Max: {np.max(all_durations):.3f} s"
        )

        if left_turns:
            print(f"\nLeft turns:")
            print(f"  Average duration: {np.mean(left_turns):.3f} seconds")
            print(f"  Median duration: {np.median(left_turns):.3f} seconds")
            print(f"  Std deviation: {np.std(left_turns):.3f} seconds")
            print(f"  Min: {np.min(left_turns):.3f} s, Max: {np.max(left_turns):.3f} s")

        if right_turns:
            print(f"\nRight turns:")
            print(f"  Average duration: {np.mean(right_turns):.3f} seconds")
            print(f"  Median duration: {np.median(right_turns):.3f} seconds")
            print(f"  Std deviation: {np.std(right_turns):.3f} seconds")
            print(
                f"  Min: {np.min(right_turns):.3f} s, Max: {np.max(right_turns):.3f} s"
            )

        # Show first few turns
        print(f"\nFirst 10 turns:")
        for i, t in enumerate(turns[:10]):
            print(
                f"  {i+1}. {t['direction']:>5} turn: {t['start_value']:+.2f}° → {t['end_value']:+.2f}° "
                f"(avg={t['avg_steering']:+.2f}°, max_offset={t['max_offset']:.2f}°) "
                f"duration: {t['duration']:.3f}s"
            )

        # Additional statistics about turn intensity
        avg_max_offsets = [t["max_offset"] for t in turns]
        print(f"\nTurn intensity (max steering offset during turns):")
        print(f"  Average max offset: {np.mean(avg_max_offsets):.2f}°")
        print(f"  Median max offset: {np.median(avg_max_offsets):.2f}°")

    else:
        print("\nNo significant turns detected.")
        print(f"Steering remains relatively close to mean ({mean_steering:.2f}°)")


# Analyze both files
file1 = (
    "/home/jellepoland/ownCloud/phd/code/EKF-AWE/data/flight_logs/v3/2019-10-08_11.csv"
)
file2 = "/home/jellepoland/ownCloud/phd/code/EKF-AWE/data/flight_logs/v3/2025-10-09_58-33-00.csv"

analyze_steering_turns(file1)
analyze_steering_turns(file2)
