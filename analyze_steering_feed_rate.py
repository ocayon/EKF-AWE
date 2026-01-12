import pandas as pd
import numpy as np


def analyze_steering_feed_rate(csv_file):
    """Analyze how fast the steering tape is fed (rate of change during active steering)."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {csv_file}")
    print(f"{'='*60}")

    # Read the CSV file
    try:
        df = pd.read_csv(csv_file, sep=",", low_memory=False)
        if len(df.columns) == 1:
            df = pd.read_csv(csv_file, sep=r"\s+", low_memory=False)
    except:
        df = pd.read_csv(csv_file, sep=r"\s+", low_memory=False)

    # Find steering column
    steering_col = None
    for name in ["kcu_actual_steering", "kite_actual_steering"]:
        if name in df.columns:
            steering_col = name
            break

    if steering_col is None:
        print("Steering column not found")
        return

    print(f"Using column: '{steering_col}'")

    # Get data
    time_col = "time" if "time" in df.columns else df.columns[0]
    time = df[time_col].values
    steering = df[steering_col].values

    print(
        f"Min steering: {np.nanmin(steering):.2f}°, Max steering: {np.nanmax(steering):.2f}°"
    )
    print(
        f"Mean: {np.nanmean(steering):.2f}°, Range: {np.nanmax(steering) - np.nanmin(steering):.2f}°"
    )

    # Calculate rate of change (degrees per second)
    dt = np.diff(time)
    dsteering = np.diff(steering)
    rate_of_change = dsteering / dt

    # Filter out near-zero time differences to avoid division issues
    valid_mask = dt > 0.01
    rate_of_change_filtered = rate_of_change[valid_mask]

    # Find periods of significant steering change (active feeding)
    # Threshold: steering rate > 5 deg/s (adjust based on data)
    threshold = 5.0  # degrees per second

    active_feeding = np.abs(rate_of_change_filtered) > threshold

    # Group consecutive active feeding events
    feeding_events = []
    in_event = False
    event_start_idx = None

    valid_indices = np.where(valid_mask)[0]

    for i, (idx, is_active) in enumerate(zip(valid_indices, active_feeding)):
        if is_active and not in_event:
            # Start of feeding event
            in_event = True
            event_start_idx = i
            event_start_global_idx = idx
            start_steering = steering[idx]

        elif not is_active and in_event:
            # End of feeding event
            event_end_idx = i - 1
            event_end_global_idx = valid_indices[event_end_idx] + 1

            # Calculate statistics for this event
            event_duration = time[event_end_global_idx] - time[event_start_global_idx]
            event_steering_change = abs(steering[event_end_global_idx] - start_steering)
            event_rates = rate_of_change_filtered[event_start_idx : event_end_idx + 1]
            event_avg_rate = np.mean(np.abs(event_rates))
            event_max_rate = np.max(np.abs(event_rates))

            # Only keep events with significant change (at least 10 degrees)
            if event_steering_change > 10.0 and event_duration > 0.5:
                feeding_events.append(
                    {
                        "duration": event_duration,
                        "steering_change": event_steering_change,
                        "avg_rate": event_avg_rate,
                        "max_rate": event_max_rate,
                        "start_value": start_steering,
                        "end_value": steering[event_end_global_idx],
                    }
                )

            in_event = False

    # Analyze feeding events
    if feeding_events:
        durations = [e["duration"] for e in feeding_events]
        changes = [e["steering_change"] for e in feeding_events]
        avg_rates = [e["avg_rate"] for e in feeding_events]
        max_rates = [e["max_rate"] for e in feeding_events]

        print(f"\nTotal active feeding events found: {len(feeding_events)}")
        print(f"  (Events where steering changes >10° at >5°/s)")

        print(f"\nFeeding duration (time to move tape):")
        print(f"  Average: {np.mean(durations):.3f} seconds")
        print(f"  Median: {np.median(durations):.3f} seconds")
        print(f"  Std: {np.std(durations):.3f} seconds")
        print(f"  Range: {np.min(durations):.3f} - {np.max(durations):.3f} seconds")

        print(f"\nSteering deflection magnitude:")
        print(f"  Average: {np.mean(changes):.2f}°")
        print(f"  Median: {np.median(changes):.2f}°")
        print(f"  Range: {np.min(changes):.2f} - {np.max(changes):.2f}°")

        print(f"\nSteering feed rate:")
        print(f"  Average rate: {np.mean(avg_rates):.2f} °/s")
        print(f"  Median rate: {np.median(avg_rates):.2f} °/s")
        print(f"  Average max rate: {np.mean(max_rates):.2f} °/s")

        # Calculate time for typical full deflection
        typical_full_deflection = np.max(steering) - np.min(steering)
        typical_feed_time = typical_full_deflection / np.median(avg_rates)
        print(f"\nEstimated time for full deflection ({typical_full_deflection:.1f}°):")
        print(f"  At median rate: {typical_feed_time:.2f} seconds")

        print(f"\nFirst 10 feeding events:")
        for i, e in enumerate(feeding_events[:10]):
            print(
                f"  {i+1}. {e['start_value']:+6.2f}° → {e['end_value']:+6.2f}° "
                f"(Δ={e['steering_change']:5.2f}°) in {e['duration']:.2f}s "
                f"@ {e['avg_rate']:.1f}°/s"
            )
    else:
        print("\nNo active feeding events detected.")
        print("The steering might be changing more gradually.")


# Analyze both files
file1 = (
    "/home/jellepoland/ownCloud/phd/code/EKF-AWE/data/flight_logs/v3/2019-10-08_11.csv"
)
file2 = "/home/jellepoland/ownCloud/phd/code/EKF-AWE/data/flight_logs/v3/2025-10-09_58-33-00.csv"

analyze_steering_feed_rate(file1)
analyze_steering_feed_rate(file2)
