#!/usr/bin/env python3
"""
Analyze kite_actual_depower column from raw flight CSV files.
Computes min, max, and mean values.
"""

import csv
import os


def analyze_depower(filepath):
    """
    Read CSV file and compute statistics for kite_actual_depower column.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (2019 or 2025 format)
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return False

    # Read the file and find the column
    try:
        with open(filepath, "r") as f:
            # Detect delimiter by reading first line
            first_line = f.readline()
            delimiter = "," if "," in first_line else " "
            f.seek(0)

            reader = csv.DictReader(f, delimiter=delimiter)

            # Check if required columns exist
            if "kite_actual_depower" not in reader.fieldnames:
                print(f"Error: 'kite_actual_depower' column not found in {filepath}")
                print(f"Available columns: {reader.fieldnames}")
                return False

            if "flight_phase" not in reader.fieldnames:
                print(f"Error: 'flight_phase' column not found in {filepath}")
                print(f"Available columns: {reader.fieldnames}")
                return False

            # Read all values, separating by flight phase
            pp_ro_values = []
            pp_ri_values = []
            total_rows = 0

            for row in reader:
                total_rows += 1

                # Get flight phase
                phase = row.get("flight_phase", "").strip()

                # Get depower value
                val_str = row["kite_actual_depower"].strip()
                if val_str and val_str.lower() != "nan":
                    try:
                        depower_val = float(val_str)

                        # Categorize by flight phase
                        if phase == "pp-ro":
                            pp_ro_values.append(depower_val)
                        elif phase == "pp-ri":
                            pp_ri_values.append(depower_val)
                    except ValueError:
                        pass

            # Print results
            print(f"File: {os.path.basename(filepath)}")
            print(f"Total rows: {total_rows}")
            print()

            # pp-ro phase statistics
            if pp_ro_values:
                print(f"Flight Phase: pp-ro")
                print(f"  Valid kite_actual_depower values: {len(pp_ro_values)}")
                print(f"  Min:  {min(pp_ro_values):.6f}")
                print(f"  Max:  {max(pp_ro_values):.6f}")
                print(f"  Mean: {sum(pp_ro_values) / len(pp_ro_values):.6f}")
            else:
                print(f"Flight Phase: pp-ro")
                print(f"  No valid kite_actual_depower values found")
            print()

            # pp-ri phase statistics
            if pp_ri_values:
                print(f"Flight Phase: pp-ri")
                print(f"  Valid kite_actual_depower values: {len(pp_ri_values)}")
                print(f"  Min:  {min(pp_ri_values):.6f}")
                print(f"  Max:  {max(pp_ri_values):.6f}")
                print(f"  Mean: {sum(pp_ri_values) / len(pp_ri_values):.6f}")
            else:
                print(f"Flight Phase: pp-ri")
                print(f"  No valid kite_actual_depower values found")
            print()

            return True

    except Exception as e:
        print(f"Error reading file: {e}")
        return False


if __name__ == "__main__":
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data", "flight_logs", "v3")

    # Define default files to analyze
    files = [
        os.path.join(data_dir, "2019-10-08_11.csv"),
        os.path.join(data_dir, "2025-10-09_58-33-00.csv"),
    ]

    print("=" * 60)
    print("Analyzing kite_actual_depower from flight CSV files")
    print("By flight phase: pp-ro and pp-ri")
    print("=" * 60)
    print()

    for filepath in files:
        analyze_depower(filepath)
