#!/usr/bin/env python3
"""
Analyze multiple columns from raw flight CSV files by flight phase.
Computes min, max, and mean values for pp-ro and pp-ri phases.
"""

import csv
import os


def print_stats(phase, column_name, values):
    """Helper function to print statistics for a column."""
    if values:
        print(f"    {column_name}:")
        print(f"      Count: {len(values)}")
        print(f"      Min:   {min(values):.6f}")
        print(f"      Max:   {max(values):.6f}")
        print(f"      Mean:  {sum(values) / len(values):.6f}")
    else:
        print(f"    {column_name}: No valid values")


def analyze_flight_data(filepath):
    """
    Read CSV file and compute statistics for multiple columns by flight phase.

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

            # Columns to analyze
            columns_to_analyze = [
                "kite_actual_depower",
                # "l_over_d",
                "kite_actual_steering",
                # "load_cell_main_force",
            ]

            # Check if required columns exist
            for col in columns_to_analyze:
                if col not in reader.fieldnames:
                    print(f"Error: '{col}' column not found in {filepath}")
                    return False

            if "flight_phase" not in reader.fieldnames:
                print(f"Error: 'flight_phase' column not found in {filepath}")
                return False

            # Read all values, separating by flight phase
            pp_ro_data = {col: [] for col in columns_to_analyze}
            pp_ri_data = {col: [] for col in columns_to_analyze}
            total_rows = 0

            for row in reader:
                total_rows += 1

                # Get flight phase
                phase = row.get("flight_phase", "").strip()

                # Collect values for all columns
                for col in columns_to_analyze:
                    val_str = row[col].strip()
                    if val_str and val_str.lower() != "nan":
                        try:
                            val = float(val_str)

                            # Categorize by flight phase
                            if phase == "pp-ro":
                                pp_ro_data[col].append(val)
                            elif phase == "pp-ri":
                                pp_ri_data[col].append(val)
                        except ValueError:
                            pass

            # Print results
            print(f"File: {os.path.basename(filepath)}")
            print(f"Total rows: {total_rows}")
            print()

            # pp-ro phase statistics
            print(f"Flight Phase: pp-ro")
            for col in columns_to_analyze:
                print_stats("pp-ro", col, pp_ro_data[col])
            print()

            # pp-ri phase statistics
            print(f"Flight Phase: pp-ri")
            for col in columns_to_analyze:
                print_stats("pp-ri", col, pp_ri_data[col])
            print()

    except Exception as e:
        print(f"Error reading file: {e}")
        return False


if __name__ == "__main__":
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data", "flight_logs", "v3")

    # Define default files to analyze
    files = [
        # os.path.join(data_dir, "2019-10-08_11.csv"),
        os.path.join(data_dir, "2025-10-09_58-33-00.csv"),
    ]

    print("=" * 60)
    print("Analyzing flight data by phase: pp-ro and pp-ri")
    print("Columns: kite_actual_depower, l_over_d, kite_actual_steering,")
    print("         load_cell_main_force")
    print("=" * 60)
    print()

    for filepath in files:
        analyze_flight_data(filepath)
