from awes_ekf.load_data.read_data import read_results_from_hdf5
from pathlib import Path
import pandas as pd

# Assume imported plotting functions for each category
from awes_ekf.plotting import plot_aerodynamics, plot_kinematics, plot_tether, plot_wind_velocity, plot_ekf_performance, personalized_plot

def select_plot_category():
    # Provide user with category options
    print("Select a category to plot:")
    categories = {
        "1": ("Aerodynamics", plot_aerodynamics),
        "2": ("Kinematics", plot_kinematics),
        "3": ("Tether", plot_tether),
        "4": ("Wind Velocity", plot_wind_velocity),
        "5": ("Personalized", personalized_plot), 
        "6": ("EKF performance", plot_ekf_performance)
    }
    for key, (label, _) in categories.items():
        print(f"{key}: {label}")
    
    # Get user input and validate
    selection = input("Enter the number corresponding to the category: ").strip()
    if selection not in categories:
        print("Invalid selection.")
        return None
    
    return categories[selection][1]  # Return the selected plotting function

def get_time_mask(data: pd.DataFrame) -> pd.Series:
    """
    Prompts the user for start and end times to filter the DataFrame by time.
    
    :param data: DataFrame containing the 'time' column.
    :return: Boolean mask for the time range.
    """
    min_time = data['time'].min()
    max_time = data['time'].max()
    print(f"Data covers a time range from {min_time:.2f} to {max_time:.2f} seconds.")

    start_time = float(input(f"Enter the start time for plotting (seconds, or leave empty for {min_time:.2f}): ") or min_time)
    end_time = float(input(f"Enter the end time for plotting (seconds, or leave empty for {max_time:.2f}): ") or max_time)

    if start_time < min_time or end_time > max_time or start_time > end_time:
        print("Invalid time range. Please try again.")
        return get_time_mask(data)

    mask = (data['time'] >= start_time) & (data['time'] <= end_time)
    return mask

def main():
    default_results_dir = Path('./results/v3/')
    results_dir = Path(input(f"Enter the directory with the results files [default: {default_results_dir}]: ").strip() or default_results_dir)

    # List HDF5 files in the results directory
    hdf5_files = list(results_dir.glob("*.h5"))
    if not hdf5_files:
        print("No HDF5 result files found.")
        return
    
    # Display files and prompt for selection
    print("Available result files:")
    for idx, file in enumerate(hdf5_files, start=1):
        print(f"{idx}: {file.name}")
    selected_idx = int(input("Select a file to open: ")) - 1
    selected_file = hdf5_files[selected_idx]

    # Read data from the selected HDF5 file
    ekf_output_df, flight_data_df, config_data = read_results_from_hdf5(selected_file)
    print("Data successfully loaded.")

    # Apply time filtering
    time_mask = get_time_mask(flight_data_df)
    filtered_ekf_output_df = ekf_output_df[time_mask].reset_index(drop=True)
    filtered_flight_data_df = flight_data_df[time_mask].reset_index(drop=True)

    # Loop for selecting and plotting multiple categories
    while True:
        plot_function = select_plot_category()
        if plot_function:
            plot_function(filtered_ekf_output_df, filtered_flight_data_df, config_data)  # Call the selected function with filtered data
        
        # Ask the user if they want to plot another category
        another_plot = input("Would you like to plot another category? (yes/no): ").strip().lower()
        if another_plot != 'yes':
            print("Exiting the plotting tool. Goodbye!")
            break

if __name__ == "__main__":
    main()
