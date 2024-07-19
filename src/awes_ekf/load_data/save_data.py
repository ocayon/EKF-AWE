from awes_ekf.ekf.ekf_output import convert_ekf_output_to_df
import os

def save_results(ekf_output_df, flight_data, kite_model, year, month, day, addition=""):
    # Construct the file name and path
    file_name = f"{kite_model}_{year}-{month}-{day}"
    path = os.path.join("results", kite_model)

    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # Save the results DataFrame to a CSV file
    csv_filename = file_name + "_res" + addition + ".csv"
    ekf_output_df.to_csv(os.path.join(path, csv_filename), index=False)

    # Save the flight data DataFrame to a CSV file
    csv_filename = file_name + "_fd.csv"
    flight_data.to_csv(os.path.join(path, csv_filename), index=False)

# Example usage
# save_results(ekf_output_df, flight_data, "kite_model_name", 2024, 7, 19)

