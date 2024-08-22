import h5py
import pandas as pd
import os

def save_results(ekf_output_df, flight_data, kite_model, year, month, day, config_data, addition=""):
    # Construct the file name and path
    file_name = f"{kite_model}_{year}-{month}-{day}"
    path = os.path.join("results", kite_model)

    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # # Save the results DataFrame to a CSV file
    # csv_filename = file_name + "_res" + addition + ".csv"
    # ekf_output_df.to_csv(os.path.join(path, csv_filename), index=False)

    # # Save the flight data DataFrame to a CSV file
    # csv_filename = file_name + "_fd.csv"
    # flight_data.to_csv(os.path.join(path, csv_filename), index=False)

    # Convert object columns to strings with fixed length
    def convert_df(df):
        for col in df.columns:
            if df[col].dtype == 'O':  # Object type, usually means string or mixed
                max_len = df[col].map(lambda x: len(str(x))).max()  # Find max length
                df[col] = df[col].astype(f'S{max_len}')  # Convert to fixed-length string
        return df

        # Helper function to encode strings in DataFrame
    def encode_strings(df):
        for col in df.columns:
            if df[col].dtype == object:  # Check if the column is of object type (likely strings)
                df[col] = df[col].astype('S')  # Convert strings to byte strings
        return df
    
    # Convert string columns in both DataFrames
    ekf_output_df = encode_strings(ekf_output_df)
    flight_data = encode_strings(flight_data)
    
    # Save the DataFrames to an HDF5 file
    hdf5_filename = file_name + addition + ".h5"
    hdf5_path = os.path.join(path, hdf5_filename)
    
    with h5py.File(hdf5_path, 'w') as hf:
        # Save the ekf_output_df DataFrame
        ekf_group = hf.create_group('ekf_output')
        ekf_group.attrs['description'] = 'Extended Kalman Filter output, including system parameters derived from the postrocessing of the EKF state vector with the experimental data.'
        for col in ekf_output_df.columns:
            ekf_group.create_dataset(col, data=ekf_output_df[col].values)
        
        # Save the flight_data DataFrame
        flight_group = hf.create_group('flight_data')
        flight_group.attrs['description'] = 'Experimental data collected during the flight test. Offsets are applied to orientation data and tether length.'
        for col in flight_data.columns:
            flight_group.create_dataset(col, data=flight_data[col].values)
        
        # Save config_data as nested groups
        def save_dict_as_group(group, config_dict):
            for key, value in config_dict.items():
                if isinstance(value, dict):
                    # Recursively save nested dictionary
                    subgroup = group.create_group(key)
                    save_dict_as_group(subgroup, value)
                else:
                    # Save the value as an attribute or dataset
                    if isinstance(value, str):
                        value = value.encode('utf-8')  # Convert strings to bytes
                    group.attrs[key] = value

        config_group = hf.create_group('config_data')
        config_group.attrs['description'] = 'Configuration data used for the simulation and postprocessing.'
        save_dict_as_group(config_group, config_data)




# Example usage
# save_results(ekf_output_df, flight_data, "kite_model_name", 2024, 7, 19)

