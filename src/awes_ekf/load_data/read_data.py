from pathlib import Path
import pandas as pd
import h5py


def read_processed_flight_data(year, month, day, kite_model):
    # File path
    file_name = f"{kite_model}_{year}-{month}-{day}"
    file_path = Path("processed_data/flight_data") / kite_model / (file_name + ".csv")
    flight_data = pd.read_csv(file_path)
    flight_data = flight_data.reset_index()
    return flight_data

def read_dict_from_group(group):
    config_dict = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode('utf-8')  # Decode byte strings back to regular strings
        config_dict[key] = value

    for subgroup_name in group:
        subgroup = group[subgroup_name]
        config_dict[subgroup_name] = read_dict_from_group(subgroup)
    
    return config_dict

def read_results(year, month, day, kite_model, addition="", path_to_main=""):
    path = "results/" + str(kite_model) + "/"
    date = str(year) + "-" + str(month) + "-" + str(day)
    file_name = str(kite_model) + "_" + date
    hdf5_path = path_to_main + path + file_name + addition + ".h5"
    with h5py.File(hdf5_path, 'r') as hf:
        # Read the ekf_output_df DataFrame
        ekf_group = hf['ekf_output']
        ekf_output_df = pd.DataFrame({col: ekf_group[col][:].astype(str) if ekf_group[col].dtype.kind == 'S' else ekf_group[col][:] 
                                      for col in ekf_group.keys()})
        
        # Read the flight_data DataFrame
        flight_group = hf['flight_data']
        flight_data_df = pd.DataFrame({col: flight_group[col][:].astype(str) if flight_group[col].dtype.kind == 'S' else flight_group[col][:] 
                                       for col in flight_group if isinstance(flight_group[col], h5py.Dataset)})
        
        # Read config_data
        config_group = hf['config_data']
        config_data = read_dict_from_group(config_group)

    return ekf_output_df, flight_data_df, config_data