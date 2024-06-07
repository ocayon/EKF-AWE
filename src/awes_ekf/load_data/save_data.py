from awes_ekf.ekf.ekf_output import convert_ekf_output_to_df
def save_results(ekf_output_df, flight_data, kite_model, year, month, day, addition=''):
    
    file_name = f"{kite_model}_{year}-{month}-{day}"
    path = 'results/'+kite_model+'/'

    # Save the results DataFrame to a CSV file
    csv_filename = file_name+'_res'+addition+'.csv'
    ekf_output_df.to_csv(path+csv_filename, index=False)

    # Save the flight data DataFrame to a CSV file
    csv_filename = file_name+'_fd.csv'
    flight_data.to_csv(path+csv_filename, index=False)
    