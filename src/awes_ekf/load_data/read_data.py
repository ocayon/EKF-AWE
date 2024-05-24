
from pathlib import Path
import pandas as pd
def read_processed_flight_data(year,month,day,kite_model):
    # File path
    file_name = f"{kite_model}_{year}-{month}-{day}"
    file_path = Path('processed_data/flight_data') / kite_model / (file_name + '.csv')
    flight_data = pd.read_csv(file_path)
    flight_data = flight_data.reset_index()
    return flight_data

def read_results(year,month,day,kite_model,addition = ''):
    path = 'results/'+kite_model+'/'
    file_name = kite_model+'_'+year+'-'+month+'-'+day
    date = year+'-'+month+'-'+day

    results = pd.read_csv(path+file_name+'_res'+addition+'.csv')
    flight_data = pd.read_csv(path+file_name+'_fd.csv')

    # results = results.dropna()
    rows_to_keep = results.index
    flight_data = flight_data.loc[rows_to_keep]
    return results, flight_data