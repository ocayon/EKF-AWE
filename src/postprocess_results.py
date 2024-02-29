import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import kappa, z0, kite_model, year, month, day
from run_EKF import create_kite
import seaborn as sns
import plot_utils as pu
from postprocessing import compute_mse, postprocess_results, calculate_wind_speed_airborne_sensors

path = '../results/'+kite_model+'/'
file_name = kite_model+'_'+year+'-'+month+'-'+day
date = year+'-'+month+'-'+day

results = pd.read_csv(path+file_name+'_res_GPS.csv')
flight_data = pd.read_csv(path+file_name+'_fd.csv')

kite = create_kite(kite_model)


results, flight_data = postprocess_results(results,flight_data, kite, imus = [0], remove_IMU_offsets=True, correct_IMU_deformation = False)

flight_data = calculate_wind_speed_airborne_sensors(results,flight_data, imus = [0])

