import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import kappa, z0, kite_model, year, month, day
from utils import R_EG_Body, calculate_angle,project_onto_plane, create_kite
import seaborn as sns
import plot_utils as pu

path = '../results/'+kite_model+'/'
file_name = kite_model+'_'+year+'-'+month+'-'+day
date = year+'-'+month+'-'+day

results = pd.read_csv(path+file_name+'_res_GPS.csv')
flight_data = pd.read_csv(path+file_name+'_fd.csv')

offset_df = pd.read_csv('../processed_data/IMU_offset.csv')

for i in range(len(offset_df)):
    if offset_df['date'].iloc[i] == date:
        offset = offset_df.iloc[i]
        break

plot_lidar_heights= [100,160,200,250]

kite = create_kite(kite_model)

IMU_0 = True
IMU_1 = False
EKF_tether = False

#%%
flight_data = pu.remove_offsets_IMU_data(flight_data, offset)
# function calculate angles of attack, finnsih postprocess
results, flight_data = pu.postprocess_results(results,flight_data, kite, IMU_0=IMU_0, IMU_1=IMU_1, EKF_tether=EKF_tether)
print('Postprocess done')
#%%
flight_data = pu.correct_aoa_ss_measurements(results,flight_data)
#%%
flight_data = pu.calculate_wind_speed_airborne_sensors(results,flight_data, IMU_0=IMU_0, IMU_1=IMU_1, EKF_tether=EKF_tether)
print('Wind speed calculated from airborne sensors')

# Postprocess done

#%%Plot results wind speed

pu.plot_wind_speed(results.iloc[6000:-6000],flight_data.iloc[6000:-6000], plot_lidar_heights,IMU_0=IMU_0, IMU_1=IMU_1, EKF_tether=EKF_tether, savefig=True) # PLot calculated wind speed against lidar

#%%
pu.plot_wind_profile(flight_data.iloc[6000:-6000], results.iloc[6000:-6000], savefig=True) # Plot wind profile





#%% Plot results aerodynamic coefficients

################## Time series ##################
cycles_plotted = np.arange(0,60,1)
pu.plot_aero_coeff_vs_aoa_ss(results, flight_data, cycles_plotted,IMU_0=IMU_0, IMU_1=IMU_1, EKF_tether=EKF_tether,savefig=True) # Plot aero coeff vs aoa_ss
pu.plot_aero_coeff_vs_up_us(results, flight_data, cycles_plotted,IMU_0=IMU_0, IMU_1=IMU_1, EKF_tether=EKF_tether,savefig=True) # Plot aero coeff vs up_used
#%%
################## Density plots ##################
flight_data = flight_data.iloc[12000:-12000]
results = results.iloc[12000:-12000]
mask = (flight_data['turn_straight'] == 'straight')&(flight_data['powered'] == 'powered')
pu.plot_CL_CD_aoa(results, mask, 'IMU_0') # Plot CL vs CD for different aoa
pu.plot_CL_CD_ss(results, mask, 'IMU_0')    # Plot CL vs CD for different aoa_ss
# pu.plot_prob_coeff_vs_aoa_ss(results, results.CL**3/results.CD**2, mask, 'IMU_0') # Plot CL^3/CD^2 vs aoa_ss
# pu.plot_prob_coeff_vs_aoa_ss(results, results.CL/results.CD, mask, 'IMU_0') # Plot CL/CD vs aoa_ss


#%% Time series
# fig,ax = plt.subplots()
# pu.plot_time_series(flight_data, flight_data['kite_apparent_windspeed'],'Apparent windspeed(m/s)', ax, color='blue', label='Measured',plot_phase=False)
# pu.plot_time_series(flight_data,results['va_kite'],'Apparent windspeed(m/s)', ax, color='red', label='Estimated',plot_phase=True)
# ax.grid()

# fig,ax = plt.subplots()
# r_kite = np.vstack((np.array(flight_data['kite_0_rx']),np.array(flight_data['kite_0_ry']),np.array(flight_data['kite_0_rz']))).T
# r_kite = np.linalg.norm(r_kite,axis = 1)
# pu.plot_time_series(flight_data, r_kite,'Kite radius', ax, color='blue', label='GPS radius',plot_phase=False)
# pu.plot_time_series(flight_data,flight_data['ground_tether_length'],'Tether length', ax, color='red', label='Tether length',plot_phase=True)
# ax.grid()


