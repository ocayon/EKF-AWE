import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import kappa, z0, kite_model
from run_EKF import create_kite
import seaborn as sns
import plot_utils as pu
from postprocessing import calculate_wind_speed_airborne_sensors, postprocess_results
year = '2024'
month = '02'
day = '16'
plt.close('all')
path = '../results/'+kite_model+'/'
file_name = kite_model+'_'+year+'-'+month+'-'+day
date = year+'-'+month+'-'+day

results = pd.read_csv(path+file_name+'_res_GPS.csv')
flight_data = pd.read_csv(path+file_name+'_fd.csv')

plot_lidar_heights= [100,160,200,250]

kite = create_kite(kite_model)

imus = [0]

#%%
results, flight_data = postprocess_results(results,flight_data, kite, imus = [0], remove_IMU_offsets=True, 
                                           correct_IMU_deformation = True,remove_vane_offsets=True,estimate_kite_angle=True)
#%%
flight_data = calculate_wind_speed_airborne_sensors(results,flight_data, imus = [0])
# Postprocess done

#%%Plot results wind speed

pu.plot_wind_speed(results.iloc[6000:-6000],flight_data.iloc[6000:-6000], plot_lidar_heights,IMU_0=False, IMU_1=False, savefig=False) # PLot calculated wind speed against lidar
#%%
pu.plot_wind_speed_height_bins(results.iloc[6000:-6000],flight_data.iloc[6000:-6000], plot_lidar_heights, savefig=True) # Plot calculated wind speed against lidar

#%%
pu.plot_wind_profile(flight_data.iloc[6000:-6000], results.iloc[6000:-6000], savefig=False) # Plot wind profile





#%% Plot results aerodynamic coefficients

################## Time series ##################
cycles_plotted = np.arange(0,60,1)
pu.plot_aero_coeff_vs_aoa_ss(results, flight_data, cycles_plotted,IMU_0=True,savefig=False) # Plot aero coeff vs aoa_ss
pu.plot_aero_coeff_vs_up_us(results, flight_data, cycles_plotted,IMU_0=True,savefig=False) # Plot aero coeff vs up_used
#%%
################## Density plots ##################
# flight_data = flight_data.iloc[1000::]
# results = results.iloc[1000::]
mask = (flight_data['turn_straight'] == 'straight')&(flight_data['powered'] == 'powered')
pu.plot_CL_CD_aoa(results,flight_data, mask, 'EKF') # Plot CL vs CD for different aoa
pu.plot_CL_CD_up(results,flight_data, mask, 'EKF') # Plot CL vs CD for different aoa
pu.plot_CL_CD_ss(results,flight_data, mask, 'meas')    # Plot CL vs CD for different aoa_ss
pu.plot_prob_coeff_vs_aoa_ss(results, results.CL**3/results.CD**2, mask, 'EKF') # Plot CL^3/CD^2 vs aoa_ss
pu.plot_prob_coeff_vs_aoa_ss(results, results.CL/results.CD, mask, 'EKF') # Plot CL/CD vs aoa_ss


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

# fig,ax = plt.subplots()
# pu.plot_time_series(flight_data, flight_data['kite_0_rz'], ax, color='blue', label='Measured',plot_phase=False)
# pu.plot_time_series(flight_data,results['z'], ax, color='red', label='Estimated',plot_phase=False)
# ax.grid()
# ax.legend()
fig,ax = plt.subplots()
pu.plot_time_series(flight_data, flight_data['kite_0_pitch'], ax, color='blue', label='Measured',plot_phase=False)
pu.plot_time_series(flight_data,results['pitch'], ax, color='red', label='Estimated',plot_phase=False)
ax.grid()
ax.legend()

fig,ax = plt.subplots()
pu.plot_time_series(flight_data, flight_data['kite_0_roll'], ax, color='blue', label='Measured',plot_phase=False)
pu.plot_time_series(flight_data,results['roll'], ax, color='red', label='Estimated',plot_phase=False)
ax.grid()
ax.legend()


fig,ax = plt.subplots()
pu.plot_time_series(flight_data, flight_data['kite_0_yaw'], ax, color='blue', label='Measured',plot_phase=False)
pu.plot_time_series(flight_data,results['yaw'], ax, color='red', label='Estimated',plot_phase=False)
ax.grid()
ax.legend()
#%%
import plot_utils as pu
mask = range(3600,4000)
pu.plot_kite_reference_frame(results.iloc[mask], flight_data.iloc[mask], imus)