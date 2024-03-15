import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import kappa, z0, kite_model
from run_EKF import create_kite
import seaborn as sns
import plot_utils as pu
from postprocessing import calculate_wind_speed_airborne_sensors, postprocess_results
year = '2019'
month = '10'
day = '08'
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

VSM_results = pd.read_csv(r'C:\Users\ocayon\Desktop\EKF\processed_data\aerostructural\v3_aero_coeffs_VSM.csv')
cycles_plotted = np.arange(10, 65, step=1)
mask = np.any(
    [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
mask = (results['CD']>0.03)&mask&(flight_data['powered'] == 'powered')
CL_VSM = VSM_results['CL']
CD_VSM = VSM_results['CD']
aoa_VSM = VSM_results['aoa']

#%%
cycles_plotted = np.arange(30, 65, step=1)
mask = np.any(
    [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
mask = (flight_data['turn_straight'] == 'straight')&(flight_data['powered'] == 'powered')&mask&(results['CD']>0.01)
fig, ax = plt.subplots(figsize = (10,6))
coeff = results.CL
pu.plot_probability_density(results['aoa'][mask]+3,coeff[mask],fig,ax,xlabel='aoa [deg]',ylabel='CL []')
plt.scatter(aoa_VSM*180/np.pi,CL_VSM,color ='orange')
plt.legend(['Experimental','VSM'])
fig, ax = plt.subplots(figsize = (10,6))
coeff = results.CD
pu.plot_probability_density(results['aoa'][mask]+3,coeff[mask],fig,ax,xlabel='aoa [deg]',ylabel='CL []')
plt.scatter(aoa_VSM*180/np.pi,CD_VSM,color ='orange')
plt.legend(['Experimental','VSM'])
fig, ax = plt.subplots()
coeff = results.CL**3/results.CD**2
pu.plot_probability_density(results['aoa'][mask]+3,coeff[mask],fig,ax,xlabel='aoa')
plt.scatter(aoa_VSM*180/np.pi,CL_VSM**3/CD_VSM**2,color ='orange')