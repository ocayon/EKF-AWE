import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_tether_end_position, state_noise_matrices, observation_matrices, R_EG_Body, calculate_angle,project_onto_plane ,read_data, rank_observability_matrix,read_data_new,get_measurements


#%%

plt.close('all')

model = 'v9'
year = '2023'
month = '11'
day = '16'

onlyGPS = True
GPSacc = False
GPSva = False
GPSaccva = False
GPSvw = False
GPSaccvw = False
GPSvavw = False

if model == 'v3':
    from v3_properties import *
elif model == 'v9':
    from v9_properties import *

path = './results/'+model+'/'
file_name = model+'_'+year+'-'+month+'-'+day

flight_data = pd.read_csv(path+file_name+'_fd.csv')

if onlyGPS:
    resonlyGPS = pd.read_csv(path+file_name+'_res_GPS.csv')
if GPSacc:
    resGPSacc = pd.read_csv(path+file_name+'_res_GPSacc.csv')
if GPSva:
    resGPSva = pd.read_csv(path+file_name+'_res_GPS_va.csv')
if GPSaccva:
    resGPSaccva = pd.read_csv(path+file_name+'_res_GPSacc_va.csv')
if GPSvw:
    resGPSvw = pd.read_csv(path+file_name+'_res_GPS_vw.csv')
if GPSaccvw:
    resGPSaccvw = pd.read_csv(path+file_name+'_res_GPSacc_vw.csv')
if GPSvavw:
    resGPSvavw = pd.read_csv(path+file_name+'_res_GPS_va_vw.csv')


#%% Plot Wind Speed



plt.figure()
if onlyGPS:
    wvel = resonlyGPS['uf']/kappa*np.log(resonlyGPS['z']/z0)
    plt.plot(flight_data['time'],wvel,'r',label='onlyGPS')
if GPSacc:
    wvel = resGPSacc['uf']/kappa*np.log(resGPSacc['z']/z0)
    plt.plot(flight_data['time'],wvel,'b',label='GPSacc')
if GPSva:
    wvel = resGPSva['uf']/kappa*np.log(resGPSva['z']/z0)
    plt.plot(flight_data['time'],wvel,'g',label='GPSva')
if GPSaccva:
    wvel = resGPSaccva['uf']/kappa*np.log(resGPSaccva['z']/z0)
    plt.plot(flight_data['time'],wvel,'c',label='Sensor Fusion')
if GPSvw:
    wvel = resGPSvw['uf']/kappa*np.log(resGPSvw['z']/z0)
    plt.plot(flight_data['time'],wvel,'m',label='GPSvw')
if GPSaccvw:
    wvel = resGPSaccvw['uf']/kappa*np.log(resGPSaccvw['z']/z0)
    plt.plot(flight_data['time'],wvel,'y',label='GPSaccvw')
if GPSvavw:
    wvel = resGPSvavw['uf']/kappa*np.log(resGPSvavw['z']/z0)
    plt.plot(flight_data['time'],wvel,'k',label='GPSvavw')

plt.plot(flight_data['time'],flight_data['ground_wind_velocity'],'k',label='Ground Sensor')

for column in flight_data.columns:
    if 'Wind Speed' in column:
        height = ''.join(filter(str.isdigit, column))
        height = int(height)
        if height in [115,160,200]:
            for i in range(len(flight_data)):
                if flight_data[column].iloc[i] != flight_data[column].iloc[i-1]:
                    plt.scatter(flight_data['time'].iloc[i],flight_data[column].iloc[i])
            plt.plot(flight_data['time'],flight_data[column],label = column)
plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Wind Speed [m/s]')

#%% Plot Wind Direction

plt.figure()
if onlyGPS:
    plt.plot(flight_data['time'],resonlyGPS['wdir']*180/np.pi,'r',label='onlyGPS')
if GPSacc:
    plt.plot(flight_data['time'],resGPSacc['wdir']*180/np.pi,'b',label='GPSacc')
if GPSva:
    plt.plot(flight_data['time'],resGPSva['wdir']*180/np.pi,'g',label='GPSva')
if GPSaccva:
    plt.plot(flight_data['time'],resGPSaccva['wdir']*180/np.pi,'c',label='Sensor Fusion')
if GPSvw:
    plt.plot(flight_data['time'],resGPSvw['wdir']*180/np.pi,'m',label='GPSvw')
if GPSaccvw:
    plt.plot(flight_data['time'],resGPSaccvw['wdir']*180/np.pi,'y',label='GPSaccvw')
if GPSvavw:
    plt.plot(flight_data['time'],resGPSvavw['wdir']*180/np.pi,'k',label='GPSvavw')
    
plt.plot(flight_data['time'],360-90-flight_data['ground_wind_direction'],'k',label='Ground Sensor')

for column in flight_data.columns:
    if 'Wind Direction' in column:
        height = ''.join(filter(str.isdigit, column))
        height = int(height)
        if height in [115,160,200]:
            for i in range(len(flight_data)):
                if flight_data[column].iloc[i] != flight_data[column].iloc[i-1]:
                    plt.scatter(flight_data['time'].iloc[i],360-90-flight_data[column].iloc[i])
            plt.plot(flight_data['time'],360-90-flight_data[column],label = column)
            


plt.legend()
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel('Wind Direction [deg]')

#%%


# Define the subplot layout
fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # Adjust the figsize as needed

# Plot Wind Speed
axs[0].plot(flight_data['time'], flight_data['ground_wind_velocity'], 'k', label='Ground Sensor',alpha = 0.5)
wvel = resGPSaccva['uf']/kappa*np.log(resGPSaccva['z']/z0)
axs[0].plot(flight_data['time'],wvel,'c',label='Sensor Fusion',alpha = 0.5)

for column in flight_data.columns:
    if 'Wind Speed' in column:
        height = ''.join(filter(str.isdigit, column))
        height = int(height)
        if height in [115, 160, 200]:
            # for i in range(len(flight_data)):
            #     if flight_data[column].iloc[i] != flight_data[column].iloc[i-1]:
            #         axs[0].scatter(flight_data['time'].iloc[i], flight_data[column].iloc[i])
            axs[0].plot(flight_data['time'], flight_data[column], label=column)

axs[0].legend()
axs[0].grid()
axs[0].set_xlabel('Time [s]')
axs[0].set_ylabel('Wind Speed [m/s]')

# Plot Wind Direction
axs[1].plot(flight_data['time'], 360 - 90 - flight_data['ground_wind_direction'], 'k', label='Ground Sensor',alpha = 0.5)
axs[1].plot(flight_data['time'],resGPSaccva['wdir']*180/np.pi,'c',label='Sensor Fusion',alpha = 0.5)
for column in flight_data.columns:
    if 'Wind Direction' in column:
        height = ''.join(filter(str.isdigit, column))
        height = int(height)
        if height in [115, 160, 200]:
            # for i in range(len(flight_data)):
            #     if flight_data[column].iloc[i] != flight_data[column].iloc[i-1]:
            #         axs[1].scatter(flight_data['time'].iloc[i], 360 - 90 - flight_data[column].iloc[i])
            axs[1].plot(flight_data['time'], 360 - 90 - flight_data[column], label=column)

axs[1].legend()
axs[1].grid()
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Wind Direction [deg]')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save or display the figure
plt.savefig('combined_figure.png',dpi =300)  # You can adjust the filename and format as needed
plt.show()