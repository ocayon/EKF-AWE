import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import R_EG_Body
from config import kappa, z0, kite_model, year, month, day

#%%

plt.close('all')



onlyGPS = True
GPSacc = False
GPSva = False
GPSaccva = False
GPSvw = False
GPSaccvw = False
GPSvavw = False


path = '../results/'+kite_model+'/'
file_name = kite_model+'_'+year+'-'+month+'-'+day

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


#%% Calculate wind speed with pitot tube and wind vanes
wvel_calc = []
wdir_calc = []

measured_aoa = flight_data['kite_angle_of_attack']
measured_ss = flight_data['kite_sideslip_angle']
measured_va = flight_data['kite_apparent_windspeed']
v_kite = flight_data[['kite_0_vx','kite_0_vy','kite_0_vz']].values

meas_roll = flight_data['kite_0_roll']
meas_pitch = flight_data['kite_0_pitch']
meas_yaw = flight_data['kite_0_yaw']-90

measured_aoa = measured_aoa+2
measured_ss = measured_ss
for i in range(len(measured_aoa)):

    
    # Calculate angle of attack based on orientation angles and estimated wind speed
    Transform_Matrix=R_EG_Body(meas_roll[i]/180*np.pi,meas_pitch[i]/180*np.pi,(meas_yaw[i])/180*np.pi)
    #    Transform_Matrix=R_EG_Body(kite_roll[i]/180*np.pi,kite_pitch[i]/180*np.pi,kite_yaw_modified[i])
    Transform_Matrix=Transform_Matrix.T
    
    #X_vector
    ex_kite=Transform_Matrix.dot(np.array([-1,0,0]))
    #Y_vector
    ey_kite=Transform_Matrix.dot(np.array([0,-1,0]))
    #Z_vector
    ez_kite=Transform_Matrix.dot(np.array([0,0,1]))
    
    va_calc= ex_kite*measured_va[i]*np.cos(measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ey_kite*measured_va[i]*np.sin(measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ez_kite*measured_va[i]*np.sin(measured_aoa[i]/180*np.pi)
    vw = va_calc+v_kite[i]
    wvel_calc.append(np.linalg.norm(vw))
    wdir_calc.append(np.arctan2(vw[1],vw[0]))




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


plot_heights = [115,160,200,220,240,250]

for column in flight_data.columns:
    if 'Wind Speed (m/s)' in column:
        height = ''.join(filter(str.isdigit, column))
        height = int(height)
        if height in plot_heights:
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
    
plt.plot(flight_data['time'],flight_data['ground_wind_direction'],'k',label='Ground Sensor')

for column in flight_data.columns:
    if 'Wind Direction' in column:
        height = ''.join(filter(str.isdigit, column))
        height = int(height)
        if height in plot_heights:
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
wvel = resonlyGPS['uf']/kappa*np.log(resonlyGPS['z']/z0)
axs[0].plot(flight_data['time'],wvel,'c',label='Sensor Fusion',alpha = 0.9)

axs[0].plot(flight_data['time'],wvel_calc,'grey',label='Pitot tube + vanes',alpha = 0.3)

for column in flight_data.columns:
    if 'Wind Speed (m/s)' in column:
        height = ''.join(filter(str.isdigit, column))
        height = int(height)
        if height in plot_heights:
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
axs[1].plot(flight_data['time'],resonlyGPS['wdir']*180/np.pi,'c',label='Sensor Fusion',alpha = 0.9)

axs[1].plot(flight_data['time'],np.array(wdir_calc)*180/np.pi,'grey',label='Pitot tube + vanes',alpha = 0.3)
for column in flight_data.columns:
    if 'Wind Direction' in column:
        height = ''.join(filter(str.isdigit, column))
        height = int(height)
        if height in plot_heights:
            # for i in range(len(flight_data)):
            #     if flight_data[column].iloc[i] != flight_data[column].iloc[i-1]:
            #         axs[1].scatter(flight_data['time'].iloc[i], 360 - 90 - flight_data[column].iloc[i])
            axs[1].plot(flight_data['time'], 360 - 90 - flight_data[column], label=column)

axs[1].legend()
axs[1].grid()
axs[1].set_xlabel('Time [s]')
axs[1].set_ylabel('Wind Direction [deg]')
axs[1].set_ylim([np.min(resonlyGPS['wdir'])*180/np.pi-40,np.max(resonlyGPS['wdir'])*180/np.pi+40])

# Adjust layout to prevent overlap
plt.tight_layout()

# Save or display the figure
plt.savefig(file_name+'wind_estimations.png',dpi =300)  # You can adjust the filename and format as needed
plt.show()