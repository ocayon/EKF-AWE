import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from awes_ekf.setup.kite import Kite
from awes_ekf.setup.settings import load_config
import seaborn as sns
import awes_ekf.plotting.plot_utils as pu
from awes_ekf.postprocess.postprocessing import  postprocess_results
from awes_ekf.load_data.read_data import read_results
#%%
year = '2024'
month = '06'
day = '05'
kite_model = 'v9'                   

plt.close('all')

results,flight_data = read_results(year, month, day, kite_model)

plot_lidar_heights= [100,160,200,250]

config_data = load_config('examples/v9_config.yaml')

kite = Kite(**config_data['kite'])
imus = [0]

# flight_data['kite_0_pitch'] = (flight_data['kite_0_pitch']+flight_data['kite_1_pitch'])/2
#%%
results, flight_data = postprocess_results(results,flight_data, kite, imus = [0], remove_IMU_offsets=True, 
                                            correct_IMU_deformation = True,remove_vane_offsets=True,estimate_kite_angle=True)

for column in results.columns:
    if 'pitch' in column or 'roll' in column or 'yaw' in column:
        results[column] = np.degrees(results[column])

for column in flight_data.columns:
    if 'pitch' in column or 'roll' in column or 'yaw' in column:
        flight_data[column] = np.degrees(flight_data[column])

# # #%%
# flight_data = calculate_wind_speed_airborne_sensors(results,flight_data, imus = [0])
# Postprocess done

#%%Plot results wind speed

pu.plot_wind_speed(results,flight_data, plot_lidar_heights,savefig=False) # PLot calculated wind speed against lidar
#%%
pu.plot_wind_speed_height_bins(results,flight_data, plot_lidar_heights, savefig=False) # Plot calculated wind speed against lidar

#%%
# pu.plot_wind_profile(flight_data, results, savefig=False) # Plot wind profile

axs = pu.plot_wind_profile_bins(flight_data.iloc[2000:-2000], results.iloc[2000:-2000], step = 10, savefig = False)

# windpath = '../processed_data/era5_data/'
# windfile = 'era5_data_'+year+'_'+month+'_'+day+'.npy'

# data_dict = np.load(windpath+windfile, allow_pickle=True)

# # Extract arrays and information
# era5_hours = data_dict.item()['hours']
# era5_heights = data_dict.item()['heights']
# era5_wvel = data_dict.item()['wvel']
# era5_wdir = data_dict.item()['wdir']
# for i in range(len(era5_hours)-1):
#     axs[0].fill_betweenx(era5_heights[:-2], era5_wvel[i,:-2], era5_wvel[i+1,:-2], color='lightblue', alpha=0.5)
#     axs[1].fill_betweenx(era5_heights[:-2], era5_wdir[i,:-2], era5_wdir[i+1,:-2], color='lightblue', alpha=0.5)

# %% Plot results aerodynamic coefficients

# ################## Time series ##################
cycles_plotted = np.arange(0,100,1)
pu.plot_aero_coeff_vs_aoa_ss(results, flight_data, cycles_plotted,IMU_0=True,savefig=False) # Plot aero coeff vs aoa_ss
pu.plot_aero_coeff_vs_up_us(results, flight_data, cycles_plotted,IMU_0=False,savefig=False) # Plot aero coeff vs up_used
#%%
################## Density plots ##################
# flight_data = flight_data.iloc[1000::]
# results = results.iloc[1000::]
# cycles_plotted = np.arange(0, 65, step=1)
# mask = np.any(
#     [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
# mask = (flight_data['up'] < 0.2)&mask#&(flight_data['up'] < 0.04)&(flight_data['turn_straight'] == 'straight')#&(results['CD']>0.03)
# pu.plot_CL_CD_aoa(results,flight_data, mask, 'EKF',savefig=False) # Plot CL vs CD for different aoa
# pu.plot_CL_CD_up(results,flight_data, mask, 'EKF',savefig=False) # Plot CL vs CD for different aoa
# pu.plot_CL_CD_ss(results,flight_data, mask, 'EKF')    # Plot CL vs CD for different aoa_ss
# pu.plot_prob_coeff_vs_aoa_ss(results, results.CL**3/results.CD**2, mask, 'EKF') # Plot CL^3/CD^2 vs aoa_ss
# pu.plot_prob_coeff_vs_aoa_ss(results, results.CL/(results.CD), mask, 'EKF') # Plot CL/CD vs aoa_ss


#%% Time series
fig,ax = plt.subplots()
pu.plot_time_series(flight_data, flight_data['kite_apparent_windspeed'], ax, color='blue', label='Measured',plot_phase=False)
pu.plot_time_series(flight_data,results['va_kite'], ax, color='red', label='Estimated',plot_phase=True)
ax.grid()

fig,ax = plt.subplots()
pu.plot_time_series(flight_data, flight_data['kite_apparent_windspeed'], ax, color='blue', label='Measured',plot_phase=False)
pu.plot_time_series(flight_data,results['va_kite'], ax, color='red', label='Estimated',plot_phase=True)
ax.grid()


# fig,ax = plt.subplots()
# r_kite = np.vstack((np.array(flight_data['kite_0_rx']),np.array(flight_data['kite_0_ry']),np.array(flight_data['kite_0_rz']))).T
# r_kite = np.linalg.norm(r_kite,axis = 1)
# pu.plot_time_series(flight_data, r_kite,'Kite radius', ax, color='blue', label='GPS radius',plot_phase=False)
# pu.plot_time_series(flight_data,flight_data['ground_tether_length'],'Tether length', ax, color='red', label='Tether length',plot_phase=True)
# ax.grid()
fig,ax = plt.subplots()
pu.plot_time_series(flight_data, flight_data['kite_0_rx'], ax, color='blue', label='Measured',plot_phase=False)
pu.plot_time_series(flight_data,results['kite_pos_x'], ax, color='red', label='Estimated',plot_phase=False)
ax.grid()
ax.legend()
fig,ax = plt.subplots()
pu.plot_time_series(flight_data, flight_data['kite_0_ry'], ax, color='blue', label='Measured',plot_phase=False)
pu.plot_time_series(flight_data,results['kite_pos_y'], ax, color='red', label='Estimated',plot_phase=False)
ax.grid()
ax.legend()
fig,ax = plt.subplots()
pu.plot_time_series(flight_data, flight_data['kite_0_rz'], ax, color='blue', label='Measured',plot_phase=False)
pu.plot_time_series(flight_data,results['kite_pos_z'], ax, color='red', label='Estimated',plot_phase=False)
ax.grid()
ax.legend()
#%%
fig,ax = plt.subplots()
pu.plot_time_series(flight_data, flight_data['kite_0_pitch'], ax, color='blue', label='Measured',plot_phase=False)
# pu.plot_time_series(flight_data, flight_data['kite_1_pitch'], ax, color='blue', label='Measured',plot_phase=False)
pu.plot_time_series(flight_data,results['kite_pitch'], ax, color='red', label='Estimated',plot_phase=False)
ax.grid()
ax.legend()

fig,ax = plt.subplots()
pu.plot_time_series(flight_data, flight_data['kite_0_roll'], ax, color='blue', label='Measured',plot_phase=False)
pu.plot_time_series(flight_data,results['kite_roll'], ax, color='red', label='Estimated',plot_phase=False)
ax.grid()
ax.legend()


fig,ax = plt.subplots()
pu.plot_time_series(flight_data, flight_data['kite_0_yaw'], ax, color='blue', label='Measured',plot_phase=False)
pu.plot_time_series(flight_data,results['kite_yaw'], ax, color='red', label='Estimated',plot_phase=False)
ax.grid()
ax.legend()
#%%

mask = range(3600,4000)
pu.plot_kite_reference_frame(results.iloc[mask], flight_data.iloc[mask], imus)


#%%
r = np.sqrt(results.kite_pos_x**2+results.kite_pos_y**2+results.kite_pos_z**2)
mechanic_power = []
slack = []
for cycle in range(0,int(max(np.array(flight_data['cycle'])))):
    mask = flight_data['cycle'] == cycle
    mask1 = mask#&(flight_data['powered'] == 'powered')
    mechanic_power.append(np.mean(flight_data['ground_tether_reelout_speed'][mask1]*flight_data['ground_tether_force'][mask1]))

    
    mask= mask&(flight_data['powered'] == 'depowered')
    slack.append(np.max(results['tether_length'][mask]-r[mask]+kite.distance_kcu_kite))

x = np.arange(0,max(np.array(flight_data['cycle'])))
y1 = mechanic_power
y2 = slack
# Create figure and first axis
fig, ax1 = plt.subplots()

# Plot the first dataset with the first y-axis
ax1.plot(x, y1, 'g-', label='Mechanical power (W)')  # 'g-' for green solid line
ax1.set_xlabel('X data')
ax1.set_ylabel('Y1 data', color='g')  # Set the color of y-axis to match the plot
ax1.tick_params('y', colors='g')

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()
# Plot the second dataset with the second y-axis
ax2.plot(x, y2, 'b-', label='Max. slack reelin')  # 'b-' for blue solid line
ax2.set_ylabel('Y2 data', color='b')  # Set the color of y-axis to match the plot
ax2.tick_params('y', colors='b')

# Optional: add a legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.title('Double Y Axis Example')
plt.show()
#%%
azimuth_escape =[]
elevation_escape = []
for i in range(len(flight_data)):
    if flight_data['powered'].iloc[i] == 'depowered' and flight_data['powered'].iloc[i-1] == 'powered':
        azimuth_escape.append(flight_data.kite_azimuth[i])
        elevation_escape.append(flight_data.kite_elevation[i])

plt.figure()
# plt.plot(abs(np.array(azimuth_escape)))
plt.plot(abs(np.array(elevation_escape)))
#%%
fs = 10
mask = range(len(results))#(results['kite_pos_z']>140)&(results['kite_pos_z']<180)
signal = np.array(results['wind_velocity'][mask])
from scipy.signal import detrend
from scipy.stats import linregress

# Assuming your signal is stored in `signal`
fs = 10  # Sampling frequency in Hz, adjusted to 10 Hz based on the timestep of 0.1s
T = len(signal) / fs  # Duration in seconds, calculated based on your signal length
t = np.linspace(0, T, int(T*fs), endpoint=False)  # Time vector, adjusted

# Subtract the mean to focus on fluctuations
signal_fluctuations = signal - np.mean(signal)

# Compute FFT and frequencies
fft_result = np.fft.fft(signal_fluctuations)
fft_freq = np.fft.fftfreq(signal_fluctuations.size, d=1/fs)

# Compute energy spectrum (magnitude squared)
energy_spectrum = np.abs(fft_result)**2/flight_data['time'].iloc[-1]

# Select positive frequencies for plotting and analysis
pos_freq = fft_freq > 0
pos_energy_spectrum = energy_spectrum[pos_freq]
pos_fft_freq = fft_freq[pos_freq]

# Log-log plot
plt.figure(figsize=(10, 6))
plt.loglog(pos_fft_freq, pos_energy_spectrum, label='Energy Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density ($m^2/s^2$)')

# Determine an appropriate subrange and calculate the slope
# Adjust subrange_start and subrange_end based on your data
subrange_start = 1e-2  # Example start frequency
subrange_end = 2e-1   # Example end frequency
subrange_mask = (pos_fft_freq > subrange_start) & (pos_fft_freq < subrange_end)

if np.any(subrange_mask):
    slope, intercept, r_value, p_value, std_err = linregress(np.log(pos_fft_freq[subrange_mask]), np.log(pos_energy_spectrum[subrange_mask]))
    plt.plot(pos_fft_freq[subrange_mask], np.exp(intercept) * pos_fft_freq[subrange_mask] ** slope, 'r--', label=f'Fitted Slope: {slope:.2f}')
    print(f"The calculated slope is: {slope:.2f}")
else:
    print("No data in the specified subrange. Please adjust the subrange criteria.")

slope = -5/3
plt.plot(pos_fft_freq[subrange_mask], np.exp(intercept) * pos_fft_freq[subrange_mask] ** slope, 'g--', label=f'Kolmogorov: -5/3')
plt.legend()
plt.show()

#%%
#read the data
data = pd.read_csv('processed_data/aerostructural/v3_aero_coeffs_VSM.csv')
cl_VSM = data['CL']
cd_VSM = data['CD']
aoa_VSM = np.degrees(data['aoa'])


aoa_plot = results['kite_aoa']
# aoa_plot = results['aoa_IMU_0']
# aoa_plot = flight_data['kite_angle_of_attack']
cycles_plotted = np.arange(2,55, step=1)
mask = np.any(
    [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
mask_angles =mask&(flight_data['kite_angle_of_attack']>0) & (flight_data['kite_angle_of_attack']<35)#& (flight_data['powered'] == 'powered')

# mask_angles =(results['aoa']>0) & (results['aoa']<20)
fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
mask = (flight_data['turn_straight'] == 'straight') & mask_angles & (flight_data.index>5000)#&(flight_data.index<len(results)-15000)
pu.plot_cl_curve(np.sqrt((results['CL']**2+results['CS']**2)), results['CD'], aoa_plot, mask,axs, label = "Straight")
mask = (flight_data['turn_straight'] == 'turn')& mask_angles& (flight_data.index>5000)#&(flight_data.index<len(results)-15000)
pu.plot_cl_curve(np.sqrt((results['CL']**2+results['CS']**2)), results['CD'], aoa_plot, mask,axs, label = "Turn")

axs[0,0].axvline(x = np.mean(aoa_plot[flight_data['powered'] == 'powered']), color = 'k',linestyle = '--', label = 'Mean reel-out angle of attack')
axs[0,0].axvline(x = np.mean(aoa_plot[flight_data['powered'] == 'depowered']), color = 'b',linestyle = '--', label = 'Mean reel-in angle of attack')
# axs[0,0].axvline(x = np.mean(results['aoa'][flight_data['powered'] == 'powered']), color = 'k',linestyle = '--', label = 'Mean reel-out angle of attack')
# axs[0,0].axvline(x = np.mean(results['aoa'][flight_data['powered'] == 'depowered']), color = 'b',linestyle = '--', label = 'Mean reel-in angle of attack')
# pu.plot_cl_curve(cl_VSM, cd_VSM, aoa_VSM,(cl_VSM>0) ,axs, label = "VSM")
# axs[0,0].scatter(aoa_VSM,cl_VSM)
# axs[0,1].scatter(aoa_VSM,cd_VSM)
# axs[0,0].fill_betweenx(y=np.linspace(0.4, 1, 100), x1=15, x2=40, color='red', alpha=0.3)
# axs[0,1].fill_betweenx(y=np.linspace(0, 0.4, 100), x1=15, x2=40, color='red', alpha=0.3)

axs[0,0].legend()
fig.suptitle('CL vs CD of the kite wing (without KCU and tether drag)')
plt.show()

#%%

cl_VSM = data['CL']
cd_VSM = data['CD']
aoa_VSM = np.degrees(data['aoa'])
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
mask = (flight_data['turn_straight'] == 'straight') & (flight_data.index>5000)& mask_angles
# pu.plot_cl_curve(results['CL'], results['CD']+results['cd_tether']+results['cd_kcu'], results['aoa'], mask,axs, label = "Straight")
pu.plot_cl_curve(np.sqrt((results['CL']**2+results['CS']**2)), results['CD']+results['cd_tether']+results['cd_kcu'], flight_data['kite_angle_of_attack'], mask,axs, label = "Straight")
mask = (flight_data['turn_straight'] == 'turn') & (flight_data.index>5000)& mask_angles
# pu.plot_cl_curve(results['CL'], results['CD']+results['cd_tether']+results['cd_kcu'], results['aoa'], mask,axs, label = "Turn")
pu.plot_cl_curve(np.sqrt((results['CL']**2+results['CS']**2)), results['CD']+results['cd_tether']+results['cd_kcu'], flight_data['kite_angle_of_attack'], mask,axs, label = "Turn")
# pu.plot_cl_curve(cl_VSM, cd_VSM, aoa_VSM,(cl_VSM>0) ,axs, label = "VSM")
# axs[0,0].fill_betweenx(y=np.linspace(0.4, 1, 100), x1=15, x2=40, color='red', alpha=0.3)
# axs[0,1].fill_betweenx(y=np.linspace(0, 0.4, 100), x1=15, x2=40, color='red', alpha=0.3)

axs[0,0].legend()

fig.suptitle('CL vs CD of the system (incl. KCU and tether drag)')

plt.show()



#%%
fig,ax = plt.subplots()
pu.plot_time_series(flight_data, results['kite_roll']-results['kcu_roll'], ax, color='blue', label='Roll',plot_phase=False)
pu.plot_time_series(flight_data,results['kite_pitch']-results['kcu_pitch'], ax, color='red', label='Pitch',plot_phase=True)

ax.legend()


#%% Compute orietnation errors

pitch_error = abs(flight_data['kite_0_pitch']-results['kite_pitch'])
roll_error = abs(flight_data['kite_0_roll']-results['kite_roll'])
yaw_error = abs(flight_data['kite_0_yaw']-results['kite_yaw'])

mean_pitch_error = np.mean(pitch_error)
mean_roll_error = np.mean(roll_error)
mean_yaw_error = np.mean(yaw_error)

std_pitch_error = np.std(pitch_error)
std_roll_error = np.std(roll_error)
std_yaw_error = np.std(yaw_error)

print(f'Mean pitch error: {mean_pitch_error:.2f} deg, std: {std_pitch_error:.2f} deg')
print(f'Mean roll error: {mean_roll_error:.2f} deg, std: {std_roll_error:.2f} deg')
print(f'Mean yaw error: {mean_yaw_error:.2f} deg, std: {std_yaw_error:.2f} deg')


#%% Plot turbulence intensity
mask = (results['kite_pos_z']>150)&(results['kite_pos_z']<170)
TI_160m = []
for i in range(len(results)):
    if i<600:
        std = np.std(results['wind_velocity'].iloc[0:i][mask])
        mean = np.mean(results['wind_velocity'].iloc[0:i][mask])
    else:
        std = np.std(results['wind_velocity'].iloc[i-600:i][mask])
        mean = np.mean(results['wind_velocity'].iloc[i-600:i][mask])

    TI_160m.append(std/mean)
#%%
TI_160m_lidar = flight_data['160m Wind Speed Dispersion (m/s)']/flight_data['160m Wind Speed (m/s)']
plt.figure()
plt.plot(TI_160m)
plt.plot(TI_160m_lidar)
plt.legend(['EKF','Lidar'])
plt.xlabel('Time (s)')
plt.ylabel('Turbulence intensity')
plt.title('Turbulence intensity at 160m altitude')



