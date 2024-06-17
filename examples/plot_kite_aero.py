import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu

# Example usage
plt.close('all')
config_file_name = "v3_config.yaml"
config = load_config("examples/" + config_file_name)

# Load results and flight data and plot kite reference frame
results, flight_data = read_results(str(config['year']), str(config['month']), str(config['day']), config['kite']['model_name'])

cycles_plotted = np.arange(6,50, step=1)
# %% Plot results aerodynamic coefficients
pu.plot_aero_coeff_vs_aoa_ss(results, flight_data, cycles_plotted,IMU_0=False,savefig=False) # Plot aero coeff vs aoa_ss
pu.plot_aero_coeff_vs_up_us(results, flight_data, cycles_plotted,IMU_0=False,savefig=False) # Plot aero coeff vs up_used


#%% Polars
aoa_plot = results['kite_aoa']
# aoa_plot = results['aoa_IMU_0']
# aoa_plot = flight_data['kite_angle_of_attack']
mask = np.any(
    [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
mask_angles =mask#&((aoa_plot>0) & (aoa_plot<15))

# mask_angles =(results['aoa']>0) & (results['aoa']<20)
fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
mask = (flight_data['turn_straight'] == 'straight') & mask_angles 
pu.plot_cl_curve(np.sqrt((results['CL']**2+results['CS']**2)), results['CD'], aoa_plot, mask,axs, label = "Straight")
mask = (flight_data['turn_straight'] == 'turn')& mask_angles
pu.plot_cl_curve(np.sqrt((results['CL']**2+results['CS']**2)), results['CD'], aoa_plot, mask,axs, label = "Turn")

axs[0,0].axvline(x = np.mean(aoa_plot[flight_data['powered'] == 'powered']), color = 'k',linestyle = '--', label = 'Mean reel-out angle of attack')
axs[0,0].axvline(x = np.mean(aoa_plot[flight_data['powered'] == 'depowered']), color = 'b',linestyle = '--', label = 'Mean reel-in angle of attack')


axs[0,0].legend()
fig.suptitle('CL vs CD of the kite wing (without KCU and tether drag)')
#plt.show()

#%%
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
aoa_plot=flight_data['kite_angle_of_attack']
mask = (flight_data['turn_straight'] == 'straight') & mask_angles & (flight_data['up'] > 0.9) | (flight_data['up'] < 0.1)
pu.plot_cl_curve(np.sqrt((results['CL']**2+results['CS']**2)), results['CD']+results['cd_tether']+results['cd_kcu'], aoa_plot, mask,axs, label = "Straight")
mask = (flight_data['turn_straight'] == 'turn') & mask_angles
pu.plot_cl_curve(np.sqrt((results['CL']**2+results['CS']**2)), results['CD']+results['cd_tether']+results['cd_kcu'], aoa_plot, mask,axs, label = "Turn")
axs[0,0].axvline(x = np.mean(aoa_plot[flight_data['powered'] == 'powered']), color = 'k',linestyle = '--', label = 'Mean reel-out angle of attack')
axs[0,0].axvline(x = np.mean(aoa_plot[flight_data['powered'] == 'depowered']), color = 'b',linestyle = '--', label = 'Mean reel-in angle of attack')
axs[0,0].legend()
fig.suptitle('CL vs CD of the system (incl. KCU and tether drag)')

#%%
plt.figure()
plt.plot(flight_data['time'],flight_data['ground_tether_force'],label = 'Measured ground')
plt.plot(results['time'],results['tether_force_kite'],label = 'Estimated at kite')
for column in flight_data.columns:
    if 'load_cell' in column:
        plt.plot(flight_data['time'],flight_data[column]*9.81,label = column)
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.title('Tether force comparison')
#plt.show()

plt.show()

#%% Find delay CS with us

def find_time_delay(signal_1,signal_2):
    # Compute the cross-correlation
    cross_corr = np.correlate(signal_2, signal_1, mode='full')

    # Find the index of the maximum value in the cross-correlation
    max_corr_index = np.argmax(cross_corr)

    # Compute the time delay
    time_delay = (max_corr_index - (len(signal_1) - 1))*0.1

    # Print the time delay
    print(f'Time delay between the two signals is {time_delay} seconds.')

    return time_delay, cross_corr

        
signal_1 = -flight_data['us']
signal_2 = results['CS']

time_delay,cross_corr = find_time_delay(signal_1, signal_2)
# Plot the signals and their cross-correlation
fig, axs = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1, 1.5]})

# Share x-axis between the first two subplots
axs[0].plot(signal_1, label='us')
axs[0].legend()
axs[0].set_title('Signal 1')

axs[1].plot(signal_2, label='CS')
axs[1].legend()
axs[1].set_title('Signal 2')
axs[1].sharex(axs[0])

# For cross-correlation, set the x-axis to match the number of samples
x_corr = np.arange(-len(signal_1) + 1, len(signal_1))
axs[2].plot(x_corr, cross_corr, label='Cross-correlation')
axs[2].axvline(x=time_delay*10, color='r', linestyle='--', label='Max correlation index')
axs[2].legend()
axs[2].set_title('Cross-correlation')

plt.tight_layout()

