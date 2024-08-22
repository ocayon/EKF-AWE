import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style


def plot_kite_aero(config_data: dict):
    # Load results and flight data and plot kite reference frame
    results, flight_data,_ = read_results(str(config_data['year']), str(config_data['month']), str(config_data['day']), config_data['kite']['model_name'])

    kite_sensors = config_data['kite']['sensor_ids']
    cycles_plotted = np.arange(6,70, step=1)
    # %% Plot results aerodynamic coefficients
    pu.plot_aero_coeff_vs_aoa_ss(results, flight_data, cycles_plotted,kite_sensors,savefig=False) # Plot aero coeff vs aoa_ss
    pu.plot_aero_coeff_vs_up_us(results, flight_data, cycles_plotted,IMU_0=False,savefig=False) # Plot aero coeff vs up_used


    #%% Polars
    mask = np.any(
        [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
    pu.plot_cl_curve(np.sqrt((results["wing_lift_coefficient"]**2+results["wing_sideforce_coefficient"]**2)), results["wing_drag_coefficient"], results['wing_angle_of_attack'], mask,axs, label = "Wing from EKF")
    pu.plot_cl_curve(np.sqrt((results["wing_lift_coefficient"]**2+results["wing_sideforce_coefficient"]**2)), results["wing_drag_coefficient"], results['wing_angle_of_attack_bridle'], mask,axs, label = "Wing from bridle")
    pu.plot_cl_curve(np.sqrt((results["wing_lift_coefficient"]**2+results["wing_sideforce_coefficient"]**2)), results["wing_drag_coefficient"], (results['wing_angle_of_attack_imu_0']), mask,axs, label = "IMU 0")


    # axs[0,0].axvline(x = np.mean(aoa_plot[flight_data['powered'] == 'powered']), color = 'k',linestyle = '--', label = 'Mean reel-out angle of attack')
    # axs[0,0].axvline(x = np.mean(aoa_plot[flight_data['powered'] == 'depowered']), color = 'b',linestyle = '--', label = 'Mean reel-in angle of attack')


    axs[0,0].legend()
    fig.suptitle('cl_wing vs cd_wing of the kite wing (without KCU and tether drag)')
    #plt.show()

    #%%
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    aoa_plot=results['wing_angle_of_attack_bridle']
    mask =  mask & (flight_data['up'] > 0.9) | (flight_data['up'] < 0.1)
    pu.plot_cl_curve(np.sqrt((results["wing_lift_coefficient"]**2+results["wing_sideforce_coefficient"]**2)), results["wing_drag_coefficient"], aoa_plot, mask,axs, label = "Straight")
    # mask = (flight_data['turn_straight'] == 'turn') & mask
    # pu.plot_cl_curve(np.sqrt((results["wing_lift_coefficient"]**2+results["wing_sideforce_coefficient"]**2)), results["wing_drag_coefficient"]+results["tether_drag_coefficient"]+results["kcu_drag_coefficient"], aoa_plot, mask,axs, label = "Turn")
    axs[0,0].axvline(x = np.mean(aoa_plot[flight_data['powered'] == 'powered']), color = 'k',linestyle = '--', label = 'Mean reel-out angle of attack')
    axs[0,0].axvline(x = np.mean(aoa_plot[flight_data['powered'] == 'depowered']), color = 'b',linestyle = '--', label = 'Mean reel-in angle of attack')
    axs[0,0].legend()
    fig.suptitle('cl_wing vs cd_wing of the system (incl. KCU and tether drag)')

    #%%
    plt.figure()
    plt.plot(flight_data['time'],flight_data['ground_tether_force'],label = 'Measured ground')
    plt.plot(results['time'],results['tether_force_kite'],label = 'Estimated at kite')
    for column in flight_data.columns:
        if 'load_cell' in column:
            plt.plot(flight_data['time'],flight_data[column]*9.81,label = column)
    plt.plot(flight_data['time'],flight_data['tether_reelout_speed'],label = 'Reelout speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.title('Tether force comparison')
    #plt.show()

    plt.show()

    #%% Find delay cs_wing with us

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

    mask = flight_data['up'] > 0.8
    signal_1 = -flight_data['us'][mask]
    signal_2 = results["wing_sideforce_coefficient"][mask]

    time_delay,cross_corr = find_time_delay(signal_1, signal_2)
    # Plot the signals and their cross-correlation
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1, 1.5]})

    # Share x-axis between the first two subplots
    axs[0].plot(signal_1, label='us')
    axs[0].legend()
    axs[0].set_title('Signal 1')

    axs[1].plot(signal_2, label='cs_wing')
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

    #%% Plot kite velocity
    plt.figure()
    kite_speed = np.sqrt(results['kite_velocity_x']**2+results['kite_velocity_y']**2+results['kite_velocity_z']**2)
    meas_kite_speed = np.sqrt(flight_data['kite_velocity_x']**2+flight_data['kite_velocity_y']**2+flight_data['kite_velocity_z']**2)
    plt.plot(results['time'],kite_speed,label = 'Estimated')
    plt.plot(flight_data['time'],meas_kite_speed,label = 'Measured')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.title('Kite speed comparison')


if __name__ == "__main__":
    # Example usage
    plt.close('all')
    config_file_name = "v3_config.yaml"
    config = load_config("examples/" + config_file_name)
    plot_kite_aero(config)
    plt.show()


