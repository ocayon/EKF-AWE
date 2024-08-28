import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
from awes_ekf.plotting.color_palette import get_color_list, visualize_palette, set_plot_style

set_plot_style()
def plot_wind_results(config_data: dict) -> None:
    # Load results and flight data and plot kite reference frame
    results, flight_data,_ = read_results(str(config_data['year']), str(config_data['month']), str(config_data['day']), config_data['kite']['model_name'])
    # res_va, fd_va, _ = read_results(str(config_data['year']), str(config_data['month']), str(config_data['day']), config_data['kite']['model_name'],addition='_va')
    # res_log, fd_log, _ = read_results(str(config_data['year']), str(config_data['month']), str(config_data['day']), config_data['kite']['model_name'],addition='_log')
    # res_min, fd_min, _ = read_results(str(config_data['year']), str(config_data['month']), str(config_data['day']), config_data['kite']['model_name'],addition='_min')

    results.loc[results['wind_direction'] > np.radians(300), 'wind_direction'] -= np.radians(360)
    cut = 1000
    results = results.iloc[cut::]
    flight_data = flight_data.iloc[cut::]
 
    results = results.reset_index(drop=True)
    flight_data = flight_data.reset_index(drop=True)

    #%%Plot results wind speed
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    pu.plot_wind_speed(results,flight_data,axs,savefig=False) # PLot calculated wind speed against lidar


    # flight_data = pu.interpolate_lidar_data(flight_data, results)
    # fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    # pu.plot_wind_speed_log_interpolation(results,flight_data,axs,savefig=False) # PLot calculated wind speed against lidar


    #%%
    # pu.plot_wind_speed_height_bins(results,flight_data, savefig=False) # Plot calculated wind speed against lidar
    
    #%%
    fig, axs = plt.subplots(1, 2, figsize=(8, 6))
    pu.plot_wind_profile_bins(flight_data, results, axs, step=10, color="Blue", label="Min. + $l_t$", lidar_data=True)
    # pu.plot_wind_profile_bins(flight_data, res_va, axs, step=10, color="Orange", label="Min. + $v_a$")
    # pu.plot_wind_profile_bins(flight_data, res_log, axs, step=10, color="Green", label="Min. + Log Profile Law")

    
    # fig, axs = plt.subplots(1, 2, figsize=(8, 6))

    # axs[0].plot(flight_data["interp_wind_speed"], results["wind_speed_horizontal"], '.', label="Min. + $l_t$", alpha=0.1)
    # axs[0].plot(flight_data["interp_wind_speed"], flight_data["interp_wind_speed"], 'k--', label="1:1 line")
    # flight_data["interp_wind_direction"] = 270-np.array(flight_data["interp_wind_direction"])
    # axs[1].plot(flight_data["interp_wind_direction"], results["wind_direction"]*180/np.pi, '.', label="Min. + $l_t$", alpha=0.1)
    # axs[1].plot(flight_data["interp_wind_direction"], flight_data["interp_wind_direction"], 'k--', label="1:1 line")


    plt.show()
    #%% Plot wind energy spectrum
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    pu.plot_kinetic_energy_spectrum(results, flight_data,ax, savefig=False)    


    #%% Plot turbulence intensity
    # mask = (results['kite_pos_z']>150)&(results['kite_pos_z']<170)
    # TI_160m = []
    # for i in range(len(results)):
    #     if i<600:
    #         std = np.std(results["wind_speed_horizontal"].iloc[0:i][mask])
    #         mean = np.mean(results["wind_speed_horizontal"].iloc[0:i][mask])
    #     else:
    #         std = np.std(results["wind_speed_horizontal"].iloc[i-600:i][mask])
    #         mean = np.mean(results["wind_speed_horizontal"].iloc[i-600:i][mask])

    #     TI_160m.append(std/mean)
    # #%%
    # TI_160m_lidar = flight_data['160m Wind Speed Dispersion (m/s)']/flight_data['160m Wind Speed (m/s)']
    # plt.figure()
    # plt.plot(TI_160m)
    # plt.plot(TI_160m_lidar)
    # plt.legend(['EKF','Lidar'])
    # plt.xlabel('Time (s)')
    # plt.ylabel('Turbulence intensity')
    # plt.title('Turbulence intensity at 160m altitude')


if __name__ == "__main__":
    # Example usage
    plt.close('all')
    config_file_name = "v3_config.yaml"
    config = load_config("examples/" + config_file_name)
    plot_wind_results(config)
    plt.show()
