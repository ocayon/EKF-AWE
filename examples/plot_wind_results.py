import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
from awes_ekf.plotting.color_palette import get_color_list, visualize_palette, set_plot_style


def plot_wind_results(config_data: dict) -> None:
    # Load results and flight data and plot kite reference frame
    results, flight_data = read_results(str(config_data['year']), str(config_data['month']), str(config_data['day']), config_data['kite']['model_name'])

    #%%Plot results wind speed

    pu.plot_wind_speed(results,flight_data,savefig=False) # PLot calculated wind speed against lidar
    #%%
    pu.plot_wind_speed_height_bins(results,flight_data, savefig=False) # Plot calculated wind speed against lidar
    #%%
    axs = pu.plot_wind_profile_bins(flight_data, results, step = 10, savefig = False)


    #%% Plot wind energy spectrum
    fs = 10
    mask = range(len(results))#(results['kite_pos_z']>140)&(results['kite_pos_z']<180)
    signal = np.array(results["wind_speed_horizontal"][mask])
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
