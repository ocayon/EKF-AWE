import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
from awes_ekf.utils import calculate_reference_frame_euler
from awes_ekf.plotting import plot_kite_trajectory
import awes_ekf.plotting.plot_utils as pu


def plot_kite_trajectories(config_data: dict) -> None:
    # Load results and flight data and plot kite reference frame
    results, flight_data,_ = read_results(str(config_data['year']), str(config_data['month']), str(config_data['day']), config_data['kite']['model_name'])

    mask = (flight_data.cycle == 65)
    # mask = (flight_data.index > 1000)
    mask = (flight_data.time > 10)
    flight_data = flight_data[mask]
    results = results[mask]
    # Calculate variables and vectors for plotting
    ex = []
    for i in np.arange(0, len(flight_data)):
        dcm = calculate_reference_frame_euler(
            results.kite_roll.iloc[i],
            results.kite_pitch.iloc[i],
            results.kite_yaw.iloc[i],
            eulerFrame="NED",
            outputFrame="ENU",
        )
        ex.append(dcm[:, 0])

    label_variables = [['kite_velocity'], ['CL'], ['Tether_force'], ["Mechanic power"],['us'],['up']]
    t = flight_data['time'].values
    x = results['kite_position_x'].values
    y = results['kite_position_y'].values
    z = results['kite_position_z'].values
    kite_velocity = np.sqrt(results['kite_velocity_x']**2 + results['kite_velocity_y']**2 + results['kite_velocity_z']**2)
    variables = [
        kite_velocity.values,
        results['wing_lift_coefficient'].values,
        [flight_data['ground_tether_force'].values],
        [flight_data['ground_tether_force'].values*flight_data['tether_reelout_speed'].values],
        flight_data['us'].values,
        flight_data['up'].values,
    ]

    # Plot kite trajectory
    plot_kite_trajectory(t, x, y, z, variables=variables, labels=label_variables, vecs=[ex])

    label_variables = [['kite_elevation'], ['kite_azimuth'], ['Reelout_speed'], ["Wind speed"]]
    t = flight_data['time'].values
    x = results['kite_position_x'].values
    y = results['kite_position_y'].values
    z = results['kite_position_z'].values
    kite_velocity = np.sqrt(results['kite_velocity_x']**2 + results['kite_velocity_y']**2 + results['kite_velocity_z']**2)
    variables = [
        np.degrees(flight_data['kite_elevation'].values),
        np.degrees(flight_data['kite_azimuth'].values),
        flight_data['tether_reelout_speed'].values,
        results['wind_speed_horizontal'].values,
    ]

    # Plot kite trajectory
    plot_kite_trajectory(t, x, y, z, variables=variables, labels=label_variables, vecs=[ex])

    # Plot kite trajectory
    fig,ax = plt.subplots()
    pu.plot_time_series(flight_data, flight_data['kite_position_z'], ax, color='blue',ylabel='Height', label='Measured',plot_phase=False)
    pu.plot_time_series(flight_data,results['kite_position_z'], ax, color='red', label='Estimated',plot_phase=True)
    ax.grid()

    fig,ax = plt.subplots()
    pu.plot_time_series(flight_data, flight_data['kite_position_x'], ax, color='blue', ylabel='x-east', label='Measured',plot_phase=False)
    pu.plot_time_series(flight_data,results['kite_position_x'], ax, color='red', label='Estimated',plot_phase=True)
    ax.grid()

    fig,ax = plt.subplots()
    pu.plot_time_series(flight_data, flight_data['kite_position_y'], ax, color='blue', ylabel='y-north', label='Measured',plot_phase=False)
    pu.plot_time_series(flight_data,results['kite_position_y'], ax, color='red', label='Estimated',plot_phase=True)
    ax.grid()


if __name__ == "__main__":
    # Example usage
    plt.close('all')
    config_file_name = "v3_config.yaml"
    config = load_config("examples/" + config_file_name)
    plot_kite_trajectories(config)
    plt.show()
