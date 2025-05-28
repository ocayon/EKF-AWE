import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style_no_latex
from awes_ekf.plotting.plot_utils import plot_time_series

colors = get_color_list()

def plot_system_performance(results, flight_data, config_data):

    set_plot_style_no_latex()

    plot_mechanical_power(flight_data, results)

    plt.show()




def plot_mechanical_power(flight_data, results):
    """
    Plots the mechanical power of the kite over time.
    
    Parameters:
        flight_data (DataFrame): Data frame containing flight data, including time.
        results (DataFrame): Data frame with results, including mechanical power.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Plot mechanical power
    plot_time_series(
        flight_data,
        flight_data["ground_tether_force"] * flight_data["tether_reelout_speed"],
        ax,
        label="Mechanical Power",
        plot_phase=False,
        color=colors[0],
    )
    ax.legend()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mechanical Power [W]")
    plt.tight_layout()

    dt = np.diff(flight_data["time"], prepend=flight_data["time"].iloc[0])
    mechanical_power = flight_data["ground_tether_force"] * flight_data["tether_reelout_speed"]
    energy = np.sum(mechanical_power * dt)
    print("Mechanical Energy [J]:", energy)
    print("Mechanical Power [W]:", energy / flight_data["time"].iloc[-1])

    cycle_powers = []
    cycle_energys = []
    cycle_energies_reelin = []
    cycle_energies_reelout = []
    # Calculate mechanical power per cycle
    for cycle in range(0,int(flight_data["cycle"].max())+1):
        cycle_mask = flight_data["cycle"] == cycle
        cycle_time = flight_data.loc[cycle_mask, "time"]
        cycle_power = mechanical_power[cycle_mask]
        cycle_energy = np.sum(cycle_power * dt[cycle_mask])
        cycle_energy_reelin = np.sum(cycle_power[cycle_power > 0] * dt[cycle_mask][cycle_power > 0])
        cycle_energy_reelout = np.sum(cycle_power[cycle_power < 0] * dt[cycle_mask][cycle_power < 0])
        cycle_energys.append(cycle_energy)
        cycle_powers.append(cycle_energy / (cycle_time.iloc[-1] - cycle_time.iloc[0]))
        cycle_energies_reelin.append(cycle_energy_reelin)
        cycle_energies_reelout.append(cycle_energy_reelout)
    
    print("Mean Cycle Power [W]:", np.mean(cycle_powers))
    print("Std Cycle Power [W]:", np.std(cycle_powers))
    print("Mean Cycle Energy [J]:", np.mean(cycle_energys))
    print("Std Cycle Energy [J]:", np.std(cycle_energys))

    print("Mean Cycle Energy Reelin [J]:", np.mean(cycle_energies_reelin))
    print("Std Cycle Energy Reelin [J]:", np.std(cycle_energies_reelin))
    print("Mean Cycle Energy Reelout [J]:", np.mean(cycle_energies_reelout))
    print("Std Cycle Energy Reelout [J]:", np.std(cycle_energies_reelout))