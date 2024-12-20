import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.plotting.color_palette import get_color_list, set_plot_style_no_latex
from awes_ekf.plotting.plot_utils import plot_time_series

colors = get_color_list()

def plot_tether(results, flight_data, config_data):

    set_plot_style_no_latex()
   
    plot_kite_tether_angles(flight_data, results)

    plot_slack_tether_force(results, flight_data)

    plt.show()


def plot_kite_tether_angles(flight_data, results, plot_phase_roll=True, plot_phase_pitch=False):
    """
    Plots the roll and pitch angles between kite and tether over time.
    
    Parameters:
        flight_data (DataFrame): Data frame containing flight data, including time.
        results (DataFrame): Data frame with results, including kite and tether roll and pitch angles.
        colors (list): List of colors for the roll and pitch plots.
        plot_phase_roll (bool): Whether to plot phase information for roll angle.
        plot_phase_pitch (bool): Whether to plot phase information for pitch angle.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    # Plot roll angle between kite and tether
    plot_time_series(
        flight_data,
        np.rad2deg(results["kite_roll"] - results["tether_roll"]),
        ax,
        label="Roll kite-tether",
        plot_phase=plot_phase_roll,
        color=colors[0],
    )

    # Plot pitch angle between kite and tether
    plot_time_series(
        flight_data,
        np.rad2deg(results["kite_pitch"] - results["tether_pitch"]),
        ax,
        label="Pitch kite-tether",
        plot_phase=plot_phase_pitch,
        color=colors[1],
    )

    ax.legend()
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle [deg]")
    plt.tight_layout()

def plot_slack_tether_force(results, flight_data,kcu = None):
    
    # Assuming 'slack' is in the 'results' DataFrame and 'tether_force' is in 'flight_data'
    time = flight_data['time']  # Replace with the appropriate time variable
    # r = np.sqrt(results["kite_position_x"]**2 + results["kite_position_y"]**2 + results["kite_position_z"]**2)
    # slack = flight_data['tether_length']+kcu.distance_kcu_kite-r
    slack = results['slack']  # Replace with the appropriate slack variable
    tether_force = flight_data['ground_tether_force']  # Replace with the appropriate tether force variable


    fig, ax1 = plt.subplots(figsize=(9, 3))

    # Plotting Slack on the first axis
    ax1 = plot_time_series(flight_data, slack, ax1, color=colors[0], ylabel='Sag (m)', plot_phase=True)
    ax1.tick_params(axis='y', labelcolor=colors[0])

    # Creating a second y-axis for Tether Force
    ax2 = ax1.twinx()
    ax2.set_ylabel('Tether Force (N)', color=colors[1])
    ax2.plot(time, tether_force, color=colors[1], linestyle='--')
    ax2.tick_params(axis='y', labelcolor=colors[1])
    
    ax1.set_xlabel('Time (s)')
    # Get the current handles and labels
    handles, labels = ax1.get_legend_handles_labels()

    # from matplotlib.patches import Patch
    # # Create a new patch for the legend
    # reel_out_straight_patch = Patch(color=colors[5], alpha=0.2, label="Reel-out - Straight")
    # reel_out_turn_patch = Patch(color=colors[7], alpha=0.2, label="Reel-out - Turn")
    # reel_in_patch = Patch(facecolor='white', alpha=1, edgecolor='black', label="Reel-in")

    # Select starting from the second element
    # ax1.legend(
    #     [reel_out_straight_patch, reel_out_turn_patch, reel_in_patch],
    #     ["Reel-out - Straight", "Reel-out - Turn", "Reel-in"],
    #     loc='upper left',
    #     frameon=True,
    #     bbox_to_anchor=(0.075, 1)  # Adjust the x-coordinate to move the legend to the right
    # )
    from matplotlib.patches import Patch
    # Create a new patch for the legend
    reel_out_straight_patch = Patch(color=colors[5], alpha=0.2, label="Reel-out - Straight")
    reel_out_turn_patch = Patch(color=colors[7], alpha=0.2, label="Reel-out - Turn")
    reel_in_patch = Patch(color='white', alpha=1, label="Reel-in")
    ax1.legend(
        [reel_out_straight_patch, reel_out_turn_patch, reel_in_patch],
        ["Reel-out - Straight", "Reel-out - Turn", "Reel-in"],
        loc='upper left',
        frameon=True,
        bbox_to_anchor=(0.075, 1)  # Adjust the x-coordinate to move the legend to the right
    )
    fig.tight_layout()  # Adjust layout to prevent overlap

    # Title and Grid
    ax2.grid(True, linestyle='--', alpha=0.8)