import numpy as np
import matplotlib.pyplot as plt
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu
import pandas as pd

# Example usage
plt.close("all")
config_file_name = "v3_config.yaml"
config = load_config("examples/" + config_file_name)

# Load results and flight data and plot kite reference frame
cut = 10000
results, flight_data,_ = read_results(
    str(config["year"]),
    str(config["month"]),
    str(config["day"]),
    config["kite"]["model_name"],
)
# results1, flight_data1 = read_results('2023','10','26','v9')
results = results.iloc[cut:-cut]
# results1 = results1.iloc[cut:-cut]

# # concatenate dataframe pandas
# results = pd.concat([results,results1])

flight_data = flight_data.iloc[cut:-cut]
# flight_data1 = flight_data1.iloc[cut:-cut]
# flight_data = pd.concat([flight_data,flight_data1])

# results = results[flight_data['kite_angle_of_attack']<14]
# flight_data = flight_data[flight_data['kite_angle_of_attack']<14]

results = results.reset_index(drop=True)
flight_data = flight_data.reset_index(drop=True)


# %% AERO COEFFICIENTS IDENTIFICATION
def calculate_weighted_least_squares(y, A, W):
    x_hat = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ y
    return x_hat
# def calculate_ls_estimation(alphas, us, up, coeffs):


### Create file with results
alpha = np.array(results["wing_angle_of_attack_bridle"])
ss = np.array(results["wing_sideslip_angle_bridle"])
plt.figure()
plt.plot(alpha)
# alpha = np.array(results['kite_aoa'])
# alpha = results['aoa_IMU_0']
mean_alpha = np.mean(alpha[flight_data['powered'] == 'powered'])
std_alpha = np.std(alpha[flight_data['powered'] == 'powered'])
print(f"Mean alpha powered: {mean_alpha}")
print(f"Std alpha depowered: {std_alpha}")

plt.plot(alpha)
us = np.array(flight_data["us"])
us = np.concatenate((np.zeros(12), us[:-12]))
up = np.array(flight_data["up"])
up = np.concatenate((np.zeros(12), up[:-12]))
p = np.array(results["roll_rate"])
q = np.array(results["pitch_rate"])
yaw_rate = np.array(results["yaw_rate"])
# Create least squares matrix
mass = config['kite']['mass']
yaw = flight_data['kite_yaw_0']
elevation = flight_data['kite_elevation']
# x: [1, alpha, alpha^2, us, us^2, up, up^2]
def construct_A_matrix(us,up,mass,yaw,elevation):
    A = np.vstack([np.ones_like(us), us, us**2, up, up**2, mass*9.81*np.sin(yaw)*np.cos(elevation)]).T
    return A
A = construct_A_matrix(us,up,mass,yaw,elevation)
def construct_A_matrix_alpha_sideslip(alpha,sideslip):
    A = np.vstack([np.ones_like(alpha), alpha, alpha**2, sideslip, sideslip**2]).T
    return A
A = construct_A_matrix_alpha_sideslip(alpha,ss)
W = np.eye(len(alpha))
mask = flight_data['up'] > 0.8
# W[mask, mask] *= 10
# Solve for coefficients
coeffs_CL = calculate_weighted_least_squares(results["wing_lift_coefficient"], A, W)
coeffs_CD = calculate_weighted_least_squares(results["wing_drag_coefficient"], A, W)

# Calculate estimated CL
CL_est = A @ coeffs_CL
CD_est = A @ coeffs_CD

# calculate the mean squared error
mse = np.mean((results["wing_lift_coefficient"] - CL_est) ** 2)
print(f"MSE: {mse}")
print(f"MRSE: {np.sqrt(mse)}")

# calculate the mean squared error
mse = np.mean((results["wing_drag_coefficient"] - CD_est) ** 2)
print(f"MSE: {mse}")
print(f"MRSE: {np.sqrt(mse)}")


# print(
#     f"CL = {coeffs_CL[0]:.3f} + {coeffs_CL[1]:.3f}*alpha + {coeffs_CL[2]:.3f}*alpha^2 + {coeffs_CL[3]:.3f}*us + {coeffs_CL[4]:.3f}*us^2 + {coeffs_CL[5]:.3f}*up + {coeffs_CL[6]:.3f}*up^2 + {coeffs_CL[7]:.3f}*p + {coeffs_CL[8]:.3f}*q"# + {coeffs_CL[9]:.3f}*r"
# )
# print(
#     f"CD = {coeffs_CD[0]:.3f} + {coeffs_CD[1]:.3f}*alpha + {coeffs_CD[2]:.3f}*alpha^2 + {coeffs_CD[3]:.3f}*us + {coeffs_CD[4]:.3f}*us^2 + {coeffs_CD[5]:.3f}*up + {coeffs_CD[6]:.3f}*up^2 + {coeffs_CD[7]:.3f}*p + {coeffs_CD[8]:.3f}*q"# + {coeffs_CD[9]:.3f}*r"
# )


# %%
# Plot the results
plt.figure()
mask = np.bool_(np.ones_like(alpha))

plt.plot(
    flight_data["time"][mask],
    results["wing_lift_coefficient"][mask],
    label="Measured CL",
    color="b",
    alpha=0.5,
)
plt.plot(
    flight_data["time"][mask], CL_est[mask], label="Estimated CL", color="r", alpha=0.5
)
plt.plot(
    flight_data["time"][mask],
    p[mask],
    label="Omega p",
    color="g",
    alpha=0.5,
)
plt.plot(
    flight_data["time"][mask],
    q[mask],
    label="Omega q",
    color="y",
    alpha=0.5,
)
plt.plot(
    flight_data["time"][mask],
    yaw_rate[mask],
    label="Omega r",
    color="k",
    alpha=0.5,
)
plt.xlabel("Time [s]")
plt.ylabel("Lift coefficient")
plt.legend()
plt.grid(True)

# Plot cl angle of attack
plt.figure()
plt.scatter(alpha[mask], results["wing_lift_coefficient"][mask], label="Measured CL", color="b", alpha=0.5)
plt.scatter(alpha[mask], CL_est[mask], label="Estimated CL", color="r", alpha=0.5)
plt.xlabel("Angle of attack [rad]")
plt.ylabel("Lift coefficient")

#Plot cl vs alpha
plt.figure()
aoas_plot = np.arange(0,20)
sideslip = np.zeros_like(aoas_plot)
yaw_rate = np.zeros_like(aoas_plot)
A = construct_A_matrix_alpha_sideslip(aoas_plot,sideslip)
CLest = A @ coeffs_CL
plt.plot(aoas_plot, CLest, label = 'CL vs alpha')

# Plot cd vs alpha
plt.figure()
CDest = A @ coeffs_CD
plt.plot(aoas_plot, CDest, label = 'CD vs alpha')
# plt.show()

# Plot cl/cd
plt.figure()
plt.plot(aoas_plot, CLest/CDest, label = 'CL/CD vs alpha')
plt.show()




# # Plot CL vs control inputs
# us = np.linspace(-1,1,41)
# up = np.zeros_like(us)
# yaw = np.zeros_like(up)
# elevation = np.zeros_like(up)
# A_us = construct_A_matrix(us,up,mass,yaw,elevation)
# CL_us = A_us @ coeffs_CL
# CD_us = A_us @ coeffs_CD
# up = np.linspace(0,1,41)
# us = np.zeros_like(up)
# yaw = np.zeros_like(up)
# elevation = np.zeros_like(up)
# A_up = construct_A_matrix(us,up,mass,yaw,elevation)
# CL_up = A_up @ coeffs_CL
# CD_up = A_up @ coeffs_CD

# us = np.linspace(-1,1,41)
# fig,axs = plt.subplots(2,2,figsize=(12,8))
# axs[0,0].plot(up,CL_up,label='CL vs up')
# axs[0,0].set_ylabel('CL')
# axs[1,0].plot(up,CD_up,label='CD vs up')
# axs[1,0].set_ylabel('CD')
# axs[1,0].set_xlabel('up')
# axs[0,1].plot(up,CL_us,label='CL vs us')
# axs[1,1].plot(up,CD_us,label='CD vs us')
# axs[1,1].set_xlabel('us')
# plt.show()








# # Plot the residuals
# plt.figure()
# plt.plot(
#     flight_data["time"][mask],
#     results["wing_lift_coefficient"][mask] - CL_est[mask],
#     label="Residuals",
#     color="b",
#     alpha=0.5,
# )
# plt.xlabel("Time [s]")
# plt.ylabel("Residuals")
# plt.legend()
# plt.grid(True)

# plt.show()


# # # %% TURN LAW IDENTIFICATION
# # from awes_ekf.old_code.turn_law import calculate_mse
# # def find_time_delay(signal_1,signal_2):
# #     # Compute the cross-correlation
# #     cross_corr = np.correlate(signal_2, signal_1, mode='full')

# #     # Find the index of the maximum value in the cross-correlation
# #     max_corr_index = np.argmax(cross_corr)

# #     # Compute the time delay
# #     time_delay = (max_corr_index - (len(signal_1) - 1))*0.1

# #     # Print the time delay
# #     print(f'Time delay between the two signals is {time_delay} seconds.')

# #     return time_delay, cross_corr

# # init_coeffs = [0,1, 1, 1, 0]
# # us = flight_data['us']
# # # us = np.concatenate((np.zeros(12),us[:-12]))
# # va = results['va_kite']
# # elevation = np.radians(flight_data['kite_elevation'])
# # yaw = flight_data['kite_yaw_s0']
# # v = np.linalg.norm([results['kite_vel_x'], results['kite_vel_y'], results['kite_vel_z']], axis=0)
# # radius = flight_data['radius_turn']
# # forces = ['weight', 'centripetal']

# # kite_yaw = np.deg2rad(flight_data['kite_yaw_s0'])
# # yaw_rate = np.diff(kite_yaw) / 0.1
# # yaw_rate = np.concatenate((yaw_rate, [0]))
# # window_size = 20
# # yaw_rate = np.convolve(yaw_rate, np.ones(window_size)/window_size, mode='same')

# # # Create least squares matrix
# # # x: [area*us*(va**2)/(v*mass)+Cn_i*us*(va)/(span**2), ( mass*v**2/(rho*area*span**2*va*radius)+mass*gravity_constant*np.sin(yaw)*np.cos(beta)/(rho*area*span**2*va))]


# # def calculate_yaw_rate( area,mass,span, us, va, beta, yaw,v,radius,yaw_rate=None,coeffs = None, offset=False):

# #     c1 = us*(va)/span
# #     c2 = ( mass*v**2/(1.225*area*span**2*va*radius)-mass*9.81*np.sin(yaw)*np.cos(beta)/(1.225*area*span**2*va))
# #     A = np.vstack([c1, c2]).T
# #     if offset == True:
# #         c3 = va/span
# #         A = A = np.vstack([c1, c2, c3]).T
# #     # Solve for coefficients
# #     if coeffs is None:
# #         W = np.eye(len(us))
# #         coeffs = calculate_weighted_least_squares(yaw_rate, A, W)


# #     yaw_rate = A@coeffs

# #     return yaw_rate, coeffs

# # yaw_rate_identified, coeffs = calculate_yaw_rate(config['kite']['area'], config['kite']['mass'], config['kite']['span'], us, va, elevation, yaw,v,radius, yaw_rate, offset = True)

# # print(f"coeffs = C_us:{coeffs[0]}+ C_i:{coeffs[1]}")

# # # Calculate the mean squared error
# # mse = calculate_mse(yaw_rate, yaw_rate_identified)
# # print(f"MSE: {mse}")
# # print(f"MRSE: {np.sqrt(mse)}")

# # #%%
# # signal_1 = np.degrees(yaw_rate_identified)
# # # signal_2 = -flight_data['us']*results['va_kite']
# # signal_2 = np.degrees(yaw_rate)

# # time_delay, cross_corr = find_time_delay(signal_1, signal_2)

# # # Plot the signals and their cross-correlation
# # fig, axs = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1, 1.5]})

# # # Share x-axis between the first two subplots
# # axs[0].plot(signal_1, label='Identified yaw rate')
# # axs[0].plot(np.degrees(yaw_rate), label='Measured')
# # axs[0].legend()
# # axs[0].set_title('Signal 1')

# # axs[1].plot(signal_2, label='Signal 2')
# # axs[1].legend()
# # axs[1].set_title('Signal 2')
# # axs[1].sharex(axs[0])

# # # For cross-correlation, set the x-axis to match the number of samples
# # x_corr = np.arange(-len(signal_1) + 1, len(signal_1))
# # axs[2].plot(x_corr, cross_corr, label='Cross-correlation')
# # axs[2].axvline(x=time_delay, color='r', linestyle='--', label='Max correlation index')
# # axs[2].legend()
# # axs[2].set_title('Cross-correlation')

# # plt.tight_layout()

# # kite = config['kite']
# # rho = 1.225
# # area = kite['area']
# # mass = kite['mass']
# # gravity_constant = -9.81
# # span = kite['span']
# # Cn_us = coeffs[0]
# # Cn_i = coeffs[1]

# # yaw_rate_steering = Cn_us*us*(va)/span

# # yaw_rate_inertia = Cn_i *( mass*v**2/(rho*area*span**2*va*radius))
# # yaw_rate_mass = Cn_i*mass*gravity_constant*np.sin(yaw)*np.cos(elevation)/(rho*area*span**2*va)


# # plt.figure()
# # plt.plot(flight_data['time'],yaw_rate_identified*180/np.pi, label = 'yaw rate identified')
# # plt.plot(flight_data['time'],yaw_rate_inertia*180/np.pi, label = 'fict')
# # plt.plot(flight_data['time'],yaw_rate_mass*180/np.pi, label = 'mass')
# # plt.plot(flight_data['time'],yaw_rate_steering*180/np.pi, label = 'steering')
# # plt.plot(flight_data['time'],yaw_rate*180/np.pi, label = 'Measured')
# # plt.legend()
# # plt.grid(True)
# # plt.show()
# # #%%
# # import copy
# # kite1 = copy.deepcopy(config['kite'])
# # kite2 = copy.deepcopy(config['kite'])
# # kite3 = copy.deepcopy(config['kite'])


# # mass = kite['mass'] + 30
# # yaw_rate_more_mass, _ = calculate_yaw_rate(config['kite']['area'], mass, config['kite']['span'], us, va, elevation, yaw,v,radius, coeffs = coeffs)
# # span = kite['span'] - 10
# # yaw_rate_more_span, _ = calculate_yaw_rate(config['kite']['area'], config['kite']['mass'], span, us, va, elevation, yaw,v,radius, coeffs = coeffs)
# # area = kite['area'] + 50
# # yaw_rate_more_area, _ = calculate_yaw_rate(area, config['kite']['mass'], config['kite']['span'], us, va, elevation, yaw,v,radius, coeffs = coeffs)


# # plt.figure()
# # plt.plot(flight_data['time'],yaw_rate_identified*180/np.pi, label = 'yaw rate identified')
# # plt.plot(flight_data['time'],yaw_rate_more_span*180/np.pi, label = 'yaw rate +10m span')
# # plt.plot(flight_data['time'],yaw_rate_more_mass*180/np.pi, label = 'yaw rate +10kg mass')
# # plt.plot(flight_data['time'],yaw_rate_more_area*180/np.pi, label = 'yaw rate +10m area')
# # plt.legend()
# # plt.grid(True)


# # #%%


# # i = 460
# # mass_array = np.linspace(20,100,100)
# # yaw_rate_mass, _ = calculate_yaw_rate(config['kite']['area'], mass_array, config['kite']['span'], np.ones_like(mass_array)*us[i], va[i], elevation[i], yaw[i],v[i],radius[i],coeffs=coeffs)


# # plt.figure()
# # plt.plot(mass_array,np.degrees(yaw_rate_mass))
# # plt.grid(True)
# # plt.xlabel('Mass [kg]')
# # plt.ylabel('Yaw rate [deg/s]')

# # #%%
# # span_array = np.linspace(1,50,100)
# # yaw_rate_span, _ = calculate_yaw_rate(config['kite']['area'], config['kite']['mass'], span_array , np.ones_like(span_array)*us[i], va[i], elevation[i], yaw[i],v[i],radius[i],coeffs=coeffs)

# # plt.figure()
# # plt.plot(span_array,np.degrees(yaw_rate_span))
# # plt.grid(True)
# # plt.xlabel('Span [m]')
# # plt.ylabel('Yaw rate [deg/s]')

# # #%%
# # area_array = np.linspace(20,100,100)
# # yaw_rate_area, _ = calculate_yaw_rate(area_array, config['kite']['mass'], config['kite']['span'] , np.ones_like(area_array)*us[i], va[i], elevation[i], yaw[i],v[i],radius[i],coeffs=coeffs)

# # plt.figure()
# # plt.plot(area_array,np.degrees(yaw_rate_area))
# # plt.grid(True)
# # plt.xlabel('Area [m^2]')
# # plt.ylabel('Yaw rate [deg/s]')
# # plt.show()
# # #%%
# # # i = 3851
# # # va = va[i]
# # # us = us[i]
# # # elevation = elevation[i]
# # # yaw = yaw[i]
# # # v = v[i]
# # # radius = radius[i]
# # # mass = config['kite']['mass']
# # # span = config['kite']['span']
# # # area = config['kite']['area']

# # # yaw_rate_opt_AR = []
# # # for AR in [1,2,5,10,20,40]:
# # #     span = np.sqrt(AR*area)
# # #     yaw_rate = calculate_yaw_rate_new(coeffs, area, mass, span, us, va, np.radians(elevation), yaw,v,radius)
# # #     yaw_rate_opt_AR.append(yaw_rate)

# # # R = v/np.array(yaw_rate_opt_AR)
# # # plt.figure()
# # # plt.plot([1,2,5,10,20,40],np.degrees(yaw_rate_opt_AR))
# # # plt.plot([1,2,5,10,20,40],R)
# # # plt.grid(True)
# # # plt.xlabel('Aspect ratio')
# # # plt.ylabel('Yaw rate [deg/s]')


# # # mse = calculate_mse(yaw_rate, yaw_rate_identified)
# # # # print(f"MAE: {mae}")
# # # print(f"MSE: {mse}")

# # # mask = np.any(
# # #     [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
# # # plt.figure()
# # # plt.scatter(np.degrees(yaw_rate[mask]), np.degrees(yaw_rate_identified[mask]), label='Turn law',
# # #             color='b', alpha=0.5, s=5)
# # # plt.plot(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100),
# # #           label='Ground Truth', color='k', linewidth=0.5, linestyle='--')
# # # plt.xlabel('Estimated Yaw rate [deg/s]')
# # # plt.ylabel('Measured Yaw rate [deg/s]')
# # # plt.legend()
