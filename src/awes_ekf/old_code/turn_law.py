import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from awes_ekf.run_EKF import create_kite
import awes_ekf.plot_utils as pu
from awes_ekf.postprocessing import postprocess_results
from scipy.optimize import minimize, least_squares
from scipy.integrate import cumtrapz
from scipy.stats import pearsonr
from scipy import stats
from awes_ekf.utils import Rz
def calculate_mae(measured, calculated):
    return np.mean(np.abs(measured - calculated))


def calculate_mse(measured, calculated):
    return np.mean((measured - calculated) ** 2)


def calculate_rmse(measured, calculated):
    return np.sqrt(calculate_mse(measured, calculated))

def calculate_yaw_rate(x, kite, us, va, beta, yaw,v,radius,forces):

    rho = 1.225
    area = kite.area
    mass = kite.mass
    gravity_constant = -9.81
    B = kite.span
    
    Cn_d = x[0]
    Cn_us = x[1]
    d_k = x[2]
    Cn_ass = x[3]
    
    k_d = rho*area*B**2*Cn_d/12
    k_us = 0.5*rho*area*Cn_us
    k_c = d_k*mass
    k_g = k_c*gravity_constant


    yaw_rate = k_us*us*(va**2)
    if 'weight' in forces:
        yaw_rate += k_g*np.sin(yaw)*np.cos(beta)
    if 'tether' in forces:
        yaw_rate += (va**2)*Cn_ass
    if 'centripetal' in forces:  
        yaw_rate += k_c*v**2/radius
    

    if 'centripetal' in forces:  
        yaw_rate = yaw_rate/(k_d*va)
    else:
        yaw_rate = yaw_rate/(k_d*va+k_c*v)

    if 'simple' in forces:
        yaw_rate = k_us*us*(va)
    # yaw_rate = (-k_g*np.sin(yaw)*np.cos(beta)+k_us*us*(va**2)+k_c*v**2/radius+(va**2)*Cn_ass)/(k_d*va)

    return yaw_rate

def obj_yaw_rate(x,kite,us_data, va_data, beta,yaw,v_kite,radius,forces, observed_yaw_rates):
    estimated_yaw_rates = calculate_yaw_rate(x,kite,us_data, va_data, beta, yaw,v_kite,radius,forces)
    return np.sum((observed_yaw_rates - estimated_yaw_rates) ** 2)

# %%
plt.close('all')

year = '2024'
month = '02'
day = '16'
kite_model = 'v9'                   # Kite model name, if Costum, change the kite parameters next
kcu_model = 'KP2'                   # KCU model name
tether_diameter = 0.014            # Tether diameter [m]


path = '../results/'+kite_model+'/'
file_name = kite_model+'_'+year+'-'+month+'-'+day
date = year+'-'+month+'-'+day

results = pd.read_csv(path+file_name+'_res_GPS.csv')
flight_data = pd.read_csv(path+file_name+'_fd.csv')

# results = results.dropna()
rows_to_keep = results.index
flight_data = flight_data.loc[rows_to_keep]

kite = create_kite(kite_model)
#%%
results, flight_data = postprocess_results(results,flight_data, kite, imus = [0], remove_IMU_offsets=True, 
                                            correct_IMU_deformation = False,remove_vane_offsets=True,estimate_kite_angle=True)

#%% Define relevant variables

v_kite = np.vstack((np.array(results['vx']), np.array(
    results['vy']), np.array(results['vz']))).T
r_kite = np.vstack((np.array(results['x']), np.array(
    results['y']), np.array(results['z']))).T

azimuth0 = np.arctan2(r_kite[:,1],r_kite[:,0])*180/np.pi

radius = np.array(results['radius_turn'])
wdir = results['wind_direction']

r_kite = np.array([Rz(-angle).dot(vector) for vector, angle in zip(r_kite, wdir)])
v_kite = np.array([Rz(-angle).dot(vector) for vector, angle in zip(v_kite, wdir)])

yaw = np.arctan2(v_kite[:,1],v_kite[:,0])
v_kite = np.linalg.norm(v_kite, axis=1)
va_kite = results['va_kite']
elevation = np.arctan2(r_kite[:,2],np.sqrt(r_kite[:,0]**2+r_kite[:,1]**2))
azimuth = np.arctan2(r_kite[:,1],r_kite[:,0])*180/np.pi
us = flight_data['kcu_actual_steering']

window_size = 30
yaw_rate = np.diff(np.unwrap(flight_data['kite_0_yaw']/180*np.pi)) / 0.1
yaw_rate = np.concatenate((yaw_rate, [0]))
# yaw_rate = flight_data['kite_1_yaw_rate']
yaw_rate = np.convolve(yaw_rate, np.ones(window_size)/window_size, mode='same')
radius[yaw_rate > 0] = -radius[yaw_rate > 0]

#%%

cycles_plotted = np.arange(5, 50, step=1)
mask = np.any(
    [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
forces = ['simple']
mask = mask&(flight_data['powered'] == 'powered')
pow = (flight_data['powered'] == 'powered')
x0 = [0.1, 0, 0, 0, 0]
# Perform the optimization
result = minimize(obj_yaw_rate, x0, args=(kite,us[mask], va_kite[mask], elevation[mask], (yaw[mask]), v_kite[mask], radius[mask], forces, yaw_rate[mask]))
optimal_x = result.x
y = calculate_yaw_rate(optimal_x,kite,us, va_kite, elevation , yaw, v_kite, radius, forces)





plt.figure()
plt.plot(flight_data['time'],y, label = 'Only aerodynamic')
plt.plot(flight_data['time'],yaw_rate, label = 'Only aerodynamic')




# Calculate accuracy metrics for each method
mae = calculate_mae(yaw_rate[pow], y[pow])*180/np.pi
mse = calculate_mse(yaw_rate[mask], y[mask])
rmse = calculate_rmse(yaw_rate[pow], y[pow])*180/np.pi
# print(f"MAE: {mae}")
print(f"MSE: {mse}")
# print(f"RMSE: {rmse}")

rho = 1.225
area = kite.area
mass = kite.mass
gravity_constant = -9.81
B = kite.span
Cn_d = optimal_x[0]
Cn_us = optimal_x[1]
d_k = optimal_x[2]
Cn_ass = optimal_x[3]
k_d = rho*area*B**2*Cn_d/12
k_us = 0.5*rho*area*Cn_us
k_c = d_k*mass
k_g = k_c*gravity_constant


plt.figure()
plt.plot(flight_data['time'],y, label = 'yaw rate')
plt.plot(flight_data['time'],-k_g*np.sin(yaw)*np.cos(elevation)/(k_d*va_kite), label = 'weight')
plt.plot(flight_data['time'],k_c*v_kite**2/radius/(k_d*va_kite), label = 'fict')
plt.plot(flight_data['time'],k_us*us*(va_kite**2)/(k_d*va_kite), label = 'steering')
plt.plot(flight_data['time'],(va_kite**2)*Cn_ass/(k_d*va_kite), label = 'assymetry')
plt.plot(flight_data['time'],yaw_rate, label = 'Measured')
plt.legend()


#%%
kite = create_kite(kite_model)
forces = ['weight','centripetal']
va =  20
v = 20
us = 0.8
beta = 30/180*np.pi
yaw = np.pi/2
r = 30
ratio = kite.mass/kite.area
AR = kite.span**2/kite.area

span_array = np.linspace(5,40)
ratios = []
yr_span = []
for span in span_array:
    kite.span = span
    kite.area = span**2/AR
    ratios.append(kite.mass/kite.area)
    yr_span.append(calculate_yaw_rate(optimal_x,kite,us, va, beta , yaw, v, r, forces))

#%%
plt.figure()
plt.plot(span_array, yr_span)
#%%
kite = create_kite(kite_model)
forces = ['weight','centripetal']
va =  20
v = 20
us = 0.8
beta = 30/180*np.pi
yaw = -np.pi/2
r = 20
ratio = kite.mass/kite.area
AR = kite.span**2/kite.area

mass_array = np.linspace(5,100)
ratios = []
yr_span = []
yr_span1 = []
for mass in mass_array:
    kite.mass = mass
    ratios.append(kite.mass/kite.area)
    yr_span.append(calculate_yaw_rate(optimal_x,kite,us, va, beta , yaw, v, r, forces))
    yr_span1.append(calculate_yaw_rate(optimal_x,kite,us, va, beta , -yaw, v, r, forces))
#%%
plt.figure()
plt.plot(ratios, yr_span)
plt.plot(ratios, yr_span1)

#%%
kite = create_kite(kite_model)
forces = ['weight']
va =  20
v = 20
us = -0.8
beta = 30/180*np.pi
yaw = np.pi/2
r = 30
ratio = kite.mass/kite.area
AR_array = np.linspace(9, 40)
ratios = []
yr_span = []
yr_span1 = []
for AR in AR_array:
    kite.area = kite.span**2/AR
    ratios.append(kite.mass/kite.area)
    yr_span.append(calculate_yaw_rate(optimal_x,kite,us, va, beta , yaw, v, r, forces))
    yr_span1.append(calculate_yaw_rate(optimal_x,kite,us, va, beta , -yaw, v, r, forces))
#%%
plt.figure()
plt.plot(AR_array, yr_span)
plt.plot(AR_array, yr_span1)

plt.figure()
plt.plot(AR_array, v/np.array(yr_span))
plt.plot(AR_array, v/np.array(yr_span1))


# t = flight_data.time
# # Results from EKF
# x = results.x
# y = results.y
# z = results.z
# vx = results.vx
# vy = results.vy
# vz = results.vz
# wdir = results.wind_direction
# elevation = np.arctan2(z, np.sqrt(x**2+y**2))*180/np.pi




# v_kite = np.vstack((np.array(vx), np.array(vy), np.array(vz))).T
# window_size = 20

# yaw_rate = flight_data['kite_1_yaw_rate']

# # yaw_rate = np.diff(np.unwrap(meas_yaw/180*np.pi)) / 0.1
# # yaw_rate = np.concatenate((yaw_rate, [0]))
# yaw_rate = np.convolve(yaw_rate, np.ones(window_size)/window_size, mode='same')
# # yaw_rate = np.unwrap(yaw_rate)
# # mask = yaw_rate<-2
# # yaw_rate[mask] += np.pi
# # mask = yaw_rate>2
# # yaw_rate[mask] += -np.pi

# # %%
# cycle_count = 50



# us = flight_data['us']
# meas_yaw = flight_data['kite_1_yaw']
# radius = np.array(results['radius_turn'])
# va = results['va_kite']
# radius[yaw_rate > 0] = -radius[yaw_rate > 0]
# pow = flight_data['powered'] == 'powered'
# # %%
# cycles_plotted = np.arange(0, cycle_count-5, step=1)
# mask = np.any(
#     [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
# mask = mask & pow
# x0 = [0, 0, 0, 0.1, 0.1]
# forces = []

# # Perform the optimization
# result = minimize(pu.obj_yaw_rate, x0, args=(
#     us[mask], va[mask], elevation[mask]/180*np.pi, (meas_yaw[mask])/180*np.pi, v_kite[mask], radius[mask], forces, yaw_rate[mask]))
# optimal_x = result.x
# y = pu.calculate_yaw_rate(optimal_x, us, va, elevation /
#                               180*np.pi, (meas_yaw)/180*np.pi, v_kite, radius, forces)

# forces = ['weight']

# # Perform the optimization
# result = minimize(pu.obj_yaw_rate, x0, args=(
#     us[mask], va[mask], elevation[mask]/180*np.pi, (meas_yaw[mask])/180*np.pi, v_kite[mask], radius[mask], forces, yaw_rate[mask]))
# optimal_x1 = result.x
# y1 = pu.calculate_yaw_rate(
#     optimal_x1, us, va, elevation/180*np.pi, (meas_yaw)/180*np.pi, v_kite, radius, forces)

# forces = ['weight', 'centripetal']

# # Perform the optimization
# result = minimize(pu.obj_yaw_rate, x0, args=(
#     us[mask], va[mask], elevation[mask]/180*np.pi, (meas_yaw[mask])/180*np.pi, v_kite[mask], radius[mask], forces, yaw_rate[mask]))
# optimal_x2 = result.x
# y2 = pu.calculate_yaw_rate(
#     optimal_x2, us, va, elevation/180*np.pi, (meas_yaw)/180*np.pi, v_kite, radius, forces)

# forces = ['weight', 'centripetal', 'tether']

# # Perform the optimization
# result = minimize(pu.obj_yaw_rate, x0, args=(
#     us[mask], va[mask], elevation[mask]/180*np.pi, (meas_yaw[mask])/180*np.pi, v_kite[mask], radius[mask], forces, yaw_rate[mask]))
# optimal_x3 = result.x
# y3 = pu.calculate_yaw_rate(
#     optimal_x3, us, va, elevation/180*np.pi, (meas_yaw)/180*np.pi, v_kite, radius, forces)


# # %%
# fig, ax = plt.subplots()
# cycles_plotted = np.arange(20, 50, step=1)
# mask = np.any(
#     [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
# mask = mask & pow
# plt.scatter(us[mask], yaw_rate[mask], label='Data')
# plt.scatter(us[mask], y3[mask], color='red', label='Fitted line')
# # pu.plot_probability_density(us[mask], yaw_rate[mask], fig, ax)

# plt.xlabel('us')
# plt.ylabel('yaw_rate/va')
# plt.legend()
# plt.show()




# # %%
# # Calculate accuracy metrics for each method
# mae = calculate_mae(yaw_rate[pow], y[pow])*180/np.pi
# mse = calculate_mse(yaw_rate[pow], y[pow])
# rmse = calculate_rmse(yaw_rate[pow], y[pow])*180/np.pi
# # print(f"MAE: {mae}")
# print(f"MSE: {mse}")
# print(f"RMSE: {rmse}")

# # Calculate accuracy metrics for each method
# mae = calculate_mae(yaw_rate[pow], y1[pow])*180/np.pi
# mse = calculate_mse(yaw_rate[pow], y1[pow])
# rmse = calculate_rmse(yaw_rate[pow], y1[pow])*180/np.pi
# # print(f"MAE: {mae}")
# print(f"MSE: {mse}")
# print(f"RMSE: {rmse}")

# # Calculate accuracy metrics for each method
# mae = calculate_mae(yaw_rate[pow], y2[pow])*180/np.pi
# mse = calculate_mse(yaw_rate[pow], y2[pow])
# rmse = calculate_rmse(yaw_rate[pow], y2[pow])*180/np.pi
# # print(f"MAE: {mae}")
# print(f"MSE: {mse}")
# print(f"RMSE: {rmse}")

# # Calculate accuracy metrics for each method
# mae = calculate_mae(yaw_rate[pow], y3[pow])*180/np.pi
# mse = calculate_mse(yaw_rate[pow], y3[pow])
# rmse = calculate_rmse(yaw_rate[pow], y3[pow])*180/np.pi
# # print(f"MAE: {mae}")
# print(f"MSE: {mse}")
# print(f"RMSE: {rmse}")

# #%%
# cycles_plotted = np.arange(65, 67, step=1)

# mask = np.any(
#     [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
# v_mod = np.linalg.norm(v_kite,axis = 1)
# r_yaw_rate = abs(v_mod/yaw_rate)
# r_turn_law = abs(v_mod/y3)
# plt.figure()
# plt.plot(t[mask], abs(radius[mask]))
# plt.plot(t[mask], r_yaw_rate[mask])
# plt.plot(t[mask], r_turn_law[mask])

# plt.ylim([0,400])
# # %%
# # us = 0.4
# # va = np.zeros((1,3))
# # va[0] = 30
# # beta = 40/180*np.pi
# # yaw = 90/180*np.pi
# # heading = 90*90/180*np.pi
# # yrup = pu.calculate_yaw_rate_awebook(x, us, va, beta, yaw)
# # rup = pu.calculate_radius_yaw_rate(yrup, np.linalg.norm(va))
# # print(yrup,rup)
# # yaw = -90/180*np.pi
# # heading = 90*90/180*np.pi
# # va[0] = 30
# # us = 0.405
# # beta = 30/180*np.pi
# # yrdown = pu.calculate_yaw_rate_awebook(x, us, va, beta, yaw, heading)
# # rdown = pu.calculate_radius_yaw_rate(yrdown, np.linalg.norm(va))
# # print(yrdown,rdown)


# # %% Plot true vs estimated yaw rate
# cycles_plotted = np.arange(5, cycle_count-5, step=1)

# mask = np.any(
#     [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
# mask = mask & pow
# # yaw_rate = np.convolve(yaw_rate, np.ones(window_size)/window_size, mode='same')

# plt.figure()
# plt.scatter(yaw_rate[mask], y3[mask], label='Turn law',
#             color='b', alpha=0.5, s=5)
# plt.plot(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100),
#          label='Ground Truth', color='k', linewidth=0.5, linestyle='--')
# plt.xlabel('Estimated Yaw rate [deg/s]')
# plt.ylabel('Measured Yaw rate [deg/s]')
# plt.legend()

# rho, p_value = pearsonr(yaw_rate[mask], y[mask])
# std = np.std(y2[mask])

# print("Pearson Correlation Coefficient (rho):", rho)
# print("std-value:", std)
# slope, intercept, r_value, p_value, std_err = stats.linregress(
#     yaw_rate[mask], y3[mask])

# # Calculate R squared
# r_squared = r_value**2

# # Print the R squared value
# print(f"Slope: {slope}")
# print(f"Intercept: {intercept}")
# print(f"R-squared: {r_squared}")
# # %%

# norm_va = va
# norm_v = np.linalg.norm(v_kite,axis = 1)
# beta = elevation/180*np.pi
# yaw = (meas_yaw-90)/180*np.pi
# plt.figure()
# plt.plot(flight_data['time'],y, label = 'Only aerodynamic')
# plt.plot(flight_data['time'],y1, label = 'Incl. Weight')
# plt.plot(flight_data['time'],y2, label = 'Incl. Weight, Fictitious')
# plt.plot(flight_data['time'],y3, label = 'Incl. Weight, Fictitious, Assymetry')
# plt.plot(flight_data['time'],yaw_rate, label = 'Measured yaw rate')
# plt.xlabel('Time (s)')
# plt.ylabel('Yaw rate [rad/s]')
# plt.grid()
# plt.legend()
# # plt.plot()
# # mask = turn
# # plt.fill_between(flight_data['time'], -2,2,where=mask, color='red', alpha=0.2, label='Turn')

# # %%

# x = optimal_x3
# norm_va = va
# norm_v = np.linalg.norm(v_kite, axis=1)
# beta = elevation/180*np.pi
# yaw = (meas_yaw-90)/180*np.pi
# plt.figure()
# plt.plot(flight_data['time'], x[0]*norm_va**2*us/(norm_va*x[3]), label = 'Steering')
# plt.plot(flight_data['time'], x[2]*norm_v**2/(radius)/(x[3]*norm_va), label = 'Ficticious forces')
# plt.plot(flight_data['time'], 9.81*x[2]*np.cos(beta)*np.sin(yaw)/(x[3]*norm_va), label = 'Weight')
# plt.plot(flight_data['time'], x[4]*norm_va**2/(x[3]*norm_va), label = 'Assymetry Kite')
# plt.plot(flight_data['time'], y3, label = 'Fitted Yaw rate')
# # plt.plot(flight_data['time'], (x[4]*norm_va**2+x[1]*np.cos(beta) *
# #          np.sin(yaw)+x[2]*norm_v**2/(radius)+x[0]*norm_va**2*us)/(norm_va*x[3]))
# plt.plot(flight_data['time'],yaw_rate, label = 'Measured Yaw rate')
# plt.legend()
# # plt.plot()
# # mask = turn
# # plt.fill_between(flight_data['time'], -2, 2,
#                  # where=mask, color='red', alpha=0.2, label='Turn')


# # %% Plot vKcu and v_kite
# v_kite = np.vstack((np.array(flight_data['kite_0_vx']), np.array(
#     flight_data['kite_0_vy']), np.array(flight_data['kite_0_vz']))).T
# v_kcu = np.vstack((np.array(flight_data['kite_1_vx']), np.array(
#     flight_data['kite_1_vy']), np.array(flight_data['kite_1_vz']))).T
# v_kite_mod = np.linalg.norm(v_kite, axis=1)
# v_kcu_mod = np.linalg.norm(v_kcu, axis=1)
# plt.figure()
# plt.plot(flight_data['time'], v_kite_mod, label='v_kite')
# plt.plot(flight_data['time'], v_kcu_mod, label='v_kcu')
# # plt.plot(flight_data['time'],va_mod, label = 'va')
# plt.legend()
# plt.xlabel('Time [s]')
# plt.ylabel('Velocity [m/s]')

# # %%
# meas_yaw[meas_yaw < -180] += 360


# yaw = cumtrapz(y2, t, initial=meas_yaw[0]/180*np.pi)
# v_tan = np.zeros((len(v_kite), 3))
# va_tan = np.zeros((len(v_kite), 3))
# for i in range(len(v_kite)):
#     v_tan[i, :] = pu.project_onto_plane(
#         v_kite[i, :], r_kite[i, :]/np.linalg.norm(r_kite[i, :]))
#     va_tan[i, :] = pu.project_onto_plane(-va[i, :],
#                                          r_kite[i, :]/np.linalg.norm(r_kite[i, :]))
# plt.figure()
# plt.plot(flight_data['time'], np.degrees(
#     np.arctan2(-v_tan[:, 1], v_tan[:, 0])))
# plt.plot(flight_data['time'], np.degrees(
#     np.arctan2(-va_tan[:, 1], va_tan[:, 0])))
# plt.plot(flight_data['time'], np.degrees(yaw))
# plt.plot(flight_data['time'], meas_yaw)
# # mask = turn
# # plt.fill_between(flight_data['time'], -150, 150,
# #                  where=mask, color='red', alpha=0.2, label='Turn')
# plt.grid()

# plt.figure()
# plt.plot(flight_data['time'], np.degrees(
#     np.arctan2(-v_tan[:, 1], v_tan[:, 0]))-meas_yaw)
# plt.plot(flight_data['time'], np.degrees(
#     np.arctan2(-va_tan[:, 1], va_tan[:, 0]))-meas_yaw)

# va_angle_hz = np.zeros(len(flight_data))

    