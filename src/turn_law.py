from scipy.integrate import cumtrapz
from scipy.stats import pearsonr
from scipy.optimize import least_squares, minimize
from scipy.optimize import curve_fit
import plot_utils as pu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import kappa, z0, kite_model, year, month, day
from utils import get_tether_end_position,  R_EG_Body, calculate_angle, project_onto_plane, create_kite
import seaborn as sns
from scipy import stats

# %%
plt.close('all')


path = '../results/'+kite_model+'/'
file_name = kite_model+'_'+year+'-'+month+'-'+day
date = year+'-'+month+'-'+day

results = pd.read_csv(path+file_name+'_res_GPS.csv')
flight_data = pd.read_csv(path+file_name+'_fd.csv')

offset_df = pd.read_csv('../processed_data/IMU_offset.csv')

for i in range(len(offset_df)):
    if offset_df['date'].iloc[i] == date:
        offset = offset_df.iloc[i]
        break

kite = create_kite(kite_model)
# %% Define flight phases and count cycles
up = (flight_data['kcu_actual_depower']-min(flight_data['kcu_actual_depower'])) / \
    (max(flight_data['kcu_actual_depower']) -
     min(flight_data['kcu_actual_depower']))
us = (flight_data['kcu_actual_steering']) / \
    max(abs(flight_data['kcu_actual_steering']))
dep = (up > 0.25)
pow = (flight_data['ground_tether_reelout_speed'] > 0) & (up < 0.115)
trans = ~pow & ~dep
turn = pow & (abs(us) > 0.5)
turn_right = pow & (us > 0.25)
tun_left = pow & (us < -0.25)
straight = pow & ~turn

cycle_count = 0
in_cycle = False
ip = 0

flight_data['cycle'] = np.zeros(len(flight_data))
for i in range(len(pow)):
    if dep[i] and not in_cycle:
        flight_data.loc[ip:i, 'cycle'] = cycle_count
        ip = i
        # Entering a new cycle
        cycle_count += 1
        in_cycle = True
    elif not dep[i] and in_cycle:
        # Exiting the current cycle
        in_cycle = False

print("Number of cycles:", cycle_count)


# %% Declare variables

# Measured variables
measured_wdir = -flight_data['ground_wind_direction']-90+360
measured_wvel = flight_data['ground_wind_velocity']
measured_uf = measured_wvel*kappa/np.log(10/z0)
measured_va = flight_data['kite_apparent_windspeed']
measured_Ft = flight_data['ground_tether_force']
measured_aoa = flight_data['kite_angle_of_attack']
if 'v9' in kite_model:
    measured_ss = flight_data['kite_sideslip_angle']
else:
    measured_ss = np.zeros(len(flight_data))
measured_aoa = np.array(measured_aoa)
# Sensor 0 is the one on the kite, sensor 1 is the one on the KCU
# 2023-11-27 : pitch0,1 +3,+4 yaw0,1 0,0 roll0,1 -7,0
# 2019-10-08 : pitch0,1 -3,-5 yaw0,1 -2,0 roll0,1 -5,0
meas_pitch = flight_data['kite_0_pitch']+offset['pitch0']
meas_pitch1 = flight_data['kite_1_pitch']+offset['pitch1']
meas_roll = flight_data['kite_0_roll'] + offset['roll0']
meas_roll1 = flight_data['kite_1_roll'] + offset['roll1']
meas_yaw = flight_data['kite_0_yaw'] + offset['yaw0']
meas_yaw1 = flight_data['kite_1_yaw'] + offset['yaw1']
meas_tetherlen = flight_data['ground_tether_length']
course = flight_data['kite_course']
azimuth = flight_data['kite_azimuth']

roll = -results.roll
pitch = -results.pitch
yaw = -results.yaw-180


t = flight_data.time
# Results from EKF
x = results.x
y = results.y
z = results.z
vx = results.vx
vy = results.vy
vz = results.vz
uf = results.uf
wdir = results.wdir
elevation = np.arctan2(z, np.sqrt(x**2+y**2))*180/np.pi
CL_EKF = results.CL
CD_EKF = results.CD

cd_kcu = results.cd_kcu
CS_EKF = results.CS
CLw = results.CLw
CDw = results.CDw
CSw = results.CSw
aoa = results.aoa
sideslip = results.sideslip-90
tether_len = results.tether_len
Ft = np.array([results.Ftx, results.Fty, results.Ftz]).T
Ft_mod = np.linalg.norm(Ft, axis=1)
wvel = uf/kappa*np.log(z/z0)
r_kite = np.vstack((np.array(x), np.array(y), np.array(z))).T
vw_EKF = np.vstack(
    (wvel*np.cos(wdir), wvel*np.sin(wdir), np.zeros(len(wvel)))).T
v_kite = np.vstack((np.array(vx), np.array(vy), np.array(vz))).T
window_size = 20
flight_data['kite_1_ax'] = np.convolve(
    flight_data['kite_1_ax'], np.ones(window_size)/window_size, mode='same')
flight_data['kite_1_ay'] = np.convolve(
    flight_data['kite_1_ay'], np.ones(window_size)/window_size, mode='same')
flight_data['kite_1_az'] = np.convolve(
    flight_data['kite_1_az'], np.ones(window_size)/window_size, mode='same')
meas_ax = flight_data.kite_1_ax
meas_ay = flight_data.kite_1_ay
meas_az = flight_data.kite_1_az
acc = np.vstack((np.array(meas_ax), np.array(meas_ay), np.array(meas_az))).T

ax = np.concatenate((np.diff(vx)/0.1, [0]))
ay = np.concatenate((np.diff(vy)/0.1, [0]))
az = np.concatenate((np.diff(vz)/0.1, [0]))
ax = np.convolve(ax, np.ones(window_size)/window_size, mode='same')
ay = np.convolve(ay, np.ones(window_size)/window_size, mode='same')
az = np.convolve(az, np.ones(window_size)/window_size, mode='same')
acc = np.vstack((np.array(ax), np.array(ay), np.array(az))).T

# meas_vx = flight_data.kite_1_vx
# meas_vy = flight_data.kite_1_vy
# meas_vz = flight_data.kite_1_vz
# v_kite = np.vstack((np.array(meas_vx),np.array(meas_vy),np.array(meas_vz))).T

va = vw_EKF-v_kite
a_kite = acc
yaw_rate = flight_data['kite_1_yaw_rate']
roll_rate = flight_data['kite_1_roll_rate']
pitch_rate = flight_data['kite_1_pitch_rate']

# yaw_rate = np.diff(np.unwrap(meas_yaw/180*np.pi)) / 0.1
# yaw_rate = np.concatenate((yaw_rate, [0]))
yaw_rate = np.convolve(yaw_rate, np.ones(window_size)/window_size, mode='same')
# yaw_rate = np.unwrap(yaw_rate)
# mask = yaw_rate<-2
# yaw_rate[mask] += np.pi
# mask = yaw_rate>2
# yaw_rate[mask] += -np.pi
# %%
# roll_rate = np.diff(meas_roll) / 0.1
# roll_rate = np.concatenate((roll_rate, [0]))
# roll_rate=np.convolve(roll_rate/180*np.pi, np.ones(window_size)/window_size, mode='same')

# pitch_rate = np.diff(pitch) / 0.1
# pitch_rate = np.concatenate((pitch_rate, [0]))
# pitch_rate=np.convolve(pitch_rate/180*np.pi, np.ones(window_size)/window_size, mode='same')

# Calculate wind speed based on KCU orientation and wind speed and direction
aoacalc = []
sideslipcalc = []
va_mod = []
slack = []
wvel_calc = []
wdir_calc = []
v_radial = []
radius = []
omega = []
y_axis = []
x_axis = []
rad_dir = []
turn_rate = []
kite_rot = []
angle = []
measured_aoa = measured_aoa+4
measured_ss = -measured_ss-5
v_tau = []
rad_dir1 = []
angle1 = []


def calculate_angle(vector_a, vector_b, reference_vector=None, deg=True):
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = np.arccos(cos_theta)

    if reference_vector is not None:
        # Calculate the cross product
        cross_product = np.cross(vector_a, vector_b)

        # Determine the sign of the angle using the reference vector
        if np.dot(cross_product, reference_vector) < 0:
            angle_rad = -angle_rad

    angle_deg = np.degrees(angle_rad)

    if deg:
        return angle_deg
    else:
        return angle_rad


for i in range(len(CL_EKF)):

    va_mod.append(np.linalg.norm(va[i]))
    q = 0.5*1.225*kite.area*va_mod[i]**2
    slack.append(tether_len[i]+kite.distance_kcu_kite -
                 np.sqrt(x[i]**2+y[i]**2+z[i]**2))

    at = np.dot(a_kite[i], np.array(v_kite[i])/np.linalg.norm(v_kite[i])
                )*np.array(v_kite[i])/np.linalg.norm(v_kite[i])
    omega_kite = np.cross(a_kite[i]-at, v_kite[i]) / \
        (np.linalg.norm(v_kite[i])**2)
    ICR = np.cross(v_kite[i], omega_kite)/(np.linalg.norm(omega_kite)**2)

    rdir = np.array(r_kite[i])/np.linalg.norm(r_kite[i])
    angle.append(calculate_angle(project_onto_plane(
        va[i], rdir), project_onto_plane(-vw_EKF[i], rdir), rdir))
    # angle1.append(calculate_angle(v_kite[i], vw_EKF[i]))
    # angle.append(course[i])
    rad_dir.append(-ICR/np.linalg.norm(ICR))
    ICR1 = np.cross(rdir, project_onto_plane(v_kite[i], rdir))
    rad_dir1.append(-ICR1/np.linalg.norm(ICR1))

    radius.append(np.linalg.norm(ICR))
    omega.append(np.linalg.norm(omega_kite))

    # Calculate tether orientation based on kite sensor measurements
    Transform_Matrix = R_EG_Body(
        meas_roll[i]/180*np.pi, meas_pitch[i]/180*np.pi, (meas_yaw[i])/180*np.pi)
    #    Transform_Matrix=R_EG_Body(kite_roll[i]/180*np.pi,kite_pitch[i]/180*np.pi,kite_yaw_modified[i])
    Transform_Matrix = Transform_Matrix.T
    # X_vector
    ex_kite = Transform_Matrix.dot(np.array([-1, 0, 0]))
    # Y_vector
    ey_kite = Transform_Matrix.dot(np.array([0, -1, 0]))
    # Z_vector
    ez_kite = Transform_Matrix.dot(np.array([0, 0, 1]))

    rotations = Transform_Matrix.dot(np.array(omega_kite))

    kite_rot.append(rotations)
    turn_rate.append(rotations[2])

    # Calculate apparent wind velocity based on KCU orientation and apparent wind speed and aoa and ss
    va_calc = ex_kite*measured_va[i]*np.cos(measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ey_kite*measured_va[i]*np.sin(
        measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ez_kite*measured_va[i]*np.sin(measured_aoa[i]/180*np.pi)
    # Calculate wind velocity based on KCU orientation and wind speed and direction
    vw = va_calc+v_kite[i]
    wvel_calc.append(np.linalg.norm(vw))
    wdir_calc.append(np.arctan2(vw[1], vw[0]))

    # Projected apparent wind velocity onto kite y axis
    va_proj = project_onto_plane(va[i], ey_kite)
    aoacalc.append(90-calculate_angle(ez_kite, va_proj)
                   )            # Angle of attack
    # Projected apparent wind velocity onto kite z axis
    va_proj = project_onto_plane(va[i], ez_kite)
    sideslipcalc.append(90-calculate_angle(ey_kite, va_proj)
                        )        # Sideslip angle

    v_radial.append(np.dot(v_kite[i], r_kite[i]/np.linalg.norm(r_kite[i])))
    v_tau.append(project_onto_plane(va[i], rdir))
v_tau = np.array(v_tau)
turn_rate = np.array(turn_rate)
omega = np.array(omega)
radius = np.array(radius)
azimuth_rate = np.concatenate((np.diff(azimuth), [0]))
# pitch_rate = np.concatenate((np.diff(meas_pitch), [0]))
sideslipcalc = np.array(sideslipcalc)
aoacalc = np.array(aoacalc)
va_mod = np.array(va_mod)
turn = pow & (vz < 0)
turn = pow & (abs(us) > 0.5)
straight = pow & ~turn
turn_right = turn & (azimuth < 0)
turn_left = turn & (azimuth > 0)
straight_right = straight & (azimuth_rate < 0)
straight_left = straight & (azimuth_rate > 0)
omega = np.convolve(omega, np.ones(window_size)/window_size, mode='same')
rad_dir = np.array(rad_dir)
kite_rot = np.array(kite_rot)
# %% Plot Steering law linear assumption

# us = flight_data['kcu_actual_steering']/100

# mask = roll_rate<-2
# roll_rate[mask] += np.pi
# mask = roll_rate>2
# roll_rate[mask] += -np.pi

heading = np.arctan2(vy, vx)-wdir
# turn_rate = yaw_rate-roll_rate*np.cos((90-meas_pitch)/180*np.pi)
cycles_plotted = np.arange(5, cycle_count-5, step=1)
# cycles_plotted = np.arange(2,cycle_count-2, step=1)

mask = np.any(
    [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)

mask_poly = pow & mask
x = us[mask_poly]
y = (yaw_rate[mask_poly]/va_mod[mask_poly])
omega[us < 0] = -omega[us < 0]
# y = omega[mask_poly]/va_mod[mask_poly]
# y = yaw_rate[mask_poly]/va_mod[mask_poly]

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Calculate R squared
r_squared = r_value**2

# Print the R squared value
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_squared}")

# Plotting (optional)
plt.figure()
plt.scatter(x, y, label='Data')
plt.plot(x, slope * x + intercept, color='red', label='Fitted line')
plt.xlabel('us')
plt.ylabel('yaw_rate/va')
plt.legend()
plt.show()

# %% Plot steering law parabolic assumption
plt.figure()
x = us[mask]
y = CS_EKF[mask]
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Calculate R squared
r_squared = r_value**2

# Print the R squared value
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_squared}")
plt.scatter(x, y)
plt.plot(x, slope * x + intercept, color='red', label='Fitted line')
plt.xlabel('Steering input (us)')
plt.ylabel('CS')

# %%
angles = np.array(angle)
mask = pow & (abs(us) > 0.4) & (azimuth > 0)
fig, ax = plt.subplots()

plt.scatter(angles[mask], CS_EKF[mask]/us[mask])
# pu.plot_probability_density(angles[mask], abs(CS_EKF)[mask]/us[mask], fig, ax)

# Define your sinusoidal function


# turn = (abs(us)>0.2)&pow
# straight = pow&~turn
x = np.cos(angles[turn])  # (angles[turn]*180/np.pi)%360
y = CS_EKF[turn]/us[turn]
# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Calculate R squared
r_squared = r_value**2

# Print the R squared value
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_squared}")

CS_linreg = slope * x + intercept
plt.figure()
# Plotting the data and the fit
plt.scatter(x, y, label='Data')
plt.plot(x, CS_linreg, color='red', label='Fitted line')
# plt.scatter(angles, sinusoidal_func(angles, *popt), label='Fitted function', color='red')
plt.xlabel('Side Slip Angle')
plt.ylabel('Side Force')
plt.legend()
plt.show()

# %%
# radius[yaw_rate<0] = -radius[yaw_rate<0]
radius[yaw_rate > 0] = -radius[yaw_rate > 0]
# %%
cycles_plotted = np.arange(5, cycle_count-5, step=1)
mask = np.any(
    [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
mask = mask & pow
x0 = [0, 0, 0, 0.1, 0.1]
forces = []

# Perform the optimization
result = minimize(pu.obj_yaw_rate, x0, args=(
    us[mask], va[mask], elevation[mask]/180*np.pi, (meas_yaw[mask]-90)/180*np.pi, v_kite[mask], radius[mask], forces, yaw_rate[mask]))
optimal_x = result.x
y = pu.calculate_yaw_rate(optimal_x, us, va, elevation /
                              180*np.pi, (meas_yaw-90)/180*np.pi, v_kite, radius, forces)

forces = ['weight']

# Perform the optimization
result = minimize(pu.obj_yaw_rate, x0, args=(
    us[mask], va[mask], elevation[mask]/180*np.pi, (meas_yaw[mask]-90)/180*np.pi, v_kite[mask], radius[mask], forces, yaw_rate[mask]))
optimal_x1 = result.x
y1 = pu.calculate_yaw_rate(
    optimal_x1, us, va, elevation/180*np.pi, (meas_yaw-90)/180*np.pi, v_kite, radius, forces)

forces = ['weight', 'centripetal']

# Perform the optimization
result = minimize(pu.obj_yaw_rate, x0, args=(
    us[mask], va[mask], elevation[mask]/180*np.pi, (meas_yaw[mask]-90)/180*np.pi, v_kite[mask], radius[mask], forces, yaw_rate[mask]))
optimal_x2 = result.x
y2 = pu.calculate_yaw_rate(
    optimal_x2, us, va, elevation/180*np.pi, (meas_yaw-90)/180*np.pi, v_kite, radius, forces)

forces = ['weight', 'centripetal', 'tether']

# Perform the optimization
result = minimize(pu.obj_yaw_rate, x0, args=(
    us[mask], va[mask], elevation[mask]/180*np.pi, (meas_yaw[mask]-90)/180*np.pi, v_kite[mask], radius[mask], forces, yaw_rate[mask]))
optimal_x3 = result.x
y3 = pu.calculate_yaw_rate(
    optimal_x3, us, va, elevation/180*np.pi, (meas_yaw-90)/180*np.pi, v_kite, radius, forces)


# %%
fig, ax = plt.subplots()
cycles_plotted = np.arange(20, 50, step=1)
mask = np.any(
    [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
mask = mask & pow
plt.scatter(us[mask], yaw_rate[mask], label='Data')
plt.scatter(us[mask], y3[mask], color='red', label='Fitted line')
# pu.plot_probability_density(us[mask], yaw_rate[mask], fig, ax)

plt.xlabel('us')
plt.ylabel('yaw_rate/va')
plt.legend()
plt.show()

# # turn_rate = yaw_rate-roll_rate*np.cos((90-meas_pitch)/180*np.pi)
# cycles_plotted = np.arange(20,25, step=1)
# # cycles_plotted = np.arange(2,cycle_count-2, step=1)

# # mask = np.any([flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
# # mask = mask
# plt.figure()
# plt.plot(t[mask], yaw_rate[mask])
# plt.plot(t[mask], y[mask])
# plt.plot(t[mask], y1[mask])
# plt.plot(t[mask], y2[mask])


def calculate_mae(measured, calculated):
    return np.mean(np.abs(measured - calculated))


def calculate_mse(measured, calculated):
    return np.mean((measured - calculated) ** 2)


def calculate_rmse(measured, calculated):
    return np.sqrt(calculate_mse(measured, calculated))


# %%
# Calculate accuracy metrics for each method
mae = calculate_mae(yaw_rate[pow], y[pow])*180/np.pi
mse = calculate_mse(yaw_rate[pow], y[pow])
rmse = calculate_rmse(yaw_rate[pow], y[pow])*180/np.pi
# print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Calculate accuracy metrics for each method
mae = calculate_mae(yaw_rate[pow], y1[pow])*180/np.pi
mse = calculate_mse(yaw_rate[pow], y1[pow])
rmse = calculate_rmse(yaw_rate[pow], y1[pow])*180/np.pi
# print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Calculate accuracy metrics for each method
mae = calculate_mae(yaw_rate[pow], y2[pow])*180/np.pi
mse = calculate_mse(yaw_rate[pow], y2[pow])
rmse = calculate_rmse(yaw_rate[pow], y2[pow])*180/np.pi
# print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# Calculate accuracy metrics for each method
mae = calculate_mae(yaw_rate[pow], y3[pow])*180/np.pi
mse = calculate_mse(yaw_rate[pow], y3[pow])
rmse = calculate_rmse(yaw_rate[pow], y3[pow])*180/np.pi
# print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

#%%
cycles_plotted = np.arange(65, 67, step=1)

mask = np.any(
    [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
v_mod = np.linalg.norm(v_kite,axis = 1)
r_yaw_rate = abs(v_mod/yaw_rate)
r_turn_law = abs(v_mod/y3)
plt.figure()
plt.plot(t[mask], abs(radius[mask]))
plt.plot(t[mask], r_yaw_rate[mask])
plt.plot(t[mask], r_turn_law[mask])

plt.ylim([0,400])
# %%
# us = 0.4
# va = np.zeros((1,3))
# va[0] = 30
# beta = 40/180*np.pi
# yaw = 90/180*np.pi
# heading = 90*90/180*np.pi
# yrup = pu.calculate_yaw_rate_awebook(x, us, va, beta, yaw)
# rup = pu.calculate_radius_yaw_rate(yrup, np.linalg.norm(va))
# print(yrup,rup)
# yaw = -90/180*np.pi
# heading = 90*90/180*np.pi
# va[0] = 30
# us = 0.405
# beta = 30/180*np.pi
# yrdown = pu.calculate_yaw_rate_awebook(x, us, va, beta, yaw, heading)
# rdown = pu.calculate_radius_yaw_rate(yrdown, np.linalg.norm(va))
# print(yrdown,rdown)


# %% Plot true vs estimated yaw rate
cycles_plotted = np.arange(5, cycle_count-5, step=1)

mask = np.any(
    [flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
mask = mask & pow
# yaw_rate = np.convolve(yaw_rate, np.ones(window_size)/window_size, mode='same')

plt.figure()
plt.scatter(yaw_rate[mask], y3[mask], label='Turn law',
            color='b', alpha=0.5, s=5)
plt.plot(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100),
         label='Ground Truth', color='k', linewidth=0.5, linestyle='--')
plt.xlabel('Estimated Yaw rate [deg/s]')
plt.ylabel('Measured Yaw rate [deg/s]')
plt.legend()

rho, p_value = pearsonr(yaw_rate[mask], y[mask])
std = np.std(y2[mask])

print("Pearson Correlation Coefficient (rho):", rho)
print("std-value:", std)
slope, intercept, r_value, p_value, std_err = stats.linregress(
    yaw_rate[mask], y3[mask])

# Calculate R squared
r_squared = r_value**2

# Print the R squared value
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_squared}")
# %%

norm_va = np.linalg.norm(va,axis = 1)
norm_v = np.linalg.norm(v_kite,axis = 1)
angle_vw = angles/180*np.pi
beta = elevation/180*np.pi
yaw = (meas_yaw-90)/180*np.pi
plt.figure()
plt.plot(flight_data['time'],y, label = 'Only aerodynamic')
plt.plot(flight_data['time'],y1, label = 'Incl. Weight')
plt.plot(flight_data['time'],y2, label = 'Incl. Weight, Fictitious')
plt.plot(flight_data['time'],y3, label = 'Incl. Weight, Fictitious, Assymetry')
plt.plot(flight_data['time'],yaw_rate, label = 'Measured yaw rate')
plt.xlabel('Time (s)')
plt.ylabel('Yaw rate [rad/s]')
plt.grid()
plt.legend()
# plt.plot()
mask = turn
plt.fill_between(flight_data['time'], -2,2,where=mask, color='red', alpha=0.2, label='Turn')

# %%

x = optimal_x3
norm_va = np.linalg.norm(va, axis=1)
norm_v = np.linalg.norm(v_kite, axis=1)
beta = elevation/180*np.pi
yaw = (meas_yaw-90)/180*np.pi
plt.figure()
plt.plot(flight_data['time'], x[0]*norm_va**2*us/(norm_va*x[3]), label = 'Steering')
plt.plot(flight_data['time'], x[2]*norm_v**2/(radius)/(x[3]*norm_va), label = 'Ficticious forces')
plt.plot(flight_data['time'], x[1]*np.cos(beta)*np.sin(yaw)/(x[3]*norm_va), label = 'Weight')
plt.plot(flight_data['time'], x[4]*norm_va**2/(x[3]*norm_va), label = 'Assymetry Kite')
plt.plot(flight_data['time'], y3, label = 'Fitted Yaw rate')
# plt.plot(flight_data['time'], (x[4]*norm_va**2+x[1]*np.cos(beta) *
#          np.sin(yaw)+x[2]*norm_v**2/(radius)+x[0]*norm_va**2*us)/(norm_va*x[3]))
plt.plot(flight_data['time'],yaw_rate, label = 'Measured Yaw rate')
plt.legend()
# plt.plot()
mask = turn
plt.fill_between(flight_data['time'], -2, 2,
                 where=mask, color='red', alpha=0.2, label='Turn')


# %% Plot vKcu and v_kite
v_kite = np.vstack((np.array(flight_data['kite_0_vx']), np.array(
    flight_data['kite_0_vy']), np.array(flight_data['kite_0_vz']))).T
v_kcu = np.vstack((np.array(flight_data['kite_1_vx']), np.array(
    flight_data['kite_1_vy']), np.array(flight_data['kite_1_vz']))).T
v_kite_mod = np.linalg.norm(v_kite, axis=1)
v_kcu_mod = np.linalg.norm(v_kcu, axis=1)
plt.figure()
plt.plot(flight_data['time'], v_kite_mod, label='v_kite')
plt.plot(flight_data['time'], v_kcu_mod, label='v_kcu')
# plt.plot(flight_data['time'],va_mod, label = 'va')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')

# %%
meas_yaw[meas_yaw < -180] += 360


yaw = cumtrapz(y2, t, initial=meas_yaw[0]/180*np.pi)
v_tan = np.zeros((len(v_kite), 3))
va_tan = np.zeros((len(v_kite), 3))
for i in range(len(v_kite)):
    v_tan[i, :] = pu.project_onto_plane(
        v_kite[i, :], r_kite[i, :]/np.linalg.norm(r_kite[i, :]))
    va_tan[i, :] = pu.project_onto_plane(-va[i, :],
                                         r_kite[i, :]/np.linalg.norm(r_kite[i, :]))
plt.figure()
plt.plot(flight_data['time'], np.degrees(
    np.arctan2(-v_tan[:, 1], v_tan[:, 0])))
plt.plot(flight_data['time'], np.degrees(
    np.arctan2(-va_tan[:, 1], va_tan[:, 0])))
plt.plot(flight_data['time'], np.degrees(yaw))
plt.plot(flight_data['time'], meas_yaw)
mask = turn
plt.fill_between(flight_data['time'], -150, 150,
                 where=mask, color='red', alpha=0.2, label='Turn')
plt.grid()

plt.figure()
plt.plot(flight_data['time'], np.degrees(
    np.arctan2(-v_tan[:, 1], v_tan[:, 0]))-meas_yaw)
plt.plot(flight_data['time'], np.degrees(
    np.arctan2(-va_tan[:, 1], va_tan[:, 0]))-meas_yaw)

va_angle_hz = np.zeros(len(flight_data))
for i in range(len(flight_data)):
    va_angle_hz[i] = calculate_angle(va[i], Ft[i] , deg=False)
    