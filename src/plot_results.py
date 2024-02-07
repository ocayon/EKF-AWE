import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import kappa, z0, kite_model, year, month, day
from utils import get_tether_end_position,  R_EG_Body, calculate_angle,project_onto_plane, create_kite
import seaborn as sns

#%%
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
#%% Define flight phases and count cycles
up = (flight_data['kcu_actual_depower']-min(flight_data['kcu_actual_depower']))/(max(flight_data['kcu_actual_depower'])-min(flight_data['kcu_actual_depower']))
us = (flight_data['kcu_actual_steering'])/max(abs(flight_data['kcu_actual_steering']))
dep = (up>0.25)
pow = (flight_data['ground_tether_reelout_speed'] > 0) & (up<0.25)
trans = ~pow & ~dep
turn = pow & (abs(us) > 0.5)
turn_right = pow & (us > 0.25)
tun_left = pow & (us < -0.25)
straight = pow & (abs(us) < 0.5)

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




#%% Declare variables

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
Ft = np.array([results.Ftx,results.Fty,results.Ftz]).T
Ft_mod = np.linalg.norm(Ft,axis = 1)
wvel = uf/kappa*np.log(z/z0)
r_kite = np.vstack((x,y,z)).T
vw = np.vstack((wvel*np.cos(wdir),wvel*np.sin(wdir),np.zeros(len(wvel)))).T
v_kite = np.vstack((np.array(vx),np.array(vy),np.array(vz))).T
meas_ax = flight_data.kite_0_ax
meas_ay = flight_data.kite_0_ay
meas_az = flight_data.kite_0_az
acc = np.vstack((np.array(meas_ax),np.array(meas_ay),np.array(meas_az))).T
va = vw-v_kite
a_kite = acc

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

measured_aoa = measured_aoa+4
measured_ss = -measured_ss-5
for i in range(len(CL_EKF)):

    va_mod.append(np.linalg.norm(va[i]))
    q = 0.5*1.225*kite.area*va_mod[i]**2
    slack.append(tether_len[i]+kite.distance_kcu_kite-np.sqrt(x[i]**2+y[i]**2+z[i]**2))
    
    at = np.dot(a_kite[i],np.array(v_kite[i])/np.linalg.norm(v_kite[i]))*np.array(v_kite[i])/np.linalg.norm(v_kite[i])
    omega_kite = np.cross(a_kite[i]-at,v_kite[i])/(np.linalg.norm(v_kite[i])**2)
    ICR = np.cross(v_kite[i],omega_kite)/(np.linalg.norm(omega_kite)**2)    
    
    radius.append(np.linalg.norm(ICR))
    omega.append(np.linalg.norm(omega_kite))
    
    # Calculate tether orientation based on kite sensor measurements
    Transform_Matrix=R_EG_Body(meas_roll[i]/180*np.pi,meas_pitch[i]/180*np.pi,(meas_yaw[i])/180*np.pi)
    #    Transform_Matrix=R_EG_Body(kite_roll[i]/180*np.pi,kite_pitch[i]/180*np.pi,kite_yaw_modified[i])
    Transform_Matrix=Transform_Matrix.T
    #X_vector
    ex_kite=Transform_Matrix.dot(np.array([-1,0,0]))
    #Y_vector
    ey_kite=Transform_Matrix.dot(np.array([0,-1,0]))
    #Z_vector
    ez_kite=Transform_Matrix.dot(np.array([0,0,1]))

    # Calculate apparent wind velocity based on KCU orientation and apparent wind speed and aoa and ss
    va_calc= ex_kite*measured_va[i]*np.cos(measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ey_kite*measured_va[i]*np.sin(measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ez_kite*measured_va[i]*np.sin(measured_aoa[i]/180*np.pi)
    # Calculate wind velocity based on KCU orientation and wind speed and direction
    vw = va_calc+v_kite[i]
    wvel_calc.append(np.linalg.norm(vw))
    wdir_calc.append(np.arctan2(vw[1],vw[0]))
    
    va_proj = project_onto_plane(va[i], ey_kite)           # Projected apparent wind velocity onto kite y axis
    aoacalc.append(90-calculate_angle(ez_kite,va_proj) )            # Angle of attack
    va_proj = project_onto_plane(va[i], ez_kite)           # Projected apparent wind velocity onto kite z axis
    sideslipcalc.append(90-calculate_angle(ey_kite,va_proj))        # Sideslip angle
    
    v_radial.append( np.dot(v_kite[i],r_kite[i]/np.linalg.norm(r_kite[i])))
    
aoa = np.array(aoacalc)
omega = np.array(omega)
radius = np.array(radius)
azimuth_rate = np.concatenate((np.diff(azimuth), [0]))
pitch_rate = np.concatenate((np.diff(meas_pitch), [0]))
sideslipcalc = np.array(sideslipcalc)
aoacalc = np.array(aoacalc)
va_mod = np.array(va_mod)
turn = pow & (vz<0)
turn = pow & (abs(us) > 0.5)
straight = pow & ~turn
turn_right = turn  & (azimuth<0)
turn_left = turn  & (azimuth>0)
straight_right = straight  & (azimuth_rate<0)
straight_left = straight  & (azimuth_rate>0)
#%% Create mask for plotting

cycles_plotted = np.arange(20,22, step=1)
# cycles_plotted = np.arange(0,cycle_count, step=1)

mask = np.any([flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)

#%% Plot slack vs tether force
# Find linear regression
from scipy import stats
# mask = mask&pow
slack = np.array(slack)
slope, intercept, r_value, p_value, std_err = stats.linregress(measured_Ft[mask],results.bias_lt[mask])
print(f"R^2: {r_value**2}")
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
plt.figure()
plt.scatter(measured_Ft[mask],results.bias_lt[mask])
plt.plot(measured_Ft[mask],slope*measured_Ft[mask]+intercept)
plt.xlabel('Tether force [N]')
plt.ylabel('Slack [m]')
plt.grid()


#%% AOA vs measured AOA

colors = ['lightblue', 'lightgreen', 'lightcoral', (0.75, 0.6, 0.8)]
plt.figure()
plt.plot(t,aoa,label = 'AoA')
plt.plot(t,aoacalc,label = 'AoA imposed orientation')
plt.plot(t,measured_aoa,label = 'AoA measured')
plt.fill_between(t, 40, where=straight, color=colors[0], alpha=0.2)
plt.fill_between(t, 40, where=turn, color=colors[1], alpha=0.2)
plt.fill_between(t, 40, where=dep, color=colors[2], alpha=0.2)
plt.fill_between(t, 40, where=trans, color=colors[3], alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Angle of attack [deg]')
plt.legend()
plt.grid()

#%% Sideslip vs measured sideslip
plt.figure()
plt.plot(t,sideslip,label = 'Sideslip from EKF')
plt.plot(t,-measured_ss,label = 'Sideslip measured')
plt.plot(t,sideslipcalc,label = 'AoA imposed orientation')
plt.fill_between(t, sideslipcalc, where=straight, color=colors[0], alpha=0.2)
plt.fill_between(t, sideslipcalc, where=turn, color=colors[1], alpha=0.2)
plt.fill_between(t, sideslipcalc, where=dep, color=colors[2], alpha=0.2)
plt.fill_between(t, sideslipcalc, where=trans, color=colors[3], alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Sideslip angle [deg]')
plt.grid()
plt.legend()


#%% Sideslip & aoa
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(18, 10))

# Plot lift coefficient
axs[0].plot(t[mask], aoa[mask])
# axs[0].plot(t[mask], CLw[mask])
axs[0].fill_between(t[mask], aoa[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
axs[0].fill_between(t[mask], aoa[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
axs[0].fill_between(t[mask], aoa[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')
axs[0].fill_between(t[mask], aoa[mask], where=trans[mask], color='red', alpha=0.2, label='Transition')
axs[0].set_ylabel('Angle of attack')
axs[0].grid()
axs[0].legend()

# Plot drag coefficient
axs[1].plot(t[mask], sideslipcalc[mask], label='Sensor fusion')
# axs[1].plot(t[mask],sideslip[mask],label = 'Sideslip from EKF')
axs[1].set_ylabel('Sideslip')
axs[1].fill_between(t[mask], sideslipcalc[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
axs[1].fill_between(t[mask], sideslipcalc[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
axs[1].fill_between(t[mask], sideslipcalc[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')
axs[1].fill_between(t[mask], sideslipcalc[mask], where=trans[mask], color='red', alpha=0.2, label='Transition')
axs[1].grid()
axs[1].legend()


correlation_matrix = np.corrcoef(aoa[straight],sideslipcalc[straight])
correlation_coefficient = correlation_matrix[0,1]
print(f"Correlation Coefficient aoa & ss: {correlation_coefficient}")
#%% Plot wind speed

# Plot horizontal wind speed
plt.figure()
plt.plot(t[mask],uf[mask])
plt.fill_between(t, 1, where=straight, color=colors[0], alpha=0.2)
plt.fill_between(t, 1, where=turn, color=colors[1], alpha=0.2)
plt.fill_between(t, 1, where=dep, color=colors[2], alpha=0.2)
plt.fill_between(t, 1, where=trans, color=colors[3], alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Friction velocity')
plt.grid()

# Plot horizontal wind speed
plt.figure()
plt.plot(t,wvel,label = 'EKF')
# plt.plot(t,wvel_calc,label = 'Va probe and vanes')
plt.fill_between(t, 15, where=straight, color=colors[0], alpha=0.2)
plt.fill_between(t, 15, where=turn, color=colors[1], alpha=0.2)
plt.fill_between(t, 15, where=dep, color=colors[2], alpha=0.2)
plt.fill_between(t, 15, where=trans, color=colors[3], alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Horizontal Wind Speed')
plt.legend()
plt.grid()

plt.figure()
plt.plot(t,wdir*180/np.pi,label = 'EKF')
# plt.plot(t,np.array(wdir_calc)*180/np.pi,label = 'Va probe and vanes')
plt.fill_between(t, 250, where=straight, color=colors[0], alpha=0.2)
plt.fill_between(t, 250, where=turn, color=colors[1], alpha=0.2)
plt.fill_between(t, 250, where=dep, color=colors[2], alpha=0.2)
plt.fill_between(t, 250, where=trans, color=colors[3], alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Horizontal Wind Direction')
plt.legend()
plt.grid()

# Plot apparent velocity
plt.figure()
plt.plot(t,va_mod,label = 'EKF')
plt.plot(t,measured_va,label = 'Va probe')
plt.xlabel('Time')
plt.ylabel('Apparent Velocity')
plt.legend()
plt.grid()

#%%
# Create a figure with three subplots stacked vertically
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(18, 10))

# Plot lift coefficient
axs[0].plot(t[mask], CL_EKF[mask])
# axs[0].plot(t[mask], CLw[mask])
axs[0].fill_between(t[mask], CL_EKF[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
axs[0].fill_between(t[mask], CL_EKF[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
axs[0].fill_between(t[mask], CL_EKF[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')
axs[0].fill_between(t[mask], CL_EKF[mask], where=trans[mask], color='red', alpha=0.2, label='Transition')
axs[0].set_ylabel('Lift Coefficient')
axs[0].grid()
axs[0].legend()

# Plot drag coefficient
axs[1].plot(t[mask], CD_EKF[mask], label='Sensor fusion')
# axs[1].plot(t[mask], CDw[mask], label='Sensor fusion')
axs[1].set_ylabel('Drag Coefficient')
axs[1].fill_between(t[mask], CD_EKF[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
axs[1].fill_between(t[mask], CD_EKF[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
axs[1].fill_between(t[mask], CD_EKF[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')
axs[1].fill_between(t[mask], CD_EKF[mask], where=trans[mask], color='red', alpha=0.2, label='Transition')
axs[1].grid()
axs[1].legend()

# Plot angle of attack
axs[2].plot(t[mask], aoa[mask])
axs[2].fill_between(t[mask], aoa[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
axs[2].fill_between(t[mask], aoa[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
axs[2].fill_between(t[mask], aoa[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')
axs[2].fill_between(t[mask], aoa[mask], where=trans[mask], color='red', alpha=0.2, label='Transition')
axs[2].set_ylabel('Angle of attack [deg]')
axs[2].grid()


# Plot sideslip angle
axs[3].plot(t[mask], sideslipcalc[mask], label='Sensor fusion')
axs[3].set_ylabel('Sideslip angle [deg]')
axs[3].fill_between(t[mask], sideslipcalc[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
axs[3].fill_between(t[mask], sideslipcalc[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
axs[3].fill_between(t[mask], sideslipcalc[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')
axs[3].set_ylabel('Sideslip angle [deg]')
axs[3].grid()


# Add a common x-axis label
fig.text(0.5, 0.04, 'Time', ha='center', va='center')

# Add a title to the entire figure
fig.suptitle('Aerodynamic Coefficients')

# # plt.savefig('aerodynamic_coefficients_plot.png', dpi=300)
from mpl_toolkits.mplot3d import Axes3D
# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cycle_bool = flight_data['cycle'] == 28

# Plot the trajectory
ax.plot(x[cycle_bool], y[cycle_bool], z[cycle_bool], label='Cycle 28') 

cycle_bool = flight_data['cycle'] == 27

# Plot the trajectory   
ax.plot(x[cycle_bool], y[cycle_bool], z[cycle_bool], label='Cycle 27')


# Set labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis (Altitude)')

plt.legend()

#%% Plot Ft with dependent variables
# Create a figure with three subplots stacked vertically
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(12, 6))

# Plot tether force
axs[0].plot(t[mask], measured_Ft[mask])
axs[0].fill_between(t[mask], measured_Ft[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
axs[0].fill_between(t[mask], measured_Ft[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
axs[0].fill_between(t[mask], measured_Ft[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')
axs[0].set_ylabel('Tether force [N]')
axs[0].grid()
axs[0].legend()

# Plot apparent wind speed
axs[1].plot(t[mask], va_mod[mask], label = 'Apparent wind speed')
axs[1].plot(t[mask], np.linalg.norm(v_kite[mask],axis = 1),label = 'Kite speed')
axs[1].fill_between(t[mask], va_mod[mask], where=straight[mask], color='blue', alpha=0.2)
axs[1].fill_between(t[mask], va_mod[mask], where=turn[mask], color='green', alpha=0.2)
axs[1].fill_between(t[mask], va_mod[mask], where=dep[mask], color='orange', alpha=0.2)
axs[1].set_ylabel('Velocity [m/s]')
axs[1].grid()
axs[1].legend()


# Plot CL
axs[2].plot(t[mask], CL_EKF[mask])
axs[2].fill_between(t[mask], CL_EKF[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
axs[2].fill_between(t[mask], CL_EKF[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
axs[2].fill_between(t[mask], CL_EKF[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')

axs[2].set_ylabel('CL [-]')
axs[2].grid()

azimuth = np.arctan2(y,x)
elevation = np.arctan2(z,np.sqrt(x**2+y**2))*180/np.pi

# Plot angle of attack
axs[3].plot(t[mask], elevation[mask])
axs[3].fill_between(t[mask], elevation[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
axs[3].fill_between(t[mask], elevation[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
axs[3].fill_between(t[mask], elevation[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')

axs[3].set_ylabel('Elevation [deg]')
axs[3].grid()


# Add a common x-axis label
fig.text(0.5, 0.04, 'Time', ha='center', va='center')

# Add a title to the entire figure
fig.suptitle('Tether force and dependent variables')

correlation_matrix = np.corrcoef(measured_Ft[mask],va_mod[mask])
correlation_coefficient = correlation_matrix[0,1]
print(f"Correlation Coefficient Ft & va: {correlation_coefficient}")

plt.savefig('tether_force.png', dpi = 300)


# Plot side force coefficient with us
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 10))

axs[0].plot(t[mask], CS_EKF[mask])
axs[0].plot(t[mask], CSw[mask])
axs[0].set_ylabel('Side Force Coefficient')
axs[0].grid()

axs[1].plot(t[mask], us[mask])
axs[1].set_ylabel('Steering input')
axs[1].grid()

# Add a common x-axis label
fig.text(0.5, 0.04, 'Time', ha='center', va='center')

#%%
# plt.figure()
# plt.scatter(us,CS_EKF)

# correlation_matrix = np.corrcoef(CS_EKF[mask&pow],us[mask&pow])
# correlation_coefficient = correlation_matrix[0,1]
# print(f"Correlation Coefficient u_s & C_S: {correlation_coefficient}")


# mask = mask&pow&(CD_EKF>0.05)#&(sideslipcalc<10)
# # Plot CL^3/CD^2 and aoa and sideslip
# fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))

# axs[0].plot(t[mask], CL_EKF[mask]**3/CD_EKF[mask]**2)
# axs[0].set_ylim([0, 50])
# axs[0].set_ylabel('CL^3/CD^2')
# axs[0].grid()

# axs[1].plot(t[mask], aoa[mask])
# axs[1].set_ylabel('Angle of attack [deg]')
# axs[1].grid()

# axs[2].plot(t[mask], sideslipcalc[mask])
# axs[2].grid()

# # Add a common x-axis label
# fig.text(0.5, 0.04, 'Time', ha='center', va='center')




# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import gaussian_kde

# # Assuming 'aoa', 'sideslipcalc', 'CL_EKF', 'CD_EKF', and 'mask' are already defined

# fig, axs = plt.subplots(2, 1, sharex=False, figsize=(8, 10))

# # Function to calculate densities
# def calculate_densities(x, y):
#     xy = np.vstack([x, y])
#     z = gaussian_kde(xy)(xy)
#     return z


# # Calculate densities for Angle of Attack plot
# z1 = calculate_densities(aoa[mask], CL_EKF[mask]**3 / CD_EKF[mask]**2)
# sc1 = axs[0].scatter(aoa[mask], CL_EKF[mask]**3 / CD_EKF[mask]**2, c=z1, cmap='viridis')
# fig.colorbar(sc1, ax=axs[0], label='Density')
# axs[0].set_ylabel('CL^3/CD^2')
# axs[0].set_ylim([0, 100])
# axs[0].set_xlabel('Angle of attack [deg]')
# axs[0].grid()
# axs[0].legend()

# # Calculate densities for Sideslip Angle plot
# z2 = calculate_densities(sideslipcalc[mask], CL_EKF[mask]**3 / CD_EKF[mask]**2)
# sc2 = axs[1].scatter(sideslipcalc[mask], CL_EKF[mask]**3 / CD_EKF[mask]**2, c=z2, cmap='viridis')
# fig.colorbar(sc2, ax=axs[1], label='Density')
# axs[1].set_ylabel('CL^3/CD^2')
# axs[1].set_ylim([0, 100])
# axs[1].set_xlabel('Sideslip angle [deg]')
# axs[1].grid()
# axs[1].legend()

# plt.show()


# # Plot Ft vs power_factor
# q = 0.5*1.225*kite.area*np.array(va_mod)**2
# plt.figure()
# plt.scatter(CL_EKF[mask]**3/CD_EKF[mask]**2, measured_Ft[mask]/q[mask])
# plt.xlabel('CL^3/CD^2')

# plt.ylabel('Tether force [N]')

# correlation_matrix = np.corrcoef(CL_EKF[mask]**3/CD_EKF[mask]**2,Ft_mod[mask])
# correlation_coefficient = correlation_matrix[0,1]
# print(f"Correlation Coefficient CL^3/CD^2 & Ft: {correlation_coefficient}")


# # Plotaoa vs sideslip
# # Calculate densities for Angle of Attack plot
# z1 = calculate_densities(aoa[mask], sideslipcalc[mask])
# plt.figure()
# sc1 = plt.scatter(aoa[mask], sideslipcalc[mask], c=CL_EKF[mask]**3 / CD_EKF[mask]**2, cmap='viridis')
# # plt.colorbar(sc1, label='Density')
# plt.ylabel('Sideslip angle [deg]')
# plt.xlabel('Angle of attack [deg]')


# # Plot CL vs aoa
# # Calculate densities for Angle of Attack plot
# z1 = calculate_densities(aoa[mask], CL_EKF[mask])
# plt.figure()
# sc1 = plt.scatter(aoa[mask], CL_EKF[mask], c=z1, cmap='viridis')
# # plt.colorbar(sc1, label='Density')
# plt.ylabel('CL')
# plt.xlabel('Angle of attack [deg]')

# # Plot CL vs aoa
# # Calculate densities for Angle of Attack plot
# z1 = calculate_densities(aoa[mask], CD_EKF[mask])
# plt.figure()
# sc1 = plt.scatter(aoa[mask], CD_EKF[mask], c=z1, cmap='viridis')
# # plt.colorbar(sc1, label='Density')
# plt.ylabel('CD')
# plt.xlabel('Angle of attack [deg]')

# # Plot CL vs ss
# # Calculate densities for Angle of Attack plot
# z1 = calculate_densities(sideslipcalc[mask], CL_EKF[mask])
# plt.figure()
# sc1 = plt.scatter(sideslipcalc[mask], CL_EKF[mask], c=z1, cmap='viridis')
# # plt.colorbar(sc1, label='Density')
# plt.ylabel('CL')
# plt.xlabel('Angle of attack [deg]')

# # Plot CD vs ss
# # Calculate densities for Angle of Attack plot
# z1 = calculate_densities(sideslipcalc[mask], CD_EKF[mask])
# plt.figure()
# sc1 = plt.scatter(sideslipcalc[mask], CD_EKF[mask], c=z1, cmap='viridis')
# # plt.colorbar(sc1, label='Density')
# plt.ylabel('CD')
# plt.xlabel('Angle of attack [deg]')


# # Plot efficiency
# plt.figure()
# plt.plot(t[mask], CL_EKF[mask]/CD_EKF[mask])
# plt.fill_between(t[mask], CL_EKF[mask]/CD_EKF[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
# plt.fill_between(t[mask], CL_EKF[mask]/CD_EKF[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
# plt.fill_between(t[mask], CL_EKF[mask]/CD_EKF[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')
# plt.fill_between(t[mask], CL_EKF[mask]/CD_EKF[mask], where=trans[mask], color='red', alpha=0.2, label='Transition')
# plt.ylabel('Efficiency')
# plt.xlabel('Time')
# plt.ylim([0, 20])
# plt.grid()
# plt.legend()

# #%% Plot bias tether length

# plt.figure()
# plt.plot(t, results.bias_lt)
# plt.fill_between(t[mask], results.bias_lt[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
# plt.fill_between(t[mask], results.bias_lt[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
# plt.fill_between(t[mask], results.bias_lt[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')

# plt.ylabel('Bias tether length')
# plt.xlabel('Time')
# plt.grid()
# plt.legend()


# #%%
# # Plot slack
# plt.figure()
# plt.plot(t[mask],flight_data['ground_tether_reelout_speed'][mask]-np.array(v_radial)[mask])
# plt.fill_between(t[mask],1 , where=straight[mask], color='blue', alpha=0.2, label='Straight')
# plt.fill_between(t[mask], 1, where=turn[mask], color='green', alpha=0.2, label='Turn')
# plt.fill_between(t[mask], 1, where=dep[mask], color='orange', alpha=0.2, label='Depower')
# plt.fill_between(t[mask], 1, where=trans[mask], color='red', alpha=0.2, label='Transition')

# slack = np.array(slack)
# plt.figure()
# plt.plot(t[mask], slack[mask]/tether_len[mask]*100)
# plt.fill_between(t[mask], slack[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
# plt.fill_between(t[mask], slack[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
# plt.fill_between(t[mask], slack[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')
# plt.fill_between(t[mask], slack[mask], where=trans[mask], color='red', alpha=0.2, label='Transition')
# plt.ylabel('Slack [m]')
# plt.xlabel('Time')
# plt.grid()
# plt.legend()


# # Plot tether force
# plt.figure()
# plt.plot(t[mask], Ft_mod[mask], label='Tether force kite')
# plt.plot(t[mask], measured_Ft[mask], label='Measured tether force ground')
# plt.fill_between(t[mask], Ft_mod[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
# plt.fill_between(t[mask], Ft_mod[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
# plt.fill_between(t[mask], Ft_mod[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')

# plt.xlabel('Time')
# plt.ylabel('Tether force [N]')
# plt.grid()
# plt.legend()

# tether_loss = (Ft_mod-measured_Ft)/Ft_mod*100

# plt.figure()
# plt.plot(t[mask], tether_loss[mask], label='Tether force kite')
# plt.fill_between(t[mask], tether_loss[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
# plt.fill_between(t[mask], tether_loss[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
# plt.fill_between(t[mask], tether_loss[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')
# plt.fill_between(t[mask], tether_loss[mask], where=trans[mask], color='red', alpha=0.2, label='Transition')
# plt.xlabel('Time')
# plt.ylabel('Tether force loss (%)')
# plt.grid()
# plt.legend()


# #%% Plot trajectory
# meas_z = flight_data['kite_0_rz']
# # Plot height
# plt.figure()
# plt.plot(t[mask], z[mask])
# plt.plot(t[mask],meas_z[mask])
# plt.fill_between(t[mask], z[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
# plt.fill_between(t[mask], z[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
# plt.fill_between(t[mask], z[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')
# plt.fill_between(t[mask], z[mask], where=trans[mask], color='red', alpha=0.2, label='Transition')
# plt.xlabel('Time')
# plt.ylabel('Height [m]')
# plt.grid()
# plt.legend()


# #%% Plot euler angles

# # Unwrap the phase
# unwrapped_angles = np.unwrap(np.radians(meas_yaw))
# # Convert back to degrees
# meas_yaw = np.degrees(unwrapped_angles)

# # Unwrap the phase
# unwrapped_angles = np.unwrap(np.radians(meas_yaw1))
# # Convert back to degrees
# meas_yaw1 = np.degrees(unwrapped_angles)

# # Unwrap the phase
# unwrapped_angles = np.unwrap(np.radians(yaw+90))
# # Convert back to degrees
# yaw = np.degrees(unwrapped_angles)


# # Plot pitch
# plt.figure()
# plt.plot(t[mask], pitch[mask], label='Pitch EKF respect to v_kite')
# plt.plot(t[mask], meas_pitch[mask], label='Pitch kite sensor')
# plt.plot(t[mask], meas_pitch1[mask], label='Pitch KCU sensor')
# plt.fill_between(t, 15, where=straight, color=colors[0], alpha=0.2)
# plt.fill_between(t, 15, where=turn, color=colors[1], alpha=0.2)
# plt.fill_between(t, 15, where=dep, color=colors[2], alpha=0.2)
# plt.fill_between(t, 15, where=trans, color=colors[3], alpha=0.2)

# plt.xlabel('Time')
# plt.ylabel('Pitch [deg]')
# plt.grid()
# plt.legend()

# plt.figure()
# plt.plot(t[mask], roll[mask]-90, label='Roll EKF respect to v_kite')
# plt.plot(t[mask], meas_roll[mask]+90, label='Roll kite sensor')
# # plt.plot(t[mask], meas_roll1[mask], label='Roll KCU sensor')

# plt.xlabel('Time')
# plt.ylabel('Roll [deg]')
# plt.grid()
# plt.legend()

# # Plot yaw
# plt.figure()
# plt.plot(t[mask], yaw[mask]%360, label='Yaw EKF respect to v_kite')
# plt.plot(t[mask], meas_yaw[mask]%360, label='Yaw kite sensor')
# plt.plot(t[mask], meas_yaw1[mask]%360, label='Yaw KCU sensor')

# plt.xlabel('Time')
# plt.ylabel('Yaw [deg]')
# plt.grid()
# plt.legend()

# #%% 
# r = np.sqrt(x**2+y**2+z**2)

# plt.figure()
# plt.plot(t,r,label = 'GPS radius')
# plt.plot(t,meas_tetherlen,label = 'Meas. tether length')
# plt.plot(t,tether_len,label = 'Tether length')
# plt.xlabel('Time')
# plt.ylabel('Distance [m]')
# plt.grid()
# plt.legend()
# #%%
# # start = 2000
# # end = start+36000
# # hours = [13,14,15]

# # h_ticks = np.arange(0,350,50)
# # heights = np.arange(100, 301, step=50)
# # ground_wdir = measured_wdir[start:end]
# # ground_wvel = measured_wvel[start:end]
# # mindir = []
# # maxdir = []
# # minvel = []
# # maxvel = []

# # fig_vel, ax_vel = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
# # fig_dir, ax_dir = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
# # for i in range(len(hours)):


# #     ax_vel[i].fill_betweenx(era5_heights[:-2], era5_wvel[i,:-2], era5_wvel[i+1,:-2], color='lightgrey', alpha=0.5)
# #     ax_vel[i].scatter(wvel[start:end],z[start:end],color = 'lightblue',alpha = 0.5)

# #     ax_vel[i].boxplot([measured_wvel[start:end]],positions = [10],vert = False,widths=(20))
# #     ax_vel[i].set_title(str(hours[i])+'h')
# #     ax_vel[i].set_xlabel('Wind velocity')
# #     ax_vel[i].set_yticks(h_ticks)
# #     ax_vel[i].set_yticklabels(h_ticks)
# #     ax_vel[i].grid()
    

# #     ax_dir[i].fill_betweenx(era5_heights[:-2], era5_wdir[i,:-2], era5_wdir[i+1,:-2], color='lightgrey', alpha=0.5)
# #     ax_dir[i].scatter(wdir[start:end]*180/np.pi,z[start:end],color = 'lightblue',alpha = 0.5)

# #     ax_dir[i].boxplot([measured_wdir[start:end]],positions = [10],vert = False,widths=(20))
# #     ax_dir[i].set_title(str(hours[i])+'h')
# #     ax_dir[i].set_xlabel('Wind direction')
# #     ax_dir[i].set_yticks(h_ticks)
# #     ax_dir[i].set_yticklabels(h_ticks)
# #     ax_dir[i].grid()
    
# #     # ax_dir[i].set_xlim([150, 220])
# #     ax_vel[i].set_xlim([0, 14])
# #     if i == 0:
# #         ax_dir[i].set_ylabel('Height')
# #         ax_vel[i].set_ylabel('Height')
        
# #     start = end
# #     end = start+36000
    
# # fig_vel.legend(['ERA5','EKF (GPS&groundwvel)','EKF with va','Ground measurement'])
# # fig_dir.legend(['ERA5','EKF (GPS&groundwvel)','EKF with va','Ground measurement'])
# # # fig_dir.savefig('wind_direction.png',dpi = 300)
# # # fig_vel.savefig('wind_velocity.png',dpi = 300)
# #%%

# ax = np.array(flight_data['kite_0_ax'])
# ay = np.array(flight_data['kite_0_ay'])
# az = np.array(flight_data['kite_0_az'])
# ax1 = np.array(flight_data['kite_1_ax'])
# ay1 = np.array(flight_data['kite_1_ay'])
# az1 = np.array(flight_data['kite_1_az'])

# plt.figure()
# plt.plot(t[mask],ax[mask],label = 'ax')
# plt.plot(t[mask],ax[mask],label = 'ax')
# plt.plot(t[mask],ay[mask],label = 'ay')
# plt.plot(t[mask],az[mask],label = 'az')
# plt.fill_between(t[mask], ax[mask], where=straight[mask], color='blue', alpha=0.2)
# plt.fill_between(t[mask], ax[mask], where=turn[mask], color='green', alpha=0.2)
# plt.fill_between(t[mask], ax[mask], where=dep[mask], color='orange', alpha=0.2)

# plt.legend()
# plt.grid()

# # a0 = np.sqrt(ax**2+ay**2+az**2)
# # a1 = np.sqrt(ax1**2+ay1**2+az1**2)
# # plt.figure()
# # plt.plot(t,a0)
# # plt.plot(t,a1)



# #%%
# vx = np.array(flight_data['kite_0_vx'])
# vy = np.array(flight_data['kite_0_vy'])
# vz = np.array(flight_data['kite_0_vz'])
# vx1 = np.array(flight_data['kite_1_vx'])
# vy1 = np.array(flight_data['kite_1_vy'])
# vz1 = np.array(flight_data['kite_1_vz'])

# vk = np.linalg.norm(v_kite,axis=1)
# v0 = np.sqrt(vx**2+vy**2+vz**2)
# v1 = np.sqrt(vx1**2+vy1**2+vz1**2)
# plt.figure()
# plt.plot(t,v0,label = 'Kite measurement')
# plt.plot(t,v1,label =  'KCU measurement')
# plt.plot(t,vk,label =  'EKF')

# plt.legend()
# plt.grid()

# #%% Make polynomial fits to aerodynamic coefficients
# # aoa = measured_aoa
# # sideslipcalc = measured_ss

# mask = pow & (t>100) 

# CLt = np.sqrt(CL_EKF**2+CS_EKF**2)

# coefficients = np.polyfit(aoa[mask], CLt[mask], 2)
# polynomial = np.poly1d(coefficients)
# alpha_fit = np.linspace(min(aoa[mask]), max(aoa[mask]), 100)  # Create a range of alpha values for the trendline
# plt.figure()
# plt.scatter(aoa[mask],CLt[mask],alpha = 0.5)   
# plt.plot(alpha_fit, polynomial(alpha_fit), label='Trendline (Degree 1)', color='r') 

# coefficients = np.polyfit(aoa[mask], CD_EKF[mask], 2)
# polynomial = np.poly1d(coefficients)
# alpha_fit = np.linspace(min(aoa[mask]), max(aoa[mask]), 100)  # Create a range of alpha values for the trendline
# plt.figure()
# plt.scatter(aoa[mask],CD_EKF[mask],alpha = 0.5)   
# plt.plot(alpha_fit, polynomial(alpha_fit), label= 'Trendline (Degree 1)', color='r') 


# mask = pow & (t>100) 
# coefficients = np.polyfit(sideslipcalc[mask], CLt[mask], 2)
# polynomial = np.poly1d(coefficients)
# alpha_fit = np.linspace(min(sideslipcalc[mask]), max(sideslipcalc[mask]), 100)  # Create a range of alpha values for the trendline
# plt.figure()
# plt.scatter(sideslipcalc[mask],CLt[mask],alpha = 0.5)   
# plt.plot(alpha_fit, polynomial(alpha_fit), label='Trendline (Degree 1)', color='r') 

# coefficients = np.polyfit(sideslipcalc[mask], CD_EKF[mask], 2)
# polynomial = np.poly1d(coefficients)
# alpha_fit = np.linspace(min(sideslipcalc[mask]), max(sideslipcalc[mask]), 100)  # Create a range of alpha values for the trendline
# plt.figure()
# plt.scatter(sideslipcalc[mask],CD_EKF[mask],alpha = 0.5)   
# plt.plot(alpha_fit, polynomial(alpha_fit), label='Trendline (Degree 1)', color='r') 


# mask = pow & (t>100)
# coefficients = np.polyfit(us[mask], CL_EKF[mask], 2)
# polynomial = np.poly1d(coefficients)
# alpha_fit = np.linspace(min(us[mask]), max(us[mask]), 100)  # Create a range of alpha values for the trendline
# plt.figure()
# plt.scatter(us[mask],CL_EKF[mask],alpha = 0.5)   
# plt.plot(alpha_fit, polynomial(alpha_fit), label='Trendline (Degree 2)', color='r') 

# coefficients = np.polyfit(us[mask], CD_EKF[mask], 2)
# polynomial = np.poly1d(coefficients)
# alpha_fit = np.linspace(min(us[mask]), max(us[mask]), 100)  # Create a range of alpha values for the trendline
# plt.figure()
# plt.scatter(us[mask],CD_EKF[mask],alpha = 0.5)   
# plt.plot(alpha_fit, polynomial(alpha_fit), label='Trendline (Degree 2)', color='r') 



# mask = pow & (t>100) 
# coefficients = np.polyfit(us[mask], CS_EKF[mask], 2)
# polynomial = np.poly1d(coefficients)
# alpha_fit = np.linspace(min(us[mask]), max(us[mask]), 100)  # Create a range of alpha values for the trendline
# plt.figure()
# mask = pow & (t>100) & (pitch_rate<0)
# plt.scatter(us[mask],abs(CS_EKF[mask]),alpha = 0.5)   
# mask = pow & (t>100) & (pitch_rate>0)
# plt.scatter(us[mask],abs(CS_EKF[mask]),alpha = 0.5)   
# plt.plot(alpha_fit, polynomial(alpha_fit), label='Trendline (Degree 2)', color='r') 

# mask = pow & (t>100) 
# coefficients = np.polyfit(us[mask], CL_EKF[mask], 2)
# polynomial = np.poly1d(coefficients)
# alpha_fit = np.linspace(min(us[mask]), max(us[mask]), 100)  # Create a range of alpha values for the trendline
# plt.figure()
# mask = pow & (t>100) & (pitch_rate<0)
# plt.scatter(us[mask],abs(CL_EKF[mask]),alpha = 0.5)   
# mask = pow & (t>100) & (pitch_rate>0)
# plt.scatter(us[mask],abs(CL_EKF[mask]),alpha = 0.5)   
# plt.plot(alpha_fit, polynomial(alpha_fit), label='Trendline (Degree 2)', color='r') 

# mask = np.any([flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)



#%% Plot wind misalignment angle
mean_az = []
mean_wdir = []
az = np.array(flight_data['kite_azimuth'])+flight_data['ground_wind_direction']
az = (np.arctan2(y,x)*180/np.pi)
CSright = []
CSleft = []
CDleft = []
CDright = []
CLleft = []
CLright = []
mean_E = []
el = np.arctan2(z,np.sqrt(x**2+y**2))
mean_measFt = []
mean_CL = []
mean_el = []
mean_wvel = []
for cycle in range(cycle_count):

    # mean_az.append(flight_data['kite_azimuth'].values[(flight_data['cycle'] == cycle)&pow].mean()+flight_data['ground_wind_direction'].values[(flight_data['cycle'] == cycle)].mean())
    mean_az.append(az[(flight_data['cycle'] == cycle)&pow].mean())
    mean_wdir.append(wdir[(flight_data['cycle'] == cycle)&pow].mean()*180/np.pi)
    CSright.append(CS_EKF[(flight_data['cycle'] == cycle)&turn_right].mean())
    CSleft.append(CS_EKF[(flight_data['cycle'] == cycle)&tun_left].mean())
    CDleft.append(CD_EKF[(flight_data['cycle'] == cycle)&tun_left].max())
    CDright.append(CD_EKF[(flight_data['cycle'] == cycle)&turn_right].max())
    CLleft.append(CL_EKF[(flight_data['cycle'] == cycle)&tun_left].mean())
    CLright.append(CL_EKF[(flight_data['cycle'] == cycle)&turn_right].mean())
    mean_E.append(CL_EKF[(flight_data['cycle'] == cycle)&pow].mean()/CD_EKF[(flight_data['cycle'] == cycle)&pow].mean())
    mean_measFt.append(measured_Ft[(flight_data['cycle'] == cycle)&pow].mean())
    mean_CL.append(CL_EKF[(flight_data['cycle'] == cycle)&pow].mean())
    mean_el.append(el[(flight_data['cycle'] == cycle)&pow].mean())
    mean_wvel.append(wvel[(flight_data['cycle'] == cycle)&pow].mean())

mean_az = np.array(mean_az)
mean_wdir = np.array(mean_wdir)
mean_CL = np.array(mean_CL)
mean_E = np.array(mean_E)
#%%

plt.figure()
plt.scatter(np.arange(0,cycle_count), mean_wdir-mean_az, label='Wind misalignment angle')
# plt.plot( mean_wdir, label='Wind misalignment angle')
plt.xlabel('Time')
plt.ylabel('Wind misalignment angle [deg]')
plt.grid()
plt.figure()
plt.plot(az)

# plt.figure()
# plt.scatter( mean_wdir-mean_az,np.array(CSright)+np.array(CSleft), label='CS turn diff')
# plt.scatter( mean_wdir-mean_az,np.array(CDright)-np.array(CDleft), label='CD turn diff')
# plt.scatter( mean_wdir-mean_az,np.array(CLright)-np.array(CLleft), label='CL turn diff')
# plt.xlabel('Wind misalignment angle [deg]')
# plt.ylabel('Side force coefficient')
# plt.grid()
# plt.legend()


# #%% Plot validation Eduardo Schimdt. et. al. 2020

# cycle = 30
# res_cycle = results[(flight_data['cycle'] == cycle)&pow]
# fd_cycle = flight_data[(flight_data['cycle'] == cycle)&pow]
# res_cycle = res_cycle.reset_index()
# fd_cycle = fd_cycle.reset_index()
# new_orbit = False
# wvel = res_cycle.uf/kappa*np.log(res_cycle.z/z0)
# az = np.arctan2(res_cycle.y,res_cycle.x)
# el = np.arctan2(res_cycle.z,np.sqrt(res_cycle.x**2+res_cycle.y**2))
# i_start = 0
# mean_wvel = []
# mean_wdir = []
# mean_az = []    
# mean_CL = []
# mean_E = []
# mean_measFt = []
# mean_el = []

# for i in range(len(res_cycle)):
    
#     if -2*np.pi+res_cycle.wdir.iloc[i]-az[i]>0 and new_orbit == True:
#         mean_wvel.append(wvel[i_start:i].mean())
#         mean_wdir.append(wdir[i_start:i].mean())
#         mean_az.append(az[i_start:i].mean())
#         mean_CL.append(res_cycle.CL[i_start:i].mean())
#         mean_E.append(res_cycle.CL[i_start:i].mean()/res_cycle.CD[i_start:i].mean())
#         mean_measFt.append(fd_cycle['ground_tether_force'].iloc[i_start:i].mean())
#         mean_el.append(el[i_start:i].mean())
#         i_start = i
#         new_orbit = False
#     if -2*np.pi+res_cycle.wdir.iloc[i]-az[i]<-25/180*np.pi and new_orbit == False:
#         new_orbit = True
# mean_az = np.array(mean_az)
# mean_wdir = np.array(mean_wdir)
# mean_CL = np.array(mean_CL)
# mean_E = np.array(mean_E)
# mean_wvel = np.array(mean_wvel)




        


# Ft_teo = 0.5*1.225*kite.area*mean_CL*mean_E**2*(1+1/mean_E**2)**1.5*(mean_wvel*np.sin(mean_el)*np.cos((-2*np.pi+mean_wdir-mean_az)))**2

# plt.figure()
# plt.scatter(mean_measFt,Ft_teo)

# plt.figure()
# mask = (up>0.9) & (t>50)
# plt.scatter(t[mask],meas_pitch[mask]-pitch[mask])
# plt.scatter(t[mask],meas_pitch[mask]-meas_pitch1[mask])

#%%
fig = plt.figure()
ax1 = plt.axes(projection='3d')

#Computation of the kite's reference frame

xx_kite=[]
xy_kite=[]
xz_kite=[]

yx_kite=[]
yy_kite=[]
yz_kite=[]

zx_kite=[]
zy_kite=[]
zz_kite=[]

for i in range(0,len(roll)):
    Transform_Matrix=R_EG_Body(meas_roll[i]/180*np.pi,pitch[i]/180*np.pi,meas_yaw[i]/180*np.pi)
#    Transform_Matrix=R_EG_Body(kite_roll[i]/180*np.pi,kite_pitch[i]/180*np.pi,kite_yaw_modified[i])
    Transform_Matrix=Transform_Matrix.T
    
    #X_vector
    res=Transform_Matrix.dot(np.array([-1,0,0]))
    xx_kite.append(res[0])
    xy_kite.append(res[1])
    xz_kite.append(res[2])
    
    #Y_vector
    res=Transform_Matrix.dot(np.array([0,-1,0]))
    yx_kite.append(res[0])
    yy_kite.append(res[1])
    yz_kite.append(res[2])
    
    #Z_vector
    res=Transform_Matrix.dot(np.array([0,0,1]))
    zx_kite.append(res[0])
    zy_kite.append(res[1])
    zz_kite.append(res[2])
    
    
        
# zx_kite=-1*np.array(zx_kite)
# zy_kite=-1*np.array(zy_kite)
# zz_kite=-1*np.array(zz_kite)


start=1000
step=5
end=5000
#ax1.plot3D(x[start:end:step], y[start:end:step], z[start:end:step], 'gray')
ax1.plot3D(x[start:end:step], y[start:end:step], z[start:end:step], 'red')
#ax1.quiver(x[start:end-2*step:step], y[start:end-2*step:step], z[start:end-2*step:step], tx[start:end-2*step:step], ty[start:end-2*step:step], tz[start:end-2*step:step], length=10, normalize=True)
ax1.quiver(x[start:end-2*step:step], y[start:end-2*step:step], z[start:end-2*step:step], xx_kite[start:end-2*step:step], xy_kite[start:end-2*step:step], xz_kite[start:end-2*step:step], length=10, normalize=True,color='green')
ax1.quiver(x[start:end-2*step:step], y[start:end-2*step:step], z[start:end-2*step:step], yx_kite[start:end-2*step:step], yy_kite[start:end-2*step:step], yz_kite[start:end-2*step:step], length=10, normalize=True,color='blue')
ax1.quiver(x[start:end-2*step:step], y[start:end-2*step:step], z[start:end-2*step:step], zx_kite[start:end-2*step:step], zy_kite[start:end-2*step:step], zz_kite[start:end-2*step:step], length=10, normalize=True,color='red')
plt.show()
#%%

turn = pow & (vz<0)  & (abs(us) > 0.5)
mask = np.any([flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
fig = plt.figure()
ax1 = plt.axes(projection='3d')
ax1.scatter(x[turn&mask],y[turn&mask],z[turn&mask])
#%% 
mean_el = []
mean_Ft = []
for cycle in range(1,cycle_count):
    mask = (flight_data['cycle'] == cycle)&pow&straight
    mean_el.append(el[mask].mean())
    mean_Ft.append(measured_Ft[mask].mean())

mean_el = np.array(mean_el)
mean_Ft = np.array(mean_Ft)
plt.figure()
plt.scatter(mean_Ft,mean_el*180/np.pi)
plt.xlabel('Mean Cycle Tether force [N]')
plt.ylabel('Mean Cycle Elevation angle [deg]')
plt.grid()

plt.savefig('mean_el.png', dpi = 300)

#%%
cycles_plotted = np.arange(60,75, step=1)
mask = np.any([flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
plt.scatter(aoa[(measured_Ft>4000)&straight&mask],va_mod[(measured_Ft>4000)&straight&mask])