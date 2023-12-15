import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import kappa, z0, kite_model, year, month, day
from utils import get_tether_end_position,  R_EG_Body, calculate_angle,project_onto_plane, create_kite

#%%
plt.close('all')


path = '../results/'+kite_model+'/'
file_name = kite_model+'_'+year+'-'+month+'-'+day

results = pd.read_csv(path+file_name+'_res_GPS.csv')
flight_data = pd.read_csv(path+file_name+'_fd.csv')

kite = create_kite(kite_model)
#%% Define flight phases and count cycles
up = (flight_data['kcu_actual_depower']-min(flight_data['kcu_actual_depower']))/(max(flight_data['kcu_actual_depower'])-min(flight_data['kcu_actual_depower']))
us = (flight_data['kcu_actual_steering'])/max(abs(flight_data['kcu_actual_steering']))
dep = (flight_data['ground_tether_reelout_speed'] < 0) & (up>0.3)
pow = (flight_data['ground_tether_reelout_speed'] > 0) & (up<0.3)
trans = ~pow & ~dep
turn = pow & (abs(us) > 0.25)
straight = pow & (abs(us) < 0.25)

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
meas_pitch = flight_data['kite_0_pitch']
meas_pitch1 = flight_data['kite_1_pitch']
meas_roll = flight_data['kite_0_roll']
meas_roll1 = flight_data['kite_1_roll']
meas_yaw = flight_data['kite_0_yaw']-90
meas_yaw1 = flight_data['kite_1_yaw']-90
meas_tetherlen = flight_data['ground_tether_length']


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
pitch = results.pitch
roll = results.roll
yaw = results.yaw
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
meas_ax = flight_data.kite_1_ax
meas_ay = flight_data.kite_1_ay
meas_az = flight_data.kite_1_az
acc = np.vstack((np.array(meas_ax),np.array(meas_ay),np.array(meas_az))).T
va = vw-v_kite


# Calculate wind speed based on KCU orientation and wind speed and direction
aoacalc = []
sideslipcalc = []
va_mod = []
slack = []
wvel_calc = []
wdir_calc = []

measured_aoa = measured_aoa
measured_ss = -measured_ss-5
for i in range(len(CL_EKF)):

    va_mod.append(np.linalg.norm(va[i]))
    q = 0.5*1.225*kite.area*va_mod[i]**2
    slack.append(tether_len[i]+kite.distance_kcu_kite-np.sqrt(x[i]**2+y[i]**2+z[i]**2))

    # Calculate tether orientation based on kite sensor measurements
    Transform_Matrix=R_EG_Body(roll[i]/180*np.pi,(pitch[i]-90)/180*np.pi,(meas_yaw[i])/180*np.pi)
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
    sideslipcalc.append(90-calculate_angle(ex_kite,va_proj) )        # Sideslip angle

pitch_rate = np.concatenate((np.diff(meas_pitch), [0]))
sideslipcalc = np.array(sideslipcalc)
aoacalc = np.array(aoacalc)
turn = pow & (meas_pitch<0)
straight = pow & (meas_pitch>0)
#%% Create mask for plotting

cycles_plotted = np.arange(1,6, step=1)
cycles_plotted = np.arange(0,cycle_count, step=1)

mask = np.any([flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)

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


correlation_matrix = np.corrcoef(aoa[pow],sideslipcalc[pow])
correlation_coefficient = correlation_matrix[0,1]
print(f"Correlation Coefficient aoa & ss: {correlation_coefficient}")
#%% Plot wind speed

# Plot horizontal wind speed
plt.figure()
plt.plot(t[mask],uf[mask])
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
plt.fill_between(t, 140, where=straight, color=colors[0], alpha=0.2)
plt.fill_between(t, 140, where=turn, color=colors[1], alpha=0.2)
plt.fill_between(t, 140, where=dep, color=colors[2], alpha=0.2)
plt.fill_between(t, 140, where=trans, color=colors[3], alpha=0.2)
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

# plt.savefig('aerodynamic_coefficients_plot.png', dpi=300)



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


plt.figure()
plt.scatter(us,CS_EKF)

correlation_matrix = np.corrcoef(CS_EKF[mask&pow],us[mask&pow])
correlation_coefficient = correlation_matrix[0,1]
print(f"Correlation Coefficient u_s & C_S: {correlation_coefficient}")





# Plot efficiency
plt.figure()
plt.plot(t[mask], CL_EKF[mask]/CD_EKF[mask])
plt.fill_between(t[mask], CL_EKF[mask]/CD_EKF[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
plt.fill_between(t[mask], CL_EKF[mask]/CD_EKF[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
plt.fill_between(t[mask], CL_EKF[mask]/CD_EKF[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')
plt.fill_between(t[mask], CL_EKF[mask]/CD_EKF[mask], where=trans[mask], color='red', alpha=0.2, label='Transition')
plt.ylabel('Efficiency')
plt.xlabel('Time')
plt.ylim([0, 20])
plt.grid()
plt.legend()

#%% Plot bias tether length

plt.figure()
plt.plot(t, results.bias_lt)
plt.fill_between(t[mask], results.bias_lt[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
plt.fill_between(t[mask], results.bias_lt[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
plt.fill_between(t[mask], results.bias_lt[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')

plt.ylabel('Bias tether length')
plt.xlabel('Time')
plt.grid()
plt.legend()


#%%
# Plot slack
slack = np.array(slack)
plt.figure()
plt.plot(t[mask], slack[mask])
plt.fill_between(t[mask], slack[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
plt.fill_between(t[mask], slack[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
plt.fill_between(t[mask], slack[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')
plt.fill_between(t[mask], slack[mask], where=trans[mask], color='red', alpha=0.2, label='Transition')
plt.ylabel('Slack [m]')
plt.xlabel('Time')
plt.grid()
plt.legend()


# Plot tether force
plt.figure()
plt.plot(t[mask], Ft_mod[mask], label='Tether force kite')
plt.plot(t[mask], measured_Ft[mask], label='Measured tether force ground')
plt.xlabel('Time')
plt.ylabel('Tether force [N]')
plt.grid()
plt.legend()

tether_loss = (Ft_mod-measured_Ft)/Ft_mod*100

plt.figure()
plt.plot(t[mask], tether_loss[mask], label='Tether force kite')
plt.fill_between(t[mask], tether_loss[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
plt.fill_between(t[mask], tether_loss[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
plt.fill_between(t[mask], tether_loss[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')
plt.fill_between(t[mask], tether_loss[mask], where=trans[mask], color='red', alpha=0.2, label='Transition')
plt.xlabel('Time')
plt.ylabel('Tether force loss (%)')
plt.grid()
plt.legend()


#%% Plot trajectory
meas_z = flight_data['kite_0_rz']
# Plot height
plt.figure()
plt.plot(t[mask], z[mask])
plt.plot(t[mask],meas_z[mask])
plt.fill_between(t[mask], z[mask], where=straight[mask], color='blue', alpha=0.2, label='Straight')
plt.fill_between(t[mask], z[mask], where=turn[mask], color='green', alpha=0.2, label='Turn')
plt.fill_between(t[mask], z[mask], where=dep[mask], color='orange', alpha=0.2, label='Depower')
plt.fill_between(t[mask], z[mask], where=trans[mask], color='red', alpha=0.2, label='Transition')
plt.xlabel('Time')
plt.ylabel('Height [m]')
plt.grid()
plt.legend()


#%% Plot euler angles

# Unwrap the phase
unwrapped_angles = np.unwrap(np.radians(meas_yaw))
# Convert back to degrees
meas_yaw = np.degrees(unwrapped_angles)

# Unwrap the phase
unwrapped_angles = np.unwrap(np.radians(meas_yaw1))
# Convert back to degrees
meas_yaw1 = np.degrees(unwrapped_angles)

# Unwrap the phase
unwrapped_angles = np.unwrap(np.radians(yaw))
# Convert back to degrees
yaw = np.degrees(unwrapped_angles)


# Plot pitch
plt.figure()
plt.plot(t[mask], pitch[mask], label='Pitch EKF respect to v_kite')
plt.plot(t[mask], meas_pitch[mask], label='Pitch kite sensor')
plt.plot(t[mask], meas_pitch1[mask], label='Pitch KCU sensor')

plt.xlabel('Time')
plt.ylabel('Pitch [deg]')
plt.grid()
plt.legend()

# Plot roll
plt.figure()
plt.plot(t[mask], roll[mask], label='Roll EKF respect to v_kite')
plt.plot(t[mask], meas_roll[mask]-5, label='Roll kite sensor')
# plt.plot(t[mask], meas_roll1[mask], label='Roll KCU sensor')

plt.xlabel('Time')
plt.ylabel('Roll [deg]')
plt.grid()
plt.legend()

# Plot yaw
plt.figure()
plt.plot(t[mask], yaw[mask]-360, label='Yaw EKF respect to v_kite')
plt.plot(t[mask], meas_yaw[mask]-90, label='Yaw kite sensor')
plt.plot(t[mask], meas_yaw1[mask]-90, label='Yaw KCU sensor')

plt.xlabel('Time')
plt.ylabel('Yaw [deg]')
plt.grid()
plt.legend()

#%% 
r = np.sqrt(x**2+y**2+z**2)

plt.figure()
plt.plot(t,r,label = 'GPS radius')
plt.plot(t,meas_tetherlen-results.bias_lt,label = 'Meas. tether length')
plt.plot(t,tether_len,label = 'Tether length')
plt.xlabel('Time')
plt.ylabel('Distance [m]')
plt.grid()
plt.legend()
#%%
# start = 2000
# end = start+36000
# hours = [13,14,15]

# h_ticks = np.arange(0,350,50)
# heights = np.arange(100, 301, step=50)
# ground_wdir = measured_wdir[start:end]
# ground_wvel = measured_wvel[start:end]
# mindir = []
# maxdir = []
# minvel = []
# maxvel = []

# fig_vel, ax_vel = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
# fig_dir, ax_dir = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
# for i in range(len(hours)):


#     ax_vel[i].fill_betweenx(era5_heights[:-2], era5_wvel[i,:-2], era5_wvel[i+1,:-2], color='lightgrey', alpha=0.5)
#     ax_vel[i].scatter(wvel[start:end],z[start:end],color = 'lightblue',alpha = 0.5)

#     ax_vel[i].boxplot([measured_wvel[start:end]],positions = [10],vert = False,widths=(20))
#     ax_vel[i].set_title(str(hours[i])+'h')
#     ax_vel[i].set_xlabel('Wind velocity')
#     ax_vel[i].set_yticks(h_ticks)
#     ax_vel[i].set_yticklabels(h_ticks)
#     ax_vel[i].grid()
    

#     ax_dir[i].fill_betweenx(era5_heights[:-2], era5_wdir[i,:-2], era5_wdir[i+1,:-2], color='lightgrey', alpha=0.5)
#     ax_dir[i].scatter(wdir[start:end]*180/np.pi,z[start:end],color = 'lightblue',alpha = 0.5)

#     ax_dir[i].boxplot([measured_wdir[start:end]],positions = [10],vert = False,widths=(20))
#     ax_dir[i].set_title(str(hours[i])+'h')
#     ax_dir[i].set_xlabel('Wind direction')
#     ax_dir[i].set_yticks(h_ticks)
#     ax_dir[i].set_yticklabels(h_ticks)
#     ax_dir[i].grid()
    
#     # ax_dir[i].set_xlim([150, 220])
#     ax_vel[i].set_xlim([0, 14])
#     if i == 0:
#         ax_dir[i].set_ylabel('Height')
#         ax_vel[i].set_ylabel('Height')
        
#     start = end
#     end = start+36000
    
# fig_vel.legend(['ERA5','EKF (GPS&groundwvel)','EKF with va','Ground measurement'])
# fig_dir.legend(['ERA5','EKF (GPS&groundwvel)','EKF with va','Ground measurement'])
# # fig_dir.savefig('wind_direction.png',dpi = 300)
# # fig_vel.savefig('wind_velocity.png',dpi = 300)
#%%

ax = np.array(flight_data['kite_0_ax'])
ay = np.array(flight_data['kite_0_ay'])
az = np.array(flight_data['kite_0_az'])
ax1 = np.array(flight_data['kite_1_ax'])
ay1 = np.array(flight_data['kite_1_ay'])
az1 = np.array(flight_data['kite_1_az'])

plt.figure()
plt.plot(t[mask],ax[mask],label = 'ax')
plt.plot(t[mask],ax[mask],label = 'ax')
plt.plot(t[mask],ay[mask],label = 'ay')
plt.plot(t[mask],az[mask],label = 'az')
plt.fill_between(t[mask], ax[mask], where=straight[mask], color='blue', alpha=0.2)
plt.fill_between(t[mask], ax[mask], where=turn[mask], color='green', alpha=0.2)
plt.fill_between(t[mask], ax[mask], where=dep[mask], color='orange', alpha=0.2)

plt.legend()
plt.grid()

# a0 = np.sqrt(ax**2+ay**2+az**2)
# a1 = np.sqrt(ax1**2+ay1**2+az1**2)
# plt.figure()
# plt.plot(t,a0)
# plt.plot(t,a1)



#%%
vx = np.array(flight_data['kite_0_vx'])
vy = np.array(flight_data['kite_0_vy'])
vz = np.array(flight_data['kite_0_vz'])
vx1 = np.array(flight_data['kite_1_vx'])
vy1 = np.array(flight_data['kite_1_vy'])
vz1 = np.array(flight_data['kite_1_vz'])

plt.legend()
plt.grid()
vk = np.linalg.norm(v_kite,axis=1)
v0 = np.sqrt(vx**2+vy**2+vz**2)
v1 = np.sqrt(vx1**2+vy1**2+vz1**2)
plt.figure()
plt.plot(t,v0,label = 'Kite measurement')
plt.plot(t,v1,label =  'KCU measurement')
plt.plot(t,vk,label =  'EKF')

plt.legend()
plt.grid()

#%% Make polynomial fits to aerodynamic coefficients
# aoa = measured_aoa
# sideslipcalc = measured_ss

mask = pow & (t>100) 

CLt = np.sqrt(CL_EKF**2+CS_EKF**2)

coefficients = np.polyfit(aoa[mask], CLt[mask], 2)
polynomial = np.poly1d(coefficients)
alpha_fit = np.linspace(min(aoa[mask]), max(aoa[mask]), 100)  # Create a range of alpha values for the trendline
plt.figure()
plt.scatter(aoa[mask],CLt[mask],alpha = 0.5)   
plt.plot(alpha_fit, polynomial(alpha_fit), label='Trendline (Degree 1)', color='r') 

coefficients = np.polyfit(aoa[mask], CD_EKF[mask], 2)
polynomial = np.poly1d(coefficients)
alpha_fit = np.linspace(min(aoa[mask]), max(aoa[mask]), 100)  # Create a range of alpha values for the trendline
plt.figure()
plt.scatter(aoa[mask],CD_EKF[mask],alpha = 0.5)   
plt.plot(alpha_fit, polynomial(alpha_fit), label= 'Trendline (Degree 1)', color='r') 


mask = pow & (t>100) 
coefficients = np.polyfit(sideslipcalc[mask], CLt[mask], 2)
polynomial = np.poly1d(coefficients)
alpha_fit = np.linspace(min(sideslipcalc[mask]), max(sideslipcalc[mask]), 100)  # Create a range of alpha values for the trendline
plt.figure()
plt.scatter(sideslipcalc[mask],CLt[mask],alpha = 0.5)   
plt.plot(alpha_fit, polynomial(alpha_fit), label='Trendline (Degree 1)', color='r') 

coefficients = np.polyfit(sideslipcalc[mask], CD_EKF[mask], 2)
polynomial = np.poly1d(coefficients)
alpha_fit = np.linspace(min(sideslipcalc[mask]), max(sideslipcalc[mask]), 100)  # Create a range of alpha values for the trendline
plt.figure()
plt.scatter(sideslipcalc[mask],CD_EKF[mask],alpha = 0.5)   
plt.plot(alpha_fit, polynomial(alpha_fit), label='Trendline (Degree 1)', color='r') 


mask = pow & (t>100)
coefficients = np.polyfit(us[mask], CL_EKF[mask], 2)
polynomial = np.poly1d(coefficients)
alpha_fit = np.linspace(min(us[mask]), max(us[mask]), 100)  # Create a range of alpha values for the trendline
plt.figure()
plt.scatter(us[mask],CL_EKF[mask],alpha = 0.5)   
plt.plot(alpha_fit, polynomial(alpha_fit), label='Trendline (Degree 2)', color='r') 

coefficients = np.polyfit(us[mask], CD_EKF[mask], 2)
polynomial = np.poly1d(coefficients)
alpha_fit = np.linspace(min(us[mask]), max(us[mask]), 100)  # Create a range of alpha values for the trendline
plt.figure()
plt.scatter(us[mask],CD_EKF[mask],alpha = 0.5)   
plt.plot(alpha_fit, polynomial(alpha_fit), label='Trendline (Degree 2)', color='r') 



mask = pow & (t>100) 
coefficients = np.polyfit(us[mask], CS_EKF[mask], 2)
polynomial = np.poly1d(coefficients)
alpha_fit = np.linspace(min(us[mask]), max(us[mask]), 100)  # Create a range of alpha values for the trendline
plt.figure()
mask = pow & (t>100) & (pitch_rate<0)
plt.scatter(us[mask],CS_EKF[mask],alpha = 0.5)   
mask = pow & (t>100) & (pitch_rate>0)
# plt.scatter(us[mask],CS_EKF[mask],alpha = 0.5)   
plt.plot(alpha_fit, polynomial(alpha_fit), label='Trendline (Degree 2)', color='r') 

mask = np.any([flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
