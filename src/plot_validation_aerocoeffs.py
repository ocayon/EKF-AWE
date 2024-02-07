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


path = '../processed_data/aerostructural/'
aero_coeffs = pd.read_csv(path+'v3_aero_coeffs_VSM.csv')

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
turn_right = pow & (us > 0.5)
tun_left = pow & (us < -0.5)
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


aoa_trim_dep = 9.88
aoa_trim_pow = 10.68

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
meas_ax = flight_data.kite_1_ax
meas_ay = flight_data.kite_1_ay
meas_az = flight_data.kite_1_az
a_kcu = np.vstack((np.array(meas_ax),np.array(meas_ay),np.array(meas_az))).T
va = vw-v_kite
meas_ax = flight_data.kite_0_ax
meas_ay = flight_data.kite_0_ay
meas_az = flight_data.kite_0_az
a_kite = np.vstack((np.array(meas_ax),np.array(meas_ay),np.array(meas_az))).T

azimuth =flight_data['kite_azimuth']


# Calculate wind speed based on KCU orientation and wind speed and direction
aoacalc = []
sideslipcalc = []
va_mod = []
slack = []
wvel_calc = []
wdir_calc = []
radius = []
measured_aoa = measured_aoa
measured_ss = -measured_ss-5
for i in range(len(CL_EKF)):

    va_mod.append(np.linalg.norm(va[i]))
    q = 0.5*1.225*kite.area*va_mod[i]**2
    slack.append(tether_len[i]+kite.distance_kcu_kite-np.sqrt(x[i]**2+y[i]**2+z[i]**2))
    
    # at = np.dot(a_kite[i],np.array(v_kite[i])/np.linalg.norm(v_kite[i]))*np.array(v_kite[i])/np.linalg.norm(v_kite[i])
    # # at = np.dot(a_kite[i],np.array(v_kite-vtau_kite)/np.linalg.norm(v_kite-vtau_kite))*np.array(v_kite-vtau_kite)/np.linalg.norm(v_kite-vtau_kite)
    
    # omega_kite = np.cross(a_kite[i]-at,v_kite[i])/(np.linalg.norm(v_kite[i])**2)
    # ICR = np.cross(v_kite[i],omega_kite)/(np.linalg.norm(omega_kite)**2)    
    # radius.append(np.linalg.norm(ICR))

    # Calculate tether orientation based on kite sensor measurements
    Transform_Matrix=R_EG_Body(roll[i]/180*np.pi,pitch[i]/180*np.pi,(meas_yaw[i])/180*np.pi)
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

# radius = np.array(radius)

#%%
azimuth_rate = np.concatenate((np.diff(azimuth), [0]))
course_rate = np.concatenate((np.diff(course), [0]))
pitch_rate = np.concatenate((np.diff(meas_pitch), [0]))
sideslipcalc = np.array(sideslipcalc)
aoacalc = np.array(aoacalc)
turn = pow & (vz<0)
straight = pow & ~turn
turn_right = turn  & (azimuth<0)
turn_left = turn  & (azimuth>0)
straight_right = straight  & (azimuth_rate>0)
straight_left = straight  & (azimuth_rate<0)

#%% Create mask for plotting
# aoa = measured_aoa
# sideslipcalc = measured_ss

cycles_plotted = np.arange(10,60, step=1)
cycles_plotted = np.arange(1,cycle_count, step=1)

mask = np.any([flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
mask = mask & (CD_EKF>0.01)
#%% Plotting
aoa = aoa
mask = mask&straight
aoa[dep] = aoacalc[dep]#-1.927-2.88
aoa[pow] = measured_aoa[pow]
sideslipcalc = measured_ss
from scipy.stats import gaussian_kde
# Function to calculate densities
def calculate_densities(x, y):
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    return z
def plot_probability_density(x,y,axs,xlabel=None,ylabel=None):
    z1 = calculate_densities(x, y)
    sc1 = axs.scatter(x, y, c=z1, cmap='viridis', label = 'Sensor Fusion')
    fig.colorbar(sc1, ax=axs, label='Probability Density')
    axs.set_ylabel(ylabel)
    axs.set_xlabel(xlabel)
    axs.grid()
    axs.legend()
#%%
#%%
# Plot CL^3/CD^2 vs aoa and ss
fig, axs = plt.subplots(2, 1, sharex=False, figsize=(10, 6))

plot_probability_density(aoa[mask], CL_EKF[mask]**3 / CD_EKF[mask]**2, axs[0])
# axs[0].scatter(aero_coeffs['aoa']*180/np.pi,aero_coeffs['CL']**3/aero_coeffs['CD']**2,label='VSM',color = 'orange')
axs[0].set_ylabel('CL^3/CD^2')
axs[0].set_xlabel('Angle of Attack [deg]')
axs[0].set_ylim([0,250])
# Calculate densities for Sideslip Angle plot
plot_probability_density(sideslipcalc[mask], CL_EKF[mask]**3 / CD_EKF[mask]**2, axs[1])
# axs[1].scatter(aero_coeffs['ss']*180/np.pi,aero_coeffs['CL']**3/aero_coeffs['CD']**2,label='VSM',color = 'orange')
axs[1].set_ylabel('CL^3/CD^2')
axs[1].set_xlabel('Sideslip Angle [deg]')
axs[1].set_ylim([0,250])
plt.show()
plt.savefig('CLcube.png',dpi = 300, bbox_inches='tight')

# Plot CL/CD vs aoa and ss
fig, axs = plt.subplots(2, 1, sharex=False, figsize=(10, 6))

plot_probability_density(aoa[mask], CL_EKF[mask]/CD_EKF[mask], axs[0])
# axs[0].scatter(aero_coeffs['aoa']*180/np.pi,aero_coeffs['CL']/aero_coeffs['CD'],label='VSM',color = 'orange')
axs[0].set_ylabel('CL/CD')
axs[0].set_xlabel('Angle of Attack [deg]')
axs[0].set_ylim([0,30])
# Calculate densities for Sideslip Angle plot
plot_probability_density(sideslipcalc[mask], CL_EKF[mask]/CD_EKF[mask], axs[1])
# axs[1].scatter(aero_coeffs['ss']*180/np.pi,aero_coeffs['CL']/aero_coeffs['CD'],label='VSM',color = 'orange')
axs[1].set_ylabel('CL/CD')
axs[1].set_xlabel('Sideslip Angle [deg]')
axs[1].set_ylim([0,30])
plt.show()
plt.savefig('eff.png',dpi = 300, bbox_inches='tight')

# Plot CL vs aoa
# Calculate densities for Angle of Attack plot
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
plot_probability_density(aoa[mask], CL_EKF[mask], axs[0])
# plt.scatter(aero_coeffs['aoa']*180/np.pi,aero_coeffs['CL'],label='VSM',color = 'orange')
axs[0].set_ylabel('CL')
# Plot CL vs aoa
# Calculate densities for Angle of Attack plot
plot_probability_density(aoa[mask], CD_EKF[mask], axs[1])
# plt.scatter(aero_coeffs['aoa']*180/np.pi,aero_coeffs['CD'],label='VSM',color = 'orange')
axs[1].set_ylabel('CD')
axs[1].set_xlabel('Angle of Attack [deg]')
plt.savefig('CLCDalpha.png',dpi = 300, bbox_inches='tight')

# Plot CL vs ss
# Calculate densities for Angle of Attack plot
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10,6))
plot_probability_density(sideslipcalc[mask], CL_EKF[mask], axs[0])
# plt.scatter(aero_coeffs['ss']*180/np.pi,aero_coeffs['CL'],label='VSM',color = 'orange')
axs[0].set_ylabel('CL')

# Plot CD vs ss
# Calculate densities for Angle of Attack plot
plot_probability_density(sideslipcalc[mask], CD_EKF[mask], axs[1])
# plt.scatter(aero_coeffs['ss']*180/np.pi,aero_coeffs['CD'],label='VSM',color = 'orange')
axs[1].set_ylabel('CD')
axs[1].set_xlabel('Sideslip Angle [deg]')
plt.savefig('CLCDss.png',dpi = 300, bbox_inches='tight')

#%%


# Plotaoa vs sideslip
fig, ax = plt.subplots(figsize=(10,6))
plot_probability_density(aoa[mask], sideslipcalc[mask], ax)
ax.set_xlabel('Angle of Attack [deg]')
ax.set_ylabel('Sideslip Angle [deg]')
plt.savefig('aoass.png',dpi = 300, bbox_inches='tight')

# mask = np.any([flight_data['cycle'] == cycle for cycle in cycles_plotted], axis=0)
# Calculate densities for Angle of Attack plot
# fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 10))

# plot_probability_density(aoa[mask&pow], sideslipcalc[mask&pow], axs[0])
# axs[0].axvline(x = aoa_trim_pow, color = 'orange', label = 'Powered trim angle', linestyle = '--')
# plot_probability_density(aoa[mask&dep], sideslipcalc[mask&dep], axs[1])
# axs[1].axvline(x = aoa_trim_dep, color = 'orange', label = 'Depowered trim angle', linestyle = '--')
# axs[0].legend()
# axs[1].legend()


#%% Plot different flight regimes (turns, straight flight, etc.)
# fig, axs = plt.subplots(2, 4, sharex=False, figsize=(15, 10))
# plot_probability_density(aoa[mask&turn_left], CL_EKF[mask&turn_left], axs[0,0],xlabel='Angle of Attack [deg]',ylabel='Lift Coefficient')
# plot_probability_density(aoa[mask&turn_left], CD_EKF[mask&turn_left], axs[1,0],xlabel='Angle of Attack [deg]',ylabel='Drag Coefficient')
# axs[0,0].set_title('Left Turn')
# plot_probability_density(aoa[mask&turn_right], CL_EKF[mask&turn_right], axs[0,1],xlabel='Angle of Attack [deg]',ylabel='Lift Coefficient')
# plot_probability_density(aoa[mask&turn_right], CD_EKF[mask&turn_right], axs[1,1],xlabel='Angle of Attack [deg]',ylabel='Drag Coefficient')
# axs[0,1].set_title('Right Turn')
# plot_probability_density(aoa[mask&straight_left], CL_EKF[mask&straight_left], axs[0,2],xlabel='Angle of Attack [deg]',ylabel='Lift Coefficient')
# plot_probability_density(aoa[mask&straight_left], CD_EKF[mask&straight_left], axs[1,2],xlabel='Angle of Attack [deg]',ylabel='Drag Coefficient')
# axs[0,2].set_title('Left Straight Flight')
# plot_probability_density(aoa[mask&straight_right], CL_EKF[mask&straight_right], axs[0,3],xlabel='Angle of Attack [deg]',ylabel='Lift Coefficient')
# plot_probability_density(aoa[mask&straight_right], CD_EKF[mask&straight_right], axs[1,3],xlabel='Angle of Attack [deg]',ylabel='Drag Coefficient')
# axs[0,3].set_title('Right Straight Flight')
# y_lims = []
# x_lim = [min(aoa[mask]),max(aoa[mask])]
# # Now, set these limits to all plots
# for i in range(4):
#     axs[0, i].set_ylim([0,1.2])
#     axs[1, i].set_ylim([0,0.3])
#     axs[0, i].set_xlim(x_lim)
#     axs[1, i].set_xlim(x_lim)

# #%% Plot different flight regimes (turns, straight flight, etc.) with sideslip
# fig, axs = plt.subplots(2, 4, sharex=False, figsize=(15, 10))
# plot_probability_density(sideslipcalc[mask&turn_left], CL_EKF[mask&turn_left], axs[0,0],xlabel='Sideslip Angle [deg]',ylabel='Lift Coefficient')
# plot_probability_density(sideslipcalc[mask&turn_left], CD_EKF[mask&turn_left], axs[1,0],xlabel='Sideslip Angle [deg]',ylabel='Drag Coefficient')
# axs[0,0].set_title('Left Turn')
# plot_probability_density(sideslipcalc[mask&turn_right], CL_EKF[mask&turn_right], axs[0,1],xlabel='Sideslip Angle [deg]',ylabel='Lift Coefficient')
# plot_probability_density(sideslipcalc[mask&turn_right], CD_EKF[mask&turn_right], axs[1,1],xlabel='Sideslip Angle [deg]',ylabel='Drag Coefficient')
# axs[0,1].set_title('Right Turn')
# plot_probability_density(sideslipcalc[mask&straight_left], CL_EKF[mask&straight_left], axs[0,2],xlabel='Sideslip Angle [deg]',ylabel='Lift Coefficient')
# plot_probability_density(sideslipcalc[mask&straight_left], CD_EKF[mask&straight_left], axs[1,2],xlabel='Sideslip Angle [deg]',ylabel='Drag Coefficient')
# axs[0,2].set_title('Left Straight Flight')
# plot_probability_density(sideslipcalc[mask&straight_right], CL_EKF[mask&straight_right], axs[0,3],xlabel='Sideslip Angle [deg]',ylabel='Lift Coefficient')
# plot_probability_density(sideslipcalc[mask&straight_right], CD_EKF[mask&straight_right], axs[1,3],xlabel='Sideslip Angle [deg]',ylabel='Drag Coefficient')
# axs[0,3].set_title('Right Straight Flight')

# x_lim = [min(sideslipcalc[mask]),max(sideslipcalc[mask])]
# # set these limits to all plots
# for i in range(4):
#     axs[0, i].set_ylim([0,1.2])
#     axs[1, i].set_ylim([0,0.3])
#     axs[0, i].set_xlim(x_lim)
    # axs[1, i].set_xlim(x_lim)
