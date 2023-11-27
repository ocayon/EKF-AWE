import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_tether_end_position, state_noise_matrices, observation_matrices, R_EG_Body, calculate_angle,project_onto_plane ,read_data, rank_observability_matrix,read_data_new,get_measurements


#%%
plt.close('all')

model = 'v9'
year = '2023'
month = '10'
day = '26'

if model == 'v3':
    from v3_properties import *
elif model == 'v9':
    from v9_properties import *

path = './results/'+model+'/'
file_name = model+'_'+year+'-'+month+'-'+day

results = pd.read_csv(path+file_name+'_res_GPS.csv')
flight_data = pd.read_csv(path+file_name+'_fd.csv')

# results = results.iloc[8000:50000].reset_index()
# flight_data = flight_data.iloc[8000:50000].reset_index()


# windpath = './data/'
# windfile = 'era5_data_'+year+'_'+month+'_'+day+'.npy'

# data_dict = np.load(windpath+windfile, allow_pickle=True)

# # Extract arrays and information
# era5_hours = data_dict.item()['hours']
# era5_heights = data_dict.item()['heights']
# era5_wvel = data_dict.item()['wvel']
# era5_wdir = data_dict.item()['wdir']

#%%
pow = (flight_data['ground_tether_reelout_speed'] > 0) & (flight_data['kcu_set_depower'] < 23)
turn = (pow) & (abs(flight_data['kcu_set_steering']) > 20)
straight = (pow) & (abs(flight_data['kcu_set_steering']) < 20)

# str_left = (straight) & (np.gradient(flight_data['kite_course'])>0 )
# str_right = (straight) & (np.gradient(flight_data['kite_course'])<0 )

measured_wdir = -flight_data['ground_wind_direction']-90+360
measured_wvel = flight_data['ground_wind_velocity']
measured_uf = measured_wvel*kappa/np.log(10/z0)
measured_va = flight_data['kite_apparent_windspeed']
measured_Ft = flight_data['ground_tether_force']
measured_aoa = flight_data['kite_angle_of_attack']
# measured_ss = flight_data['kite_sideslip_angle']
measured_aoa = np.array(measured_aoa)
# measured_aoa = np.convolve(measured_aoa, np.ones(10)/10, mode='same')
meas_pitch = flight_data['kite_0_pitch']
meas_pitch1 = flight_data['kite_1_pitch']
meas_roll = flight_data['kite_0_roll']+4
meas_roll1 = flight_data['kite_1_roll']+4
meas_yaw = flight_data['kite_0_yaw']-90
meas_yaw1 = flight_data['kite_1_yaw']-90
# meas_v = np.vstack((np.array(flight_data['vx1']),np.array(flight_data['vy1']),np.array(flight_data['vz1']))).T
# meas_a = np.vstack((np.array(flight_data['ax']),np.array(flight_data['ay']),np.array(flight_data['az']))).T
t = flight_data.time



#%%
x = results.x
y = results.y
z = results.z
vx = results.vx
vy = results.vy
vz = results.vz

uf = results.uf
wdir = results.wdir
pitch = results.pitch


CL_EKF = results.CL
CD_EKF = results.CD
# cd_kcu = results.cd_kcu
CS_EKF = results.CS
CLw = results.CLw
CDw = results.CDw
aoa = results.aoa
tether_len = results.Lt
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
vwh = []
angle = []
L = []
D = []
aoacalc = []
sideslipcalc = []

va_mod = []
slack = []
# acc = []
Fa = []
omega = []
CL = np.zeros(len(x))
CD = np.zeros(len(x))
CS = np.zeros(len(x))
for i in range(len(CL)):

    va_mod.append(np.linalg.norm(va[i]))
    q = 0.5*1.225*A_kite*va_mod[i]**2
    # L.append(CL[i]*q)
    # D.append(CD[i]*q)
    slack.append(tether_len[i]+l_bridle-np.sqrt(x[i]**2+y[i]**2+z[i]**2))
    # Fa.append(np.sqrt(CL[i]**2+CD[i]**2+CS[i]**2)*q)
    dir_D = va[i]/va_mod[i]

    r_mod = np.linalg.norm(r_kite[i])

    omega.append(np.cross(r_kite[i], v_kite[i])/r_mod**2)
    Fa = -Ft[i,:]+m_kite*acc[i,:]+m_kite*np.array([0,0,-9.81])

    D = np.dot(Fa,dir_D)
    dir_L = Ft[i]/Ft_mod[i] - np.dot(Ft[i]/Ft_mod[i],dir_D)*dir_D
    L = np.dot(Fa,dir_L)
    dir_S = np.cross(dir_L,dir_D)
    S = np.dot(Fa,dir_S)

    CD[i] = np.linalg.norm(D)/(0.5*rho*A_kite*va_mod[i]**2)
    CL[i] = np.linalg.norm(L)/(0.5*rho*A_kite*va_mod[i]**2)
    CS[i] = np.linalg.norm(S)/(0.5*rho*A_kite*va_mod[i]**2)
    # Calculate angle of attack based on orientation angles and estimated wind speed
    Transform_Matrix=R_EG_Body(meas_roll1[i]/180*np.pi,meas_pitch1[i]/180*np.pi,(meas_yaw1[i])/180*np.pi)
    #    Transform_Matrix=R_EG_Body(kite_roll[i]/180*np.pi,kite_pitch[i]/180*np.pi,kite_yaw_modified[i])
    Transform_Matrix=Transform_Matrix.T
    
    #X_vector
    ex_kite=Transform_Matrix.dot(np.array([1,0,0]))
    #Y_vector
    ey_kite=Transform_Matrix.dot(np.array([0,1,0]))
    #Z_vector
    ez_kite=Transform_Matrix.dot(np.array([0,0,1]))

    va_proj = project_onto_plane(va[i], -ey_kite)           # Projected apparent wind velocity onto kite y axis
    aoacalc.append(calculate_angle(-ex_kite,va_proj))        # Angle of attack
    va_proj = project_onto_plane(va[i], ez_kite)           # Projected apparent wind velocity onto kite z axis
    sideslipcalc.append(calculate_angle(ey_kite,va_proj)-90)   # Sideslip angle





CLt = np.sqrt(CL_EKF**2+   CS_EKF**2)


#%%

up = (flight_data['kcu_actual_depower']-min(flight_data['kcu_actual_depower']))/(max(flight_data['kcu_actual_depower'])-min(flight_data['kcu_actual_depower']))
us = (flight_data['kcu_actual_steering'])/max(abs(flight_data['kcu_actual_steering']))
print(min(us))
turn = (flight_data['ground_tether_reelout_speed'] > 0) & (abs(us) > 0.5)
straight = (flight_data['ground_tether_reelout_speed'] > 0) & (abs(us) < 0.5)
dep = (flight_data['ground_tether_reelout_speed'] < 0) & (up>0.1)
trans = (flight_data['ground_tether_reelout_speed'] < 0) & (up<0.1)
pow = (flight_data['ground_tether_reelout_speed'] > 0) & (up<0.1)
# turn_right = (flight_data['ground_tether_reelout_speed'] > 0) & (abs(flight_data['kcu_set_steering']) > 10) & (np.gradient(flight_data['kite_course'])<0 )
# turn_left = (flight_data['ground_tether_reelout_speed'] > 0) & (abs(flight_data['kcu_set_steering']) > 10) & (np.gradient(flight_data['kite_course'])>0 )

plt.figure()
plt.plot(aoacalc)
plt.plot(aoa)
plt.plot(measured_aoa)
plt.grid()

plt.figure()
plt.plot(sideslipcalc)
# plt.plot(measured_ss)
plt.grid()


#%% Plot aero coeffs

# pow_res = results[flight_data['ground_tether_reelout_speed']>0]
# pow_straight = pow_res[abs(flight_data['kcu_set_steering'])<10]
# pow_turn = pow_res[abs(flight_data['kcu_set_steering'])>10]
# dep_res = results[(flight_data['ground_tether_reelout_speed'] < 0) & (flight_data['kcu_set_depower'] > 25)]

# degree = 2  # Choose the degree of the polynomial (you can adjust this)
# coefficients = np.polyfit(pow_res.aoa, pow_res.CLw, degree)
# polynomial = np.poly1d(coefficients)
# alpha_fit = np.linspace(min(pow_res.aoa), max(pow_res.aoa), 100)  # Create a range of alpha values for the trendline
# Cl_fit = polynomial(alpha_fit)  # Calculate Cl values for the trendline


# plt.figure()
# plt.scatter(pow_res.aoa,pow_res.CLw)
# plt.scatter(dep_res.aoa,dep_res.CLw)
# plt.scatter(pow_straight.aoa,pow_straight.CLw)
# # plt.scatter(pow_turn.aoa,pow_turn.CLw)
# plt.plot(alpha_fit, Cl_fit, label=f'Trendline (Degree {degree})', color='r')

# plt.figure()
# plt.scatter(aoa[str_left],CDw[str_left],alpha = 0.5)
# plt.scatter(aoa[str_right],CDw[str_right],alpha = 0.5)

# Add a color map to represent point density
# plt.hist2d(aoa[str_left], CDw[str_left], bins=(50, 50), cmap=plt.cm.jet)

# Add a color bar to the plot
# plt.colorbar()

# plt.figure()
# # plt.scatter(aoa[turn],CLw[turn],alpha = 0.5)

# # Add a color map to represent point density
# # plt.hist2d(aoa[str_right], CDw[str_right], bins=(50, 50), cmap=plt.cm.jet)

# # Add a color bar to the plot
# plt.colorbar()




#%%
# plt.scatter(pow_turn.aoa,pow_turn.CDw)


plt.figure()
plt.plot(t,cd_kcu/CDw*100)
plt.xlabel('Time')
plt.ylabel('Cd kcu/Cd wing [%]')
#%% AOA vs pitch

# Smooth measured aoa

colors = ['lightblue', 'lightgreen', 'lightcoral', (0.75, 0.6, 0.8)]
plt.figure()
plt.plot(t,aoa,label = 'AoA')
plt.plot(t,aoacalc,label = 'AoA imposed orientation')
plt.plot(t,measured_aoa+ 10,label = 'AoA measured')
plt.fill_between(t, 40, where=straight, color=colors[0], alpha=0.2)
plt.fill_between(t, 40, where=turn, color=colors[1], alpha=0.2)
plt.fill_between(t, 40, where=dep, color=colors[2], alpha=0.2)
plt.fill_between(t, 40, where=trans, color=colors[3], alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Angle of attack [deg]')
plt.legend()
plt.grid()
colors = ['lightblue', 'lightgreen', 'lightcoral', (0.75, 0.6, 0.8)]


#%%
plt.figure()
plt.plot(t,sideslipcalc)
plt.fill_between(t, sideslipcalc, where=straight, color=colors[0], alpha=0.2)
plt.fill_between(t, sideslipcalc, where=turn, color=colors[1], alpha=0.2)
plt.fill_between(t, sideslipcalc, where=dep, color=colors[2], alpha=0.2)
plt.fill_between(t, sideslipcalc, where=trans, color=colors[3], alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Sideslip angle [deg]')
plt.grid()


sideslipcalc = np.array(sideslipcalc)
aoacalc = np.array(aoa)

mask = (sideslipcalc<0)& (sideslipcalc>-25)&(pow)#&(aoacalc>3)&(aoacalc<5)
mask = (up<0.1)&(aoacalc<10)&(z>180)
coefficients = np.polyfit(sideslipcalc[mask], CL_EKF[mask], 3)
polynomial = np.poly1d(coefficients)
alpha_fit = np.linspace(min(sideslipcalc[mask]), max(sideslipcalc[mask]), 100)  # Create a range of alpha values for the trendline
plt.figure()
plt.scatter(sideslipcalc[mask],CL_EKF[mask],alpha = 0.5)   
plt.plot(alpha_fit, polynomial(alpha_fit), label=f'Trendline (Degree 2)', color='r') 

coefficients = np.polyfit(sideslipcalc[mask], CD_EKF[mask], 2)
polynomial = np.poly1d(coefficients)
alpha_fit = np.linspace(min(sideslipcalc[mask]), max(sideslipcalc[mask]), 100)  # Create a range of alpha values for the trendline
plt.figure()
plt.scatter(sideslipcalc[mask],CD_EKF[mask],alpha = 0.5)   
plt.plot(alpha_fit, polynomial(alpha_fit), label=f'Trendline (Degree 2)', color='r') 



coefficients = np.polyfit(aoacalc[mask], CL[mask], 2)
polynomial = np.poly1d(coefficients)
alpha_fit = np.linspace(min(aoacalc[mask]), max(aoacalc[mask]), 100)  # Create a range of alpha values for the trendline
plt.figure()
plt.scatter(aoacalc[mask],CL[mask],alpha = 0.5)   
plt.plot(alpha_fit, polynomial(alpha_fit), label=f'Trendline (Degree 1)', color='r') 



# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# start = 0
# end = 10000
# ax.plot(x[start:end], y[start:end], z[start:end], color='b')
        
# # x = pow_turn.x
# # y = pow_turn.y
# # z = pow_turn.z
# # for i in range(1, len(pow_turn)):
# #     if pow_turn.index[i] == pow_turn.index[i - 1] + 1:
# #         ax.plot([x.iloc[i - 1], x.iloc[i]], [y.iloc[i - 1], y.iloc[i]], [z.iloc[i - 1], z.iloc[i]], color='orange')

# # Set labels for the axes



#%%
# custom_labels = ['13', '14', '15','16']
# x_ticks = [0, 36000, 72000,108000]

# Plot horizontal wind speed
plt.figure()
plt.plot(uf)

plt.plot(measured_uf)
plt.xlabel('Time')
plt.ylabel('Friction velocity')
# plt.xticks(x_ticks, custom_labels)
plt.grid()

# Plot horizontal wind speed
plt.figure()
plt.plot(wvel)

# plt.plot(measured_uf/kappa*np.log(z/z0))
plt.xlabel('Time')
plt.ylabel('Horizontal Wind Speed')
# plt.xticks(x_ticks, custom_labels)
plt.grid()

plt.figure()
plt.plot(t,wdir*180/np.pi)
plt.fill_between(t, 140, where=straight, color=colors[0], alpha=0.2)
plt.fill_between(t, 140, where=turn, color=colors[1], alpha=0.2)
plt.fill_between(t, 140, where=dep, color=colors[2], alpha=0.2)
plt.fill_between(t, 140, where=trans, color=colors[3], alpha=0.2)

plt.xlabel('Time')
plt.ylabel('Horizontal Wind Direction')
# plt.xticks(x_ticks, custom_labels)
plt.grid()


#%%
# Plot lift coefficient
plt.figure()
plt.plot(t,CLw, label = 'Tether model')
plt.plot(t,CL_EKF,label = 'Sensor fusion')
plt.plot(t,CLt)

plt.xlabel('Time')
plt.ylabel('Lift Coefficient')
plt.fill_between(t, CLw, where=straight, color=colors[0], alpha=0.2)
plt.fill_between(t, CLw, where=turn, color=colors[1], alpha=0.2)
plt.fill_between(t, CLw, where=dep, color=colors[2], alpha=0.2)
plt.fill_between(t, CLw, where=trans, color=colors[3], alpha=0.2)
# plt.xticks(x_ticks, custom_labels)
plt.grid()
plt.legend()

plt.figure()
plt.plot(t,CDw, label = 'Tether model')
plt.plot(t,CD_EKF,label = 'Sensor fusion')
plt.xlabel('Time')
plt.ylabel('Drag Coefficient')
# plt.xticks(x_ticks, custom_labels)
plt.grid()
plt.legend()
# Plot drag coefficient
plt.figure()
plt.plot(CS)

plt.xlabel('Time')
plt.ylabel('Side Coefficient')
# plt.xticks(x_ticks, custom_labels)
plt.grid()

# Plot side force coefficient
plt.figure()
plt.plot(CLw/CDw)
plt.xlabel('Time')
plt.ylabel('CL/CD')
# plt.xticks(x_ticks, custom_labels)
plt.grid()

# # Plot lift
# plt.figure()
# plt.plot(L)
# plt.xlabel('Time')
# plt.ylabel('Lift')
# # plt.xticks(x_ticks, custom_labels)
# plt.grid()

# # Plot drag
# plt.figure()
# plt.plot(D)
# plt.xlabel('Time')
# plt.ylabel('Drag')
# # plt.xticks(x_ticks, custom_labels)
# plt.grid()

# Plot apparent velocity
plt.figure()
plt.plot(va_mod)
plt.plot(measured_va)
plt.xlabel('Time')
plt.ylabel('Apparent Velocity')
# plt.xticks(x_ticks, custom_labels)
plt.grid()

# Plot slack
fig, ax1 = plt.subplots()
ax1.plot(slack, color='tab:blue', label='Horizontal Wind Speed')
ax1.set_xlabel('Time')
ax1.set_ylabel('Slack')
ax2 = ax1.twinx()
# ax2.plot(measured_Ft, color='tab:red', label='Vertical Wind Speed')
ax2.plot(Ft_mod, color='tab:orange', label='Tether Force Kite')
ax2.set_ylabel('Tether Force Kite', color='tab:orange')
plt.show()

# plt.grid()

# fig, ax1 = plt.subplots()
# ax1.plot(Fa, color='tab:blue', label='Horizontal Wind Speed')
# ax1.set_xlabel('Time')
# ax1.set_ylabel('Aerodynamic force')
# ax2 = ax1.twinx()
# # ax2.plot(measured_Ft, color='tab:red', label='Vertical Wind Speed')
# ax2.plot(Ft_mod, color='tab:orange', label='Tether Force Kite')
# ax2.set_ylabel('Tether Force Kite', color='tab:orange')

# plt.grid()

start = 0
end = 40000
# Plot apparent velocity
plt.figure()
plt.plot(z[start:end])

plt.plot(flight_data['kite_0_rz'].iloc[start:end])
plt.xlabel('Time')
plt.ylabel('Height')
plt.grid()

plt.figure()
plt.plot(x[start:end])

plt.plot(flight_data['kite_0_rx'].iloc[start:end])
plt.xlabel('Time')
plt.ylabel('Pos east')
plt.grid()

plt.figure()
plt.plot(y[start:end])

plt.plot(flight_data['kite_0_ry'].iloc[start:end])
plt.xlabel('Time')
plt.ylabel('Pos north')
plt.grid()



# Plot Tether length
plt.figure()
plt.plot(t[start:end],tether_len[start:end])    
plt.plot(t[start:end],flight_data['ground_tether_length'].iloc[start:end])
plt.fill_between(t[start:end], 400, where=turn[start:end], color='lightblue', alpha=0.2)
plt.xlabel('Time')
plt.ylabel('Tether_len')
plt.grid()
plt.show()

plt.figure()
plt.plot(t[start:end],flight_data['kcu_set_depower'].iloc[start:end],label = 'Up')
# plt.plot(t[start:end],flight_data['kite_actual_depower'].iloc[start:end],label = 'Up')
plt.plot(t[start:end],flight_data['kcu_set_steering'].iloc[start:end],label = 'Us')
# plt.plot(t[start:end],flight_data['kite_actual_steering'].iloc[start:end],label = 'Us')
plt.xlabel('Time')
plt.ylabel('Control inputs')
plt.legend()
plt.grid()
# # Plot apparent velocity
# plt.figure()
# plt.plot(vz[start:end])
# plt.xlabel('Time')
# plt.ylabel('OMega')
# plt.grid()

#%%
plt.figure()
plt.plot(pitch-90)
plt.plot(meas_pitch)
plt.plot(meas_pitch1)

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