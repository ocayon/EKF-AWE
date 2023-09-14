import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from v3_properties import *




#%%
plt.close('all')
# results = pd.read_csv('EKF_resultsnowind1.csv')
# flight_data = pd.read_csv('flight_data.csv')
# results = pd.read_csv('EKFresults_cont.csv')
results = pd.read_csv('EKFresults_temp.csv')
flight_data = pd.read_csv('flightdata_temp.csv')
# ti = 36000+8000
# te = 42000+8000
# results = results.iloc[ti:te]
# flight_data = flight_data.iloc[ti:te]
# results = results.reset_index()
# flight_data = flight_data.reset_index()
#%%
x = results.x
y = results.y
z = results.z
vx = results.vx
vy = results.vy
vz = results.vz
vwx = results.vwx
vwy = results.vwy
vwz = results.vwz
CL = results.CL
CD = results.CD
CS = results.CS
CLw = results.CLw
CDw = results.CDw
aoa = results.aoa
tether_len = results.Lt
Ft = np.array([results.Ftx,results.Fty,results.Ftz])
Ft_mod = np.linalg.norm(Ft,axis = 0)

va = np.vstack((np.array(vwx-vx),np.array(vwy-vy),np.array(vwz-vz))).T

vwh = []
angle = []
L = []
D = []
va_mod = []
slack = []
acc = []
Fa = []
omega = []
for i in range(len(CL)):
    vwh.append(np.sqrt(vwx[i]**2+vwy[i]**2))
    angle.append(np.arctan(vwy[i]/vwx[i])*180/np.pi)
    va_mod.append(np.linalg.norm(va[i]))
    q = 0.5*1.225*A_kite*va_mod[i]**2
    L.append(CL[i]*q)
    D.append(CD[i]*q)
    slack.append(tether_len[i]+l_bridle-np.sqrt(x[i]**2+y[i]**2+z[i]**2))
    Fa.append(np.sqrt(CL[i]**2+CD[i]**2+CS[i]**2)*q)
    dir_D = va[i]/va_mod[i]
    r = np.array([x[i],y[i],z[i]])
    v = np.array([vx[i],vy[i],vz[i]])
    r_mod = np.linalg.norm(r)
    dir_L = r/r_mod - np.dot(r/r_mod,dir_D)*dir_D
    dir_S = np.cross(dir_L,dir_D) 

    # Lmod = ca.dot(Fa,dir_L)
    # Dmod = ca.dot(Fa,dir_D)
    # Smod = ca.dot(Fa,dir_S)
    Li = CL[i]*0.5*rho*A_kite*va_mod[i]**2*dir_L
    Di = CD[i]*0.5*rho*A_kite*dir_D*va_mod[i]**2
    Si = CS[i]*0.5*rho*A_kite*va_mod[i]**2*dir_S
    Fg = np.array([0,0,-m_kite*9.8])
    acc.append((Li+Di+Si-Ft.T[i]+Fg)/m_kite)
    omega.append(np.cross(r, v)/np.linalg.norm(r)**2)

CL = np.sqrt(CL**2+   CS**2)
#%%
measured_wdir = -flight_data['est_upwind_direction']*180/np.pi-90+360
measured_wvel = flight_data['est_wind_velocity']
measured_va = flight_data['airspeed_apparent_windspeed']
measured_Ft = flight_data['ground_tether_force']
measured_aoa = flight_data['airspeed_angle_of_attack']
#%% Plot aero coeffs

# pow_res = results[flight_data['ground_tether_reelout_speed']>0]
# pow_straight = pow_res[abs(flight_data['kite_set_steering'])<10]
# pow_turn = pow_res[abs(flight_data['kite_set_steering'])>10]
# dep_res = results[(flight_data['ground_tether_reelout_speed'] < 0) & (flight_data['kite_set_depower'] > 25)]

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
# plt.scatter(pow_res.aoa,pow_res.CDw)
# plt.scatter(dep_res.aoa,dep_res.CDw)
# plt.scatter(pow_straight.aoa,pow_straight.CDw)
# # plt.scatter(pow_turn.aoa,pow_turn.CDw)


# plt.figure()
# plt.plot(aoa)
# plt.plot(measured_aoa)



# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # x = pow_straight.x
# # y = pow_straight.y
# # z = pow_straight.z
# # for i in range(1, len(pow_straight)):
# #     if pow_straight.index[i] == pow_straight.index[i - 1] + 1:
# #         ax.plot([x.iloc[i - 1], x.iloc[i]], [y.iloc[i - 1], y.iloc[i]], [z.iloc[i - 1], z.iloc[i]], color='b')
        
# # x = pow_turn.x
# # y = pow_turn.y
# # z = pow_turn.z
# # for i in range(1, len(pow_turn)):
# #     if pow_turn.index[i] == pow_turn.index[i - 1] + 1:
# #         ax.plot([x.iloc[i - 1], x.iloc[i]], [y.iloc[i - 1], y.iloc[i]], [z.iloc[i - 1], z.iloc[i]], color='orange')

# # Set labels for the axes



# #%%
# custom_labels = ['13', '14', '15','16']
# x_ticks = [0, 36000, 72000,108000]

# # Plot horizontal wind speed
# plt.figure()
# plt.plot(vwh)
# plt.xlabel('Time')
# plt.ylabel('Horizontal Wind Speed')
# # plt.xticks(x_ticks, custom_labels)
# plt.grid()

# # Plot wind direction
# plt.figure()
# plt.plot(angle)
# plt.xlabel('Time')
# plt.ylabel('Wind Direction')
# # plt.xticks(x_ticks, custom_labels)
# plt.grid()

# # Plot vertical wind speed
# plt.figure()
# plt.plot(vwz)
# plt.xlabel('Time')
# plt.ylabel('Vertical Wind Speed')
# # plt.xticks(x_ticks, custom_labels)
# plt.grid()

# # Plot lift coefficient
# plt.figure()
# plt.plot(CLw)
# plt.plot(CL)

# plt.xlabel('Time')
# plt.ylabel('Lift Coefficient')
# # plt.xticks(x_ticks, custom_labels)
# plt.grid()

# # Plot drag coefficient
# plt.figure()
# plt.plot(CDw)
# plt.plot(CD)

# plt.xlabel('Time')
# plt.ylabel('Drag Coefficient')
# # plt.xticks(x_ticks, custom_labels)
# plt.grid()

# # Plot side force coefficient
# plt.figure()
# plt.plot(CLw/CDw)
# plt.xlabel('Time')
# plt.ylabel('CL/CD')
# # plt.xticks(x_ticks, custom_labels)
# plt.grid()

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

# # Plot apparent velocity
# plt.figure()
# plt.plot(va_mod)
# plt.plot(measured_va)
# plt.xlabel('Time')
# plt.ylabel('Apparent Velocity')
# # plt.xticks(x_ticks, custom_labels)
# plt.grid()

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

start = 2000
end = -1
# Plot apparent velocity
plt.figure()
plt.plot(z[start:end])
plt.plot(flight_data['rz'].iloc[start:end])
plt.xlabel('Time')
plt.ylabel('Height')
plt.grid()

plt.figure()
plt.plot(x[start:end])
plt.plot(flight_data['rx'].iloc[start:end])
plt.xlabel('Time')
plt.ylabel('Pos east')
plt.grid()

plt.figure()
plt.plot(y[start:end])
plt.plot(flight_data['ry'].iloc[start:end])
plt.xlabel('Time')
plt.ylabel('Pos north')
plt.grid()



# Plot Tether length
plt.figure()
plt.plot(tether_len[start:end])
plt.plot(flight_data['ground_tether_length'].iloc[start:end])
plt.plot(vz[start:end])
plt.xlabel('Time')
plt.ylabel('Tether_len')
plt.grid()
plt.show()

# # Plot apparent velocity
# plt.figure()
# plt.plot(vz[start:end])
# plt.xlabel('Time')
# plt.ylabel('OMega')
# plt.grid()


#%%
start = 4000
end = start+36000
hours = [13,14,15]

h_ticks = np.arange(0,350,50)
heights = np.arange(100, 301, step=50)
ground_wdir = measured_wdir[start:end]
ground_wvel = measured_wvel[start:end]
mindir = []
maxdir = []
minvel = []
maxvel = []

fig_vel, ax_vel = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
fig_dir, ax_dir = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
for i in range(len(hours)):
    ERA5vel = np.loadtxt('data/'+str(hours[i])+'hwindvel.csv',delimiter = ',')    
    hERA5 = ERA5vel[0:-1:2,1]
    minERA5 = ERA5vel[0:-1:2,0]
    maxERA5 = ERA5vel[1::2,0]
    ax_vel[i].fill_betweenx(hERA5, minERA5, maxERA5, color='lightgrey', alpha=0.5)
    ax_vel[i].scatter(vwh[start:end],z[start:end],color = 'lightblue',alpha = 0.5)
    ax_vel[i].boxplot([measured_wvel[start:end]],positions = [10],vert = False,widths=(20))
    ax_vel[i].set_title(str(hours[i])+'h')
    ax_vel[i].set_xlabel('Wind velocity')
    ax_vel[i].set_yticks(h_ticks)
    ax_vel[i].set_yticklabels(h_ticks)
    ax_vel[i].grid()
    
    
    ERA5dir = np.loadtxt('data/'+str(hours[i])+'hwinddir.csv',delimiter = ',')
    hERA5 = ERA5dir[0:-1:2,1]
    minERA5 = ERA5dir[0:-1:2,0]
    maxERA5 = ERA5dir[1::2,0]
    ax_dir[i].fill_betweenx(hERA5, minERA5, maxERA5, color='lightgrey', alpha=0.5)
    ax_dir[i].scatter(angle[start:end],z[start:end],color = 'lightblue',alpha = 0.5)
    ax_dir[i].boxplot([measured_wdir[start:end]],positions = [10],vert = False,widths=(20))
    ax_dir[i].set_title(str(hours[i])+'h')
    ax_dir[i].set_xlabel('Wind direction')
    ax_dir[i].set_yticks(h_ticks)
    ax_dir[i].set_yticklabels(h_ticks)
    ax_dir[i].grid()
    
    
    if i == 0:
        ax_dir[i].set_ylabel('Height')
        ax_vel[i].set_ylabel('Height')
    start = end
    end = start+36000
    
# fig_dir.savefig('wind_direction.png',dpi = 300)
# fig_vel.savefig('wind_velocity.png',dpi = 300)