import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def R_EG_Body(Roll,Pitch,Yaw):#!!In radians!!
    
    #Rotational matrix for Roll
    R_Roll=np.array([[1, 0, 0],[0,np.cos(Roll),np.sin(Roll)],[0,-np.sin(Roll),np.cos(Roll)]])#OK checked with Blender
    
    #Rotational matrix for Pitch
    R_Pitch=np.array([[np.cos(Pitch), 0, np.sin(Pitch)],[0,1,0],[-np.sin(Pitch), 0, np.cos(Pitch)]])#Checked with blender
    
    #Rotational matrix for Roll
    R_Yaw= np.array([[np.cos(Yaw),-np.sin(Yaw),0],[np.sin(Yaw),np.cos(Yaw),0],[0,0,1]])#Checked with Blender
    
    #Total Rotational Matrix
    return R_Roll.dot(R_Pitch.dot(R_Yaw))
#%%
plt.close('all')

model = 'v3'
year = '2019'
month = '10'
day = '08'

if model == 'v3':
    from v3_properties import *
elif model == 'v9':
    from v9_properties import *

path = './results/'+model+'/'
file_name = model+'_'+year+'-'+month+'-'+day

res1 = pd.read_csv(path+file_name+'_res_GPS.csv')
fd1 = pd.read_csv(path+file_name+'_fd_GPS.csv')

res2 = pd.read_csv(path+file_name+'_res_GPS_vw.csv')
fd2 = pd.read_csv(path+file_name+'_fd_GPS_vw.csv')

res3 = pd.read_csv(path+file_name+'_res_GPS_va.csv')
fd3 = pd.read_csv(path+file_name+'_fd_GPS_va.csv')


roll = fd1['roll0']-5
pitch = fd1['pitch0']-5
yaw = fd1['yaw0']
aoa = fd1['airspeed_angle_of_attack']
va_mod = fd1['airspeed_apparent_windspeed']
vk = np.array([fd1['vx0'], fd1['vy0'], fd1['vz0']]).T

va = np.zeros((len(roll),3))
vw = np.zeros((len(roll),3))
for i in range(0,len(roll)):
    Transform_Matrix=R_EG_Body(roll[i]/180*np.pi,pitch[i]/180*np.pi,(yaw[i])/180*np.pi)
#    Transform_Matrix=R_EG_Body(kite_roll[i]/180*np.pi,kite_pitch[i]/180*np.pi,kite_yaw_modified[i])
    Transform_Matrix=Transform_Matrix.T
    
    #X_vector
    x_vector=Transform_Matrix.dot(np.array([1,0,0]))
    #Y_vector
    y_vector=Transform_Matrix.dot(np.array([0,1,0]))
    #Z_vector
    z_vector=Transform_Matrix.dot(np.array([0,0,1]))

    vax = np.cos(aoa[i]/180*np.pi)*va_mod[i]
    vaz = np.sin(aoa[i]/180*np.pi)*va_mod[i]

    va[i] = -vax*x_vector + -vaz*z_vector

    vw[i] = va[i] + vk[i]

#%% 
plt.figure()
plt.plot(fd1.time,res1.pitch-90,label='GPS')
plt.plot(fd2.time,res2.pitch-90,label='GPS+groundvw')
plt.plot(fd3.time,res3.pitch-90,label='GPS+va')

plt.plot(fd1.time,fd1.pitch1,label='Flight data')
plt.plot(fd1.time,fd1.pitch0,label='Flight data')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Pitch [deg]')
plt.grid()


plt.figure()
plt.plot(fd1.time,res1.wdir*180/np.pi, label='GPS')
plt.plot(fd2.time,res2.wdir*180/np.pi,label='GPS+groundvw')
plt.plot(fd3.time,res3.wdir*180/np.pi,label='GPS+va')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Wind direction [deg]')
plt.grid()

plt.figure()
plt.plot(fd1.time,res1.uf,label= 'GPS')
plt.plot(fd2.time,res2.uf,label='GPS+groundvw')
plt.plot(fd3.time,res3.uf,label='GPS+va')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('u [m/s]')
plt.grid()

wvel1 = res1.uf/kappa*np.log(res1.z/z0)
wvel2 = res2.uf/kappa*np.log(res2.z/z0)
wvel3 = res3.uf/kappa*np.log(res3.z/z0)
wvel_va = np.linalg.norm(vw,axis=1)
wdir1 = res1.wdir
wdir2 = res2.wdir
wdir3 = res3.wdir
wdir_va = np.arctan2(vw[:,1],vw[:,0])

plt.figure()
plt.plot(fd1.time,wvel1,label='GPS')
plt.plot(fd2.time,wvel2,label='GPS+groundvw')
plt.plot(fd3.time,wvel3,label='GPS+va')
plt.plot(fd1.time,wvel_va,label='va')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Wind speed [m/s]')
plt.grid()

#%%
import matplotlib as mpl  
mpl.rc('font',family='Arial')
import seaborn as sns
measured_wdir = -fd1['ground_upwind_direction']-90+360
measured_wvel = fd1['ground_wind_velocity']

windpath = './data/'
windfile = 'era5_data_'+year+'_'+month+'_'+day+'.npy'

# Define a color palette from seaborn
palette = sns.color_palette("pastel")

data_dict = np.load(windpath+windfile, allow_pickle=True)

# Extract arrays and information
era5_hours = data_dict.item()['hours']
era5_heights = data_dict.item()['heights']
era5_wvel = data_dict.item()['wvel']
era5_wdir = data_dict.item()['wdir']


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

fig_vel, ax_vel = plt.subplots(1, 3, sharey=True, figsize=(8, 4))
fig_dir, ax_dir = plt.subplots(1, 3, sharey=True, figsize=(8, 4))
fig_vel.patch.set_alpha(0) 
fig_dir.patch.set_alpha(0) 
hfont = {'fontname':'Arial'}
for i in range(len(hours)):


    lin1 = ax_vel[i].scatter(wvel_va[start:end],fd1['rz'].iloc[start:end],color = palette[0],alpha = 0.5)
    lin2 = ax_vel[i].fill_betweenx(era5_heights[:-2], era5_wvel[i,:-2], era5_wvel[i+1,:-2], color='lightgrey', alpha=0.5)
    lin3 = ax_vel[i].scatter(wvel1[start:end],res1.z.iloc[start:end],color = palette[1],alpha = 0.5)
    lin4 = ax_vel[i].scatter(wvel2[start:end],res2.z.iloc[start:end],color = palette[2],alpha = 0.5)
    lin5 = ax_vel[i].scatter(wvel3[start:end],res3.z.iloc[start:end],color = palette[3],alpha = 0.5)
    lin6 = ax_vel[i].boxplot([measured_wvel[start:end]],positions = [10],vert = False,widths=(20))
    ax_vel[i].set_title(str(hours[i])+'h')
    
    ax_vel[i].set_yticks(h_ticks)
    ax_vel[i].set_yticklabels(h_ticks)
    ax_vel[i].grid(color='black', linestyle='--', linewidth=0.5)
    

    ax_dir[i].scatter(wdir_va[start:end]*180/np.pi,fd1['rz'].iloc[start:end],color = palette[0],alpha = 0.5)
    ax_dir[i].fill_betweenx(era5_heights[:-2], era5_wdir[i,:-2], era5_wdir[i+1,:-2], color='lightgrey', alpha=0.5)
    ax_dir[i].scatter(wdir1[start:end]*180/np.pi,res1.z.iloc[start:end],color = palette[1],alpha = 0.5)
    ax_dir[i].scatter(wdir2[start:end]*180/np.pi,res2.z.iloc[start:end],color = palette[2],alpha = 0.5)
    ax_dir[i].scatter(wdir3[start:end]*180/np.pi,res3.z.iloc[start:end],color = palette[3],alpha = 0.5)
    
    ax_dir[i].boxplot([measured_wdir[start:end]],positions = [10],vert = False,widths=(20))
    ax_dir[i].set_title(str(hours[i])+'h')
    
    ax_dir[i].set_yticks(h_ticks)
    ax_dir[i].set_yticklabels(h_ticks)
    ax_dir[i].grid(color='black', linestyle='--', linewidth=0.5)

    ax_dir[i].set_xlim([np.min(wdir_va*180/np.pi), np.max(wdir_va*180/np.pi)])
    ax_vel[i].set_xlim([np.min(wvel_va), np.max(wvel_va)])  
    ax_dir[i].set_ylim([0,300])
    ax_vel[i].set_ylim([0,300])  

    if i == 0:
        ax_dir[i].set_ylabel('Height (m)',**hfont)
        ax_vel[i].set_ylabel('Height (m)',**hfont)
    
    if i == 1:
        ax_vel[i].set_xlabel('$v_w$ (m/s)',**hfont)
        ax_dir[i].set_xlabel(r'$\theta$ (°)',**hfont)
    start = end
    end = start+36000
fig_vel.tight_layout()
fig_dir.tight_layout()
# fig_vel.legend(['$v_w = v_a + v_k$','Reanalysis data*','EKF (GPS)','EKF (GPS+$v_{w_g}$)','EKF (GPS+$v_a$)','Ground wind velocity ($v_{w_g}$)'],loc='right',framealpha=1)
# fig_leg.legend([lin1,lin2,lin3,lin4,lin5,lin6],['Only $v_a$','ERA5 data','EKF (GPS)','EKF (GPS+$v_{w_g}$)','EKF (GPS+$v_a$)','Ground wind velocity ($v_{w_g}$)'])
# fig_dir.legend(['ERA5','EKF (GPS&groundwvel)','EKF with va','Ground measurement'])
fig_dir.savefig('wind_direction.png',dpi = 600,transparent = True)
fig_vel.savefig('wind_velocity.png',dpi =600,transparent = True)
#%%%
plt.figure()
plt.plot(wdir_va*180/np.pi)
plt.plot(wdir2*180/np.pi)
