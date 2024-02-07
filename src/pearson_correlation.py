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

azimuth = np.arctan2(y,x)
elevation = np.arctan2(z,np.sqrt(x**2+y**2))

# Calculate wind speed based on KCU orientation and wind speed and direction
aoacalc = []
sideslipcalc = []
va_mod = []
slack = []
wvel_calc = []
wdir_calc = []
v_radial = []
radius = []

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
    
    # Calculate tether orientation based on kite sensor measurements
    Transform_Matrix=R_EG_Body(meas_roll[i]/180*np.pi,pitch[i]/180*np.pi,(meas_yaw[i])/180*np.pi)
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

slack = np.array(slack)
radius = np.array(radius)
azimuth_rate = np.concatenate((np.diff(azimuth), [0]))
pitch_rate = np.concatenate((np.diff(meas_pitch), [0]))
sideslipcalc = np.array(sideslipcalc)
aoacalc = np.array(aoacalc)
va_mod = np.array(va_mod)
turn = pow & (vz<0)
straight = pow & ~turn
turn_right = turn  & (azimuth<0)
turn_left = turn  & (azimuth>0)
straight_right = straight  & (azimuth_rate<0)
straight_left = straight  & (azimuth_rate>0)
#%% Create mask for plotting

mask= pow
pearson_data = pd.DataFrame({   'measured_Ft':measured_Ft[mask],
                                'wvel_EKF': wvel[mask],
                                'slack':slack[mask],
                                'v_kite':np.linalg.norm(v_kite[mask],axis = 1),
                                'va':va_mod[mask],
                                'height':z[mask],
                                'pitch':pitch[mask],
                                'roll':meas_roll[mask],
                                'yaw':meas_yaw[mask],
                                'azimuth':azimuth[mask]-wdir[mask],
                                'elevation':elevation[mask], 
                                'CL':CL_EKF[mask],
                                'CD':CD_EKF[mask],
                                'CL3/CD2':CL_EKF[mask]**3/CD_EKF[mask]**2,
                                'aoa':aoa[mask],
                                'sideslip':sideslipcalc[mask]
                                })


# Calculating the Pearson correlation coefficient for the DataFrame
correlation_matrix = pearson_data.corr()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Heatmap of Pearson Correlation Coefficient')
plt.show()
plt.savefig('pearson.png', dpi=300) 