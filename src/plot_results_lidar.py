import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import R_EG_Body
from config import kappa, z0, kite_model, year, month, day
import numpy as np
from utils import R_EG_Body
import seaborn as sns

#%%
def plot_wind_direction(flight_data, res,plot_heights_lidar,res_max = None, res_min = None, onlyres = False,title = None):
    plt.figure()
    
    if res_max is not None:
        plt.fill_between(flight_data['time'],res_min['wdir']*180/np.pi,res_max['wdir']*180/np.pi,alpha = 0.3)
    plt.plot(flight_data['time'],res['wdir']*180/np.pi,label='Sensor Fusion')
    if not onlyres:
        plt.plot(flight_data['time'],flight_data['ground_wind_direction'],'k',label='Ground Sensor',alpha = 0.3)
        for column in flight_data.columns:
            if 'Wind Direction' in column:
                height = ''.join(filter(str.isdigit, column))
                height = int(height)
                if height in plot_heights_lidar:
                    # for i in range(len(flight_data)):
                    #     if flight_data[column].iloc[i] != flight_data[column].iloc[i-1]:
                    #         plt.scatter(flight_data['time'].iloc[i],360-90-flight_data[column].iloc[i])
                    plt.plot(flight_data['time'],360-90-flight_data[column],label = column)
    
    plt.legend()
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel('Wind Direction [deg]')
    if title:
        plt.title(title)

def plot_wind_speed(flight_data, res,plot_heights_lidar,res_max=None,res_min=None, onlyres = False,title = None):
    plt.figure()
    
    if not onlyres:
        plt.plot(flight_data['time'],flight_data['ground_wind_velocity'],'k',label='Ground Sensor',alpha = 0.3)
        for column in flight_data.columns:
            if 'Wind Speed (m/s)' in column:
                height = ''.join(filter(str.isdigit, column))
                vw_max_col = height+'m Wind Speed max (m/s)'
                vw_min_col = height+'m Wind Speed min (m/s)'
                height = int(height)
                
                if height in plot_heights_lidar:
                    # for i in range(len(flight_data)):
                    #     if flight_data[column].iloc[i] != flight_data[column].iloc[i-1]:
                    #         y_value = flight_data[column].iloc[i]
                    #         err_negative = y_value - flight_data[vw_min_col].iloc[i] 
                    #         err_positive = flight_data[vw_max_col].iloc[i] - y_value
                    #         # plt.scatter(flight_data['time'].iloc[i],flight_data[column].iloc[i])
                    #         # plt.errorbar(flight_data['time'].iloc[i],y_value, yerr=[[err_negative],[err_positive]],
                    #         #             fmt='-', ecolor='lightgray',label = column)
                    plt.fill_between(flight_data['time'],flight_data[vw_min_col],flight_data[vw_max_col],alpha = 0.3)
                    plt.plot(flight_data['time'],flight_data[column],label = column)
                    
                    
    if res_max is not None:
        plt.fill_between(flight_data['time'],res_min['uf']/kappa*np.log(res_min['z']/z0),res_max['uf']/kappa*np.log(res_max['z']/z0),alpha = 0.8)
    plt.plot(flight_data['time'],res['uf']/kappa*np.log(res['z']/z0),label='Sensor Fusion')
    plt.legend()
    plt.grid()
    plt.xlabel('Time [s]')
    plt.ylabel('Wind Speed [m/s]')
    if title:
        plt.title(title)

def plot_true_estimated_wind_speed(flight_data, res,plot_heights_lidar):
    plt.figure()

    wvel = res['uf']/kappa*np.log(res['z']/z0)
    for column in flight_data.columns:
        if 'Wind Speed (m/s)' in column:
            height = ''.join(filter(str.isdigit, column))
            vw_max_col = height+'m Wind Speed max (m/s)'
            vw_min_col = height+'m Wind Speed min (m/s)'
            height = int(height)
            
            if height in plot_heights_lidar:
                # for i in range(len(flight_data)):
                #     if flight_data[column].iloc[i] != flight_data[column].iloc[i-1]:
                #         y_value = flight_data[column].iloc[i]
                #         err_negative = y_value - flight_data[vw_min_col].iloc[i] 
                #         err_positive = flight_data[vw_max_col].iloc[i] - y_value
                #         # plt.scatter(flight_data['time'].iloc[i],flight_data[column].iloc[i])
                #         # plt.errorbar(flight_data['time'].iloc[i],y_value, yerr=[[err_negative],[err_positive]],
                #         #             fmt='-', ecolor='lightgray',label = column)
                
                plt.scatter(flight_data[column],wvel,label = 'EKF Estimation', color = 'b', alpha = 0.9, s = 5)
                plt.plot(flight_data[column],flight_data[column],label = 'Ground Truth', color = 'k', alpha = 0.3, linewidth = 0.5, linestyle = '--')
                
    plt.grid()
    plt.legend()


def plot_true_estimated_wind_direction(flight_data, res,plot_heights_lidar):
    plt.figure()

    for column in flight_data.columns:
        if 'Wind Direction' in column:
            height = ''.join(filter(str.isdigit, column))
            height = int(height)
            if height in plot_heights_lidar:
                # for i in range(len(flight_data)):
                #     if flight_data[column].iloc[i] != flight_data[column].iloc[i-1]:
                #         plt.scatter(flight_data['time'].iloc[i],360-90-flight_data[column].iloc[i])
                plt.scatter(flight_data[column],res['wdir']*180/np.pi,label = 'EKF Estimation', color = 'b', alpha = 0.9, s = 5)
                plt.plot(flight_data[column],360-90-flight_data[column],label = 'Ground Truth', color = 'k', alpha = 0.3, linewidth = 0.5, linestyle = '--')
    
    plt.grid()
    plt.legend()


def calculate_wind_speed_pitot_tube(flight_data, kite_model):
    
    wvel_calc = []
    wdir_calc = []

    measured_aoa = flight_data['kite_angle_of_attack']
    if 'v9' in kite_model:
        measured_ss = flight_data['kite_sideslip_angle']
    else:
        measured_ss = np.zeros(len(flight_data))
    measured_va = flight_data['kite_apparent_windspeed']
    v_kite = flight_data[['kite_0_vx','kite_0_vy','kite_0_vz']].values

    meas_roll = flight_data['kite_0_roll']
    meas_pitch = flight_data['kite_0_pitch']
    meas_yaw = flight_data['kite_0_yaw']-90

    measured_aoa = measured_aoa+2
    measured_ss = measured_ss
    for i in range(len(measured_aoa)):

        # Calculate angle of attack based on orientation angles and estimated wind speed
        Transform_Matrix=R_EG_Body(meas_roll[i]/180*np.pi,meas_pitch[i]/180*np.pi,(meas_yaw[i])/180*np.pi)
        Transform_Matrix=Transform_Matrix.T

        # X_vector
        ex_kite=Transform_Matrix.dot(np.array([-1,0,0]))
        # Y_vector
        ey_kite=Transform_Matrix.dot(np.array([0,-1,0]))
        # Z_vector
        ez_kite=Transform_Matrix.dot(np.array([0,0,1]))

        va_calc= ex_kite*measured_va[i]*np.cos(measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ey_kite*measured_va[i]*np.sin(measured_ss[i]/180*np.pi)*np.cos(measured_aoa[i]/180*np.pi)+ez_kite*measured_va[i]*np.sin(measured_aoa[i]/180*np.pi)
        vw = va_calc+v_kite[i]
        wvel_calc.append(np.linalg.norm(vw))
        wdir_calc.append(np.arctan2(vw[1],vw[0]))
    
    return wvel_calc, wdir_calc

def plot_wind_speed_and_direction(flight_data, res, plot_heights_lidar, res_max=None, res_min=None, onlyres=False, title=None):
    # Generate a colorblind-friendly palette
    palette = sns.color_palette("tab10")

    # Define the subplot layout
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)  # Adjust the figsize as needed

    # Plot Wind Speed
    wvel = res['uf']/kappa*np.log(res['z']/z0)
    axs[0].plot(flight_data['time'], wvel, color=palette[0], label='Sensor Fusion', alpha=0.9, linewidth=2.0)
    if res_max is not None:
        axs[0].fill_between(flight_data['time'], res_min['uf']/kappa*np.log(res_min['z']/z0), res_max['uf']/kappa*np.log(res_max['z']/z0), color=palette[0], alpha=0.5)

    i = 1
    for column in flight_data.columns:
        if 'Wind Speed (m/s)' in column:
            height = ''.join(filter(str.isdigit, column))
            vw_max_col = height + 'm Wind Speed max (m/s)'
            vw_min_col = height + 'm Wind Speed min (m/s)'
            label = 'Lidar ' + height +'m height'
            height = int(height)
            
            if height in plot_heights_lidar:
                
                axs[0].fill_between(flight_data['time'], flight_data[vw_min_col], flight_data[vw_max_col], color=palette[i], alpha=0.3)
                axs[0].plot(flight_data['time'], flight_data[column],color=palette[i], label=label)

                i +=1
                
    axs[0].plot(flight_data['time'], flight_data['ground_wind_velocity'], color=palette[i], label='Ground Sensor')
    axs[0].grid()
    axs[0].set_ylabel('Wind Speed [m/s]')

    

    # Plot Wind Direction
    axs[1].plot(flight_data['time'], res['wdir']*180/np.pi, color=palette[0], label='Sensor Fusion', linewidth=2.0)
    if res_max is not None:
        axs[1].fill_between(flight_data['time'], res_min['wdir']*180/np.pi, res_max['wdir']*180/np.pi, color=palette[0], alpha=0.5)

    i = 1
    for column in flight_data.columns:
        if 'Wind Direction' in column:
            height = ''.join(filter(str.isdigit, column))
            label = 'Lidar ' + height +'m height'
            height = int(height)
            if height in plot_heights_lidar:
                axs[1].plot(flight_data['time'], 360 - 90 - flight_data[column], label=label, color=palette[i])
                i +=1
                
    axs[1].plot(flight_data['time'], flight_data['ground_wind_direction'], color=palette[i], label='Ground Sensor')
    axs[1].legend()
    axs[1].grid()
    axs[1].set_xlabel('Time [min]')
    axs[1].set_ylabel('Wind Direction [deg]')
    axs[1].set_ylim([np.min(res['wdir'])*180/np.pi-40, np.max(res['wdir'])*180/np.pi+40])

    
    
    if title:
        fig.suptitle(title) 

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save or display the figure
    
    plt.show()

def log_wind_profile(z, u_star):
    k = 0.4  # von Kármán constant
    return (u_star / k) * np.log(z / z0)


def logarithmic_fit(flight_data):
    i_start = 0
    from scipy.optimize import curve_fit
    lidar_heights = []
    lidar_columns = []
    u_star = np.zeros(len(flight_data))
    z0 = np.zeros(len(flight_data))
    for column in flight_data.columns:
        if 'Wind Speed (m/s)' in column:
            height = ''.join(filter(str.isdigit, column))
            lidar_heights.append(int(height))
            lidar_columns.append(column)
    for i in range(1, len(flight_data)):  # Start from 1 to avoid -1 index
        wvel = []
        if flight_data['100m Wind Speed (m/s)'].iloc[i] != flight_data['100m Wind Speed (m/s)'].iloc[i-1]:
            for column in lidar_columns:
                wvel.append(flight_data[column].iloc[i-1])
            # Perform the curve fitting
            popt, pcov = curve_fit(log_wind_profile, lidar_heights, wvel, maxfev=10000)

            # Extracting the fitted parameters
            u_star_fitted = popt    
            u_star[i_start:i] = u_star_fitted
            i_start = i
    for column in lidar_columns:
        wvel.append(flight_data[column].iloc[i-1])
    # Perform the curve fitting
    popt, pcov = curve_fit(log_wind_profile, lidar_heights, wvel, maxfev=10000)            
    # Extracting the fitted parameters
    u_star_fitted = popt    
    u_star[i_start::] = u_star_fitted
    return u_star
                
            

def time_average_function(flight_data, res):
    # Select only numeric columns
    flight_data = flight_data.select_dtypes(include='number')
    res = res.select_dtypes(include='number')

    # Initialize empty DataFrames with the same columns
    fd_average = pd.DataFrame(columns=flight_data.columns)
    res_average = pd.DataFrame(columns=res.columns)
    
    # Process each column
    for column in flight_data.columns:
        if 'Wind Speed (m/s)' in column:
            i_start = 0
            for i in range(1, len(flight_data)):  # Start from 1 to avoid -1 index
                if flight_data[column].iloc[i] != flight_data[column].iloc[i-1]:
                    # Calculate mean and convert to DataFrame
                    fd_mean = flight_data.iloc[i_start:i].mean().to_frame().T
                    res_mean = res.iloc[i_start:i].mean().to_frame().T
                    
                    fd_mean['time'] = flight_data['time'].iloc[i]/60
                    
                    # Concatenate
                    fd_average = pd.concat([fd_average, fd_mean], ignore_index=True)
                    res_average = pd.concat([res_average, res_mean], ignore_index=True)

                    i_start = i

            # Calculate mean for the last segment
            fd_mean = flight_data.iloc[i_start:].mean().to_frame().T
            res_mean = res.iloc[i_start:].mean().to_frame().T
            
            fd_mean['time'] = flight_data['time'].iloc[i]/60
            # Concatenate
            fd_average = pd.concat([fd_average, fd_mean], ignore_index=True)
            res_average = pd.concat([res_average, res_mean], ignore_index=True)

            break  # Break after processing the specific column

    return fd_average, res_average

def time_maxmin_function(flight_data, res):
    # Select only numeric columns
    flight_data = flight_data.select_dtypes(include='number')
    res = res.select_dtypes(include='number')

    # Initialize empty DataFrames with the same columns
    fd_max = pd.DataFrame(columns=flight_data.columns)
    res_max = pd.DataFrame(columns=res.columns)
    fd_min = pd.DataFrame(columns=flight_data.columns)
    res_min = pd.DataFrame(columns=res.columns)

    # Process each column
    for column in flight_data.columns:
        if 'Wind Speed (m/s)' in column:
            i_start = 0
            for i in range(1, len(flight_data)):  # Start from 1 to avoid -1 index
                if flight_data[column].iloc[i] != flight_data[column].iloc[i-1]:
                    # Calculate max and min and convert to DataFrame
                    fd_maxi = flight_data.iloc[i_start:i].max().to_frame().T
                    res_maxi = res.iloc[i_start:i].max().to_frame().T
                    fd_mini = flight_data.iloc[i_start:i].min().to_frame().T
                    res_mini = res.iloc[i_start:i].min().to_frame().T

                    # Concatenate
                    fd_max = pd.concat([fd_max, fd_maxi], ignore_index=True)
                    res_max = pd.concat([res_max, res_maxi], ignore_index=True)
                    fd_min = pd.concat([fd_min, fd_mini], ignore_index=True)
                    res_min = pd.concat([res_min, res_mini], ignore_index=True)


                    i_start = i

            # Calculate max and min for the last segment
            fd_maxi = flight_data.iloc[i_start:].max().to_frame().T
            res_maxi = res.iloc[i_start:].max().to_frame().T
            fd_mini = flight_data.iloc[i_start:].min().to_frame().T
            res_mini = res.iloc[i_start:].min().to_frame().T

            # Concatenate
            fd_max = pd.concat([fd_max, fd_maxi], ignore_index=True)
            res_max = pd.concat([res_max, res_maxi], ignore_index=True)
            fd_min = pd.concat([fd_min, fd_mini], ignore_index=True)
            res_min = pd.concat([res_min, res_mini], ignore_index=True)

            break  # Break after processing the specific column

    return fd_max, res_max, fd_min, res_min

plt.close('all')

path = '../results/'+kite_model+'/'
file_name = kite_model+'_'+year+'-'+month+'-'+day

flight_data = pd.read_csv(path+file_name+'_fd.csv')
plot_heights_lidar = [100,160,200,250]

res = pd.read_csv(path+file_name+'_res_GPS.csv')


u_star = logarithmic_fit(flight_data)




fd_average, res_average = time_average_function(flight_data,res)
fd_max,res_max,fd_min,res_min = time_maxmin_function(flight_data,res)

#%% Plot Wind Speed
    
plot_wind_speed(flight_data, res,plot_heights_lidar, onlyres = False)

plot_wind_direction(flight_data, res,plot_heights_lidar, onlyres = False)

plot_wind_speed_and_direction(flight_data,res,plot_heights_lidar,onlyres = False)



plot_wind_speed(fd_average, res_average,plot_heights_lidar,res_max = res_max,res_min=res_min, onlyres = False,title = 'Time Averaged Wind Speed')

plot_wind_direction(fd_average, res_average,plot_heights_lidar, res_max = res_max,res_min=res_min,onlyres = False, title = 'Time Averaged Wind Direction')

title = 'Kitepower flight Ireland ' + year+'-'+month+'-'+day
 
plot_wind_speed_and_direction(fd_average, res_average,plot_heights_lidar, res_max = res_max,res_min=res_min,onlyres = False, title =title)

plt.savefig('wind_estimations.png', dpi=300)  # You can adjust the filename and format as needed


mask = (res['z']>180)&(res['z']<220)
fd_average, res_average = time_average_function(flight_data[mask],res[mask])
fd_max,res_max,fd_min,res_min = time_maxmin_function(flight_data,res)

plot_heights_lidar = [200]
plot_true_estimated_wind_speed(fd_average, res_average,plot_heights_lidar)
plot_true_estimated_wind_direction(fd_average, res_average,plot_heights_lidar)

# Plot max min heights
plt.figure()
plt.plot(fd_average['time'],res_max.z,'r',label='Max Height')
plt.plot(fd_average['time'],res_min.z,'b',label='Min Height')
plt.plot(fd_average['time'],res_average.z,'k',label='Average Height')
plt.legend()
plt.grid()
plt.xlabel('Time [min]')
plt.ylabel('Height [m]')

plt.savefig('height.png', dpi=300)  # You can adjust the filename and format as needed

