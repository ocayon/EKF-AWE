import casadi as ca
import numpy as np
import pandas as pd
from v3_properties import *
from scipy.optimize import least_squares
from utils import get_tether_end_position, state_noise_matrices, observation_matrices, calculate_angle,project_onto_plane ,read_data, rank_observability_matrix,read_data_new
import control
import time


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read and process data
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_tether_elements = 5
flight_data = read_data()
flight_data = flight_data.reset_index()
    
window_size=30
flight_data['ax']=np.convolve(flight_data['ax'], np.ones(window_size)/window_size, mode='same')
flight_data['ay']=np.convolve(flight_data['ay'], np.ones(window_size)/window_size, mode='same')
flight_data['az']=np.convolve(flight_data['az'], np.ones(window_size)/window_size, mode='same')

dep_fd = flight_data[flight_data['kite_set_depower']>25]
pow_fd = flight_data[flight_data['kite_set_depower']<25]
u_p = flight_data['kite_set_depower']-min(flight_data['kite_set_depower'])
u_p = u_p/max(u_p)
phi_upwind_direction = np.array(-flight_data['est_upwind_direction']-np.pi/2.+2*np.pi)
vw = [9*np.cos(0.2),9*np.sin(0.2),0]

CL = []
CD= []
CS = []
L=[]
D=[]
tether_length =[]
for i in range(2):
    row = flight_data.iloc[i]
    args = (row['ground_tether_force'], n_tether_elements, list(row[['rx', 'ry', 'rz']]), list(row[['vx','vy','vz']]),vw,
            list(row[['ax', 'ay', 'az']]),True, False)
    opt_res = least_squares(get_tether_end_position, list(row[['kite_elevation', 'kite_azimuth', 'kite_distance']]), args=args,
                            kwargs={'find_force': False}, verbose=0)
    res = get_tether_end_position(
        opt_res.x, *args, return_values=True, find_force=False)
    tether_length.append(res[1])
    aero_force = res[6]         
    va = res[7]
    FD = np.dot(va/np.linalg.norm(va),aero_force)*va/np.linalg.norm(va)
    FL = aero_force-FD
    CD.append(np.linalg.norm(FD)/(0.5*rho*A_kite*np.linalg.norm(va)**2))
    CL.append(np.linalg.norm(FL)/(0.5*rho*A_kite*np.linalg.norm(va)**2))
    L.append(FL)
    D.append(FD)

#%% Initial state vector 
x0 = np.vstack((flight_data[['rx', 'ry', 'rz']].values[0, :],flight_data[['vx', 'vy', 'vz']].values[0, :],vw))
x0 = np.append(x0,[CL[0],CD[0],0])

#%% Definition kalman filter matrices

Ts = ca.SX.sym('ts')  # timestep
up = ca.SX.sym('up')  # steering input
Ft = ca.SX.sym('Ft',3) # Tether force
el =  ca.SX.sym('el')
az =  ca.SX.sym('az')
Lt =  ca.SX.sym('Lt')
r = ca.SX.sym('r', 3) # Position
v = ca.SX.sym('v', 3) # Velocity
a = ca.SX.sym('a', 3) # Acceleration
vw = ca.SX.sym('vw', 3) # Acceleration
CD = ca.SX.sym('CD')   
CL = ca.SX.sym('CL')    
CS = ca.SX.sym('CS')
dFdup = ca.SX.sym('dFdup')


x = ca.vertcat(r,v,vw,CL,CD,CS) # Solution vector

u_sym = ca.vertcat(Ft)


#%% Measurement vectors 


Z = np.array([flight_data['rx'],flight_data['ry'],flight_data['rz'],flight_data['vx'],flight_data['vy'],flight_data['vz'],
              np.zeros(len(flight_data)),flight_data['ground_wind_velocity'],phi_upwind_direction,flight_data['ax'],flight_data['ay'],flight_data['az']]).T

dep = (flight_data['ground_tether_reelout_speed'] < 0) & (flight_data['kite_set_depower'] > 23)
#%%
########################################################################
## Set simulation parameters
########################################################################
n           =  x0.shape[0]      # state dimension
nm          =  Z.shape[1]       # number of measurements
m           =  3                # number of inputs
    
stdv_us = 0.1   

stdv_xGPS = 2.5
stdv_vGPS = 1
stdv_aGPS = 7
stdv_vwz = 0.1
stdv_vwg = 1
stdv_va = 0.5
stdv_dirw = 10/180*np.pi

R = np.zeros((nm,nm))
R[:3, :3] = np.eye(3) * stdv_xGPS**2

R[3:6, 3:6] = np.eye(3) * stdv_vGPS**2
R[6,6] = stdv_vwz**2
R[7,7] = stdv_vwg**2
R[8,8] = stdv_dirw**2
R[9:12,9:12] =  np.eye(3) * stdv_aGPS**2

# R[3,3] = 1.6**2
# R[4,4] = 3.3**2
# R[5,5] = 1.7**2
# R[9,9] = 5.8**2
# R[10,10] = 7.8**2
# R[11,11] =6.9**2

#%% Define process noise matrix
stdv_Ft = 200
stdv_CL = 0.2
stdv_CD = 0.1
stdv_CS = 0.1
stdv_vw = 0.0

# Define process noise matrix
Q = np.zeros((9, 9))
Q[:3,:3] = np.eye(3)*stdv_Ft**2
Q[3:5,3:5] = np.eye(2)*stdv_vw**2
Q[5,5] = 0.00**2
Q[6,6] = stdv_CL**2
Q[7,7] = stdv_CD**2
Q[8,8] = stdv_CS**2

# Wind correlation Ft
# x direction
# Q[0,3] = Q[3,0] = np.sqrt(Q[3,3]*Q[0,0])*0      
# Q[1,3] = Q[3,1] = np.sqrt(Q[3,3]*Q[1,1])*0 
# Q[2,3] = Q[3,2] = np.sqrt(Q[3,3]*Q[2,2])*0
# # y direction
# Q[0,4] = Q[4,0] = np.sqrt(Q[4,4]*Q[0,0])*0     
# Q[1,4] = Q[4,1] = np.sqrt(Q[4,4]*Q[1,1])*0 
# Q[2,4] = Q[4,2] = np.sqrt(Q[4,4]*Q[2,2])*0
# # z direction
# Q[0,5] = Q[5,0] = np.sqrt(Q[5,5]*Q[0,0])*0     
# Q[1,5] = Q[5,1] = np.sqrt(Q[5,5]*Q[1,1])*0
# Q[2,5] = Q[5,2] = np.sqrt(Q[5,5]*Q[2,2])*0

# Wind correlation CL
# Q[3,6] = Q[6,3] = np.sqrt(Q[3,3]*Q[6,6])*0.01     
# Q[4,6] = Q[6,4] = np.sqrt(Q[4,4]*Q[6,6])*0.01
# Q[5,6] = Q[6,5] = np.sqrt(Q[5,5]*Q[6,6])*0.01
# # Wind correlation CD
# Q[3,7] = Q[7,3] = np.sqrt(Q[3,3]*Q[7,7])*0.2     
# Q[4,7] = Q[7,4] = np.sqrt(Q[4,4]*Q[7,7])*0.2
# Q[5,7] = Q[7,5] = np.sqrt(Q[5,5]*Q[7,7])*0.2
# # Wind correlation CS
# Q[3,8] = Q[8,3] = np.sqrt(Q[3,3]*Q[8,8])*0.01      
# Q[4,8] = Q[8,4] = np.sqrt(Q[4,4]*Q[8,8])*0.01
# Q[5,8] = Q[8,5] = np.sqrt(Q[5,5]*Q[8,8])*0.01

# Ft relation
# Q[0,1] = Q[1,0] = np.sqrt(Q[0,0]*Q[1,1])*0.1        
# Q[0,2] = Q[2,0] = np.sqrt(Q[0,0]*Q[2,2])*0.01       
# Q[1,2] = Q[2,1] = np.sqrt(Q[1,1]*Q[2,2])*0.01  

# Wind relation
# Q[3,4] = Q[4,3] = np.sqrt(Q[3,3]*Q[4,4])*0.1        
# Q[3,5] = Q[5,3] = np.sqrt(Q[3,3]*Q[5,5])*0.01       
# Q[4,5] = Q[5,4] = np.sqrt(Q[4,4]*Q[5,5])*0.01        

# Aerodynamic coefficients correlation
# Q[6,7] = Q[7,6] = np.sqrt(Q[6,6]*Q[7,7])*0.1        # CL,CD
# Q[6,8] = Q[8,6] = np.sqrt(Q[6,6]*Q[8,8])*0.01       # CL,CS
# Q[7,8] = Q[8,7] = np.sqrt(Q[7,7]*Q[8,8])*0.01        # CD,CS

# Tether force correlation CL
# Q[0,6] = Q[6,0] = np.sqrt(Q[0,0]*Q[6,6])*0.9      
# Q[1,6] = Q[6,1] = np.sqrt(Q[1,1]*Q[6,6])*0.9
# Q[2,6] = Q[6,2] = np.sqrt(Q[2,2]*Q[6,6])*0.9
# # Tether force correlation CD
# Q[0,7] = Q[7,0] = np.sqrt(Q[0,0]*Q[7,7])*0.1
# Q[1,7] = Q[7,1] = np.sqrt(Q[1,1]*Q[7,7])*0.1
# Q[2,7] = Q[7,2] = np.sqrt(Q[2,2]*Q[7,7])*0.1
# # Tether force correlation CS
# Q[0,8] = Q[8,0] = np.sqrt(Q[0,0]*Q[8,8])*0.1
# Q[1,8] = Q[8,1] = np.sqrt(Q[1,1]*Q[8,8])*0.1
# Q[2,8] = Q[8,2] = np.sqrt(Q[2,2]*Q[8,8])*0.1

#%%
########################################################################
## Initialize Extended Kalman filter
########################################################################
n_intervals = flight_data.shape[0] - 1
N = n_intervals
nx = x0.shape[0]

# allocate space to store traces
XX_k1_k1    = np.zeros([nx, N])
err_meas    = np.zeros([nm, N])
z_k1_k1    = np.zeros([nm, N-1])
PP_k1_k1    = np.zeros([n, N])
STD_x_cor   = np.zeros([n, N])
STD_z       = np.zeros([nm, N])
ZZ_pred     = np.zeros([nm, N])
IEKFitcount = np.zeros([N, 1])

epsilon         = 10**(-10)         # IEKF threshold
doIEKF          = True            # If false, EKF without iterations is used
maxIterations   = 200               # maximum amount of iterations per sample

# Store Initial values
XX_k1_k1[:,0]   = x0
ZZ_pred[:,0]    = Z[0]

# Define Initial Matrix P
P_k1_k1 = np.eye(n)*1**2

x_sol = x0
x_k1_k1 = x0
us = np.array(flight_data['kite_set_steering'])
t = np.array(flight_data['time'])
u = np.zeros(3)
Ft = []
tether_len = []
CL = []
CD = []
aoa = []
yaw = []
pitch = []
roll = []
sideslip = []
flight_data['ground_tether_length'] = flight_data['ground_tether_length']+32.87
meas_lt = np.array(flight_data['ground_tether_length'])
fx,calc_Fx,calc_G = state_noise_matrices(x,u_sym,Ts)
dae = {'x': x, 'p': u_sym, 'ode': fx}
intg = ca.integrator('intg', 'cvodes', dae, {'tf': 0.1})
calc_hx,calc_Hx = observation_matrices(x,u_sym)
vw = np.array([np.array(flight_data['ground_wind_velocity']),np.zeros(len(flight_data)),np.zeros(len(flight_data))]).T
start_time = time.time()
mins = -1
for k in range(n_intervals):
    ts = t[k+1]-t[k]
    row = flight_data.iloc[k]
    args = (row['ground_tether_force'], n_tether_elements, x_k1_k1[0:3], x_k1_k1[3:6],x_k1_k1[6:9],
            list(row[['ax','ay','az']]),True, False)
    opt_res = least_squares(get_tether_end_position, opt_res.x, args=args,
                            kwargs={'find_force': False}, verbose=0)
    res = get_tether_end_position(
        opt_res.x, *args, return_values=True, find_force=False)
    
    u = np.array(res[8])
    

    dcm_b2w = res[2]
    ey_kite = dcm_b2w[:,1]
    ez_kite = dcm_b2w[:,2]
    ex_kite = dcm_b2w[:,0]
    
    
    
    
    Ft.append(res[8])
    tether_len.append(res[1])
    CL.append(res[-2])
    CD.append(res[-1])
    zi = Z[k]
    # if dep[k]:
    #     R[6,6] = stdv_va**2
    # else:
    #     R[6,6] = 5**2
    if k%600==0:
        elapsed_time = time.time() - start_time
        start_time = time.time()  # Record end time
        mins +=1
        print(f"Real time: {mins} minutes.  Elapsed time: {elapsed_time:.2f} seconds")

    sol = intg(x0=x_k1_k1, p=u)
    x_k1_k = np.array(sol["xf"].T)
    Fx = np.array(calc_Fx(x_k1_k.T,u,ts))
    G = np.array(calc_G(x_k1_k.T,u,ts))
    
    # # Convert continuous-time state-space model to discrete-time model
    sys_ct = control.ss(Fx, G, np.zeros(n), np.zeros(9))
    sys_dt = control.sample_system(sys_ct, ts, method='zoh')
    # Get discrete-time state transition and input-to-state matrices
    Phi = sys_dt.A
    Gamma = sys_dt.B

    P_k1_k = Phi@P_k1_k1@Phi.T + Gamma@Q@Gamma.T
    if (doIEKF == True):
        
        eta2    = x_k1_k
        err     = 2*epsilon
        itts    = 0
        
        while (err > epsilon):
            if (itts >= maxIterations):
                print("Terminating IEKF: exceeded max iterations (%d)\n" %(maxIterations))  
                break
            
            itts    = itts + 1
            eta1    = eta2
              
            # Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix 
            Hx = np.array(calc_Hx(eta1,u))
            
            # Observation and observation error predictions
            z_k1_k = np.array(calc_hx(eta1,u)).reshape(-1)                         # prediction of observation (for validation)   
            P_zz        = Hx@P_k1_k@Hx.T + R      # covariance matrix of observation error (for validation)   
            std_z       = np.sqrt(np.diag(P_zz))         # standard deviation of observation error (for validation)    
    
            # K(k+1) (gain)
            K           =P_k1_k @ Hx.T @ np.linalg.inv(P_zz) 
            
            # new observation
            eta2        = x_k1_k + K@(zi - z_k1_k - np.array((Hx@(x_k1_k - eta1).T)).reshape(-1))
            eta2    = np.array(eta2).reshape(-1)
            err         = np.linalg.norm(eta2-eta1)/np.linalg.norm(eta1)  
    
        IEKFitcount[k]  = itts
        x_k1_k1         = eta2
    
    else:
        Hx = np.array(calc_Hx(x_k1_k,u))
        
        # correction
        z_k1_k = np.array(calc_hx(x_k1_k,u)).reshape(-1)
        P_zz = Hx@P_k1_k@Hx.T + R     # covariance matrix of observation error (for validation)   
        std_z       = np.sqrt(np.diag(P_zz))
        # K(k+1) (gain)
        K           = P_k1_k @ Hx.T @  np.linalg.inv(P_zz)
        
        # Calculate optimal state x(k+1|k+1) 
        x_k1_k1     = np.array(x_k1_k + K@(zi - z_k1_k)).reshape(-1)
    
    # P(k|k) (correction) using the numerically stable form of P_k_1k_1 = (eye(n) - K*Hx) * P_kk_1 
    P_k1_k1 = (np.eye(n) - K @ Hx) @ P_k1_k
    std_x_cor   = np.sqrt(np.diag(P_k1_k1))        # standard deviation of state estimation error (for validation)
    
    va = x_k1_k1[6:9]-x_k1_k1[3:6] 
    va_proj = project_onto_plane(va, ey_kite)
    aoa.append(calculate_angle(ex_kite,va_proj))
    va_proj = project_onto_plane(va, ez_kite)
    sideslip.append(calculate_angle(ey_kite,va_proj))
    # pitch.append(np.arctan2(x_k1_k1[5],np.sqrt(x_k1_k1[3]**2+x_k1_k1[4]**2))*180/np.pi)
    pitch.append(calculate_angle(ex_kite, [0,0,1]))
    yaw.append(calculate_angle(ex_kite, [0,1,0]))
    roll.append(calculate_angle(ey_kite, [0,0,1]))
    # print(x_k1_k1[12:15]-x_k1_k[12:15])
    # # Next step
    
    if k ==0:
        rank_O = rank_observability_matrix(Phi,Hx)
        print(rank_O)
    # store results
    XX_k1_k1[:,k]   = np.array(x_k1_k1).reshape(-1)
    STD_x_cor[:,k]  = std_x_cor
    STD_z[:,k]      = std_z
    ZZ_pred [:,k]    = z_k1_k
    err_meas[:,k] = z_k1_k - zi
    
#%% Save results
ti = 100
results = XX_k1_k1[:,ti:k]
results = np.vstack((results,np.array(Ft)[ti:k,:].T,np.array(tether_len[ti:k]),np.array(CL[ti:k]),np.array(CD[ti:k]),np.array(aoa[ti:k])
                     ,np.array(pitch[ti:k]),np.array(yaw[ti:k]),np.array(roll[ti:k])))
column_names = ['x','y','z','vx','vy','vz','vwx','vwy','vwz','CL','CD','CS','Ftx','Fty','Ftz','Lt',
                'CLw','CDw','aoa','pitch','yaw','roll']
df = pd.DataFrame(data=results.T, columns=column_names)

flight_data = flight_data.iloc[ti:k]

# Save the DataFrame to a CSV file
csv_filename = 'EKFresults_temp.csv'
df.to_csv(csv_filename, index=False)
# Save the DataFrame to a CSV file
csv_filename = 'flightdata_temp.csv'
flight_data.to_csv(csv_filename, index=False)

