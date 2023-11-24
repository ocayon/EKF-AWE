import casadi as ca
import numpy as np
import pandas as pd

from scipy.optimize import least_squares
from utils import *
import control
import time


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read and process data
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_tether_elements = 5

model = 'v3'
year = '2019'
month = '10'
day = '08'

if model == 'v3':
    from v3_properties import *
elif model == 'v9':
    from v9_properties import *

file_name = model+'_'+year+'-'+month+'-'+day
file_path = './data/'+ file_name+'.csv'


flight_data = pd.read_csv(file_path)
flight_data = flight_data.reset_index()
    
measured_wvel = flight_data['ground_wind_velocity']
measured_uf = measured_wvel*kappa/np.log(10/z0)

# window_size=10
# flight_data['ax']=np.convolve(flight_data['ax'], np.ones(window_size)/window_size, mode='same')
# flight_data['ay']=np.convolve(flight_data['ay'], np.ones(window_size)/window_size, mode='same')
# flight_data['az']=np.convolve(flight_data['az'], np.ones(window_size)/window_size, mode='same')

dep_fd = flight_data[flight_data['kcu_set_depower']>25]
pow_fd = flight_data[flight_data['kcu_set_depower']<25]
u_p = flight_data['kcu_set_depower']-min(flight_data['kcu_set_depower'])
u_p = u_p/max(u_p)
ground_wind_dir = np.array(-flight_data['ground_wind_direction']/180*np.pi-np.pi/2.+2*np.pi)

up = (flight_data['kcu_set_depower']-min(flight_data['kcu_set_depower']))/(max(flight_data['kcu_set_depower'])-min(flight_data['kcu_set_depower']))

vw = [9*np.cos(np.mean(ground_wind_dir)),9*np.sin(np.mean(ground_wind_dir)),0]

CL = []
CD= []
CS = []
L=[]
D=[]
tether_length =[]
for i in range(2):
    row = flight_data.iloc[i]
    kite_pos = np.array([row['kite_0_rx'],row['kite_0_ry'],row['kite_0_rz']])
    kite_vel = np.array([row['kite_0_vx'],row['kite_0_vy'],row['kite_0_vz']])
    kite_acc = np.array([row['kite_1_ax'],row['kite_1_ay'],row['kite_1_az']])
    args = (row['ground_tether_force'], n_tether_elements, kite_pos, kite_vel,vw,
            kite_acc,True, False)
    opt_res = least_squares(get_tether_end_position, list(calculate_polar_coordinates(np.array(kite_pos))), args=args,
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
x0 = np.vstack((flight_data[['kite_0_rx','kite_0_ry','kite_0_rz']].values[0, :],flight_data[['kite_0_vx','kite_0_vy','kite_0_vz']].values[0, :]))
x0 = np.append(x0,[0.6,np.mean(ground_wind_dir),0.6,0.1,0])
#%% Measurement vectors 
measurements = ['GPS_pos', 'GPS_vel','apparent_wvel']
meas_dict,Z = get_measurements(flight_data,measurements,False)
# Z = np.array([flight_data['rx'],flight_data['ry'],flight_data['rz'],flight_data['vx'],flight_data['vy'],flight_data['vz'],
#               measured_uf,flight_data['airspeed_apparent_windspeed'],np.zeros(len(flight_data)),flight_data['ax'],flight_data['ay'],flight_data['az']]).T

dep = (flight_data['ground_tether_reelout_speed'] < 0) #& (up > 0.1)
#%% Definition kalman filter matrices

Ts = ca.SX.sym('ts')    # timestep
Ft = ca.SX.sym('Ft',3)  # Tether force
r = ca.SX.sym('r', 3)   # Position
v = ca.SX.sym('v', 3)   # Velocity
a = ca.SX.sym('a', 3)   # Acceleration
uf = ca.SX.sym('uf')    # Fricton velocity
wdir = ca.SX.sym('wdir')# Wind direction
CD = ca.SX.sym('CD')    # Drag coefficient
CL = ca.SX.sym('CL')    # Lift coefficient
CS = ca.SX.sym('CS')    # Side force coefficient
vz = ca.SX.sym('vz')    # Apparent wind velocity


x = ca.vertcat(r,v,uf,wdir,CL,CD,CS) # State vector symbolic
u_sym = ca.vertcat(Ft) # Input vector symbolic
fx,calc_Fx,calc_G = state_noise_matrices(x,u_sym,Ts)        # Nonlinear state transition function (fx), state transition matrix (Fx), system noise input matrix(G)

check_obs = False
if check_obs == True:
    meas_dict = {'GPS_pos': 1, 'GPS_vel': 1, 'GPS_acc': 0,'apparent_wvel':0,'ground_wvel':0}
    hx,calc_hx,calc_Hx = observation_matrices(x,u_sym,meas_dict)   # Nonlinear observation function (hx), observation matrix (Hx)
    O = observability_Lie_method(fx,hx,x)
    calc_O = ca.Function('calc_O', [x,u_sym],[O])
    O_app = calc_O(x0,res[8])

    print(np.linalg.matrix_rank(O_app))

hx,calc_hx,calc_Hx = observation_matrices(x,u_sym,meas_dict)   # Nonlinear observation function (hx), observation matrix (Hx)

#%%
########################################################################
## Set simulation parameters
########################################################################
n           =  x0.shape[0]      # state dimension
nm          =  Z.shape[1]       # number of measurements
m           =  3                # number of inputs
    
stdv_xGPS = 2.5
stdv_vGPS = 1
stdv_aGPS = 10
stdv_uf = 0.1
stdv_va = 0.2
stdv_dirw = 10/180*np.pi

R = np.zeros((nm,nm))
j = 0
jva = None
for key, value in meas_dict.items():
    if key == 'GPS_pos':
        for i in range(value):
            R[j:j+3, j:j+3] = np.eye(3) * stdv_xGPS**2
            # R[j+3,j+3] = 10**2
            j +=3
    elif key == 'GPS_vel':
        for i in range(value):
            R[j:j+3, j:j+3] = np.eye(3) * stdv_vGPS**2
            j += 3
    elif key == 'GPS_acc':
        for i in range(value):
            R[j:j+3, j:j+3] = np.eye(3) * stdv_aGPS**2
            j += 3
    elif key == 'ground_wvel':
        for i in range(value):
            R[j,j] = stdv_uf**2
            j+=1
    elif key == 'apparent_wvel':
        for i in range(value):
            R[j,j] = stdv_va**2
            jva = j
            j+=1

#%% Define process noise matrix
stdv_Ft =0
stdv_CL = 0.1**2
stdv_CD = 0.05**2
stdv_CS = 0.05**2
stdv_uf = 0.025**2
stdv_x = 0.1**2
stdv_v = 0.1**2
stdv_wdir = (2/180*np.pi)**2

# Define process noise matrix
Q = np.zeros((11,11))
Q[:3,:3] = np.eye(3)*stdv_x**2
Q[3:6,3:6] = np.eye(3)*stdv_v**2
Q[6,6] = stdv_uf**2
Q[7,7] = stdv_wdir**2
Q[8,8] = stdv_CL**2
Q[9,9] = stdv_CD**2
Q[10,10] = stdv_CS**2

# Aerodynamic coefficients correlation
# Q[6,7] = Q[7,6] = np.sqrt(Q[6,6]*Q[7,7])        # CL,CD
# Q[6,8] = Q[8,6] = np.sqrt(Q[6,6]*Q[8,8])        # CL,CD
# Q[6,9] = Q[9,6] = np.sqrt(Q[6,6]*Q[9,9])        # CL,CD
# Q[6,10] = Q[10,6] = np.sqrt(Q[6,6]*Q[10,10])        # CL,CD
# Q[6,7] = Q[7,6] = np.sqrt(Q[6,6]*Q[7,7])        # CL,CD


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
doIEKF          = True           # If false, EKF without iterations is used
maxIterations   = 200               # maximum amount of iterations per sample

# Store Initial values
XX_k1_k1[:,0]   = x0
ZZ_pred[:,0]    = Z[0]

# Define Initial Matrix P
P_k1_k1 = np.eye(n)*1**2

x_sol = x0
x_k1_k1 = x0
t = np.array(flight_data['time'])
u = res[8]      # Initial input
Ft = []         # Tether force
tether_len = [] # Tether length
CL = []         # Lift coefficient
CD = []         # Drag coefficient
aoa = []        # Angle of attack
yaw = []        # Yaw angle
pitch = []      # Pitch angle
roll = []       # Roll angle
cd_kcu = []     # Kite control unit drag coefficient
sideslip = []   # Sideslip angle
tether_pos = [] # Tether positions
flight_data['ground_tether_length'] = flight_data['ground_tether_length']+22.

fx,calc_Fx,calc_G = state_noise_matrices(x,u_sym,Ts)        # Nonlinear state transition function (fx), state transition matrix (Fx), system noise input matrix(G)
hx,calc_hx,calc_Hx = observation_matrices(x,u_sym,meas_dict)   # Nonlinear observation function (hx), observation matrix (Hx)
def print_diagonal(matrix):
    # Check if the matrix is square
    if not all(len(row) == len(matrix) for row in matrix):
        print("The matrix is not square, and therefore, it doesn't have a diagonal.")
        return

    # Print the diagonal elements
    diagonal = [matrix[i][i] for i in range(len(matrix))]
    print("Diagonal elements:", diagonal)
from scipy.stats import chi2 
    
cov_uf = []
cov_wdir = []
dae = {'x': x, 'p': u_sym, 'ode': fx}                       # Define ODE system
intg = ca.integrator('intg', 'cvodes', dae, {'tf': 0.1})    # Define integrator
start_time = time.time()
mins = -1
for k in range(n_intervals):
    wvel = x_k1_k1[6]/kappa*np.log(x_k1_k1[2]/z0)
    wdir = x_k1_k1[7]
    vw = np.array([wvel*np.cos(wdir),wvel*np.sin(wdir),0])
    ts = t[k+1]-t[k]
    row = flight_data.iloc[k]
   
    zi = Z[k]

        
    # Accuracy of the va_sensor depending on the powering state (Depowered -> aligned with flow)
    if jva:
        if dep[k]:
            R[jva,jva] = stdv_va**2
        else:
            R[jva,jva] = 2**2
            
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
    sys_ct = control.ss(Fx, G, np.zeros(n), np.zeros(len(Q)))
    sys_dt = control.sample_system(sys_ct, ts, method='zoh')
    # Get discrete-time state transition and input-to-state matrices
    Phi = sys_dt.A
    Gamma = sys_dt.B

    # P_k1_k = Phi@P_k1_k1@Phi.T + Gamma@Q@Gamma.T
    P_k1_k = Phi@P_k1_k1@Phi.T + Q
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
    # if np.trace(P_k1_k1[6::,6::])<1e-4:
    #     P_k1_k1[6::,6::]= P_k1_k1[6::,6::]*10
    # Find tether shape and force
    args = (row['ground_tether_force'], n_tether_elements, x_k1_k1[0:3], x_k1_k1[3:6],vw,
            list(row[['kite_1_ax','kite_1_ay','kite_1_az']]),True, True)
    # Find tether shape
    opt_res = least_squares(get_tether_end_position, opt_res.x, args=args,
                            kwargs={'find_force': False}, verbose=0)
    # Calculate tether force
    res = get_tether_end_position(
        opt_res.x, *args, return_values=True, find_force=False)
    
    u = np.array(res[8])        # Input next step
    
    # # Assuming normalized_innovation is a 1D array or list of normalized innovations
    # meas_err = (zi - z_k1_k) 
    
    # NIS = meas_err.T @ np.linalg.inv(P_zz) @ meas_err
    # # Calculate the chi-square statistic
    # chi_square_statistic = np.sum(NIS**2)
    
    # # Set the degrees of freedom equal to the dimension of the measurement vector
    # degrees_of_freedom = len(meas_err)
    
    # # Set the significance level (e.g., 0.05)
    # significance_level = 0.01
    
    # # Calculate the critical value from the chi-square distribution
    # critical_value = chi2.ppf(1 - significance_level, degrees_of_freedom)
    
    # # Compare the chi-square statistic to the critical value
    # if chi_square_statistic < critical_value:
    #     print("The normalized innovation follows the expected distribution.")
    # else:
    #     # P_k1_k1[6::,6::] = P_k1_k1[6::,6::]*2
    #     print("The normalized innovation does not follow the expected distribution.")
    
    
    cov_uf.append(P_k1_k1[6,6])
    cov_wdir.append(P_k1_k1[7,7])
    

    tether_pos.append(res[0])   # Tether positions
    dcm_b2w = res[2]            # DCM bridle to earth
    ey_kite = dcm_b2w[:,1]      # Kite y axis
    ez_kite = dcm_b2w[:,2]      # Kite z axis
    ex_kite = dcm_b2w[:,0]      # Kite x axis
    Ft.append(res[8])           # Tether force
    tether_len.append(res[1])   # Tether length
    CL.append(res[-2])          # Lift coefficient
    CD.append(res[-1])          # Drag coefficient
    cd_kcu.append(res[-3])      # Kite control unit drag coefficient
    va = x_k1_k1[6:9]-x_k1_k1[3:6]                      # Apparent wind velocity
    
    va_proj = project_onto_plane(va, ey_kite)           # Projected apparent wind velocity onto kite y axis
    aoa.append(calculate_angle(ex_kite,va_proj))        # Angle of attack
    va_proj = project_onto_plane(va, ez_kite)           # Projected apparent wind velocity onto kite z axis
    sideslip.append(calculate_angle(ey_kite,va_proj))   # Sideslip angle
    pitch.append(calculate_angle(ex_kite, [0,0,1]))     # Pitch angle
    yaw.append(calculate_angle(ex_kite, [0,1,0]))       # Yaw angle       
    roll.append(calculate_angle(ey_kite, [0,0,1]))      # Roll angle

    # Calculate rank observability matrix
    if k ==300:
        rank_O = rank_observability_matrix(Fx,Hx)
        print(rank_O)
        rank_O = rank_observability_matrix(Phi,Hx)
        print(rank_O)

    # # Next step
    # store results
    XX_k1_k1[:,k]   = np.array(x_k1_k1).reshape(-1)
    STD_x_cor[:,k]  = std_x_cor
    STD_z[:,k]      = std_z
    ZZ_pred [:,k]    = z_k1_k
    err_meas[:,k] = z_k1_k - zi
    
#%% Save results
ti = 0
results = XX_k1_k1[:,ti:k]
results = np.vstack((results,np.array(Ft)[ti:k,:].T,np.array(tether_len[ti:k]),np.array(CL[ti:k]),np.array(CD[ti:k]),np.array(aoa[ti:k])
                     ,np.array(pitch[ti:k]),np.array(yaw[ti:k]),np.array(roll[ti:k]),np.array(cd_kcu[ti:k])))
column_names = ['x','y','z','vx','vy','vz','uf','wdir','CL','CD','CS','Ftx','Fty','Ftz','Lt',
                'CLw','CDw','aoa','pitch','yaw','roll','cd_kcu']
df = pd.DataFrame(data=results.T, columns=column_names)

flight_data = flight_data.iloc[ti:k]

path = './results/'+model+'/'
# Save the DataFrame to a CSV file
csv_filename = file_name+'_res_GPS.csv'
df.to_csv(path+csv_filename, index=False)
# Save the DataFrame to a CSV file
csv_filename = file_name+'_fd.csv'
flight_data.to_csv(path+csv_filename, index=False)

