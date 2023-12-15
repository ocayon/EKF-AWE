import numpy as np
import casadi as ca
from config import kite_models,kcu_cylinders, tether_materials, kappa, z0, rho, g, n_tether_elements
import pandas as pd
import control
from scipy.interpolate import splrep, splev
from scipy.optimize import least_squares

#%% Class definitions
class ExtendedKalmanFilter:
    def __init__(self, stdv_x, stdv_y, ts, doIEKF=False, epsilon=1e-6, max_iterations=200):
        self.Q = self.get_state_noise_covariance(stdv_x)
        self.R = self.get_observation_noise_covariance(stdv_y)
        self.doIEKF = doIEKF
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.ts = ts
        self.n = len(stdv_x)
        self.P_k1_k1 = np.eye(self.n) * 1 ** 2
    
    def initialize(self, x_k1_k, u,z):
        self.x_k1_k = np.array(x_k1_k).reshape(-1)
        self.u = u
        self.z = z


    def predict(self):
        # Calculate Jacobians
        self.Fx = np.array(self.calc_Fx(self.x_k1_k,self.u))
        # self.G = np.array(self.calc_G(self.x_k1_k,self.u))
        nx = self.Fx.shape[0]
        
        # Calculate discrete time state transition and input-to-state matrices
        sys_ct = control.ss(self.Fx, np.zeros([nx,nx]), np.zeros(nx), np.zeros(nx))     # If process noise input matrix wants to be added ->control.ss(self.Fx, self.G, np.zeros(nx), np.zeros(nx))
        sys_dt = control.sample_system(sys_ct, self.ts, method='zoh')
        self.Phi = sys_dt.A
        # self.Gamma = sys_dt.B
    
        # Calculate covariance prediction error
        self.P_k1_k = self.Phi @ self.P_k1_k1 @ self.Phi.T + self.Q

    def update(self):
        if (self.doIEKF == True):
        
            eta2    = self.x_k1_k
            err     = 2*self.epsilon
            itts    = 0
            
            while (err > self.epsilon):
                if (itts >= self.max_iterations):
                    print("Terminating IEKF: exceeded max iterations (%d)\n" %(self.max_iterations))  
                    break
                
                itts    = itts + 1
                eta1    = eta2
                
                # Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix 
                self.Hx = np.array(self.calc_Hx(eta1,self.u))
                
                # Observation and observation error predictions
                self.z_k1_k = np.array(self.calc_hx(eta1,self.u)).reshape(-1)                         # prediction of observation (for validation)   
                self.P_zz        = self.Hx@self.P_k1_k@self.Hx.T + self.R      # covariance matrix of observation error (for validation)   
                self.std_z       = np.sqrt(np.diag(self.P_zz))         # standard deviation of observation error (for validation)    
        
                # K(k+1) (gain)
                self.K           =self.P_k1_k @ self.Hx.T @ np.linalg.inv(self.P_zz) 
                
                # new observation
                eta2        = self.x_k1_k + self.K@(self.z - self.z_k1_k - np.array((self.Hx@(self.x_k1_k - eta1).T)).reshape(-1))
                eta2    =      np.array(eta2).reshape(-1)
                err         = np.linalg.norm(eta2-eta1)/np.linalg.norm(eta1)  
        
            self.IEKF_itts = itts
            self.x_k1_k1         = eta2
        
        else:
            self.Hx = np.array(self.calc_Hx(self.x_k1_k,self.u))
            
            # correction
            self.z_k1_k = np.array(self.calc_hx(self.x_k1_k,self.u)).reshape(-1)
            self.P_zz = self.Hx@self.P_k1_k@self.Hx.T + self.R     # covariance matrix of observation error (for validation)   
            self.std_z       = np.sqrt(np.diag(self.P_zz))
            # K(k+1) (gain)
            self.K           = self.P_k1_k @ self.Hx.T @  np.linalg.inv(self.P_zz)
            
            # Calculate optimal state x(k+1|k+1) 
            self.x_k1_k1     = np.array(self.x_k1_k + self.K@(self.z - self.z_k1_k)).reshape(-1)
    
        self.P_k1_k1 = (np.eye(self.n) - self.K @ self.Hx) @ self.P_k1_k
        self.std_x_cor   = np.sqrt(np.diag(self.P_k1_k1))        # standard deviation of state estimation error (for validation)

    def get_state_noise_covariance(self, stdv_x):
        return np.diag(np.array(stdv_x)**2)
    def get_observation_noise_covariance(self, stdv_y):
        return np.diag(np.array(stdv_y)**2)
    
        
class DynamicModel:
    def __init__(self,kite,ts):
        

        self.r = ca.SX.sym('x',3)    # Kite position
        self.v = ca.SX.sym('v',3)    # Kite velocity
        self.uf = ca.SX.sym('uf')    # Ground wind velocity
        self.wdir = ca.SX.sym('wdir')# Ground wind direction
        self.CL = ca.SX.sym('CL')    # Lift coefficient
        self.CD = ca.SX.sym('CD')    # Drag coefficient
        self.CS = ca.SX.sym('CS')    # Side force coefficient
        self.Ft = ca.SX.sym('Ft',3)  # Tether force
        self.bias_lt = ca.SX.sym('bias_lt') # Bias tether length
        self.bias_aoa = ca.SX.sym('bias_aoa') # Bias angle of attack
        
        self.x = self.get_state()
        self.u = self.get_input()
        self.fx = self.get_fx(kite)
        
        # Define ODE system
        dae = {'x': self.x, 'p': self.u, 'ode': self.fx}                       # Define ODE system
        self.intg = ca.integrator('intg', 'cvodes', dae, 0,ts)    # Define integrator

    def get_state(self):    
        return ca.vertcat(self.r,self.v,self.uf,self.wdir,self.CL,self.CD,self.CS,self.bias_lt,self.bias_aoa)
    
    def get_input(self):
        return ca.vertcat(self.Ft)
    
    def get_fx(self,kite):
        wvel = self.uf/kappa*ca.log(self.r[2]/z0)
        vw = ca.vertcat(wvel*ca.cos(self.wdir),wvel*ca.sin(self.wdir),0)
        self.va = vw - self.v 

        dir_D = self.va/ca.norm_2(self.va)
        dir_L = self.Ft/ca.norm_2(self.Ft) - ca.dot(self.Ft/ca.norm_2(self.Ft),dir_D)*dir_D
        dir_S = ca.cross(dir_L,dir_D) 

        L = self.CL*0.5*rho*kite.area*ca.norm_2(self.va)**2*dir_L
        D = self.CD*0.5*rho*kite.area*ca.norm_2(self.va)**2*dir_D
        S = self.CS*0.5*rho*kite.area*ca.norm_2(self.va)**2*dir_S

        Fg = ca.vertcat(0, 0, -kite.mass*g)
        rp = self.v
        vp = (-self.Ft+L+D+S+Fg)/kite.mass
        return ca.vertcat(rp,vp,0,0,0,0,0,0,0)
    
    def get_fx_jac(self):
        return ca.jacobian(self.fx,self.x)

    def get_fx_jac_fun(self):
        return ca.Function('calc_Fx', [self.x,self.u],[self.get_fx_jac()])
    
    def propagate(self,x,u):
        return np.array(self.intg(x0=x,p=u)['xf'].T)


class ObservationModel:

    def __init__(self,x,u,measurements,KITE):
        self.x = x
        self.u = u
        self.measurements = measurements
        self.hx = self.get_hx(KITE)
    
    def get_hx(self,KITE):
        uf = self.x[6]
        wdir = self.x[7]
        wvel = uf/kappa*ca.log(self.x[2]/z0)
        vw = ca.vertcat(wvel*ca.cos(wdir),wvel*ca.sin(wdir),0)
        va = vw - self.x[3:6] 


        ez_kite = self.u/ca.norm_2(self.u)
        ey_kite = ca.cross(ez_kite, - self.x[3:6] )/ca.norm_2(ca.cross(ez_kite, - self.x[3:6] ))
        va_proj = project_onto_plane_sym(va, ey_kite)           # Projected apparent wind velocity onto kite y axis
        aoa = 90-calculate_angle_sym(ez_kite,va_proj)             # Angle of attack
        h = ca.SX()

        for key in self.measurements:
            if key == 'GPS_pos':
                h = ca.vertcat(self.x[0:3])
            elif key == 'GPS_vel':        
                h = ca.vertcat(h,self.x[3:6])
            elif key == 'GPS_acc':
                h = ca.vertcat(h,self.fx[3:6])
            elif key == 'ground_wvel':
                h = ca.vertcat(h,self.x[6])
            elif key == 'apparent_wvel':
                h = ca.vertcat(h,ca.norm_2(va))
            elif key == 'tether_len':
                h = ca.vertcat(h,ca.norm_2(self.x[0:3])-KITE.distance_kcu_kite+self.x[11])
            elif key == 'aoa':
                h = ca.vertcat(h,aoa+self.x[12])

        return h

    def get_hx_jac(self):
        return ca.jacobian(self.hx,self.x)
    
    def get_hx_jac_fun(self):
        return ca.Function('calc_Hx', [self.x,self.u],[self.get_hx_jac()])
    
    def get_hx_fun(self):
        return ca.Function('calc_hx', [self.x,self.u],[self.hx])
    
class KiteModel:
    def __init__(self, model_name, mass, area, distance_kcu_kite, total_length_bridle_lines, diameter_bridle_lines):
        self.model_name = model_name
        self.mass = mass
        self.area = area
        self.distance_kcu_kite = distance_kcu_kite
        self.total_length_bridle_lines = total_length_bridle_lines
        self.diameter_bridle_lines = diameter_bridle_lines


class KCUModel:

    # Exracted from Applied fluid dynamics handbook
    ldt_data = np.array([0,0.5,1.0,1.5,2.0,3.0,4.0,5.0])  # L/D values
    cdt_data = np.array([1.15,1.1,0.93,0.85,0.83,0.85,0.85,0.85])  # Cd values for tangential flow
    ldp_data = np.array([1,1.98,2.96,5,10,20,40,1e6])  # L/D values
    cdp_data = np.array([0.64,0.68,0.74,0.74,0.82,0.91,0.98,1.2])  # Cd values perpendicular flow

    cdt_cone_data= np.array([0.43, 0.35, 0.22, 0.19, 0.20, 0.21, 0.22, 0.24])
    ldt_cone = np.array([0.01, 0.64, 1.98, 4.57, 8.91, 12.02, 13.69, 15.29])
    # Create spline interpolations
    spline_t = splrep(ldt_data, cdt_data, s=0)
    spline_p = splrep(ldp_data, cdp_data, s=0)
    
    def __init__(self,length,diameter,mass):
        
        self.length = length
        self.diameter = diameter
        self.mass = mass

        # Example: Interpolate Cd for tangential flow at a specific L/D
        self.cdt = splev(self.length/self.diameter, KCUModel.spline_t)
        self.cdp = splev(self.length/self.diameter, KCUModel.spline_p)

        self.At = np.pi*(self.diameter/2)**2  # Calculate area of the KCU
        self.Ap = self.diameter*self.length  # Calculate area of the KCU

class TetherModel:
    def __init__(self,material,diameter,density,cd,Youngs_modulus):
        self.material = material
        self.diameter = diameter
        self.density = density
        self.cd = cd
        self.cf = 0.02
        self.E = Youngs_modulus
        self.area = np.pi*(self.diameter/2)**2
        self.EA = self.E*self.area
        
#%% Function definitions

def create_kite(model_name):
    if model_name in kite_models:
        model_params = kite_models[model_name]
        return KiteModel(model_name, model_params["mass"], model_params["area"], model_params["distance_kcu_kite"],
                     model_params["total_length_bridle_lines"], model_params["diameter_bridle_lines"])
    else:
        raise ValueError("Invalid kite model")
    
def create_kcu(model_name):
    if model_name in kcu_cylinders:
        model_params = kcu_cylinders[model_name]
        return KCUModel(model_params["length"], model_params["diameter"], model_params["mass"])
    else:
        raise ValueError("Invalid KCU model")

def create_tether(material_name,diameter):
    if material_name in tether_materials:
        material_params = tether_materials[material_name]
        return TetherModel(material_name,diameter,material_params["density"],material_params["cd"],material_params["Youngs_modulus"])
    else:
        raise ValueError("Invalid tether material")

def project_onto_plane(vector, plane_normal):
    return vector - np.dot(vector, plane_normal) * plane_normal


def project_onto_plane_sym(vector, plane_normal):
    return vector - ca.dot(vector, plane_normal) * plane_normal


def get_measurements(df, measurements,multiple_GPS = True):
    meas_dict = {}
    Z = []
    for meas in measurements:
        if meas == 'GPS_pos':
            col_rx = [col for col in df.columns if 'rx' in col]
            col_ry = [col for col in df.columns if 'ry' in col]
            col_rz = [col for col in df.columns if 'rz' in col]
            for i in range(len(col_rx)):
                Z.append(df[col_rx[i]].values)
                Z.append(df[col_ry[i]].values)
                Z.append(df[col_rz[i]].values)
            meas_dict[meas] = len(col_rx)

        elif meas == 'GPS_vel':
            col_vx = [col for col in df.columns if 'vx' in col]
            col_vy = [col for col in df.columns if 'vy' in col]
            col_vz = [col for col in df.columns if 'vz' in col]
            if multiple_GPS:
                for i in range(len(col_vx)):
                    Z.append(df[col_vx[i]].values)
                    Z.append(df[col_vy[i]].values)
                    Z.append(df[col_vz[i]].values)
                meas_dict[meas] = len(col_vx)
            else:
                Z.append(df['kite_0_vx'].values)
                Z.append(df['kite_0_vy'].values)
                Z.append(df['kite_0_vz'].values)
                meas_dict[meas] = 1
        
        elif meas == 'GPS_acc':
            col_ax = [col for col in df.columns if 'ax' in col]
            col_ay = [col for col in df.columns if 'ay' in col]
            col_az = [col for col in df.columns if 'az' in col]
            if multiple_GPS:
                for i in range(len(col_ax)):
                    Z.append(df[col_ax[i]].values)
                    Z.append(df[col_ay[i]].values)
                    Z.append(df[col_az[i]].values)
                meas_dict[meas] = len(col_ax)
            else:
                Z.append(df['kite_0_ax'].values)
                Z.append(df['kite_0_ay'].values)
                Z.append(df['kite_0_az'].values)
                meas_dict[meas] = 1
                
        
        elif meas == 'ground_wvel':
            uf = df['ground_wind_velocity']*kappa/np.log(10/z0)
            Z.append(uf.values)
            meas_dict[meas] = 1

        elif meas == 'apparent_wvel':
            col_va = [col for col in df.columns if 'apparent' in col]
            for i in range(len(col_va)):
                Z.append(df[col_va[i]].values)
                meas_dict[meas] = len(col_va)

        elif meas == 'tether_len':
            Z.append(df['ground_tether_length'].values)

        elif meas == 'aoa':
            Z.append(df['kite_angle_of_attack'].values)

    Z = np.array(Z)
    Z = Z.T

    return meas_dict, Z


def get_tether_end_position(x, set_parameter, n_tether_elements, r_kite, v_kite, vw,a_kite, kite, kcu, tether, separate_kcu_mass=True, elastic_elements=True, return_values=False, find_force=False):
    # Currently neglecting radial velocity of kite.
    if find_force:
        beta_n, phi_n, tension_ground = x
        tether_length = set_parameter
    else:
        beta_n, phi_n, tether_length = x
        tension_ground = set_parameter

    l_unstrained = tether_length/n_tether_elements
    m_s = np.pi*tether.diameter**2/4 * l_unstrained * tether.density

    n_elements = n_tether_elements
    if separate_kcu_mass:
        n_elements += 1
    
    wvel = np.linalg.norm(vw)
    wdir = vw/wvel

    vtau_kite = project_onto_plane(v_kite,r_kite/np.linalg.norm(r_kite)) # Velocity projected onto the tangent plane
    omega_tether = np.cross(r_kite,vtau_kite)/(np.linalg.norm(r_kite)**2) # Tether angular velocity, with respect to the tether attachment point

    
    # Find instantaneuous center of rotation and omega of the kite
    at = np.dot(a_kite,np.array(v_kite)/np.linalg.norm(v_kite))*np.array(v_kite)/np.linalg.norm(v_kite)
    omega_kite = np.cross(a_kite-at,v_kite)/(np.linalg.norm(v_kite)**2)
    ICR = np.cross(v_kite,omega_kite)/(np.linalg.norm(omega_kite)**2)      
    alpha = np.cross(at,ICR)/np.linalg.norm(ICR)**2
    

        

    tensions = np.zeros((n_elements, 3))
    tensions[0, 0] = np.cos(beta_n)*np.cos(phi_n)*tension_ground
    tensions[0, 1] = np.cos(beta_n)*np.sin(phi_n)*tension_ground
    tensions[0, 2] = np.sin(beta_n)*tension_ground

    positions = np.zeros((n_elements+1, 3))
    if elastic_elements:
        l_s = (tension_ground/(tether.EA)+1)*l_unstrained
        # print((tension_ground/tether_stiffness+1))
    else:
        l_s = l_unstrained
    positions[1, 0] = np.cos(beta_n)*np.cos(phi_n)*l_s
    positions[1, 1] = np.cos(beta_n)*np.sin(phi_n)*l_s
    positions[1, 2] = np.sin(beta_n)*l_s

    velocities = np.zeros((n_elements+1, 3))
    accelerations = np.zeros((n_elements+1, 3))
    non_conservative_forces = np.zeros((n_elements, 3))

    stretched_tether_length = l_s  # Stretched
    for j in range(n_elements):  # Iterate over point masses.
        last_element = j == n_elements - 1
        kcu_element = separate_kcu_mass and j == n_elements - 2

        # Determine kinematics at point mass j.
        vj = np.cross(omega_tether, positions[j+1, :])
        velocities[j+1, :] = vj
        aj = np.cross(omega_tether, vj)
        accelerations[j+1, :] = aj
        delta_p = positions[j+1, :] - positions[j, :]
        ej = delta_p/np.linalg.norm(delta_p)  # Axial direction of tether element
        vwj = wvel*np.log(positions[j+1,2]/z0)/np.log(r_kite[2]/z0)*wdir # Wind
        
        if last_element:
            vj = np.array(v_kite)
            aj = np.array(a_kite)
        if kcu_element:       
            vkcu = v_kite + np.cross(omega_kite,positions[j+1, :]-r_kite)
            vj = vkcu
            velocities[j+1, :] = vj
            akcu = a_kite+ np.cross(alpha,positions[j+1, :]-r_kite) +np.cross(omega_kite,np.cross(omega_kite,positions[j+1, :]-r_kite))
            aj = akcu
            accelerations[j+1, :] = aj

            ej = (r_kite-np.array(positions[j+1, :]))/np.linalg.norm(r_kite-np.array(positions[j+1, :]))
        # Determine flow at point mass j.
        vaj = vj - vwj  # Apparent wind velocity
        
        vajp = np.dot(vaj, ej)*ej  # Parallel to tether element
        # TODO: check whether to use vajn
        vajn = vaj - vajp  # Perpendicular to tether element

        vaj_sq = np.linalg.norm(vaj)*vaj

        # Determina angle between  va and tether
        theta = calculate_angle(-vaj,ej,False)
        # vaj_sq = np.linalg.norm(vajn)*vajn
        CD_tether = tether.cd*np.sin(theta)**3+tether.cf
        # CL_tether = tether.cd*np.sin(theta)**2*np.cos(theta)
        tether_drag_basis = rho*l_unstrained*tether.diameter*CD_tether*vaj_sq
        
        # Determine drag at point mass j.
        if not separate_kcu_mass:
            if n_tether_elements == 1:
                dj = -.125*tether_drag_basis
            elif last_element:
                dj = -.25*tether_drag_basis  # TODO: add bridle drag
            else:
                dj = -.5*tether_drag_basis
        else:
            if last_element:
                # dj = -0.25*rho*L_blines*d_bridle*vaj_sq*cd_t # Bridle lines drag
                dj = 0
                
            elif n_tether_elements == 1:
                dj = -.25*tether_drag_basis
                dp= -.5*rho*np.linalg.norm(vajp)*vajp*kcu.cdp*kcu.Ap  # Adding kcu drag perpendicular to kcu
                dt= -.5*rho*np.linalg.norm(vajn)*vajn*kcu.cdt*kcu.At  # Adding kcu drag parallel to kcu
                dj += dp+dt
                cd_kcu = (np.linalg.norm(dp+dt))/(0.5*rho*kite.area*np.linalg.norm(vaj)**2)
            elif kcu_element:
                dj = -.25*tether_drag_basis

                cd_kcu = kcu.cdt*np.sin(theta)**3+tether.cf
                cl_kcu = kcu.cdt*np.sin(theta)**2*np.cos(theta)
                # D_turbine = 0.5*rho*np.linalg.norm(vaj)**2*np.pi*0.2**2*1
                # dp= -.5*rho*np.linalg.norm(vajp)*vajp*kcu.cdp*kcu.Ap  # Adding kcu drag perpendicular to kcu
                # dt= -.5*rho*np.linalg.norm(vajn)*vajn*kcu.cdt*kcu.At  # Adding kcu drag parallel to kcu
                # th = -0.5*rho*vaj_sq*np.pi*0.2**2*0.4
                # dj += dp+dt

                # Approach described in Hoerner, taken from Paul Thedens dissertation
                dir_D = -vaj/np.linalg.norm(vaj)
                dir_L = ej - np.dot(ej,dir_D)*dir_D
                L_kcu = 0.5*rho*np.linalg.norm(vaj)**2*kcu.At*cl_kcu
                D_kcu = 0.5*rho*np.linalg.norm(vaj)**2*cd_kcu*kcu.At
                dj += L_kcu*dir_L + D_kcu*dir_D #+ D_turbine*dir_D

                # cd_kcu = (np.linalg.norm(dp+dt))/(0.5*rho*kite.area*np.linalg.norm(vaj)**2)

            else:
                dj = -.5*tether_drag_basis

        if not separate_kcu_mass:
            if last_element:
                point_mass = m_s/2 + kite.mass + kcu.mass           
            else:
                point_mass = m_s
        else:
            if last_element:
                point_mass = kite.mass          
                # aj = np.zeros(3)
            elif kcu_element:
                point_mass = m_s/2 + kcu.mass
            else:
                point_mass = m_s

        # Use force balance to infer tension on next element.
        fgj = np.array([0, 0, -point_mass*g])
        next_tension = point_mass*aj + tensions[j, :] - fgj - dj  # a_kite gave better fit
        if not last_element:
            tensions[j+1, :] = next_tension
            non_conservative_forces[j, :] = dj
        else:
            aerodynamic_force = next_tension
            non_conservative_forces[j, :] = dj + aerodynamic_force


        # Derive position of next point mass from former tension
        if kcu_element:
            positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * kite.distance_kcu_kite
            
        elif not last_element:
            if elastic_elements:
                l_s = (np.linalg.norm(tensions[j+1, :])/tether.EA+1)*l_unstrained
            else:
                l_s = l_unstrained
            stretched_tether_length += l_s
            positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * l_s

    if return_values == 2:
        return positions, velocities, accelerations, tensions, aerodynamic_force, non_conservative_forces
    elif return_values:
        va = vwj-vj  # All y-axes are defined perpendicular to apparent wind velocity.

        ez_bridle = tensions[-1, :]/np.linalg.norm(tensions[-1, :])
        ey_bridle = np.cross(ez_bridle, -vj)/np.linalg.norm(np.cross(ez_bridle, -vj))
        ex_bridle = np.cross(ey_bridle, ez_bridle)
        dcm_b2w = np.vstack(([ex_bridle], [ey_bridle], [ez_bridle])).T

        ez_tether = tensions[-2, :]/np.linalg.norm(tensions[-2, :])
        ey_tether = np.cross(ez_tether, va)/np.linalg.norm(np.cross(ez_tether, va))
        ex_tether = np.cross(ey_tether, ez_tether)
        dcm_t2w = np.vstack(([ex_tether], [ey_tether], [ez_tether])).T

        ez_f_aero = aerodynamic_force/np.linalg.norm(aerodynamic_force)
        ey_f_aero = np.cross(ez_f_aero, va)/np.linalg.norm(np.cross(ez_f_aero, va))
        ex_f_aero = np.cross(ey_f_aero, ez_f_aero)
        dcm_fa2w = np.vstack(([ex_f_aero], [ey_f_aero], [ez_f_aero])).T

        ez_tau = r_kite/np.linalg.norm(r_kite)
        ey_tau = np.cross(ez_tau, va)/np.linalg.norm(np.cross(ez_tau, va))
        ex_tau = np.cross(ey_tau, ez_tau)
        dcm_tau2w = np.vstack(([ex_tau], [ey_tau], [ez_tau])).T
        
        dir_D = va/np.linalg.norm(va)
        CD = np.dot(aerodynamic_force,dir_D)/(0.5*rho*kite.area*np.linalg.norm(va)**2)
        dir_L = tensions[-1, :]/np.linalg.norm(tensions[-1, :]) - np.dot(tensions[-1, :]/np.linalg.norm(tensions[-1, :]),dir_D)*dir_D
        CL = np.dot(aerodynamic_force,dir_L)/(0.5*rho*kite.area*np.linalg.norm(va)**2)
        dir_S = np.cross(dir_L,dir_D)
        CS = np.dot(aerodynamic_force,dir_S)/(0.5*rho*kite.area*np.linalg.norm(va)**2)

        return positions, stretched_tether_length, dcm_b2w, dcm_t2w, dcm_fa2w, \
               dcm_tau2w, aerodynamic_force, va,tensions[-1,:], dcm_t2w[:,1],cd_kcu, CL,CD,CS
    else:
        return (positions[-1, :] - r_kite) 

def rotate_vector(v, u, theta):
    # Normalize vectors
    v = v 
    u = u / ca.norm_2(u)

    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)

    v_rot = v * cos_theta + ca.cross(u, v) * sin_theta + u * ca.dot(u, v) * (1 - cos_theta)

    return v_rot    

def calculate_angle(vector_a, vector_b,deg = True):
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)
    
    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    if deg:
        return angle_deg
    else:
        return angle_rad

def calculate_angle_sym(vector_a, vector_b):
    dot_product = ca.dot(vector_a, vector_b)
    magnitude_a = ca.norm_2(vector_a)
    magnitude_b = ca.norm_2(vector_b)
    
    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = ca.arccos(cos_theta)
    angle_deg = angle_rad*180/np.pi
    
    return angle_deg

def observability_Lie_method(f,h,x,u, x0, u0):
        
    n = f.shape[0]
    m = h.shape[0]
    O = ca.SX.zeros((m*n, n))
    Li = ca.simplify(ca.jacobian(h,x))
    O[0*m:(1)*m,:] = Li
    for i in range(1,m):
        Hxf = ca.mtimes(Li, f)
        Hxxf= ca.simplify(ca.jacobian(Hxf,x))
        Li = Hxxf
        O[i*m:(i+1)*m,:] = Hxxf
        
    calc_O = ca.Function('calc_O', [x,u],[O])
    O_app = calc_O(x0,u0)
    if np.linalg.matrix_rank(O_app) == len(x0):
        print('System is observable')
    else:
        print('System is not observable')      


def rank_observability_matrix(A,C):
    # Construct the observability matrix O_numeric
    n = A.shape[1]  # Number of state variables
    m = C.shape[0]  # Number of measurements
    O = np.zeros((m * n, n))
    
    for i in range(n):
        power_of_A = np.linalg.matrix_power(A, i)
        O[i * m: (i + 1) * m, :] = C @ power_of_A

    # Compute the rank of O using NumPy
    rank_O = np.linalg.matrix_rank(O)
    return rank_O

def R_EG_Body(Roll,Pitch,Yaw):#!!In radians!!
    
    #Rotational matrix for Roll
    R_Roll=np.array([[1, 0, 0],[0,np.cos(Roll),np.sin(Roll)],[0,-np.sin(Roll),np.cos(Roll)]])#OK checked with Blender
    
    #Rotational matrix for Pitch
    R_Pitch=np.array([[np.cos(Pitch), 0, np.sin(Pitch)],[0,1,0],[-np.sin(Pitch), 0, np.cos(Pitch)]])#Checked with blender
    
    #Rotational matrix for Roll
    R_Yaw= np.array([[np.cos(Yaw),-np.sin(Yaw),0],[np.sin(Yaw),np.cos(Yaw),0],[0,0,1]])#Checked with Blender
    
    #Total Rotational Matrix
    return R_Roll.dot(R_Pitch.dot(R_Yaw))


def calculate_polar_coordinates(r):
    # Calculate azimuth and elevation angles from a vector.
    r_mod = np.linalg.norm(r)
    az = np.arctan2(r[1], r[0])
    el = np.arcsin(r[2]/r_mod)
    return el, az, r_mod

def initialize_state(flight_data,kite,kcu,tether):
    n_tether_elements = 30
    ground_wdir0 = np.mean(flight_data['ground_wind_direction'].iloc[0:3000])/180*np.pi # Initial wind direction
    ground_wvel0 = np.mean(flight_data['ground_wind_velocity'].iloc[0:3000]) # Initial wind velocity
    uf0 = ground_wvel0*kappa/np.log(10/z0)
    wvel0 = uf0/kappa*np.log(flight_data['kite_0_rz'].iloc[0]/z0)
    vw = [wvel0*np.cos(ground_wdir0),wvel0*np.sin(ground_wdir0),0] # Initial wind velocity
    row = flight_data.iloc[0] # Initial row of flight data
    kite_pos = np.array([row['kite_0_rx'],row['kite_0_ry'],row['kite_0_rz']]) # Initial kite position
    kite_vel = np.array([row['kite_0_vx'],row['kite_0_vy'],row['kite_0_vz']]) # Initial kite velocity
    kite_acc = np.array([row['kite_1_ax'],row['kite_1_ay'],row['kite_1_az']]) # Initial kite acceleration
    args = (row['ground_tether_force'], n_tether_elements, kite_pos, kite_vel,vw,
            kite_acc, kite, kcu, tether, True, True)
    opt_res = least_squares(get_tether_end_position, list(calculate_polar_coordinates(np.array(kite_pos))), args=args,
                            kwargs={'find_force': False}, verbose=0)
    res = get_tether_end_position(
        opt_res.x, *args, return_values=True, find_force=False)
    u0 = res[8]        # Tether force
    CL0 = res[-3]     # Lift coefficient
    CD0 = res[-2]     # Drag coefficient
    CS0 = res[-1]     # Side force coefficient
    dcm_b2w = res[2]            # DCM bridle to earth
    ey_kite = dcm_b2w[:,1]      # Kite y axis perpendicular to va and tether
    ez_kite = dcm_b2w[:,2]      # Kite z axis pointing in the direction of the tension
    ex_kite = dcm_b2w[:,0]      # Kite x axis 
    va = vw-kite_vel            # Apparent wind velocity
    va_proj = project_onto_plane(va, ey_kite)           # Projected apparent wind velocity onto kite y axis
    aoa = 90-calculate_angle(ez_kite,va_proj)             # Angle of attack
    bias_tether = res[1]- row['ground_tether_length']
    

    x0 = np.vstack((flight_data[['kite_0_rx','kite_0_ry','kite_0_rz']].values[0, :],flight_data[['kite_0_vx','kite_0_vy','kite_0_vz']].values[0, :]))
    x0 = np.append(x0,[0.6,ground_wdir0,CL0,CD0,CS0,bias_tether,0])

    return x0, u0, opt_res.x

def calculate_quasi_static_tether(x, fd, opt_guess, kite, kcu, tether):
    wvel = x[6]/kappa*np.log(x[2]/z0) # Wind speed
    wdir = x[7] # Wind direction
    vw = np.array([wvel*np.cos(wdir),wvel*np.sin(wdir),0]) # Wind velocity
    # Solve for tether shape and force
    args = (fd['ground_tether_force'], n_tether_elements, x[0:3], x[3:6],vw,
            list(fd[['kite_0_ax','kite_0_ay','kite_0_az']]), kite, kcu, tether, True, True)
    opt_res = least_squares(get_tether_end_position, opt_guess, args=args,
                            kwargs={'find_force': False}, verbose=0,xtol = 1e-3,ftol = 1e-3)
    opt_guess = opt_res.x
    # Get results from optimization
    res = get_tether_end_position(
        opt_res.x, *args, return_values=True, find_force=False)

    dcm_b2w = res[2]            # DCM bridle to earth
    ey_kite = dcm_b2w[:,1]      # Kite y axis perpendicular to va and tether
    ez_kite = dcm_b2w[:,2]      # Kite z axis pointing in the direction of the tension
    ex_kite = dcm_b2w[:,0]      # Kite x axis 
    Ft = np.array(res[8])                 # Tether force
    tether_len = res[1]         # Tether length
    CL = res[-3]                # Lift coefficient
    CD = res[-2]                # Drag coefficient
    CS = res[-1]                # Side force coefficient
    cd_kcu = res[-4]            # Kite control unit drag coefficient
    va = vw-x[3:6]              # Apparent wind velocity
    
    va_proj = project_onto_plane(va, ey_kite)           # Projected apparent wind velocity onto kite y axis
    aoa = 90-calculate_angle(ez_kite,va_proj)             # Angle of attack
    va_proj = project_onto_plane(va, ez_kite)           # Projected apparent wind velocity onto kite z axis
    sideslip = calculate_angle(ey_kite,va_proj)         # Sideslip angle
    pitch = 90-calculate_angle(-ex_kite, [0,0,1])           # Pitch angle
    yaw = np.arctan2(ex_kite[0],ex_kite[1])*180/np.pi   # Yaw angle       
    roll = calculate_angle(ey_kite, [0,0,1])            # Roll angle

    res_williams = np.array([roll,pitch,yaw,aoa,sideslip,CL,CD,CS,cd_kcu,tether_len])

    return Ft, res_williams, opt_guess
