
import numpy as np
import casadi as ca
from config import kappa, z0, rho, g
import control
from utils import project_onto_plane, calculate_angle_2vec
#%% 
class ExtendedKalmanFilter:
    def __init__(self, stdv_x, stdv_y, ts,dyn_model,obs_model, doIEKF=True, epsilon=1e-6, max_iterations=200):
        self.Q = self.get_state_noise_covariance(stdv_x)
        self.R = self.get_observation_noise_covariance(stdv_y)
        self.doIEKF = doIEKF
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.ts = ts
        self.n = len(stdv_x)
        self.P_k1_k1 = np.eye(self.n) * 1 ** 2
        self.calc_Fx = dyn_model.get_fx_jac_fun()
        self.calc_Hx = obs_model.get_hx_jac_fun()
        self.calc_hx = obs_model.get_hx_fun()

    
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
    def __init__(self,kite,tether,kcu,ts):
        

        self.r = ca.SX.sym('r',3)    # Kite position
        self.r_tether_model = ca.SX.sym('r_tether_model',3) # Tether attachment point
        self.v = ca.SX.sym('v',3)    # Kite velocity
        self.uf = ca.SX.sym('uf')    # Friction velocity
        self.wdir = ca.SX.sym('wdir')# Ground wind direction
        self.CL = ca.SX.sym('CL')    # Lift coefficient
        self.CD = ca.SX.sym('CD')    # Drag coefficient
        self.CS = ca.SX.sym('CS')    # Side force coefficient
        self.Ftg = ca.SX.sym('ground_tether_force')  # Tether force
        self.tether_length = ca.SX.sym('tether_length') # Tether length
        self.reelout_speed = ca.SX.sym('reelout_speed') # Tether reelout speed
        self.elevation_0 = ca.SX.sym('elevation_first_tether_element') # Elevation from ground to first tether element
        self.azimuth_0 = ca.SX.sym('azimuth_first_tether_element')   # Azimuth from ground to first tether element
        self.a_kcu =  ca.SX.sym('a_kcu',3)  # KCU acceleration
        self.uf_u = ca.SX.sym('uf_u')    # Friction velocity for input (previous timestep)
        self.wdir_u = ca.SX.sym('wdir_u')   # Ground wind direction for input (previous timestep)


        self.x = self.get_state(kite,tether,kcu)
        self.u = self.get_input()
        self.fx = self.get_fx(kite,tether,kcu)
        
        # Define ODE system
        dae = {'x': self.x, 'p': self.u, 'ode': self.fx}                       # Define ODE system
        self.intg = ca.integrator('intg', 'cvodes', dae, 0,ts)    # Define integrator

    def get_state(self,kite,tether,kcu):    
        return ca.vertcat(self.r,self.v,self.uf,self.wdir,self.CL,self.CD,self.CS,self.tether_length, self.elevation_0, self.azimuth_0)
    
    def get_input(self):
        return ca.vertcat(self.reelout_speed,self.Ftg,self.a_kcu, self.uf_u,self.wdir_u)
    
    def get_fx(self,kite,tether,kcu):
        wvel = self.uf/kappa*ca.log(self.r[2]/z0)
        vw = ca.vertcat(wvel*ca.cos(self.wdir),wvel*ca.sin(self.wdir),0)
        self.va = vw - self.v 

        # ez_kite = self.u/ca.norm_2(self.u)
        # ey_kite = ca.cross(ez_kite, - self.x[3:6] )/ca.norm_2(ca.cross(ez_kite, - self.x[3:6] ))
        # va_proj = project_onto_plane(va, ey_kite)           # Projected apparent wind velocity onto kite y axis
        # aoa = 90-calculate_angle_2vec(ez_kite,va_proj)*180/np.pi             # Angle of attack
                # Currently neglecting radial velocity of kite.
        v_reelout = self.reelout_speed
        tension_ground = self.Ftg
        a_kcu = self.a_kcu
        tether_length = self.tether_length
        elevation_0 = self.elevation_0
        azimuth_0 = self.azimuth_0

        l_unstrained = tether_length/tether.n_elements
        m_s = np.pi*tether.diameter**2/4 * l_unstrained * tether.density

        n_elements = tether.n_elements
        if kite.KCU == True:
            n_elements += 1

        tensions = ca.SX.zeros((n_elements, 3))
        tensions[0, 0] = ca.cos(elevation_0)*ca.cos(azimuth_0)*tension_ground
        tensions[0, 1] = ca.cos(elevation_0)*ca.sin(azimuth_0)*tension_ground
        tensions[0, 2] = ca.sin(elevation_0)*tension_ground

        positions =  ca.SX.zeros((n_elements+1, 3))
        if tether.elastic:
            l_s = (tension_ground/(tether.EA)+1)*l_unstrained
        else:
            l_s = l_unstrained

        positions[1, 0] = ca.cos(elevation_0)*ca.cos(azimuth_0)*l_s
        positions[1, 1] = ca.cos(elevation_0)*ca.sin(azimuth_0)*l_s
        positions[1, 2] = ca.sin(elevation_0)*l_s

        velocities  = ca.SX.zeros((n_elements+1, 3))
        accelerations  = ca.SX.zeros((n_elements+1, 3))
        non_conservative_forces  = ca.SX.zeros((n_elements+1, 3))

        stretched_tether_length = l_s  # Stretched
        for j in range(n_elements):  # Iterate over point masses.
            last_element = j == n_elements - 1
            kcu_element = kite.KCU and j == n_elements - 2


            if not kite.KCU:
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
            fgj = ca.SX.zeros((3))
            fgj[2] = -point_mass*g
            
            next_tension = tensions[j, :].T - fgj  
            if kcu_element:
                next_tension += kcu.mass*a_kcu
            if not last_element:
                tensions[j+1, :] = next_tension

            else:
                aerodynamic_force = next_tension
                non_conservative_forces[j, :] = aerodynamic_force

            # Derive position of next point mass from former tension
            if kcu_element:
                positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/ca.norm_2(tensions[j+1, :]) * kite.distance_kcu_kite
                
            elif not last_element:
                if tether.elastic:
                    l_s = (ca.norm_2(tensions[j+1, :])/tether.EA+1)*l_unstrained
                else:
                    l_s = l_unstrained
                stretched_tether_length += l_s
                positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/ca.norm_2(tensions[j+1, :]) * l_s

        tension_last_element = tensions[-1,:].T
        dir_D = self.va/ca.norm_2(self.va)
        dir_L = tension_last_element/ca.norm_2(tension_last_element) - ca.dot(tension_last_element/ca.norm_2(tension_last_element),dir_D)*dir_D
        dir_S = ca.cross(dir_L,dir_D) 

        L = self.CL*0.5*rho*kite.area*ca.norm_2(self.va)**2*dir_L
        D = self.CD*0.5*rho*kite.area*ca.norm_2(self.va)**2*dir_D
        S = self.CS*0.5*rho*kite.area*ca.norm_2(self.va)**2*dir_S

        Fg = ca.vertcat(0, 0, -kite.mass*g)
        rp = self.v
        vp = (-tension_last_element+L+D+S+Fg)/kite.mass

        return ca.vertcat(rp,vp,0,0,0,0,0,v_reelout,0,0)
    
    def get_fx_jac(self):
        return ca.jacobian(self.fx,self.x)

    def get_fx_jac_fun(self):
        return ca.Function('calc_Fx', [self.x,self.u],[self.get_fx_jac()])
    
    def propagate(self,x,u):
        return np.array(self.intg(x0=x,p=u)['xf'].T)

class ObservationModel:

    def __init__(self,x,u,opt_measurements,kite,tether,kcu):
        self.x = x
        self.u = u
        self.opt_measurements = opt_measurements
        self.hx = self.get_hx(kite,tether,kcu)
    
    def get_hx(self,kite,tether,kcu):
        uf = self.x[6]
        wdir = self.x[7]
        wvel = uf/kappa*ca.log(self.x[2]/z0)
        vw = ca.vertcat(wvel*ca.cos(wdir),wvel*ca.sin(wdir),0)
        va = vw - self.x[3:6] 


        # ez_kite = self.u/ca.norm_2(self.u)
        # ey_kite = ca.cross(ez_kite, - self.x[3:6] )/ca.norm_2(ca.cross(ez_kite, - self.x[3:6] ))
        # va_proj = project_onto_plane(va, ey_kite)           # Projected apparent wind velocity onto kite y axis
        # aoa = 90-calculate_angle_2vec(ez_kite,va_proj)*180/np.pi             # Angle of attack
                # Currently neglecting radial velocity of kite.
        a_kcu = self.u[2:5]
        tension_ground = self.u[1]

        uf = self.x[6]
        wdir = self.x[7]

        tether_length = self.x[11]
        elevation_0 = self.x[12]
        azimuth_0 = self.x[13]

        l_unstrained = tether_length/tether.n_elements
        m_s = np.pi*tether.diameter**2/4 * l_unstrained * tether.density

        n_elements = tether.n_elements
        if kite.KCU == True:
            n_elements += 1

        tensions = ca.SX.zeros((n_elements, 3))
        tensions[0, 0] = ca.cos(elevation_0)*ca.cos(azimuth_0)*tension_ground
        tensions[0, 1] = ca.cos(elevation_0)*ca.sin(azimuth_0)*tension_ground
        tensions[0, 2] = ca.sin(elevation_0)*tension_ground

        positions =  ca.SX.zeros((n_elements+1, 3))
        if tether.elastic:
            l_s = (tension_ground/(tether.EA)+1)*l_unstrained
        else:
            l_s = l_unstrained

        positions[1, 0] = ca.cos(elevation_0)*ca.cos(azimuth_0)*l_s
        positions[1, 1] = ca.cos(elevation_0)*ca.sin(azimuth_0)*l_s
        positions[1, 2] = ca.sin(elevation_0)*l_s

        velocities  = ca.SX.zeros((n_elements+1, 3))
        accelerations  = ca.SX.zeros((n_elements+1, 3))
        non_conservative_forces  = ca.SX.zeros((n_elements+1, 3))

        stretched_tether_length = l_s  # Stretched
        for j in range(n_elements):  # Iterate over point masses.
            last_element = j == n_elements - 1
            kcu_element = kite.KCU and j == n_elements - 2


            if not kite.KCU:
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
            fgj = ca.SX.zeros((3))
            fgj[2] = -point_mass*g
            
            next_tension = tensions[j, :].T - fgj  
            if kcu_element:
                next_tension += kcu.mass*a_kcu
            if not last_element:
                tensions[j+1, :] = next_tension

            else:
                aerodynamic_force = next_tension
                non_conservative_forces[j, :] = aerodynamic_force

            # Derive position of next point mass from former tension
            if kcu_element:
                positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/ca.norm_2(tensions[j+1, :]) * kite.distance_kcu_kite
                
            elif not last_element:
                if tether.elastic:
                    l_s = (ca.norm_2(tensions[j+1, :])/tether.EA+1)*l_unstrained
                else:
                    l_s = l_unstrained
                stretched_tether_length += l_s
                positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/ca.norm_2(tensions[j+1, :]) * l_s
        
        r_thether_model = positions[-1, :].T

        h = ca.SX()
        h = ca.vertcat(self.x[0:3])
        h = ca.vertcat(h,self.x[3:6])
        for key in self.opt_measurements:
            if key == 'kite_acc':
                h = ca.vertcat(h,self.fx[3:6])
            elif key == 'ground_wvel':
                h = ca.vertcat(h,self.x[6])
            elif key == 'apparent_windspeed':
                h = ca.vertcat(h,ca.norm_2(va))
            elif key == 'tether_length':
                h = ca.vertcat(h,ca.norm_2(self.x[0:3])-kite.distance_kcu_kite-self.x[11])
            # elif key == 'aoa':
            #     h = ca.vertcat(h,aoa+self.x[12])
        h = ca.vertcat(h,self.x[11])
        h = ca.vertcat(h,self.x[12])
        h = ca.vertcat(h,self.x[13])
        h = ca.vertcat(h,(self.x[0:3]-r_thether_model)**2)

        return h

    def get_hx_jac(self):
        return ca.jacobian(self.hx,self.x)
    
    def get_hx_jac_fun(self):
        return ca.Function('calc_Hx', [self.x,self.u],[self.get_hx_jac()])
    
    def get_hx_fun(self):
        return ca.Function('calc_hx', [self.x,self.u],[self.hx])
    
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