
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
        self.uf = ca.SX.sym('uf')    # Ground wind velocity
        self.wdir = ca.SX.sym('wdir')# Ground wind direction
        self.CL = ca.SX.sym('CL')    # Lift coefficient
        self.CD = ca.SX.sym('CD')    # Drag coefficient
        self.CS = ca.SX.sym('CS')    # Side force coefficient
        self.Ftg = ca.SX.sym('ground_tether_force')  # Tether force
        self.tether_length = ca.SX.sym('tether_length') # Tether length
        self.relout_speed = ca.SX.sym('reelout_speed') # Tether reelout speed
        self.elevation_0 = ca.SX.sym('elevation_first_tether_element') # Elevation from ground to first tether element
        self.azimuth_0 = ca.SX.sym('azimuth_first_tether_element')   # Azimuth from ground to first tether element
        
        self.x = self.get_state()
        self.u = self.get_input()
        self.fx = self.get_fx(kite)
        
        # Define ODE system
        dae = {'x': self.x, 'p': self.u, 'ode': self.fx}                       # Define ODE system
        self.intg = ca.integrator('intg', 'cvodes', dae, 0,ts)    # Define integrator

    def get_state(self,kite,tether,kcu):    

        return ca.vertcat(self.r,self.v,self.uf,self.wdir,self.CL,self.CD,self.CS,self.tether_length, self.elevation_0, self.azimuth_0, self.sol_least_squares)
    
    def get_input(self):
        return ca.vertcat(self.reloout_speed,self.Ft)

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
        
        return ca.vertcat(rp,vp,0,0,0,0,0,0,0,0)
    
    def get_fx_jac(self):
        return ca.jacobian(self.fx,self.x)

    def get_fx_jac_fun(self):
        return ca.Function('calc_Fx', [self.x,self.u],[self.get_fx_jac()])
    
    def propagate(self,x,u, kite, tether, kcu):
        
        r_kite = x[0:3]
        v_kite = x[3:6]
        uf = x[6]
        wdir = x[7]
        CL = x[8]
        CD = x[9]
        CS = x[10]
        tether_length = x[11]
        elevation_0 = x[12]
        azimuth_0 = x[13]
        sol_least_squares = x[14:]
        tension_ground = u[0]
        reelout_speed = u[1]
        a_kite = u[2:5]

        l_unstrained = tether_length/tether.n_elements
        m_s = np.pi*self.diameter**2/4 * l_unstrained * self.density

        n_elements = tether.n_elements
        if kite.KCU == True:
            n_elements += 1
        
        wvel = uf/kappa*np.log(r_kite[2]/z0)


        vtau_kite = project_onto_plane(v_kite,r_kite/np.linalg.norm(r_kite)) # Velocity projected onto the tangent plane
        omega_tether = np.cross(r_kite,vtau_kite)/(np.linalg.norm(r_kite)**2) # Tether angular velocity, with respect to the tether attachment point

        if a_kite is not None:
            # Find instantaneuous center of rotation and omega of the kite
            at = np.dot(a_kite,np.array(v_kite)/np.linalg.norm(v_kite))*np.array(v_kite)/np.linalg.norm(v_kite) # Tangential acceleration
            if np.linalg.norm(a_kite)<1e-3:
                omega_kite = omega_tether # If the kite is not accelerating, the ICR is at infinity, and the kite is rotating at the same rate as the tether
            else:
                omega_kite = np.cross(a_kite-at,v_kite)/(np.linalg.norm(v_kite)**2) # Angular velocity of the kite

            ICR = np.cross(v_kite,omega_kite)/(np.linalg.norm(omega_kite)**2) # Instantaneous center of rotation     
            alpha = np.cross(at,ICR)/np.linalg.norm(ICR)**2 # Angular acceleration of the kite

        tensions = np.zeros((n_elements, 3))
        tensions[0, 0] = np.cos(elevation_0)*np.cos(azimuth_0)*tension_ground
        tensions[0, 1] = np.cos(elevation_0)*np.sin(azimuth_0)*tension_ground
        tensions[0, 2] = np.sin(elevation_0)*tension_ground

        positions = np.zeros((n_elements+1, 3))
        if self.elastic:
            l_s = (tension_ground/(self.EA)+1)*l_unstrained
        else:
            l_s = l_unstrained

        positions[1, 0] = np.cos(elevation_0)*np.cos(azimuth_0)*l_s
        positions[1, 1] = np.cos(elevation_0)*np.sin(azimuth_0)*l_s
        positions[1, 2] = np.sin(elevation_0)*l_s

        velocities = np.zeros((n_elements+1, 3))
        accelerations = np.zeros((n_elements+1, 3))
        non_conservative_forces = np.zeros((n_elements, 3))

        stretched_tether_length = l_s  # Stretched
        for j in range(n_elements):  # Iterate over point masses.
            last_element = j == n_elements - 1
            kcu_element = kite.KCU and j == n_elements - 2

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
                if a_kcu is not None:
                    aj = np.array(a_kcu)
                    vj = np.array(v_kcu)
                else:
                    v_kcu = v_kite + np.cross(omega_kite,positions[j+1, :]-r_kite)
                    vj = v_kcu
                    velocities[j+1, :] = vj
                    a_kcu = a_kite+ np.cross(alpha,positions[j+1, :]-r_kite) +np.cross(omega_kite,np.cross(omega_kite,positions[j+1, :]-r_kite))
                    aj = a_kcu
                    accelerations[j+1, :] = aj

                ej = (r_kite-np.array(positions[j+1, :]))/np.linalg.norm(r_kite-np.array(positions[j+1, :]))

            # Determine flow at point mass j.
            vaj = vj - vwj  # Apparent wind velocity
            
            vajp = np.dot(vaj, ej)*ej  # Parallel to tether element
            # TODO: check whether to use vajn
            vajn = vaj - vajp  # Perpendicular to tether element

            vaj_sq = np.linalg.norm(vaj)*vaj

            # Determina angle between  va and tether
            theta = calculate_angle_2vec(-vaj,ej)
            # vaj_sq = np.linalg.norm(vajn)*vajn
            CD_tether = self.cd*np.sin(theta)**3+self.cf
            # CL_tether = self.cd*np.sin(theta)**2*np.cos(theta)
            tether_drag_basis = rho*l_unstrained*self.diameter*CD_tether*vaj_sq
            
            # Determine drag at point mass j.
            if not kite.KCU:
                if tether.n_elements == 1:
                    dj = -.125*tether_drag_basis
                elif last_element:
                    dj = -.25*tether_drag_basis  # TODO: add bridle drag
                else:
                    dj = -.5*tether_drag_basis
            else:
                if last_element:
                    # dj = -0.25*rho*L_blines*d_bridle*vaj_sq*cd_t # Bridle lines drag
                    dj = 0
                    
                elif tether.n_elements == 1:
                    dj = -.25*tether_drag_basis
                    dp= -.5*rho*np.linalg.norm(vajp)*vajp*kcu.cdp*kcu.Ap  # Adding kcu drag perpendicular to kcu
                    dt= -.5*rho*np.linalg.norm(vajn)*vajn*kcu.cdt*kcu.At  # Adding kcu drag parallel to kcu
                    dj += dp+dt
                    cd_kcu = (np.linalg.norm(dp+dt))/(0.5*rho*kite.area*np.linalg.norm(vaj)**2)
                elif kcu_element:
                    dj = -.25*tether_drag_basis
                    theta = np.pi/2-theta
                    cd_kcu = kcu.cdt*np.sin(theta)**3+self.cf
                    cl_kcu = kcu.cdt*np.sin(theta)**2*np.cos(theta)
                    # D_turbine = 0.5*rho*np.linalg.norm(vaj)**2*np.pi*0.2**2*1
                    # dp= -.5*rho*np.linalg.norm(vajp)*vajp*kcu.cdp*kcu.Ap  # Adding kcu drag perpendicular to kcu
                    # dt= -.5*rho*np.linalg.norm(vajn)*vajn*kcu.cdt*kcu.At  # Adding kcu drag parallel to kcu
                    # th = -0.5*rho*vaj_sq*np.pi*0.2**2*0.4
                    # dj += dp+dt
                    # Approach described in Hoerner, taken from Paul Thedens dissertation
                    dir_D = -vaj/np.linalg.norm(vaj)
                    dir_L = ej - np.dot(ej,dir_D)*dir_D
                    L_kcu = 0.5*rho*np.linalg.norm(vaj)**2*kcu.Ap*cl_kcu
                    D_kcu = 0.5*rho*np.linalg.norm(vaj)**2*cd_kcu*kcu.Ap
                    dj += L_kcu*dir_L + D_kcu*dir_D #+ D_turbine*dir_D

                else:
                    dir_D = -vaj/np.linalg.norm(vaj)
                    dir_L = ej - np.dot(ej,dir_D)*dir_D
                    cd_t = self.cd*np.sin(theta)**3+self.cf
                    cl_t = self.cd*np.sin(theta)**2*np.cos(theta)
                    L_t = 0.5*rho*np.linalg.norm(vaj)**2*l_unstrained*self.diameter*cl_t
                    D_t = 0.5*rho*np.linalg.norm(vaj)**2*l_unstrained*self.diameter*cd_t
                    dj = L_t*dir_L + D_t*dir_D

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
                if self.elastic:
                    l_s = (np.linalg.norm(tensions[j+1, :])/self.EA+1)*l_unstrained
                else:
                    l_s = l_unstrained
                stretched_tether_length += l_s
                positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * l_s

        x_int = np.array(self.intg(x0=x,p=u)['xf'].T)
        x_least_squares = np.array( positions[-1, :] - r_kite )

        # Concatenate the results
        x_next = np.concatenate((x_int,x_least_squares))
        return x_next

class ObservationModel:

    def __init__(self,x,u,opt_measurements,KITE):
        self.x = x
        self.u = u
        self.opt_measurements = opt_measurements
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
        aoa = 90-calculate_angle_2vec_sym(ez_kite,va_proj)*180/np.pi             # Angle of attack
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
                h = ca.vertcat(h,ca.norm_2(self.x[0:3])-KITE.distance_kcu_kite-self.x[11])
            elif key == 'aoa':
                h = ca.vertcat(h,aoa+self.x[12])

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