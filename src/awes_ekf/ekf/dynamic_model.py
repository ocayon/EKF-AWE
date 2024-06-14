import numpy as np
import casadi as ca
from awes_ekf.setup.settings import kappa, z0, rho, g

class DynamicModel:
    def __init__(self,kite,tether,kcu,simConfig):
        
        self.simConfig = simConfig

        self.r = ca.SX.sym('r',3)    # Kite position
        self.r_tether_model = ca.SX.sym('r_tether_model',3) # Tether attachment point
        self.v = ca.SX.sym('v',3)    # Kite velocity
        self.CL = ca.SX.sym('CL')    # Lift coefficient
        self.CD = ca.SX.sym('CD')    # Drag coefficient
        self.CS = ca.SX.sym('CS')    # Side force coefficient
        self.Ftg = ca.SX.sym('ground_tether_force')  # Tether force
        self.tether_length = ca.SX.sym('tether_length') # Tether length
        self.reelout_speed = ca.SX.sym('reelout_speed') # Tether reelout speed
        self.elevation_0 = ca.SX.sym('elevation_first_tether_element') # Elevation from ground to first tether element
        self.azimuth_0 = ca.SX.sym('azimuth_first_tether_element')   # Azimuth from ground to first tether element
        self.tether_offset = ca.SX.sym('tether_offset') # Tether offset
        self.yaw = ca.SX.sym('yaw') # Bias angle of attack
        self.us = ca.SX.sym('us')    # Steering input
        self.k_yaw_rate = ca.SX.sym('k_yaw_rate') # Yaw rate constant
        


        self.get_wind_velocity(simConfig.log_profile)
        self.va = self.vw - self.v

        self.u = self.get_input(kcu,kite)
        self.x = self.get_state(kite)
        if simConfig.model_yaw:
            self.x = ca.vertcat(self.x, self.yaw, self.k_yaw_rate)
        if simConfig.tether_offset:
            self.x = ca.vertcat(self.x, self.tether_offset)
        self.x0 = kite.x0  
        self.fx = self.get_fx(kite,tether,kcu)
        
        
        if simConfig.model_yaw:
            yaw_rate = self.k_yaw_rate*self.us*ca.norm_2(self.va)
            self.fx = ca.vertcat(self.fx,yaw_rate,0)
        if simConfig.tether_offset:
            self.fx = ca.vertcat(self.fx,0)
        
          # Kite position
    
        self.calc_fx = ca.Function('calc_fx', [self.x,self.u,self.x0],[self.fx])

    def get_state(self,kite):    
        
        return kite.get_state()
    
    def get_wind_velocity(self, log_profile):
        if log_profile is True:
            self.uf = ca.SX.sym('uf')    # Friction velocity
            self.wdir = ca.SX.sym('wdir')# Ground wind direction
            self.vwz = ca.SX.sym('vwz')  # Vertical wind velocity
            self.wvel = self.uf/kappa*ca.log(self.r[2]/z0)
            self.vw = ca.vertcat(self.wvel*ca.cos(self.wdir),self.wvel*ca.sin(self.wdir),self.vwz)
            self.vw_state = ca.vertcat(self.uf,self.wdir,self.vwz)
        else:
            self.vw = ca.SX.sym('vw',3)  # Wind velocity
            self.vw_state = self.vw
    
    def get_input(self,kcu,kite):
        return kite.get_input()
        
        
    
    def get_fx(self,kite,tether,kcu):
        
        v_reelout = self.u[0]

        fx_kite = kite.get_fx(tether)
        rprime = fx_kite[0:3]
        vprime = fx_kite[3:6]

        return ca.vertcat(rprime,vprime,0,0,0,0,0,0,v_reelout,0,0)
    
    def get_fx_fun(self,kite,tether,kcu):
        return ca.Function('calc_Fx', [self.x,self.u,self.x0],[self.get_fx(kite,tether,kcu)])

    def get_fx_jac(self,kite,tether,kcu):
  
        return ca.simplify(ca.jacobian(self.fx,self.x))

    def get_fx_jac_fun(self,kite,tether,kcu):
        return ca.Function('calc_Fx', [self.x,self.u,self.x0],[self.get_fx_jac(kite,tether,kcu)])
    
    def propagate(self,x,u, kite,tether,kcu,ts):
        
        
        self.fx = self.calc_fx(self.x,self.u,x)
        
        # Define ODE system
        dae = {'x': self.x, 'p': self.u, 'ode': self.fx}                       # Define ODE system
        integrator = ca.integrator('intg', 'cvodes', dae, 0,ts)    # Define integrator
        
        return np.array(integrator(x0=x,p=u)['xf'].T)