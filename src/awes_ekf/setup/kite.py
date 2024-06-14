from abc import ABC, abstractmethod
import casadi as ca
from awes_ekf.setup.settings import kappa, z0, rho, g
import numpy as np
class Kite(ABC):
    def __init__(self, **kwargs):
        self.mass = kwargs.get('mass')
        self.area = kwargs.get('area')
        self.span = kwargs.get('span')
        self.model_name = kwargs.get('model_name')
        self.thrust = kwargs.get('thrust', False)
    
    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_input(self):
        pass

    @abstractmethod
    def get_fx(self,tether):
        pass

    @abstractmethod
    def get_fx_fun(self,kite,tether,kcu):
        pass

    @abstractmethod
    def get_fx_jac(self,kite,tether,kcu):
        pass

    @abstractmethod
    def get_fx_jac_fun(self,kite,tether,kcu):
        pass
        
    @abstractmethod
    def propagate(self,x,u, kite,tether,kcu,ts):
        pass



class PointMassEKF(Kite):

    def __init__(self,simConfig, **kwargs):
        
        
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

        self.u = self.get_input()
        self.x = self.get_state()
        if simConfig.model_yaw:
            self.x = ca.vertcat(self.x, self.yaw, self.k_yaw_rate)
        if simConfig.tether_offset:
            self.x = ca.vertcat(self.x, self.tether_offset)
        self.x0 = ca.SX.sym('x0',self.x.shape[0])  # Initial state vector
        

        super().__init__(**kwargs)

    def get_state(self):    
        
        return ca.vertcat(self.r,self.v,self.vw_state,self.CL,self.CD,self.CS,self.tether_length, self.elevation_0, self.azimuth_0)
    
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
    
    def get_input(self):
        input = ca.vertcat(self.reelout_speed,self.Ftg)
        if self.simConfig.obsData.kite_acc:
            self.a_kite =  ca.SX.sym('a_kite',3)  # Kite acceleration
            input = ca.vertcat(input,self.a_kite)
        if self.simConfig.obsData.kcu_acc:
            self.a_kcu =  ca.SX.sym('a_kcu',3)  # KCU acceleration
            input = ca.vertcat(input,self.a_kcu) 
        if self.simConfig.obsData.kcu_vel:
            self.v_kcu =  ca.SX.sym('v_kcu',3) # KCU velocity
            input = ca.vertcat(input,self.v_kcu)
        if self.simConfig.obsData.thrust_force:
            self.thrust = ca.SX.sym('thrust', 3)    # Thrust force
            input =  ca.vertcat(input,self.thrust)
        
        return input
        
        
    
    def get_fx(self,tether):
        
        
        elevation_0 = self.elevation_0
        azimuth_0 = self.azimuth_0
        tether_length = self.tether_length
        r_kite = self.x0[0:3]
        v_kite = self.x0[3:6]
        tension_ground = self.Ftg

        if self.simConfig.log_profile:
            wvel = self.x0[6]/kappa*np.log(self.x0[2]/z0)
            wdir = self.x0[7]
            vw = np.array([wvel*np.cos(wdir),wvel*np.sin(wdir),self.x0[8]])
        else:
            vw = self.x0[6:9]


        args = (elevation_0, azimuth_0, tether_length, tension_ground, r_kite, v_kite, vw)
        if self.simConfig.obsData.kite_acc:
            args += (self.a_kite,)
        if self.simConfig.obsData.kcu_acc:
            args += (self.a_kcu,)
        if self.simConfig.obsData.kcu_vel:
            args += (self.v_kcu,)


        tension_last_element = tether.tether_force_kite(*args)
        
        dir_D = self.va/ca.norm_2(self.va)
        dir_L = tension_last_element/ca.norm_2(tension_last_element) - ca.dot(tension_last_element/ca.norm_2(tension_last_element),dir_D)*dir_D
        dir_S = ca.cross(dir_L,dir_D) 

        L = self.CL*0.5*rho*self.area*ca.norm_2(self.va)**2*dir_L
        D = self.CD*0.5*rho*self.area*ca.norm_2(self.va)**2*dir_D
        S = self.CS*0.5*rho*self.area*ca.norm_2(self.va)**2*dir_S

        Fg = ca.vertcat(0, 0, -self.mass*g)
        rp = self.v
        if self.thrust:
            vp = (-tension_last_element+L+D+S+Fg+self.thrust)/self.mass
        else:
            vp = (-tension_last_element+L+D+S+Fg)/self.mass

        return ca.vertcat(rp,vp)
    
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
        


