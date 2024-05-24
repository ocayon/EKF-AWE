import numpy as np
import casadi as ca
from awes_ekf.setup.settings import kappa, z0

class ObservationModel:

    def __init__(self,x,u,model_specs,kite,tether,kcu):
        self.x = x
        self.u = u
        self.x0 = ca.SX.sym('x0',self.x.shape[0])    # Kite position
        self.model_specs = model_specs
    
    def get_hx(self,kite,tether,kcu):
        
        elevation_0 = self.x[13]
        azimuth_0 = self.x[14]
        tether_length = self.x[12]
        r_kite = self.x0[0:3]
        v_kite = self.x0[3:6]
        tension_ground = self.u[1]

        if kcu.data_available:
            a_kcu = self.u[2:5]
            v_kcu = self.u[5:8]
            a_kite = None
        else:
            a_kite = self.u[2:5]
            a_kcu = None
            v_kcu = None
        if self.model_specs.log_profile:
            wvel = self.x0[6]/kappa*np.log(self.x0[2]/z0)
            wdir = self.x0[7]
            vw = np.array([wvel*np.cos(wdir),wvel*np.sin(wdir),self.x0[8]])
        else:
            vw = self.x0[6:9]
        
        r_thether_model, tension_last_element = tether.calculate_tether_shape_symbolic(elevation_0, azimuth_0, tether_length,
                                         tension_ground, r_kite, v_kite, vw, kite, kcu,tether,  
                                        a_kite = a_kite, a_kcu = a_kcu, v_kcu = v_kcu)

        if self.model_specs.log_profile:
            wvel = self.x[6]/kappa*np.log(self.x[2]/z0)
            wdir = self.x[7]
            vw = np.array([wvel*np.cos(wdir),wvel*np.sin(wdir),self.x[8]])
        else:
            vw = self.x[6:9]
            
        va = vw -self.x[3:6]
        
        h = ca.SX()
        h = ca.vertcat(self.x[0:3])
        h = ca.vertcat(h,self.x[3:6])
        if self.model_specs.tether_offset:
            h = ca.vertcat(h,self.x[12]-self.x[-1])
        else:
            h = ca.vertcat(h,self.x[12])
        h = ca.vertcat(h,self.x[13])
        h = ca.vertcat(h,self.x[14])
        h = ca.vertcat(h,(self.x[0:3]-r_thether_model)**2)
        if self.model_specs.model_yaw:
            h = ca.vertcat(h,self.x[15])
        if self.model_specs.enforce_z_wind:
            h = ca.vertcat(h,self.x[8])
        
            
        for key in self.model_specs.opt_measurements:
            if key == 'apparent_windspeed':
                h = ca.vertcat(h,ca.norm_2(va))
        
        return h

    def get_hx_jac(self,kite,tether,kcu):
        hx = self.get_hx(kite,tether,kcu)
        return ca.simplify(ca.jacobian(hx,self.x))
    
    def get_hx_jac_fun(self,kite,tether,kcu):
        return ca.Function('calc_Hx', [self.x,self.u,self.x0],[self.get_hx_jac(kite,tether,kcu)])
    
    def get_hx_fun(self,kite,tether,kcu):
        return ca.Function('calc_hx', [self.x,self.u,self.x0],[self.get_hx(kite,tether,kcu)])
    