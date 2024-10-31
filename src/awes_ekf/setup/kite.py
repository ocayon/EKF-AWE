from abc import ABC, abstractmethod
import casadi as ca
from awes_ekf.setup.settings import kappa, z0, rho, g
import numpy as np
from dataclasses import dataclass
from awes_ekf.utils import calculate_log_wind_velocity, calculate_euler_from_reference_frame, project_onto_plane

class Kite(ABC):
    def __init__(self, **kwargs):
        self.mass = kwargs.get("mass")
        self.area = kwargs.get("area")
        self.span = kwargs.get("span")
        self.model_name = kwargs.get("model_name")
        self.thrust = kwargs.get("thrust", False)

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_input(self):
        pass

    @abstractmethod
    def get_fx(self, tether):
        pass

    @abstractmethod
    def get_fx_fun(self, kite, tether, kcu):
        pass
    @abstractmethod
    def propagate(self, x, u, kite, tether, kcu, ts):
        pass


class PointMassEKF(Kite):

    def __init__(self, simConfig, **kwargs):

        self.simConfig = simConfig

        self.r = ca.SX.sym("r", 3)  # Kite position
        self.r_tether_model = ca.SX.sym("r_tether_model", 3)  # Tether attachment point
        self.v = ca.SX.sym("v", 3)  # Kite velocity
        self.CL = ca.SX.sym("CL")  # Lift coefficient
        self.CD = ca.SX.sym("CD")  # Drag coefficient
        self.CS = ca.SX.sym("CS")  # Side force coefficient
        self.Ftg = ca.SX.sym("ground_tether_force")  # Tether force
        self.tether_length = ca.SX.sym("tether_length")  # Tether length
        self.reelout_speed = ca.SX.sym("reelout_speed")  # Tether reelout speed
        self.elevation_0 = ca.SX.sym(
            "elevation_first_tether_element"
        )  # Elevation from ground to first tether element
        self.azimuth_0 = ca.SX.sym(
            "azimuth_first_tether_element"
        )  # Azimuth from ground to first tether element
        self.tether_length_offset = ca.SX.sym("tether_length_offset")  # Tether offset
        self.tether_elevation_offset = ca.SX.sym("tether_elevation_offset") # Tether offset
        self.tether_azimuth_offset = ca.SX.sym("tether_azimuth_offset") # Tether offset
        self.yaw = ca.SX.sym("yaw")  # Bias angle of attack
        self.us = ca.SX.sym("us")  # Steering input
        self.k_yaw_rate = ca.SX.sym("k_yaw_rate")  # Yaw rate constant

        self.get_wind_velocity()
        self.va = self.vw - self.v

        self.u = self.get_input()
        self.x = self.get_state()
        self.x0 = self.create_previous_state_vector()
        
        super().__init__(**kwargs)

    def get_state(self):
        self.x = ca.vertcat(
            self.r,
            self.v,
            self.vw_state,
            self.CL,
            self.CD,
            self.CS,
            self.tether_length,
            self.elevation_0,
            self.azimuth_0,
        )
        if self.simConfig.model_yaw:
            self.x = ca.vertcat(self.x, self.yaw, self.k_yaw_rate)
        if self.simConfig.obsData.tether_length:
            self.x = ca.vertcat(self.x, self.tether_length_offset)
        if self.simConfig.obsData.tether_elevation:
            self.x = ca.vertcat(self.x, self.tether_elevation_offset)
        if self.simConfig.obsData.tether_azimuth:
            self.x = ca.vertcat(self.x, self.tether_azimuth_offset)

        return self.x

    def get_wind_velocity(self):
        if self.simConfig.log_profile is True:
            self.uf = ca.SX.sym("uf")  # Friction velocity
            self.wdir = ca.SX.sym("wdir")  # Ground wind direction
            self.vwz = ca.SX.sym("vw_2")  # Vertical wind velocity
            self.vw = calculate_log_wind_velocity(self.uf, self.wdir, self.vwz, self.r[2])
            self.vw_state = ca.vertcat(self.uf, self.wdir, self.vwz)
        else:
            self.vw = ca.SX.sym("vw", 3)  # Wind velocity
            self.vw_state = self.vw

    def get_input(self):
        input = ca.vertcat(self.reelout_speed, self.Ftg)
        if self.simConfig.obsData.kite_acceleration:
            self.a_kite = ca.SX.sym("a_kite", 3)  # Kite acceleration
            input = ca.vertcat(input, self.a_kite)
        if self.simConfig.obsData.kcu_acceleration:
            self.a_kcu = ca.SX.sym("a_kcu", 3)  # KCU acceleration
            input = ca.vertcat(input, self.a_kcu)
        if self.simConfig.obsData.kcu_velocity:
            self.v_kcu = ca.SX.sym("v_kcu", 3)  # KCU velocity
            input = ca.vertcat(input, self.v_kcu)
        if self.simConfig.obsData.kite_thrust_force:
            self.thrust = ca.SX.sym("thrust", 3)  # Thrust force
            input = ca.vertcat(input, self.thrust)
        if self.simConfig.model_yaw:
            input = ca.vertcat(input, self.us)

        return input

    def get_fx(self, tether):

        elevation_0 = self.elevation_0
        azimuth_0 = self.azimuth_0
        tether_length = self.tether_length
        r_kite = self.x0[0:3]
        v_kite = self.x0[3:6]
        tension_ground = self.Ftg

        if self.simConfig.log_profile:
            wvel = self.x0[6] / kappa * np.log(self.x0[2] / z0)
            wdir = self.x0[7]
            vw = np.array([wvel * np.cos(wdir), wvel * np.sin(wdir), self.x0[8]])
        else:
            vw = self.x0[6:9]

        args = (
            elevation_0,
            azimuth_0,
            tether_length,
            tension_ground,
            r_kite,
            v_kite,
            vw,
        )
        if self.simConfig.obsData.kite_acceleration:
            args += (self.a_kite,)
        if self.simConfig.obsData.kcu_acceleration:
            args += (self.a_kcu,)
        if self.simConfig.obsData.kcu_velocity:
            args += (self.v_kcu,)

        tether_force = tether.tether_force_kite(*args)

        dir_D = self.va / ca.norm_2(self.va)
        dir_L = (
            tether_force / ca.norm_2(tether_force)
            - ca.dot(tether_force / ca.norm_2(tether_force), dir_D)
            * dir_D
        )
        dir_S = ca.cross(dir_L, dir_D)

        L = self.CL * 0.5 * rho * self.area * ca.norm_2(self.va) ** 2 * dir_L
        D = self.CD * 0.5 * rho * self.area * ca.norm_2(self.va) ** 2 * dir_D
        S = self.CS * 0.5 * rho * self.area * ca.norm_2(self.va) ** 2 * dir_S

        Fg = ca.vertcat(0, 0, -self.mass * g)
        rp = self.v
        if self.simConfig.obsData.kite_thrust_force:
            vp = (-tether_force + L + D + S + Fg + self.thrust) / self.mass
        else:
            vp = (-tether_force + L + D + S + Fg) / self.mass

        r_norm = ca.norm_2(r_kite)
        r_normal = r_kite/r_norm
        vert = ca.vertcat(0,0,1)
        vert_norm = project_onto_plane(vert, r_normal)
        horz_norm = ca.cross(vert_norm, r_normal)
        v_kite_tan_horz = ca.dot(v_kite, horz_norm)
        v_kite_tan_vert = ca.dot(v_kite, vert_norm)
        elevation_rate = v_kite_tan_vert / r_norm
        azimuth_rate = v_kite_tan_horz / (r_norm * ca.cos(elevation_0))
        

        fx = ca.vertcat(rp, vp, 0, 0, 0, 0, 0, 0, self.reelout_speed, elevation_rate, azimuth_rate)
        if self.simConfig.model_yaw:
            yaw_rate = self.k_yaw_rate * self.us * ca.norm_2(self.va)
            fx = ca.vertcat(fx, yaw_rate, 0)
        if self.simConfig.obsData.tether_length:
            fx = ca.vertcat(fx, 0)
        if self.simConfig.obsData.tether_elevation:
            fx = ca.vertcat(fx, 0)
        if self.simConfig.obsData.tether_azimuth:
            fx = ca.vertcat(fx, 0)

        return fx

    def get_fx_fun(self, tether):
        return ca.Function("calc_Fx", [self.x, self.u, self.x0], [self.get_fx(tether)])

    def get_fx_jac(self, kite, tether, kcu):

        return ca.simplify(ca.jacobian(self.fx, self.x))

    def get_fx_jac_fun(self, kite, tether, kcu):
        return ca.Function(
            "calc_Fx", [self.x, self.u, self.x0], [self.get_fx_jac(kite, tether, kcu)]
        )

    def propagate(self, x, u, ts):

        fx = self.calc_fx(self.x, self.u, x)

        # Define ODE system
        dae = {"x": self.x, "p": self.u, "ode": fx}  # Define ODE system
        integrator = ca.integrator("intg", "cvodes", dae, 0, ts)  # Define integrator

        return np.array(integrator(x0=x, p=u)["xf"].T)
    
    def create_previous_state_vector(self):
        # Extract the names of the symbolic variables
        names = [str(self.x[i]) for i in range(self.x.size1())]
        # Create new symbolic variables with names ending in _0
        x0_elements = [ca.SX.sym(f"{name}_0") for name in names]
        x0 = ca.vertcat(*x0_elements)
        return x0
    @property
    def state_index_map(self):
        # Split the CasADi matrix into individual symbolic variables
        state_variables = ca.vertsplit(self.x)

        # Create a dictionary to map variable names to their indices
        variable_index_map = {var.name(): i for i, var in enumerate(state_variables)}
        return variable_index_map
    
    @property
    def input_index_map(self):
        # Split the CasADi matrix into individual symbolic variables
        input_variables = ca.vertsplit(self.u)

        # Create a dictionary to map variable names to their indices
        variable_index_map = {var.name(): i for i, var in enumerate(input_variables)}
        return variable_index_map

class PointMass(Kite):

    def __init__(self, simConfig, **kwargs):

        self.simConfig = simConfig

        
        self.r = ca.SX.sym("r", 3)  # Kite position
        self.v = ca.SX.sym("v", 3)  # Kite velocity
        self.yaw = ca.SX.sym("yaw")  # Bias angle of attack
        self.us = ca.SX.sym("us")  # Steering input
        self.wind_velocity = ca.SX.sym("wind_velocity", 3)  # Wind velocity
        self.va = self.wind_velocity - self.v  # Apparent wind velocity
        self.up = ca.SX.sym("up")  # Pitch input
        self.tether_force = ca.SX.sym("tether_force", 3)  # Tether force

        self.u = self.get_input()
        self.x = self.get_state()
        self.x0 = ca.SX.sym("x0", self.x.shape[0])  # Initial state vector

        super().__init__(**kwargs)

    def get_state(self):
        self.x = ca.vertcat(
            self.r,
            self.v,
            self.yaw,
        )

        return self.x

    def get_input(self):
        input = ca.vertcat(self.wind_velocity, self.us, self.up, self.tether_force)

        return input

    def get_fx(self):


        tether_force = self.tether_force

        dir_D = self.va / ca.norm_2(self.va)
        dir_L = (
            -tether_force / ca.norm_2(tether_force)
            - ca.dot(-tether_force / ca.norm_2(tether_force), dir_D)
            * dir_D
        )
        dir_S = ca.cross(dir_L, dir_D)

        CL = self.calculate_CL(self.up, self.us)
        CD = self.calculate_CD(self.up, self.us)
        CS = self.calculate_CS(self.up, self.us)

        L = CL * 0.5 * rho * self.area * ca.norm_2(self.va) ** 2 * dir_L
        D = CD * 0.5 * rho * self.area * ca.norm_2(self.va) ** 2 * dir_D
        S = CS * 0.5 * rho * self.area * ca.norm_2(self.va) ** 2 * dir_S

        Fg = ca.vertcat(0, 0, -self.mass * g)
        rp = self.v

        vp = (tether_force + L + D + S + Fg) / self.mass

        yaw_rate = self.calculate_yaw_rate(self.up, self.us)
        fx = ca.vertcat(rp, vp, yaw_rate)

        return fx

    def get_fx_fun(self):

        return ca.Function("calc_Fx", [self.x, self.u], [self.get_fx()])
    
    def propagate(self, kite_input, ts):

        x = np.hstack((kite_input.kite_position, kite_input.kite_velocity, np.array([kite_input.kite_yaw]))).reshape(-1)
        u = np.hstack((kite_input.wind_velocity, np.array([kite_input.us]), np.array([kite_input.up]), kite_input.tether_force)).reshape(-1)
        calc_fx = self.get_fx_fun()

        # Define ODE system
        dae = {"x": self.x, "p": self.u, "ode": calc_fx(self.x,self.u)}  # Define ODE system

        integrator = ca.integrator("intg", "cvodes", dae, 0, ts)  # Define integrator

        return np.array(integrator(x0=x, p=u)["xf"].T)
    
    def calculate_CL(self, up, us):
        #TODO: Implement CL calculation
        return 0.9
    
    def calculate_CD(self, up, us):
        #TODO: Implement CD calculation
        return 0.1
    
    def calculate_CS(self, up, us):
        #TODO: Implement CS calculation  
        return 0.1
    
    def calculate_yaw_rate(self, up, us):
        #TODO: Implement yaw rate calculation
        return 0.1
    
    def calculate_tether_force(self, KiteInput):
        #TODO: Implement tether force calculation, linking to tether model
        return np.array([0,0,0])


@dataclass
class KiteInput:
    kite_position: np.array
    kite_velocity: np.array
    kite_yaw: float
    wind_velocity: np.array
    us: float
    up: float
    tether_force: np.array
    tether_length: float