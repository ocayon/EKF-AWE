from abc import ABC, abstractmethod
import casadi as ca
from awes_ekf.setup.settings import kappa, z0, rho, g
import numpy as np
from dataclasses import dataclass
from awes_ekf.utils import (
    calculate_log_wind_velocity,
    calculate_euler_from_reference_frame,
    project_onto_plane,
)


class Kite(ABC):
    def __init__(self, **kwargs):
        self.mass: float = float(kwargs.get("mass") or 0.0)
        self.area: float = float(kwargs.get("area") or 0.0)
        self.span: float = float(kwargs.get("span") or 0.0)
        self.model_name: str = str(kwargs.get("model_name") or "")
        self.thrust = kwargs.get("thrust", False)

    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_input(self):
        pass

    @abstractmethod
    def get_fx(self, *args, **kwargs):  # type: ignore[override]
        pass

    @abstractmethod
    def get_fx_fun(self, *args, **kwargs):  # type: ignore[override]
        pass

    @abstractmethod
    def propagate(self, *args, **kwargs):  # type: ignore[override]
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
        self.tether_elevation_offset = ca.SX.sym(
            "tether_elevation_offset"
        )  # Tether offset
        self.tether_azimuth_offset = ca.SX.sym("tether_azimuth_offset")  # Tether offset
        self.yaw = ca.SX.sym("yaw")  # Bias angle of attack
        self.us = ca.SX.sym("us")  # Steering input
        self.k_yaw_rate = ca.SX.sym("k_yaw_rate")  # Yaw rate constant
        self.k_cl_up = ca.SX.sym("k_cl_up")  # Effect of depower rate on CL
        self.k_cd_up = ca.SX.sym("k_cd_up")  # Effect of depower rate on CL
        self.k_phi_us = ca.SX.sym("k_phi_us")  # Steering-to-sideslip constant
        self.k_cl_us = ca.SX.sym("k_cl_us")  # Effect of steering magnitude on CL
        self.k_cd_us = ca.SX.sym("k_cd_us")  # Effect of squared steering on CD
        self.k_cl_us_odd = ca.SX.sym("k_cl_us_odd")  # Signed steering effect on CL
        self.k_cd_cl2 = ca.SX.sym("k_cd_cl2")  # Induced-drag factor: CL^2 on CD
        self.depower_input = ca.SX.sym("depower_input")
        self.delta_up = ca.SX.sym("delta_up")  # Change in depower setting

        self.get_wind_velocity()
        self.va = self.vw - self.v

        self.u = self.get_input()
        self.x = self.get_state()
        self.x0 = self.create_previous_state_vector()

        super().__init__(**kwargs)

    def get_state(self):
        # Concatenate state variables into the state vector
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

        # Maintain a list of state variable names
        symbols_vw = ca.symvar(self.vw_state)
        self.state_names = (
            [f"r_{i}" for i in range(3)]
            + [f"v_{i}" for i in range(3)]
            + [s.name() for s in symbols_vw]
            + ["CL", "CD", "CS", "tether_length", "elevation_0", "azimuth_0"]
        )

        # Handle optional variables based on configuration
        if self.simConfig.model_yaw:
            self.x = ca.vertcat(self.x, self.yaw, self.k_yaw_rate)
            self.state_names.extend(["yaw", "k_yaw_rate"])

        if self.simConfig.obsData.tether_length:
            self.x = ca.vertcat(self.x, self.tether_length_offset)
            self.state_names.append("tether_length_offset")

        if self.simConfig.obsData.tether_elevation:
            self.x = ca.vertcat(self.x, self.tether_elevation_offset)
            self.state_names.append("tether_elevation_offset")

        if self.simConfig.obsData.tether_azimuth:
            self.x = ca.vertcat(self.x, self.tether_azimuth_offset)
            self.state_names.append("tether_azimuth_offset")

        if self.simConfig.obsData.dynamic_depower:
            self.x = ca.vertcat(self.x, self.k_cl_up)
            self.state_names.append("k_cl_up")
            self.x = ca.vertcat(self.x, self.k_cd_up)
            self.state_names.append("k_cd_up")

        if self.simConfig.steering_dependent_cs:
            self.x = ca.vertcat(self.x, self.k_phi_us)
            self.state_names.append("k_phi_us")

        if self.simConfig.steering_dependent_clcd:
            self.x = ca.vertcat(self.x, self.k_cl_us)
            self.state_names.append("k_cl_us")
            self.x = ca.vertcat(self.x, self.k_cd_us)
            self.state_names.append("k_cd_us")

        if self.simConfig.steering_dependent_cl_asym:
            self.x = ca.vertcat(self.x, self.k_cl_us_odd)
            self.state_names.append("k_cl_us_odd")

        if self.simConfig.drag_polar:
            self.x = ca.vertcat(self.x, self.k_cd_cl2)
            self.state_names.append("k_cd_cl2")

        return self.x

    def get_wind_velocity(self):
        if self.simConfig.log_profile is True:
            self.uf = ca.SX.sym("uf")  # Friction velocity
            self.wdir = ca.SX.sym("wdir")  # Ground wind direction
            self.vwz = ca.SX.sym("vw_2")  # Vertical wind velocity
            self.vw = calculate_log_wind_velocity(
                self.uf, self.wdir, self.vwz, self.r[2]
            )
            self.vw_state = ca.vertcat(self.uf, self.wdir, self.vwz)
        else:
            self.vw = ca.SX.sym("vw", 3)  # Wind velocity
            self.vw_state = self.vw

    def get_input(self):
        # Initialize the input vector and input names list
        input = ca.vertcat(self.reelout_speed, self.Ftg)
        self.input_names = ["reelout_speed", "ground_tether_force"]

        # Optional inputs based on configuration
        if self.simConfig.obsData.kite_acceleration:
            self.a_kite = ca.SX.sym("a_kite", 3)  # Kite acceleration
            input = ca.vertcat(input, self.a_kite)
            self.input_names.extend([f"a_kite_{i}" for i in range(3)])

        if self.simConfig.obsData.kcu_acceleration:
            self.a_kcu = ca.SX.sym("a_kcu", 3)  # KCU acceleration
            input = ca.vertcat(input, self.a_kcu)
            self.input_names.extend([f"a_kcu_{i}" for i in range(3)])

        if self.simConfig.obsData.kcu_velocity:
            self.v_kcu = ca.SX.sym("v_kcu", 3)  # KCU velocity
            input = ca.vertcat(input, self.v_kcu)
            self.input_names.extend([f"v_kcu_{i}" for i in range(3)])

        if self.simConfig.obsData.kite_thrust_force:
            self.thrust = ca.SX.sym("thrust", 3)  # Thrust force
            input = ca.vertcat(input, self.thrust)
            self.input_names.extend([f"thrust_{i}" for i in range(3)])

        if (
            self.simConfig.model_yaw
            or self.simConfig.steering_dependent_cs
            or self.simConfig.steering_dependent_clcd
            or self.simConfig.steering_dependent_cl_asym
        ):
            input = ca.vertcat(input, self.us)
            self.input_names.append("us")

        if self.simConfig.obsData.dynamic_depower:
            input = ca.vertcat(input, self.depower_input)
            self.input_names.append("depower_input")
            input = ca.vertcat(input, self.delta_up)
            self.input_names.append("delta_up")

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
            - ca.dot(tether_force / ca.norm_2(tether_force), dir_D) * dir_D
        )
        dir_S = ca.cross(dir_L, dir_D)

        # With the steering-dependent stages enabled, the loop-locked part of
        # the aero coefficients is carried deterministically by the measured
        # steering input and the CL/CD/CS states keep only the residual the
        # actuation cannot explain.
        CL_eff = self.CL
        CD_eff = self.CD
        CS_eff = self.CS
        if self.simConfig.steering_dependent_clcd:
            CL_eff = CL_eff + self.k_cl_us * ca.fabs(self.us)
            CD_eff = CD_eff + self.k_cd_us * self.us * self.us
        if self.simConfig.steering_dependent_cl_asym:
            CL_eff = CL_eff + self.k_cl_us_odd * self.us
        if self.simConfig.drag_polar:
            # Induced drag follows the COMPLETE effective lift coefficient;
            # the CD state is then only the parasitic residual CD0
            CD_eff = CD_eff + self.k_cd_cl2 * CL_eff * CL_eff
        if self.simConfig.steering_dependent_cs:
            # The aerodynamic sideslip phi_a = atan(CS/CL) is proportional to
            # the steering input, so the sideforce follows the lift
            CS_eff = CL_eff * ca.tan(self.k_phi_us * self.us) + CS_eff

        L = CL_eff * 0.5 * rho * self.area * ca.norm_2(self.va) ** 2 * dir_L
        D = CD_eff * 0.5 * rho * self.area * ca.norm_2(self.va) ** 2 * dir_D
        S = CS_eff * 0.5 * rho * self.area * ca.norm_2(self.va) ** 2 * dir_S

        Fg = ca.vertcat(0, 0, -self.mass * g)
        rp = self.v
        if self.simConfig.obsData.kite_thrust_force:
            vp = (-tether_force + L + D + S + Fg + self.thrust) / self.mass
        else:
            vp = (-tether_force + L + D + S + Fg) / self.mass

        r_norm = ca.norm_2(r_kite)
        r_normal = r_kite / r_norm
        vert = ca.vertcat(0, 0, 1)
        vert_norm = project_onto_plane(vert, r_normal)
        horz_norm = ca.cross(vert_norm, r_normal)
        v_kite_tan_horz = ca.dot(v_kite, horz_norm)
        v_kite_tan_vert = ca.dot(v_kite, vert_norm)
        elevation_rate = v_kite_tan_vert / r_norm
        azimuth_rate = v_kite_tan_horz / (r_norm * ca.cos(elevation_0))

        if self.simConfig.obsData.dynamic_depower:
            k_cl_up = self.k_cl_up
            k_cd_up = self.k_cd_up
            delta_up = self.delta_up

            CLdot = k_cl_up * delta_up
            # With the drag polar, the depower-driven CD change flows through
            # CL (CLdot above, then k_cd_cl2*CL_eff^2 in the force model). An
            # independent k_cd_up path would compete for the same slow signal
            # and leave the polar constant only the corrupted fast covariance.
            CDdot = 0 if self.simConfig.drag_polar else k_cd_up * delta_up
        else:
            CLdot = 0
            CDdot = 0

        fx = ca.vertcat(
            rp,
            vp,
            0,
            0,
            0,
            CLdot,
            CDdot,
            0,
            self.reelout_speed,
            elevation_rate,
            azimuth_rate,
        )
        if self.simConfig.model_yaw:
            yaw_rate = self.k_yaw_rate * self.us * ca.norm_2(self.va)
            fx = ca.vertcat(fx, yaw_rate, 0)
        if self.simConfig.obsData.tether_length:
            fx = ca.vertcat(fx, 0)
        if self.simConfig.obsData.tether_elevation:
            fx = ca.vertcat(fx, 0)
        if self.simConfig.obsData.tether_azimuth:
            fx = ca.vertcat(fx, 0)
        if self.simConfig.obsData.dynamic_depower:
            fx = ca.vertcat(fx, 0)
            fx = ca.vertcat(fx, 0)
        if self.simConfig.steering_dependent_cs:
            fx = ca.vertcat(fx, 0)
        if self.simConfig.steering_dependent_clcd:
            fx = ca.vertcat(fx, 0)
            fx = ca.vertcat(fx, 0)
        if self.simConfig.steering_dependent_cl_asym:
            fx = ca.vertcat(fx, 0)
        if self.simConfig.drag_polar:
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

    def create_integrator(self, ts):
        """
        Creates an integrator as a CasADi function with parameters self.x, self.u, x, and ts.
        """
        x_param = ca.SX.sym("x_param", self.x.size1())  # Symbolic input for x parameter
        ts_sym = ca.SX.sym("ts")  # Symbolic input for time step
        u_param = ca.SX.sym("u_param", self.u.size1())  # Symbolic input for u parameter

        # Define ODE system with x_param as a parameter
        fx = self.calc_fx(self.x, self.u, x_param)
        dae = {"x": self.x, "p": ca.vertcat(self.u, x_param), "ode": fx}

        # Create the integrator with a symbolic time step
        self.integrator = ca.integrator("intg", "cvodes", dae, 0, ts)

        # # Convert the integrator into a CasADi function with all parameters
        # self.integrator_fn = ca.Function(
        #     "integrator_fn",
        #     [x_param, u_param],
        #     [)["xf"]],
        #     [ "x_param", "u_param","ts"],
        #     ["x_next"]
        # )

    def propagate(self, x_param, u_param, ts):
        """
        Uses the pre-built CasADi function to integrate the ODE.

        Parameters:
        - x: State variable for the ODE.
        - u: Control input.
        - x_param: Initial state of the system.
        - ts: Time step for integration.

        Returns:
        - Integrated state after the time step.
        """
        # Only three digits after the decimal point
        ts = round(ts, 3)
        if not hasattr(self, "integrator"):
            self.create_integrator(ts)
            self.ts = ts

        if ts != self.ts:
            print(ts, self.ts)
            print("Recreating integrator")
            self.ts = ts
            self.create_integrator(ts)

        result = self.integrator(x0=x_param, p=ca.vertcat(u_param, x_param))["xf"]

        return np.asarray(result).reshape(-1)

    # def propagate(self, x, u, ts):

    #     # Define ODE system
    #     dae = {"x": self.x, "p": self.u, "ode": self.calc_fx(self.x, self.u, x)}  # Define ODE system

    #     self.integrator = ca.integrator("intg", "cvodes", dae, 0,ts)

    #     return np.array(self.integrator(x0=x, p=u)["xf"].T)

    def create_previous_state_vector(self):
        # Generate previous state names by appending '_0' to current state names
        self.previous_state_names = [f"{name}_0" for name in self.state_names]
        print(self.previous_state_names)

        # Create symbolic variables with these names
        x0_elements = [ca.SX.sym(name) for name in self.previous_state_names]

        # Concatenate into a symbolic vector
        x0 = ca.vertcat(*x0_elements)
        return x0

    @property
    def state_index_map(self):

        # Create a dictionary to map variable names to their indices
        variable_index_map = {name: i for i, name in enumerate(self.state_names)}
        return variable_index_map

    @property
    def input_index_map(self):
        # Split the CasADi matrix into individual symbolic variables
        variable_index_map = {name: i for i, name in enumerate(self.input_names)}
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
        self.calc_fx = self.get_fx_fun()

        super().__init__(**kwargs)

    def get_state(self):
        self.x = ca.horzcat(
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
            - ca.dot(-tether_force / ca.norm_2(tether_force), dir_D) * dir_D
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

        x = np.hstack(
            (
                kite_input.kite_position,
                kite_input.kite_velocity,
                np.array([kite_input.kite_yaw]),
            )
        ).reshape(-1)
        u = np.hstack(
            (
                kite_input.wind_velocity,
                np.array([kite_input.us]),
                np.array([kite_input.up]),
                kite_input.tether_force,
            )
        ).reshape(-1)

        # Define ODE system
        dae = {
            "x": self.x,
            "p": self.u,
            "ode": self.calc_fx(self.x, self.u),
        }  # Define ODE system

        self.integrator = ca.integrator("intg", "cvodes", dae, 0, ts)

        return np.array(self.integrator(x0=x, p=u)["xf"].T)

    def calculate_CL(self, up, us):
        # TODO: Implement CL calculation
        return 0.9

    def calculate_CD(self, up, us):
        # TODO: Implement CD calculation
        return 0.1

    def calculate_CS(self, up, us):
        # TODO: Implement CS calculation
        return 0.1

    def calculate_yaw_rate(self, up, us):
        # TODO: Implement yaw rate calculation
        return 0.1

    def calculate_tether_force(self, KiteInput):
        # TODO: Implement tether force calculation, linking to tether model
        return np.array([0, 0, 0])


@dataclass
class KiteInput:
    kite_position: np.ndarray
    kite_velocity: np.ndarray
    kite_yaw: float
    wind_velocity: np.ndarray
    us: float
    up: float
    tether_force: np.ndarray
    tether_length: float
