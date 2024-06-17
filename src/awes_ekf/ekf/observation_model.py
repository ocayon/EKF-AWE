import numpy as np
import casadi as ca
from awes_ekf.setup.settings import kappa, z0
from awes_ekf.utils import calculate_airflow_angles


class ObservationModel:

    def __init__(self, x, u, simConfig, kite, tether, kcu):
        self.x = x
        self.u = u
        self.x0 = ca.SX.sym("x0", self.x.shape[0])  # Kite position
        self.simConfig = simConfig

    def get_hx(self, kite, tether, kcu):

        # Split the CasADi matrix into individual symbolic variables
        state_variables = ca.vertsplit(self.x)
        # Create a dictionary to map variable names to their symbolic variables
        state_map = {var.name(): var for var in state_variables}

        input_variables = ca.vertsplit(self.u)
        input_map = {var.name(): var for var in input_variables}

        elevation_0 = state_map["elevation_first_tether_element"]
        azimuth_0 = state_map["azimuth_first_tether_element"]
        tether_length = state_map["tether_length"]
        r_kite = self.x0[0:3]
        v_kite = self.x0[3:6]
        tension_ground = input_map["ground_tether_force"]

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
        if self.simConfig.obsData.kite_acc:
            a_kite = self.u[2:5]
            args += (a_kite,)
        if self.simConfig.obsData.kcu_acc:
            if self.simConfig.obsData.kite_acc:
                a_kcu = self.u[5:8]
                args += (a_kcu,)
            else:
                a_kcu = self.u[2:5]
                args += (a_kcu,)

        if self.simConfig.obsData.kcu_vel:
            if self.simConfig.obsData.kite_acc:
                v_kcu = self.u[8:11]
                args += (v_kcu,)
            else:
                v_kcu = self.u[5:8]
                args += (v_kcu,)

        r_tether_model = tether.kite_position(*args)

        if self.simConfig.log_profile:
            wvel = self.x[6] / kappa * np.log(self.x[2] / z0)
            wdir = self.x[7]
            vw = np.array([wvel * np.cos(wdir), wvel * np.sin(wdir), self.x[8]])
        else:
            vw = self.x[6:9]

        va = vw - self.x[3:6]

        dcm_b2vel = tether.bridle_frame_va(*args)

        airflow_angles = calculate_airflow_angles(dcm_b2vel, vw - v_kite)

        h = ca.SX()
        h = ca.vertcat(self.x[0:3])
        h = ca.vertcat(h, self.x[3:6])
        if self.simConfig.tether_offset:
            h = ca.vertcat(h, self.x[12] - self.x[-1])
        else:
            h = ca.vertcat(h, self.x[12])
        h = ca.vertcat(h, self.x[13])
        h = ca.vertcat(h, self.x[14])
        h = ca.vertcat(h, (self.x[0:3] - r_tether_model))
        if self.simConfig.model_yaw:
            h = ca.vertcat(h, self.x[15])
        if self.simConfig.enforce_z_wind:
            h = ca.vertcat(h, self.x[8])

        if self.simConfig.obsData.apparent_windspeed:
            h = ca.vertcat(h, ca.norm_2(va))
        if self.simConfig.obsData.angle_of_attack:
            h = ca.vertcat(h, airflow_angles[0])

        return h

    def get_hx_jac(self, kite, tether, kcu):
        hx = self.get_hx(kite, tether, kcu)
        return ca.simplify(ca.jacobian(hx, self.x))

    def get_hx_jac_fun(self, kite, tether, kcu):
        return ca.Function(
            "calc_Hx", [self.x, self.u, self.x0], [self.get_hx_jac(kite, tether, kcu)]
        )

    def get_hx_fun(self, kite, tether, kcu):
        return ca.Function(
            "calc_hx", [self.x, self.u, self.x0], [self.get_hx(kite, tether, kcu)]
        )
