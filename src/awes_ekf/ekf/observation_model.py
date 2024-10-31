import numpy as np
import casadi as ca
from awes_ekf.setup.settings import kappa, z0
from awes_ekf.utils import calculate_airflow_angles, calculate_log_wind_velocity


class ObservationModel:

    def __init__(self, x, u, simConfig, kite, tether, kcu):
        self.x = x
        self.u = u
        self.x0 = kite.x0
        self.simConfig = simConfig

    def get_hx(self, kite, tether, kcu):

        # Split the CasADi matrix into individual symbolic variables
        state_variables = ca.vertsplit(self.x)
        previous_state_variables = ca.vertsplit(self.x0)
        input_variables = ca.vertsplit(self.u)
        # Create a dictionary to map variable names to their symbolic variables
        state_map = {var.name(): var for var in state_variables}
        input_map = {var.name(): var for var in input_variables}
        previous_state_map = {var.name(): var for var in previous_state_variables}

        elevation_0 = state_map["elevation_first_tether_element"]
        azimuth_0 = state_map["azimuth_first_tether_element"]
        tether_length = state_map["tether_length"]
        r_kite = np.array([previous_state_map[f"r_{i}_0"] for i in range(3)])
        v_kite = np.array([previous_state_map[f"v_{i}_0"] for i in range(3)])
        tension_ground = input_map["ground_tether_force"]

        if self.simConfig.log_profile:
            vw = calculate_log_wind_velocity(
                previous_state_map["uf_0"],
                previous_state_map["wdir_0"],
                previous_state_map["vw_2_0"],
                previous_state_map["r_2_0"],
            )
        else:
            vw = np.array([previous_state_map[f"vw_{i}_0"] for i in range(3)])

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
            a_kite = np.array([input_map[f"a_kite_{i}"] for i in range(3)])
            args += (a_kite,)
        if self.simConfig.obsData.kcu_acceleration:
            a_kcu = np.array([input_map[f"a_kcu_{i}"] for i in range(3)])
            args += (a_kcu,)
        if self.simConfig.obsData.kcu_velocity:
            v_kcu = np.array([input_map[f"v_kcu_{i}"] for i in range(3)])
            args += (v_kcu,)

        r_tether_model = tether.kite_position(*args)

        index_map = kite.state_index_map

        if self.simConfig.log_profile:
            vw = calculate_log_wind_velocity(
                self.x[index_map["uf"]],
                self.x[index_map["wdir"]],
                self.x[index_map["vw_2"]],
                self.x[index_map["r_2"]],
            )
        else:
            vw = ca.vertcat(*[self.x[index_map[f"vw_{i}"]] for i in range(3)])

        r_kite = ca.vertcat(*[self.x[index_map[f"r_{i}"]] for i in range(3)])
        v_kite = ca.vertcat(*[self.x[index_map[f"v_{i}"]] for i in range(3)])
        va = vw - v_kite

        dcm_b2vel = tether.bridle_frame_va(*args)

        airflow_angles = calculate_airflow_angles(dcm_b2vel, vw - v_kite)

        h = ca.SX()
        if self.simConfig.obsData.kite_position:
            h = ca.vertcat(r_kite)
        if self.simConfig.obsData.kite_velocity:
            h = ca.vertcat(h, v_kite)
        h = ca.vertcat(h, (r_kite - r_tether_model))
        # Convert into a for
        if self.simConfig.model_yaw:
            h = ca.vertcat(h, self.x[index_map["yaw"]])
        if self.simConfig.obsData.tether_length:
            h = ca.vertcat(h, self.x[index_map["tether_length"]]-self.x[index_map["tether_length_offset"]])
        if self.simConfig.obsData.tether_elevation:
            h = ca.vertcat(h, self.x[index_map["elevation_first_tether_element"]]-self.x[index_map["tether_elevation_offset"]])
        if self.simConfig.obsData.tether_azimuth:
            h = ca.vertcat(h, self.x[index_map["azimuth_first_tether_element"]] - self.x[index_map["tether_azimuth_offset"]])
        if self.simConfig.enforce_vertical_wind_to_0:
            h = ca.vertcat(h, self.x[index_map["vw_2"]])
        if self.simConfig.obsData.kite_apparent_windspeed:
            h = ca.vertcat(h, ca.norm_2(va))
        if self.simConfig.obsData.bridle_angle_of_attack:
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
