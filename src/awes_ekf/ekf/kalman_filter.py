import numpy as np
import casadi as ca
from awes_ekf.setup.settings import kappa, z0, rho, g
import control
from awes_ekf.utils import project_onto_plane, calculate_angle_2vec


# %%
class ExtendedKalmanFilter:
    def __init__(
        self, stdv_x, stdv_y, ts, dyn_model, obs_model, kite, tether, kcu, simConfig
    ):
        self.Q = self.get_state_noise_covariance(stdv_x, simConfig)
        self.R = self.get_observation_noise_covariance(stdv_y)
        self.doIEKF = simConfig.doIEKF
        self.epsilon = simConfig.epsilon
        self.max_iterations = simConfig.max_iterations
        self.ts = ts
        self.n = len(stdv_x)
        self.P_k1_k1 = np.eye(self.n) * 1**2
        self.simConfig = simConfig

        self.kite = kite
        self.kcu = kcu
        self.tether = tether
        self.obs_model = obs_model
        self.dyn_model = dyn_model
        self.calc_Fx = self.dyn_model.get_fx_jac_fun()
        self.calc_Hx = self.obs_model.get_hx_jac_fun(kite, tether, kcu)
        self.calc_hx = self.obs_model.get_hx_fun(kite, tether, kcu)

    def predict(self):

        # Calculate Jacobians
        self.Fx = np.array(self.calc_Fx(self.x_k1_k, self.u, self.x_k1_k))
        # self.G = np.array(self.calc_G(self.x_k1_k,self.u))
        nx = self.Fx.shape[0]

        # Calculate discrete time state transition and input-to-state matrices
        sys_ct = control.ss(
            self.Fx, np.zeros([nx, nx]), np.zeros(nx), np.zeros(nx)
        )  # If process noise input matrix wants to be added ->control.ss(self.Fx, self.G, np.zeros(nx), np.zeros(nx))
        sys_dt = control.sample_system(sys_ct, self.ts, method="zoh")
        self.Phi = sys_dt.A
        # self.Gamma = sys_dt.B
        # Calculate covariance prediction error
        self.P_k1_k = self.Phi @ self.P_k1_k1 @ self.Phi.T + self.Q

    def update(self):

        if self.doIEKF == True:

            eta2 = self.x_k1_k
            err = 2 * self.epsilon
            itts = 0

            while err > self.epsilon:
                if itts >= self.max_iterations:
                    print(
                        "Terminating IEKF: exceeded max iterations (%d)\n"
                        % (self.max_iterations)
                    )
                    break

                itts = itts + 1
                eta1 = eta2

                # Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix
                self.Hx = np.array(self.calc_Hx(eta1, self.u, eta1))

                # Observation and observation error predictions
                self.z_k1_k = np.array(self.calc_hx(eta1, self.u, eta1)).reshape(
                    -1
                )  # prediction of observation (for validation)
                self.P_zz = (
                    self.Hx @ self.P_k1_k @ self.Hx.T + self.R
                )  # covariance matrix of observation error (for validation)
                self.std_z = np.sqrt(
                    np.diag(self.P_zz)
                )  # standard deviation of observation error (for validation)

                # K(k+1) (gain)
                self.K = self.P_k1_k @ self.Hx.T @ np.linalg.inv(self.P_zz)

                # new observation
                eta2 = self.x_k1_k + self.K @ (
                    self.z
                    - self.z_k1_k
                    - np.array((self.Hx @ (self.x_k1_k - eta1).T)).reshape(-1)
                )
                eta2 = np.array(eta2).reshape(-1)
                err = np.linalg.norm(eta2 - eta1) / np.linalg.norm(eta1)

            self.IEKF_itts = itts
            self.x_k1_k1 = eta2

        else:
            self.Hx = np.array(self.calc_Hx(self.x_k1_k, self.u, self.x_k1_k))

            # correction
            self.z_k1_k = np.array(
                self.calc_hx(self.x_k1_k, self.u, self.x_k1_k)
            ).reshape(-1)
            self.P_zz = (
                self.Hx @ self.P_k1_k @ self.Hx.T + self.R
            )  # covariance matrix of observation error (for validation)
            self.std_z = np.sqrt(np.diag(self.P_zz))
            # K(k+1) (gain)
            self.K = self.P_k1_k @ self.Hx.T @ np.linalg.inv(self.P_zz)

            # Calculate optimal state x(k+1|k+1)
            self.x_k1_k1 = np.array(
                self.x_k1_k + self.K @ (self.z - self.z_k1_k)
            ).reshape(-1)

        self.P_k1_k1 = (np.eye(self.n) - self.K @ self.Hx) @ self.P_k1_k
        self.std_x_cor = np.sqrt(
            np.diag(self.P_k1_k1)
        )  # standard deviation of state estimation error (for validation)

    def get_state_noise_covariance(self, stdv_x, simConfig):
        Q = np.diag(np.array(stdv_x) ** 2)
        return Q

    def get_observation_noise_covariance(self, stdv_y):
        return np.diag(np.array(stdv_y) ** 2)

    def update_input_vector(self, input_class):
        input = np.array([input_class.reelout_speed, input_class.tether_force])
        if self.simConfig.obsData.kite_acc:
            input = np.concatenate((input, input_class.kite_acc))
        if self.simConfig.obsData.kcu_acc:
            input = np.concatenate((input, input_class.kcu_acc))
        if self.simConfig.obsData.kcu_vel:
            input = np.concatenate((input, input_class.kcu_vel))
        if self.simConfig.obsData.thrust_force:
            input = np.concatenate((input, input_class.thrust_force))

        self.u = input

    def update_measurement_vector(self, input_class, simConfig):
        opt_measurements = simConfig.opt_measurements
        z = np.array([])  # Initialize an empty NumPy array

        # Append values to the NumPy array
        z = np.append(z, input_class.kite_pos)
        z = np.append(z, input_class.kite_vel)
        z = np.append(z, np.zeros(3))  # Add zeros for the least-squares problem
        if simConfig.model_yaw:
            z = np.append(z, input_class.kite_yaw)
        if simConfig.obsData.tether_length:
            z = np.append(z, input_class.tether_length)
        if simConfig.obsData.tether_elevation:
            z = np.append(z, input_class.tether_elevation)
        if simConfig.obsData.tether_azimuth:
            z = np.append(z, input_class.tether_azimuth)
        if simConfig.enforce_z_wind:
            z = np.append(z, 0)
        if simConfig.obsData.apparent_windspeed:
            z = np.append(z, input_class.apparent_windspeed)
        if simConfig.obsData.angle_of_attack:
            z = np.append(z, input_class.kite_aoa)

        self.z = z


def observability_Lie_method(f, h, x, u, x0, u0):

    n = f.shape[0]
    m = h.shape[0]
    O = ca.SX.zeros((m * n, n))
    Li = ca.simplify(ca.jacobian(h, x))
    O[0 * m : (1) * m, :] = Li
    for i in range(1, m):
        Hxf = ca.mtimes(Li, f)
        Hxxf = ca.simplify(ca.jacobian(Hxf, x))
        Li = Hxxf
        O[i * m : (i + 1) * m, :] = Hxxf

    calc_O = ca.Function("calc_O", [x, u], [O])
    O_app = calc_O(x0, u0)
    if np.linalg.matrix_rank(O_app) == len(x0):
        print("System is observable")
    else:
        print("System is not observable")


# %%
