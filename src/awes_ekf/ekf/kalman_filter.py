import numpy as np
import casadi as ca
from awes_ekf.setup.settings import kappa, z0, rho, g
import control
from awes_ekf.utils import project_onto_plane, calculate_angle_2vec
import logging

# Set up logging
logging.basicConfig(
    filename="ekf_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# %%
class ExtendedKalmanFilter:
    def __init__(
        self, stdv_dynamic_model, stdv_measurements, dyn_model, obs_model, kite, tether, kcu, simConfig
    ):  
        self.simConfig = simConfig
        self.stdv_dynamic_model = stdv_dynamic_model
        self.stdv_measurements = stdv_measurements
        self.Q = self.get_state_noise_covariance(stdv_dynamic_model, simConfig)
        self.R = self.get_observation_noise_covariance(stdv_measurements)
        self.doIEKF = simConfig.doIEKF
        self.epsilon = simConfig.epsilon
        self.max_iterations = simConfig.max_iterations
        self.n = len(stdv_dynamic_model)
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
    
    @property
    def obs_model(self):
        return self._obs_model
    
    @obs_model.setter
    def obs_model(self, obs_model):
        self._obs_model = obs_model
        self.calc_Hx = obs_model.get_hx_jac_fun(self.kite, self.tether, self.kcu)
        self.calc_hx = obs_model.get_hx_fun(self.kite, self.tether, self.kcu)

    @property
    def stdv_dynamic_model(self):
        return self._stdv_dynamic_model
    
    @stdv_dynamic_model.setter
    def stdv_dynamic_model(self, value):
        self.Q = self.get_state_noise_covariance(value, self.simConfig)
        self.n = len(value)
        self.P_k1_k1 = np.eye(self.n) * 1**2

    @property
    def stdv_measurements(self):
        return self._stdv_measurements
    
    @stdv_measurements.setter
    def stdv_measurements(self, value):
        self.R = self.get_observation_noise_covariance(value)


    def predict(self,ts):

        # Calculate Jacobians
        self.Fx = np.array(self.calc_Fx(self.x_k1_k, self.u, self.x_k1_k))
        # self.G = np.array(self.calc_G(self.x_k1_k,self.u))
        nx = self.Fx.shape[0]

        # Calculate discrete time state transition and input-to-state matrices
        sys_ct = control.ss(
            self.Fx, np.zeros([nx, nx]), np.zeros(nx), np.zeros(nx)
        )  # If process noise input matrix wants to be added ->control.ss(self.Fx, self.G, np.zeros(nx), np.zeros(nx))
        sys_dt = control.sample_system(sys_ct, ts, method="zoh")
        self.Phi = sys_dt.A
        # self.Gamma = sys_dt.B
        # Calculate covariance prediction error
        self.P_k1_k = self.Phi @ self.P_k1_k1 @ self.Phi.T + self.Q

    def update(self):

        # Find indices where z has None values
        none_indices = np.where(np.isnan(self.z))[0]

        # If there are None values, output a message
        if len(none_indices) > 0:
            print(f"Warning: None values found in the measurements at indices {none_indices}")

        # Create a mask for valid indices (i.e., where z is not None)
        valid_indices = [i for i in range(len(self.z)) if i not in none_indices]

        # Create a reduced version of z without None values
        self.z_valid = self.z[valid_indices]


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

                # Construct the Jacobian Hx for valid observations only
                Hx_full = np.array(self.calc_Hx(eta1, self.u, eta1))  # Full Hx
                self.Hx = Hx_full[valid_indices, :]  # Reduced Hx (remove rows)

                # Observation and observation error predictions (valid observations only)
                z_k1_k_full = np.array(self.calc_hx(eta1, self.u, eta1)).reshape(-1)
                self.z_k1_k = z_k1_k_full[valid_indices]  # Only valid predictions
                R_full = self.R  # Save the full R matrix
                self.R = self.R[np.ix_(valid_indices, valid_indices)]  # Remove rows/cols

                self.P_zz = (
                    self.Hx @ self.P_k1_k @ self.Hx.T + self.R
                )  # Covariance matrix of observation error (for valid z)
                std_z = np.sqrt(
                    np.diag(self.P_zz)
                )  # Standard deviation of observation error (for valid z)

                # Gain (K)
                self.K = self.P_k1_k @ self.Hx.T @ np.linalg.inv(self.P_zz)

                # New observation for valid z values
                eta2 = self.x_k1_k + self.K @ (
                    self.z_valid
                    - self.z_k1_k
                    - np.array((self.Hx @ (self.x_k1_k - eta1).T)).reshape(-1)
                )
                eta2 = np.array(eta2).reshape(-1)
                err = np.linalg.norm(eta2 - eta1) / np.linalg.norm(eta1)

            self.IEKF_itts = itts
            self.x_k1_k1 = eta2

        else:
            # Non-iterative case
            Hx_full = np.array(self.calc_Hx(self.x_k1_k, self.u, self.x_k1_k))  # Full Hx
            self.Hx = Hx_full[valid_indices, :]  # Reduced Hx

            # Observation and observation error predictions (valid observations only)
            z_k1_k_full = np.array(self.calc_hx(self.x_k1_k, self.u, self.x_k1_k)).reshape(-1)
            self.z_k1_k = z_k1_k_full[valid_indices]  # Only valid predictions
            R_full = self.R  # Save the full R matrix
            self.R = self.R[np.ix_(valid_indices, valid_indices)]  # Remove rows/cols

            self.P_zz = (
                self.Hx @ self.P_k1_k @ self.Hx.T + self.R
            )  # Covariance matrix of observation error (for valid z)
            self.std_z = np.sqrt(np.diag(self.P_zz))

            # Gain (K)
            self.K = self.P_k1_k @ self.Hx.T @ np.linalg.inv(self.P_zz)

            # Calculate optimal state x(k+1|k+1) for valid z
            self.x_k1_k1 = np.array(
                self.x_k1_k + self.K @ (self.z_valid - self.z_k1_k)
            ).reshape(-1)

        self.P_k1_k1 = (np.eye(self.n) - self.K @ self.Hx) @ self.P_k1_k
        self.std_x_cor = np.sqrt(
            np.diag(self.P_k1_k1)
        )  # Standard deviation of state estimation error (for validation)
        
        self.debug_info = self._output_debug_info(self.z_valid, self.z_k1_k, self.P_zz)

        # Restore full R and Hx for the next update
        self.R = R_full
        self.Hx = Hx_full

    def get_state_noise_covariance(self, stdv_dynamic_model, simConfig):
        Q = np.diag(np.array(stdv_dynamic_model) ** 2)
        return Q

    def get_observation_noise_covariance(self, stdv_measurements):
        return np.diag(np.array(stdv_measurements) ** 2)

    def update_input_vector(self, input_class):
        input = np.array([input_class.tether_reelout_speed, input_class.tether_force])
        if self.simConfig.obsData.kite_acceleration:
            input = np.concatenate((input, input_class.kite_acceleration))
        if self.simConfig.obsData.kcu_acceleration:
            input = np.concatenate((input, input_class.kcu_acceleration))
        if self.simConfig.obsData.kcu_velocity:
            input = np.concatenate((input, input_class.kcu_velocity))
        if self.simConfig.obsData.kite_thrust_force:
            input = np.concatenate((input, input_class.kite_thrust_force))

        self.u = input

    def update_measurement_vector(self, input_class, simConfig):
        opt_measurements = simConfig.opt_measurements
        z = np.array([])  # Initialize an empty NumPy array

        # Append values to the NumPy array
        if simConfig.obsData.kite_position:
            z = np.append(z, input_class.kite_position)
        if simConfig.obsData.kite_velocity:
            z = np.append(z, input_class.kite_velocity)
        z = np.append(z, np.zeros(3))  # Add zeros for the least-squares problem
        # TODO: Convert this into dict to loop through and avoid hardcoding
        if simConfig.model_yaw:
            z = np.append(z, input_class.kite_yaw)
        if simConfig.obsData.tether_length:
            z = np.append(z, input_class.tether_length)
        if simConfig.obsData.tether_elevation:
            z = np.append(z, input_class.tether_elevation_ground)
        if simConfig.obsData.tether_azimuth:
            z = np.append(z, input_class.tether_azimuth_ground)
        if simConfig.enforce_vertical_wind_to_0:
            z = np.append(z, 0)
        if simConfig.obsData.kite_apparent_windspeed:
            z = np.append(z, input_class.kite_apparent_windspeed)
        if simConfig.obsData.bridle_angle_of_attack:
            z = np.append(z, input_class.bridle_angle_of_attack)

        self.z = z

    def _output_debug_info(self, z_valid, z_k1_k, P_zz):
        epsilon = z_valid - z_k1_k
        epsilon_norm = epsilon / np.sqrt(np.diag(P_zz))
        nis = epsilon.T @ np.linalg.inv(P_zz) @ epsilon
        mahalanobis_distance = np.sqrt(nis)
        norm_epsilon_norm = np.linalg.norm(epsilon_norm)
        
        debug_info = {
            "norm_epsilon_norm": norm_epsilon_norm,
            "nis": nis,
            "mahalanobis_distance": mahalanobis_distance,
        }
        if self.simConfig.debug:
            logging.debug(f"Residual (Innovation): {epsilon}")
            logging.debug(f"Normalized Residual: {epsilon_norm}")
            logging.debug(f"NIS: {nis}")
            logging.debug(f"Mahalanobis Distance: {mahalanobis_distance}")
            logging.debug(f"Norm of Normalized Residual: {norm_epsilon_norm}")

        threshold = 1  # Example threshold, should be pre-calculated
        if norm_epsilon_norm < threshold:
            debug_info["exceeds_threshold"] = False
        else:
            debug_info["exceeds_threshold"] = True

        return debug_info


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
