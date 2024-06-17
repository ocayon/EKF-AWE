import numpy as np
import casadi as ca
from awes_ekf.setup.settings import kappa, z0, rho, g


class DynamicModel:
    def __init__(self, kite, tether, simConfig):

        self.simConfig = simConfig

        self.u = kite.get_input()
        self.x = kite.get_state()
        self.x0 = kite.x0

        self.fx = kite.get_fx(tether)
        self.calc_fx = ca.Function("calc_fx", [self.x, self.u, self.x0], [self.fx])

    def get_fx_fun(self, kite, tether, kcu):
        return ca.Function(
            "calc_Fx", [self.x, self.u, self.x0], [self.get_fx(kite, tether, kcu)]
        )

    def get_fx_jac(self):

        return ca.simplify(ca.jacobian(self.fx, self.x))

    def get_fx_jac_fun(self):
        return ca.Function("calc_Fx", [self.x, self.u, self.x0], [self.get_fx_jac()])

    def propagate(self, x, u, ts):

        fx = self.calc_fx(self.x, self.u, x)

        # Define ODE system
        dae = {"x": self.x, "p": self.u, "ode": fx}  # Define ODE system
        integrator = ca.integrator("intg", "cvodes", dae, 0, ts)  # Define integrator

        return np.array(integrator(x0=x, p=u)["xf"].T)
