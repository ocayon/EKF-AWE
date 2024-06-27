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

    def get_fx_jac(self):

        return ca.simplify(ca.jacobian(self.fx, self.x))

    def get_fx_jac_fun(self):
        return ca.Function("calc_Fx", [self.x, self.u, self.x0], [self.get_fx_jac()])
