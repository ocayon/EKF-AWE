import numpy as np
from awes_ekf.setup.settings import g, rho, z0
from scipy.optimize import least_squares
from awes_ekf.utils import project_onto_plane, calculate_angle_2vec
import casadi as ca
from dataclasses import dataclass

tether_materials = {
    "Dyneema-SK78": {
        "density": 970,
        "cd": 1.1,
        "Youngs_modulus": 132e9,
    },
    "Dyneema-SK75": {
        "density": 970,
        "cd": 1.1,
        "Youngs_modulus": 109e9,
    },
    "Kitekraft": {
        "density": 1500,
        "cd": 1.1,
        "Youngs_modulus": 70e9,
    },
}


class Tether:
    """Tether model class"""

    def __init__(self, kite, kcu, obsData, elastic=True, **kwargs):
        """ "Create tether model class from material name and diameter"""
        material_name = kwargs.get("material_name")
        diameter = kwargs.get("diameter")
        n_elements = kwargs.get("n_elements")

        if material_name in tether_materials:
            material_params = tether_materials[material_name]
            for key, value in material_params.items():
                # Set each key-value pair as an attribute of the instance
                setattr(self, key, value)
        else:
            raise ValueError("Invalid tether material")

        self.cf = 0.01  # Skin-friction drag coefficient
        self.diameter = diameter
        self.n_elements = n_elements
        self.elastic = elastic
        self.area = np.pi * (self.diameter / 2) ** 2
        self.EA = self.Youngs_modulus * self.area
        self.kcu = kcu
        self.kite = kite
        self.obsData = obsData

        res = self.calculate_tether_shape_symbolic()
        self.kite_position = res["kite_position"]
        self.tether_force_kite = res["tether_force_kite"]
        self.bridle_frame_va = res["bridle_frame_va"]
        self.bridle_frame_vk = res["bridle_frame_vk"]
        self.tether_frame = res["tether_frame"]
        self.cd_kcu = res["cd_kcu"]
        self.cd_tether = res["cd_tether"]
        self.cd_bridles = res["cd_bridles"]
        self.CL = res["CL"]
        self.CD = res["CD"]
        self.CS = res["CS"]
        self.tether_length = res["tether_length"]
        self.positions = res["positions"]

    def calculate_tether_shape_symbolic(self):
        """Calculate tether shape using symbolic expressions"""

        elevation_0 = ca.SX.sym("elevation_0")
        azimuth_0 = ca.SX.sym("azimuth_0")
        tether_length = ca.SX.sym("tether_length")
        tension_ground = ca.SX.sym("tension_ground")
        r_kite = ca.SX.sym("r_kite", 3)
        v_kite = ca.SX.sym("v_kite", 3)
        vw = ca.SX.sym("vw", 3)
        kcu = self.kcu
        kite = self.kite

        l_unstrained = tether_length / self.n_elements
        m_s = np.pi * self.diameter**2 / 4 * l_unstrained * self.density

        n_elements = self.n_elements
        if kcu is not None:
            n_elements += 1

        wvel = ca.norm_2(vw)
        wdir = vw / wvel

        vtau_kite = project_onto_plane(
            v_kite, r_kite / ca.norm_2(r_kite)
        )  # Velocity projected onto the tangent plane
        omega_tether = ca.cross(r_kite, vtau_kite) / (
            ca.norm_2(r_kite) ** 2
        )  # Tether angular velocity, with respect to the tether attachment point

        if self.obsData.kite_acceleration:
            a_kite = ca.SX.sym("a_kite", 3)
            # Find instantaneuous center of rotation and omega of the kite
            at = (
                ca.dot(a_kite, v_kite / ca.norm_2(v_kite)) * v_kite / ca.norm_2(v_kite)
            )  # Tangential acceleration
            omega_kite = -ca.cross(a_kite-at, v_kite) / (
                ca.norm_2(v_kite) ** 2
            )  # Angular velocity of the kite
            ICR = -ca.cross(v_kite, omega_kite) / (
                ca.norm_2(omega_kite) ** 2
            )  # Instantaneous center of rotation
            alpha = (
                ca.cross(at, ICR) / ca.norm_2(ICR) ** 2
            )  # Angular acceleration of the kite

        tensions = ca.SX.zeros((n_elements, 3))
        tensions[0, 0] = ca.cos(elevation_0) * ca.cos(azimuth_0) * tension_ground
        tensions[0, 1] = ca.cos(elevation_0) * ca.sin(azimuth_0) * tension_ground
        tensions[0, 2] = ca.sin(elevation_0) * tension_ground

        positions = ca.SX.zeros((n_elements + 1, 3))
        if self.elastic:
            l_s = (tension_ground / (self.EA) + 1) * l_unstrained
        else:
            l_s = l_unstrained

        positions[1, 0] = ca.cos(elevation_0) * ca.cos(azimuth_0) * l_s
        positions[1, 1] = ca.cos(elevation_0) * ca.sin(azimuth_0) * l_s
        positions[1, 2] = ca.sin(elevation_0) * l_s

        velocities = ca.SX.zeros((n_elements + 1, 3))
        accelerations = ca.SX.zeros((n_elements + 1, 3))

        drag_tether = 0
        stretched_tether_length = l_s  # Stretched
        for j in range(n_elements):  # Iterate over point masses.
            last_element = j == n_elements - 1
            kcu_element = kcu is not None and j == n_elements - 2

            # Determine kinematics at point mass j.
            vj = ca.cross(omega_tether, positions[j + 1, :].T)
            velocities[j + 1, :] = vj
            aj = ca.cross(omega_tether, vj)
            accelerations[j + 1, :] = aj
            delta_p = positions[j + 1, :] - positions[j, :]
            ej = delta_p.T / ca.norm_2(delta_p)  # Axial direction of tether element
            vwj = (
                wvel * ca.log(positions[j + 1, 2] / z0) / ca.log(r_kite[2] / z0) * wdir
            )  # Wind

            if kcu_element:
                # Determine kinematics at the KCU
                if self.obsData.kcu_acceleration:
                    a_kcu = ca.SX.sym("a_kcu", 3)
                    aj = a_kcu
                elif self.obsData.kite_acceleration:
                    a_kcu = (
                        a_kite
                        + ca.cross(alpha, positions[j + 1, :].T - r_kite)
                        + ca.cross(
                            omega_kite,
                            ca.cross(omega_kite, positions[j + 1, :].T - r_kite),
                        )
                    )
                    aj = a_kcu
                    accelerations[j + 1, :] = aj
                else:
                    a_kcu = aj
                    accelerations[j + 1, :] = a_kcu

                if self.obsData.kcu_velocity:
                    v_kcu = ca.SX.sym("v_kcu", 3)
                    vj = v_kcu
                elif self.obsData.kite_acceleration:
                    v_kcu = v_kite + ca.cross(
                        omega_kite, positions[j + 1, :].T - r_kite
                    )
                    vj = v_kcu
                    velocities[j + 1, :] = vj
                else:
                    v_kcu = vj

                ej = (r_kite - positions[j + 1, :].T) / ca.norm_2(
                    r_kite - positions[j + 1, :].T
                )

            # Determine flow at point mass j.
            vaj = vwj -vj  # Apparent wind velocity

            vajp = ca.dot(vaj, ej) * ej  # Parallel to tether element
            # TODO: check whether to use vajn
            vajn = vaj - vajp  # Perpendicular to tether element

            vaj_sq = ca.norm_2(vaj)**2

            # Determina angle between  va and tether
            theta = calculate_angle_2vec(vaj, ej)
            cd_t = self.cd * ca.sin(theta) ** 3 + np.pi*self.cf*ca.cos(theta)**3
            cl_t = self.cd * ca.sin(theta) ** 2 * ca.cos(theta)-np.pi*self.cf*ca.sin(theta)*ca.cos(theta)**2
            dir_D = vaj / ca.norm_2(vaj) # Drag direction
            dir_L = -(ej - ca.dot(ej, dir_D) * dir_D) # Lift direction
            dynamic_pressure_area = 0.5 * rho * ca.norm_2(vaj) ** 2 * l_unstrained * self.diameter

            # Calculate lift and drag using the common factor
            lift_j = dynamic_pressure_area * cl_t * dir_L
            drag_j = dynamic_pressure_area * cd_t * dir_D

            # Determine drag at point mass j.
            if kcu is None:
                if self.n_elements == 1:
                    faj = 0.5 * lift_j + 0.5 * drag_j
                elif last_element:
                    faj = 0.5 * drag_j + 0.5 * lift_j
                else:
                    faj = 0.5 * drag_j + 0.5 * lift_j
            else:
                if last_element:
                    faj = 0.5*drag_bridles +0.5*lift_bridles

                elif kcu_element:
                    drag_bridles = 0.5*rho*kcu.total_length_bridle_lines*kcu.diameter_bridle_lines*vaj_sq*cd_t # Bridle lines drag
                    drag_bridles = drag_bridles * dir_D
                    lift_bridles = 0.5*rho*kcu.total_length_bridle_lines*kcu.diameter_bridle_lines*vaj_sq*cl_t # Bridle lines lift
                    lift_bridles = lift_bridles * dir_L
                    
                    dp_kcu= .5*rho*ca.norm_2(vajp)*vajp*kcu.cdp*kcu.Ap  # Adding kcu drag perpendicular to kcu
                    dt_kcu= .5*rho*ca.norm_2(vajn)*vajn*kcu.cdt*kcu.At  # Adding kcu drag parallel to kcu
                    th = -0.5*rho*vaj_sq*ca.pi*0.2**2*0.4       # Add thrust of a wind turbine if present
                    # D_turbine = 0.5*rho*ca.norm_2(vaj)**2*ca.pi*0.2**2*1
                    D_kcu = ca.norm_2(dp_kcu+dt_kcu)
                    faj = dp_kcu+dt_kcu+0.5*drag_bridles+0.5*lift_bridles+0.5*drag_j+0.5*lift_j
                    
                else:
                    faj = lift_j  + drag_j 

                    drag_tether += ca.norm_2(drag_j)
  
            if kcu is None:
                if last_element:
                    point_mass = m_s / 2 + kite.mass
                else:
                    point_mass = m_s
            else:
                if last_element:
                    point_mass = kite.mass
                    # aj = np.zeros(3)
                elif kcu_element:
                    point_mass = m_s / 2 + kcu.mass
                else:
                    point_mass = m_s

            # Use force balance to infer tension on next element.
            fgj = ca.SX.zeros((3))
            fgj[2] = -point_mass * g
            if not last_element:
                next_tension = (
                    point_mass * aj + tensions[j, :].T - fgj - faj
                )  # a_kite gave better fit
                tensions[j + 1, :] = next_tension

            # Derive position of next point mass from former tension
            if kcu_element:
                positions[j + 2, :] = (
                    positions[j + 1, :]
                    + tensions[j + 1, :]
                    / ca.norm_2(tensions[j + 1, :])
                    * kcu.distance_kcu_kite
                )

            elif not last_element:
                if self.elastic:
                    l_s = (ca.norm_2(tensions[j + 1, :]) / self.EA + 1) * l_unstrained
                else:
                    l_s = l_unstrained
                stretched_tether_length += l_s
                positions[j + 2, :] = (
                    positions[j + 1, :]
                    + tensions[j + 1, :] / ca.norm_2(tensions[j + 1, :]) * l_s
                )
            elif last_element:
                next_tension = tensions[j, :].T - fgj - faj  # a_kite gave better fit
                if self.obsData.kite_acceleration:
                    next_tension += point_mass * aj
                aerodynamic_force = next_tension

        va = vwj - vj
        ez_bridle = -tensions[-1, :].T / ca.norm_2(
            tensions[-1, :]
        )  # Bridle direction, pointing down
        ey_bridle = ca.cross(ez_bridle, -va) / ca.norm_2(
            ca.cross(ez_bridle, -va)
        )  # y-axis of bridle frame, perpendicular to va
        ex_bridle = ca.cross(
            ey_bridle, ez_bridle
        )  # x-axis of bridle frame, perpendicular ex and ey
        dcm_b2w = ca.horzcat(ex_bridle, ey_bridle, ez_bridle)

        ez_bridle = -tensions[-1, :].T / ca.norm_2(
            tensions[-1, :]
        )  # Bridle direction, pointing down
        ey_bridle = ca.cross(ez_bridle, v_kite) / ca.norm_2(
            ca.cross(ez_bridle, v_kite)
        )  # y-axis of bridle frame, perpendicular to va
        ex_bridle = ca.cross(
            ey_bridle, ez_bridle
        )  # x-axis of bridle frame, perpendicular ex and ey
        dcm_b2vel = ca.horzcat(ex_bridle, ey_bridle, ez_bridle)

        # Tether frame at the kite
        ez_tether = -tensions[-2, :].T / ca.norm_2(tensions[-2, :])
        ey_tether = ca.cross(ez_tether, -va) / ca.norm_2(ca.cross(ez_tether, -va))
        ex_tether = ca.cross(ey_tether, ez_tether)
        dcm_t2w = ca.horzcat(ex_tether, ey_tether, ez_tether)

        tension_kite = tensions[-1, :].T
        # Calculate aerodynamic coefficients
        dir_D = va / ca.norm_2(va)
        CD = ca.dot(aerodynamic_force, dir_D) / (
            0.5 * rho * kite.area * ca.norm_2(va) ** 2
        )
        dir_L = (
            tension_kite / ca.norm_2(tension_kite)
            - ca.dot(tension_kite / ca.norm_2(tension_kite), dir_D) * dir_D
        )
        CL = ca.dot(aerodynamic_force, dir_L) / (
            0.5 * rho * kite.area * ca.norm_2(va) ** 2
        )
        dir_S = ca.cross(dir_L, dir_D)
        CS = ca.dot(aerodynamic_force, dir_S) / (
            0.5 * rho * kite.area * ca.norm_2(va) ** 2
        )
        # Parasitic drag of tether and KCU
        if kcu is not None:
            cd_kcu = D_kcu / (0.5 * rho * ca.norm_2(vaj) ** 2 * kite.area)
            cd_bridles = ca.norm_2(drag_bridles) / (0.5 * rho * ca.norm_2(vaj) ** 2 * kite.area)
        else:
            cd_kcu = 0
            cd_bridles = 0
        cd_tether = drag_tether / (0.5 * rho * ca.norm_2(vaj) ** 2 * kite.area)

        args = [
            elevation_0,
            azimuth_0,
            tether_length,
            tension_ground,
            r_kite,
            v_kite,
            vw,
        ]

        if self.obsData.kite_acceleration:
            args.append(a_kite)
        if self.obsData.kcu_acceleration:
            args.append(a_kcu)
        if self.obsData.kcu_velocity:
            args.append(v_kcu)

        res = {
            "kite_position": ca.Function("kite_position", args, [positions[-1, :].T]),
            "tether_force_kite": ca.Function(
                "tether_force_kite", args, [tensions[-1, :].T]
            ),
            "bridle_frame_va": ca.Function("bridle_frame_va", args, [dcm_b2w]),
            "bridle_frame_vk": ca.Function("bridle_frame_vk", args, [dcm_b2vel]),
            "tether_frame": ca.Function("tether_frame", args, [dcm_t2w]),
            "cd_kcu": ca.Function("cd_kcu", args, [cd_kcu]),
            "cd_bridles": ca.Function("cd_bridles", args, [cd_bridles]),
            "cd_tether": ca.Function("cd_tether", args, [cd_tether]),
            "CL": ca.Function("CL", args, [CL]),
            "CD": ca.Function("CD", args, [CD]),
            "CS": ca.Function("CS", args, [CS]),
            "tether_length": ca.Function(
                "tether_length", args, [stretched_tether_length]
            ),
            "positions": ca.Function("positions", args, [positions]),
            
        }

        return res

    def objective_function(self, x, *args, return_force=False, kcu=None):
        """Objective function for optimization"""

        r_kite = np.array(args[1])
        if return_force:
            tension_ground = x[2]
            tether_length = args[0]
        else:
            tether_length = x[2]
            tension_ground = args[0]

        args = (x[0], x[1], tether_length, tension_ground) + args[1::]
        r_tether_model = self.kite_position(*args)

        return r_kite - np.array(r_tether_model).reshape(-1)

    def solve_tether_shape(self, tetherInput):
        """Solve for the tether shape"""

        r_kite = np.array(tetherInput.kite_position)
        v_kite = np.array(tetherInput.kite_velocity)
        vw = np.array(tetherInput.wind_velocity)
        a_kite = np.array(tetherInput.kite_acceleration)
        a_kcu = np.array(tetherInput.kcu_acceleration)
        v_kcu = np.array(tetherInput.kcu_velocity)

        tension_ground = tetherInput.tether_force

        if not hasattr(self, "opt_guess"):
            elevation = np.arctan2(r_kite[2], np.linalg.norm(r_kite))
            azimuth = np.arctan2(r_kite[1], r_kite[0])
            length = np.linalg.norm(r_kite)

            self.opt_guess = [elevation, azimuth, length]

        args = (tension_ground, r_kite, v_kite, vw)

        if self.obsData.kite_acceleration:
            args += (a_kite,)
        if self.obsData.kcu_acceleration:
            args += (a_kcu,)
        if self.obsData.kcu_velocity:
            args += (v_kcu,)

        opt_res = least_squares(
            self.objective_function,
            self.opt_guess,
            args=args,
            verbose=0,
            xtol=1e-5,
            ftol=1e-5,
        )
        self.opt_guess = opt_res.x
        tetherInput.tether_length = opt_res.x[2]
        tetherInput.tether_elevation = opt_res.x[0]
        tetherInput.tether_azimuth = opt_res.x[1]

        return tetherInput


@dataclass
class TetherInput:
    kite_position: np.ndarray
    kite_velocity: np.ndarray
    tether_force: float
    tether_length: float
    tether_elevation: float
    tether_azimuth: float
    wind_velocity: np.ndarray = np.array([1e-3, 1e-3, 0])
    kite_acceleration: np.ndarray = None
    kcu_acceleration: np.ndarray = None
    kcu_velocity: np.ndarray = None

    def create_input_tuple(self, obsData):

        args = (
            self.tether_elevation,
            self.tether_azimuth,
            self.tether_length,
            self.tether_force,
            self.kite_position,
            self.kite_velocity,
            self.wind_velocity,
        )
        if obsData.kite_acceleration:
            args = args + (self.kite_acceleration,)
        if obsData.kcu_acceleration:
            args = args + (self.kcu_acceleration,)
        if obsData.kcu_velocity:
            args = args + (self.kcu_velocity,)

        return args
