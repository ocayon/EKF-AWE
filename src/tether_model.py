import numpy as np
from config import g, rho, z0
from scipy.optimize import least_squares
from config import tether_materials
from utils import project_onto_plane, calculate_angle_2vec


class TetherModel:
    def __init__(self,material,diameter,density,cd,Youngs_modulus,elastic=True):
        self.material = material
        self.diameter = diameter
        self.density = density
        self.cd = cd
        self.cf = 0.02
        self.E = Youngs_modulus
        self.area = np.pi*(self.diameter/2)**2
        self.EA = self.E*self.area
        self.elastic = elastic
    
    def calculate_tether_shape(self,x, n_tether_elements, r_kite, v_kite, vw, kite, kcu, tension_ground = None, tether_length = None,
                               a_kite = None, a_kcu = None, v_kcu = None, return_values=False):
        
        # Currently neglecting radial velocity of kite.
        if tension_ground is not None and tether_length is not None:
            beta_n, phi_n = x
        elif tension_ground is not None and tether_length is None:
            beta_n, phi_n , tether_length = x
        else:
            beta_n, phi_n, tension_ground = x
        
        l_unstrained = tether_length/n_tether_elements
        m_s = np.pi*self.diameter**2/4 * l_unstrained * self.density

        n_elements = n_tether_elements
        if kite.KCU == True:
            n_elements += 1
        
        wvel = np.linalg.norm(vw)
        wdir = vw/wvel

        vtau_kite = project_onto_plane(v_kite,r_kite/np.linalg.norm(r_kite)) # Velocity projected onto the tangent plane
        omega_tether = np.cross(r_kite,vtau_kite)/(np.linalg.norm(r_kite)**2) # Tether angular velocity, with respect to the tether attachment point

        if a_kite is not None:
            # Find instantaneuous center of rotation and omega of the kite
            at = np.dot(a_kite,np.array(v_kite)/np.linalg.norm(v_kite))*np.array(v_kite)/np.linalg.norm(v_kite) # Tangential acceleration
            if np.linalg.norm(a_kite)<1e-3:
                omega_kite = omega_tether # If the kite is not accelerating, the ICR is at infinity, and the kite is rotating at the same rate as the tether
            else:
                omega_kite = np.cross(a_kite-at,v_kite)/(np.linalg.norm(v_kite)**2) # Angular velocity of the kite

            ICR = np.cross(v_kite,omega_kite)/(np.linalg.norm(omega_kite)**2) # Instantaneous center of rotation     
            alpha = np.cross(at,ICR)/np.linalg.norm(ICR)**2 # Angular acceleration of the kite

        tensions = np.zeros((n_elements, 3))
        tensions[0, 0] = np.cos(beta_n)*np.cos(phi_n)*tension_ground
        tensions[0, 1] = np.cos(beta_n)*np.sin(phi_n)*tension_ground
        tensions[0, 2] = np.sin(beta_n)*tension_ground

        positions = np.zeros((n_elements+1, 3))
        if self.elastic:
            l_s = (tension_ground/(self.EA)+1)*l_unstrained
        else:
            l_s = l_unstrained

        positions[1, 0] = np.cos(beta_n)*np.cos(phi_n)*l_s
        positions[1, 1] = np.cos(beta_n)*np.sin(phi_n)*l_s
        positions[1, 2] = np.sin(beta_n)*l_s

        velocities = np.zeros((n_elements+1, 3))
        accelerations = np.zeros((n_elements+1, 3))
        non_conservative_forces = np.zeros((n_elements, 3))

        stretched_tether_length = l_s  # Stretched
        for j in range(n_elements):  # Iterate over point masses.
            last_element = j == n_elements - 1
            kcu_element = kite.KCU and j == n_elements - 2

            # Determine kinematics at point mass j.
            vj = np.cross(omega_tether, positions[j+1, :])
            velocities[j+1, :] = vj
            aj = np.cross(omega_tether, vj)
            accelerations[j+1, :] = aj
            delta_p = positions[j+1, :] - positions[j, :]
            ej = delta_p/np.linalg.norm(delta_p)  # Axial direction of tether element
            vwj = wvel*np.log(positions[j+1,2]/z0)/np.log(r_kite[2]/z0)*wdir # Wind
            
            if last_element:
                vj = np.array(v_kite)
                aj = np.array(a_kite)
            if kcu_element: 
                if a_kcu is not None:
                    aj = np.array(a_kcu)
                    vj = np.array(v_kcu)
                else:
                    v_kcu = v_kite + np.cross(omega_kite,positions[j+1, :]-r_kite)
                    vj = v_kcu
                    velocities[j+1, :] = vj
                    a_kcu = a_kite+ np.cross(alpha,positions[j+1, :]-r_kite) +np.cross(omega_kite,np.cross(omega_kite,positions[j+1, :]-r_kite))
                    aj = a_kcu
                    accelerations[j+1, :] = aj

                ej = (r_kite-np.array(positions[j+1, :]))/np.linalg.norm(r_kite-np.array(positions[j+1, :]))

            # Determine flow at point mass j.
            vaj = vj - vwj  # Apparent wind velocity
            
            vajp = np.dot(vaj, ej)*ej  # Parallel to tether element
            # TODO: check whether to use vajn
            vajn = vaj - vajp  # Perpendicular to tether element

            vaj_sq = np.linalg.norm(vaj)*vaj

            # Determina angle between  va and tether
            theta = calculate_angle_2vec(-vaj,ej)
            # vaj_sq = np.linalg.norm(vajn)*vajn
            CD_tether = self.cd*np.sin(theta)**3+self.cf
            # CL_tether = self.cd*np.sin(theta)**2*np.cos(theta)
            tether_drag_basis = rho*l_unstrained*self.diameter*CD_tether*vaj_sq
            
            # Determine drag at point mass j.
            if not kite.KCU:
                if n_tether_elements == 1:
                    dj = -.125*tether_drag_basis
                elif last_element:
                    dj = -.25*tether_drag_basis  # TODO: add bridle drag
                else:
                    dj = -.5*tether_drag_basis
            else:
                if last_element:
                    # dj = -0.25*rho*L_blines*d_bridle*vaj_sq*cd_t # Bridle lines drag
                    dj = 0
                    
                elif n_tether_elements == 1:
                    dj = -.25*tether_drag_basis
                    dp= -.5*rho*np.linalg.norm(vajp)*vajp*kcu.cdp*kcu.Ap  # Adding kcu drag perpendicular to kcu
                    dt= -.5*rho*np.linalg.norm(vajn)*vajn*kcu.cdt*kcu.At  # Adding kcu drag parallel to kcu
                    dj += dp+dt
                    cd_kcu = (np.linalg.norm(dp+dt))/(0.5*rho*kite.area*np.linalg.norm(vaj)**2)
                elif kcu_element:
                    dj = -.25*tether_drag_basis
                    theta = np.pi/2-theta
                    cd_kcu = kcu.cdt*np.sin(theta)**3+self.cf
                    cl_kcu = kcu.cdt*np.sin(theta)**2*np.cos(theta)
                    # D_turbine = 0.5*rho*np.linalg.norm(vaj)**2*np.pi*0.2**2*1
                    # dp= -.5*rho*np.linalg.norm(vajp)*vajp*kcu.cdp*kcu.Ap  # Adding kcu drag perpendicular to kcu
                    # dt= -.5*rho*np.linalg.norm(vajn)*vajn*kcu.cdt*kcu.At  # Adding kcu drag parallel to kcu
                    # th = -0.5*rho*vaj_sq*np.pi*0.2**2*0.4
                    # dj += dp+dt
                    # Approach described in Hoerner, taken from Paul Thedens dissertation
                    dir_D = -vaj/np.linalg.norm(vaj)
                    dir_L = ej - np.dot(ej,dir_D)*dir_D
                    L_kcu = 0.5*rho*np.linalg.norm(vaj)**2*kcu.Ap*cl_kcu
                    D_kcu = 0.5*rho*np.linalg.norm(vaj)**2*cd_kcu*kcu.Ap
                    dj += L_kcu*dir_L + D_kcu*dir_D #+ D_turbine*dir_D

                else:
                    dir_D = -vaj/np.linalg.norm(vaj)
                    dir_L = ej - np.dot(ej,dir_D)*dir_D
                    cd_t = self.cd*np.sin(theta)**3+self.cf
                    cl_t = self.cd*np.sin(theta)**2*np.cos(theta)
                    L_t = 0.5*rho*np.linalg.norm(vaj)**2*l_unstrained*self.diameter*cl_t
                    D_t = 0.5*rho*np.linalg.norm(vaj)**2*l_unstrained*self.diameter*cd_t
                    dj = L_t*dir_L + D_t*dir_D

            if not kite.KCU:
                if last_element:
                    point_mass = m_s/2 + kite.mass + kcu.mass           
                else:
                    point_mass = m_s
            else:
                if last_element:
                    point_mass = kite.mass          
                    # aj = np.zeros(3)
                elif kcu_element:
                    point_mass = m_s/2 + kcu.mass
                else:
                    point_mass = m_s

            # Use force balance to infer tension on next element.
            fgj = np.array([0, 0, -point_mass*g])
            next_tension = point_mass*aj + tensions[j, :] - fgj - dj  # a_kite gave better fit
            if not last_element:
                tensions[j+1, :] = next_tension
                non_conservative_forces[j, :] = dj
            else:
                aerodynamic_force = next_tension
                non_conservative_forces[j, :] = dj + aerodynamic_force

            # Derive position of next point mass from former tension
            if kcu_element:
                positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * kite.distance_kcu_kite
                
            elif not last_element:
                if self.elastic:
                    l_s = (np.linalg.norm(tensions[j+1, :])/self.EA+1)*l_unstrained
                else:
                    l_s = l_unstrained
                stretched_tether_length += l_s
                positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * l_s

        if return_values:
            va = vwj-vj  # All y-axes are defined perpendicular to apparent wind velocity.

            ez_bridle = tensions[-1, :]/np.linalg.norm(tensions[-1, :])
            ey_bridle = np.cross(ez_bridle, -vj)/np.linalg.norm(np.cross(ez_bridle, -vj))
            ex_bridle = np.cross(ey_bridle, ez_bridle)
            dcm_b2w = np.vstack(([ex_bridle], [ey_bridle], [ez_bridle])).T

            ez_tether = tensions[-2, :]/np.linalg.norm(tensions[-2, :])
            ey_tether = np.cross(ez_tether, va)/np.linalg.norm(np.cross(ez_tether, va))
            ex_tether = np.cross(ey_tether, ez_tether)
            dcm_t2w = np.vstack(([ex_tether], [ey_tether], [ez_tether])).T

            ez_f_aero = aerodynamic_force/np.linalg.norm(aerodynamic_force)
            ey_f_aero = np.cross(ez_f_aero, va)/np.linalg.norm(np.cross(ez_f_aero, va))
            ex_f_aero = np.cross(ey_f_aero, ez_f_aero)
            dcm_fa2w = np.vstack(([ex_f_aero], [ey_f_aero], [ez_f_aero])).T

            ez_tau = r_kite/np.linalg.norm(r_kite)
            ey_tau = np.cross(ez_tau, va)/np.linalg.norm(np.cross(ez_tau, va))
            ex_tau = np.cross(ey_tau, ez_tau)
            dcm_tau2w = np.vstack(([ex_tau], [ey_tau], [ez_tau])).T
            
            dir_D = va/np.linalg.norm(va)
            CD = np.dot(aerodynamic_force,dir_D)/(0.5*rho*kite.area*np.linalg.norm(va)**2)
            dir_L = tensions[-1, :]/np.linalg.norm(tensions[-1, :]) - np.dot(tensions[-1, :]/np.linalg.norm(tensions[-1, :]),dir_D)*dir_D
            CL = np.dot(aerodynamic_force,dir_L)/(0.5*rho*kite.area*np.linalg.norm(va)**2)
            dir_S = np.cross(dir_L,dir_D)
            CS = np.dot(aerodynamic_force,dir_S)/(0.5*rho*kite.area*np.linalg.norm(va)**2)

            return positions, stretched_tether_length, dcm_b2w, dcm_t2w, dcm_fa2w, \
                dcm_tau2w, aerodynamic_force, va,tensions[-1,:], dcm_t2w[:,1],cd_kcu, CL,CD,CS
        else:
            if len(x) == 2:
                return (positions[-1, :2] - r_kite[:2])
            else:   
                return (positions[-1, :] - r_kite) 
        
    def solve_tether_shape(self, n_tether_elements, r_kite, v_kite, vw, kite, kcu, tension_ground = None, tether_length = None,
                            a_kite = None, a_kcu = None, v_kcu = None):
        
        if not hasattr(self, 'opt_guess'):
            elevation = np.arctan2(r_kite[2], np.linalg.norm(r_kite[:2]))
            azimuth = np.arctan2(r_kite[1], r_kite[0])
            length = np.linalg.norm(r_kite)
            if tension_ground is not None and tether_length is not None:
                self.opt_guess = [elevation, azimuth]
            else:
                self.opt_guess = [elevation, azimuth, length]
            
        args = (n_tether_elements, r_kite, v_kite, vw, kite, kcu, tension_ground, tether_length, a_kite, a_kcu, v_kcu)
        opt_res = least_squares(self.calculate_tether_shape, self.opt_guess, args=args, verbose=0,xtol = 1e-3,ftol = 1e-3)
        self.opt_guess = opt_res.x
        res = self.calculate_tether_shape(self.opt_guess, *args, return_values=True)
        self.positions = res[0]
        self.kite_pos = self.positions[-1, :]
        self.stretched_tether_length = res[1]
        self.dcm_b2w = res[2]
        self.Ft_kite = np.array(res[8]) # Tether force at the kite
        self.CL = res[11]
        self.CD = res[12]
        self.CS = res[13]



