import numpy as np
from awes_ekf.setup.settings import g, rho, z0
from scipy.optimize import least_squares
from awes_ekf.utils import project_onto_plane, calculate_angle_2vec
import casadi as ca

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
        }
}
class Tether:
    """ Tether model class"""
    def __init__(self,elastic=True, **kwargs):
        """"Create tether model class from material name and diameter"""
        material_name = kwargs.get('material_name')
        diameter = kwargs.get('diameter')
        n_elements = kwargs.get('n_elements')

        if material_name in tether_materials:
            material_params = tether_materials[material_name]
            for key, value in material_params.items():
                # Set each key-value pair as an attribute of the instance
                setattr(self, key, value)
        else:
            raise ValueError("Invalid tether material")
            
        
        self.cf = 0.02
        self.diameter = diameter
        self.n_elements = n_elements
        self.elastic = elastic
        self.area = np.pi*(self.diameter/2)**2
        self.EA = self.Youngs_modulus*self.area
        
        
    
    def calculate_tether_shape_symbolic(self, elevation_0, azimuth_0, tether_length,
                                         tension_ground, r_kite, v_kite, vw, kite, kcu,tether,  
                                        a_kite = None, a_kcu = None, v_kcu = None):
        

        l_unstrained = tether_length/tether.n_elements
        m_s = np.pi*tether.diameter**2/4 * l_unstrained * tether.density

        n_elements = tether.n_elements
        if kcu is not None:
            n_elements += 1
        
        wvel = ca.norm_2(vw)
        wdir = vw/wvel

        vtau_kite = project_onto_plane(v_kite,r_kite/ca.norm_2(r_kite)) # Velocity projected onto the tangent plane
        omega_tether = ca.cross(r_kite,vtau_kite)/(ca.norm_2(r_kite)**2) # Tether angular velocity, with respect to the tether attachment point

        if a_kite is not None:
            # Find instantaneuous center of rotation and omega of the kite
            at = ca.dot(a_kite,v_kite/ca.norm_2(v_kite))*v_kite/ca.norm_2(v_kite) # Tangential acceleration
            omega_kite = ca.cross(a_kite,v_kite)/(ca.norm_2(v_kite)**2) # Angular velocity of the kite
            ICR = ca.cross(v_kite,omega_kite)/(ca.norm_2(omega_kite)**2) # Instantaneous center of rotation     
            alpha = ca.cross(at,ICR)/ca.norm_2(ICR)**2 # Angular acceleration of the kite

        tensions = ca.SX.zeros((n_elements, 3))
        tensions[0, 0] = ca.cos(elevation_0)*ca.cos(azimuth_0)*tension_ground
        tensions[0, 1] = ca.cos(elevation_0)*ca.sin(azimuth_0)*tension_ground
        tensions[0, 2] = ca.sin(elevation_0)*tension_ground

        positions =  ca.SX.zeros((n_elements+1, 3))
        if tether.elastic:
            l_s = (tension_ground/(tether.EA)+1)*l_unstrained
        else:
            l_s = l_unstrained

        positions[1, 0] = ca.cos(elevation_0)*ca.cos(azimuth_0)*l_s
        positions[1, 1] = ca.cos(elevation_0)*ca.sin(azimuth_0)*l_s
        positions[1, 2] = ca.sin(elevation_0)*l_s

        velocities  = ca.SX.zeros((n_elements+1, 3))
        accelerations  = ca.SX.zeros((n_elements+1, 3))
        non_conservative_forces  = ca.SX.zeros((n_elements+1, 3))

        stretched_tether_length = l_s  # Stretched
        for j in range(n_elements):  # Iterate over point masses.
            last_element = j == n_elements - 1
            kcu_element = kcu is not None and j == n_elements - 2

            # Determine kinematics at point mass j.
            vj = ca.cross(omega_tether, positions[j+1, :].T)
            velocities[j+1, :] = vj
            aj = ca.cross(omega_tether, vj)
            accelerations[j+1, :] = aj
            delta_p = positions[j+1, :] - positions[j, :]
            ej = delta_p.T/ca.norm_2(delta_p)  # Axial direction of tether element
            vwj = wvel*ca.log(positions[j+1,2]/z0)/ca.log(r_kite[2]/z0)*wdir # Wind
            
            if last_element:
                vj = v_kite
                aj = a_kite
            if kcu_element: 
                if a_kcu is not None:
                    aj = a_kcu
                    vj = v_kcu
                else:
                    v_kcu = v_kite + ca.cross(omega_kite,positions[j+1, :].T-r_kite)
                    vj = v_kcu
                    velocities[j+1, :] = vj
                    a_kcu = a_kite+ ca.cross(alpha,positions[j+1, :].T-r_kite) +ca.cross(omega_kite,ca.cross(omega_kite,positions[j+1, :].T-r_kite))
                    aj = a_kcu
                    accelerations[j+1, :] = aj

                ej = (r_kite-positions[j+1, :].T)/ca.norm_2(r_kite-positions[j+1, :].T)

            # Determine flow at point mass j.
            vaj = vj - vwj  # Apparent wind velocity
            
            vajp = ca.dot(vaj, ej)*ej  # Parallel to tether element
            # TODO: check whether to use vajn
            vajn = vaj - vajp  # Perpendicular to tether element

            vaj_sq = ca.norm_2(vaj)*vaj

            # Determina angle between  va and tether
            theta = calculate_angle_2vec(-vaj,ej)
            # vaj_sq = ca.norm_2(vajn)*vajn
            CD_tether = tether.cd*ca.sin(theta)**3+tether.cf
            # CL_tether = tether.cd*ca.sin(theta)**2*ca.cos(theta)
            tether_drag_basis = rho*l_unstrained*tether.diameter*CD_tether*vaj_sq
            
            # Determine drag at point mass j.
            if kcu is None:
                if tether.n_elements == 1:
                    dj = -.125*tether_drag_basis
                elif last_element:
                    dj = -.25*tether_drag_basis  # TODO: add bridle drag
                else:
                    dj = -.5*tether_drag_basis
            else:
                if last_element:
                    # dj = -0.25*rho*L_blines*d_bridle*vaj_sq*cd_t # Bridle lines drag
                    dj = 0
                    
                elif tether.n_elements == 1:
                    dj = -.25*tether_drag_basis
                    dp= -.5*rho*ca.norm_2(vajp)*vajp*kcu.cdp*kcu.Ap  # Adding kcu drag perpendicular to kcu
                    dt= -.5*rho*ca.norm_2(vajn)*vajn*kcu.cdt*kcu.At  # Adding kcu drag parallel to kcu
                    dj += dp+dt
                    cd_kcu = (ca.norm_2(dp+dt))/(0.5*rho*kite.area*ca.norm_2(vaj)**2)
                elif kcu_element:
                    dj = -.25*tether_drag_basis
                    theta = ca.pi/2-theta
                    cd_kcu = kcu.cdt*ca.sin(theta)**3+tether.cf
                    cl_kcu = kcu.cdt*ca.sin(theta)**2*ca.cos(theta)
                    # D_turbine = 0.5*rho*ca.norm_2(vaj)**2*ca.pi*0.2**2*1
                    # dp= -.5*rho*ca.norm_2(vajp)*vajp*kcu.cdp*kcu.Ap  # Adding kcu drag perpendicular to kcu
                    # dt= -.5*rho*ca.norm_2(vajn)*vajn*kcu.cdt*kcu.At  # Adding kcu drag parallel to kcu
                    # th = -0.5*rho*vaj_sq*ca.pi*0.2**2*0.4
                    # dj += dp+dt
                    # Approach described in Hoerner, taken from Paul Thedens dissertation
                    dir_D = -vaj/ca.norm_2(vaj)
                    dir_L = ej - ca.dot(ej,dir_D)*dir_D
                    L_kcu = 0.5*rho*ca.norm_2(vaj)**2*kcu.Ap*cl_kcu
                    D_kcu = 0.5*rho*ca.norm_2(vaj)**2*cd_kcu*kcu.Ap
                    dj += L_kcu*dir_L + D_kcu*dir_D #+ D_turbine*dir_D

                else:
                    dir_D = -vaj/ca.norm_2(vaj)
                    dir_L = ej - ca.dot(ej,dir_D)*dir_D
                    cd_t = tether.cd*ca.sin(theta)**3+tether.cf
                    cl_t = tether.cd*ca.sin(theta)**2*ca.cos(theta)
                    L_t = 0.5*rho*ca.norm_2(vaj)**2*l_unstrained*tether.diameter*cl_t
                    D_t = 0.5*rho*ca.norm_2(vaj)**2*l_unstrained*tether.diameter*cd_t
                    dj = L_t*dir_L + D_t*dir_D

            if kcu is None:
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
            fgj = ca.SX.zeros((3))
            fgj[2] = -point_mass*g
            if not last_element:
                next_tension = point_mass*aj + tensions[j, :].T - fgj - dj  # a_kite gave better fit
                tensions[j+1, :] = next_tension

            # Derive position of next point mass from former tension
            if kcu_element:
                positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/ca.norm_2(tensions[j+1, :]) * kcu.distance_kcu_kite
                
            elif not last_element:
                if tether.elastic:
                    l_s = (ca.norm_2(tensions[j+1, :])/tether.EA+1)*l_unstrained
                else:
                    l_s = l_unstrained
                stretched_tether_length += l_s
                positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/ca.norm_2(tensions[j+1, :]) * l_s
        
    
        return positions[-1, :].T, tensions[-1, :].T
        
        


    def calculate_tether_shape(self,x, r_kite, v_kite, vw, kite, kcu, tension_ground = None, tether_length = None,
                               a_kite = None, a_kcu = None, v_kcu = None, return_values=False):
        """ Calculate the shape of the tether given the current state of the kite and the wind.
        Possible inputs:
        - kite x and y position, tether length and ground tether tension
        - kite x,y and z position, tether length 
        - kite x,y and z position, ground tether tension
        Optional inputs:
        - kcu acceleration and velocity
        """
        # Currently neglecting radial velocity of kite.
        if tension_ground is not None and tether_length is not None:
            beta_n, phi_n = x
        elif tension_ground is not None and tether_length is None:
            beta_n, phi_n , tether_length = x
        else:
            beta_n, phi_n, tension_ground = x
        
        n_elements = self.n_elements
        l_unstrained = tether_length/n_elements
        m_s = np.pi*self.diameter**2/4 * l_unstrained * self.density

        if kcu is not None:
            n_elements += 1
        
        wvel = np.linalg.norm(vw)
        wdir = vw/wvel

        vtau_kite = project_onto_plane(v_kite,r_kite/np.linalg.norm(r_kite)) # Velocity projected onto the tangent plane
        omega_tether = np.cross(r_kite,vtau_kite)/(np.linalg.norm(r_kite)**2) # Tether angular velocity, with respect to the tether attachment point

        if a_kite is not None:
            # Find instantaneuous center of rotation and omega of the kite
            at = np.dot(a_kite,np.array(v_kite)/np.linalg.norm(v_kite))*np.array(v_kite)/np.linalg.norm(v_kite) # Tangential acceleration
            
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
        drag_tether = 0
        stretched_tether_length = l_s  # Stretched
        for j in range(n_elements):  # Iterate over point masses.
            last_element = j == n_elements - 1
            kcu_element = kcu is not None and j == n_elements - 2

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
                    v_kcu = np.cross(omega_kite,ICR)
                    vj = v_kcu
                    velocities[j+1, :] = vj
                    a_kcu = a_kite+ np.cross(alpha,positions[j+1, :]-r_kite) +np.cross(omega_kite,np.cross(omega_kite,positions[j+1, :]-r_kite))
                    a_kcu = np.cross(omega_kite,v_kite)
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
            if kcu is None:
                if n_elements == 1:
                    dj = -.125*tether_drag_basis
                elif last_element:
                    dj = -.25*tether_drag_basis  # TODO: add bridle drag
                else:
                    dj = -.5*tether_drag_basis
            else:
                if last_element:
                    # dj = -0.25*rho*L_blines*d_bridle*vaj_sq*cd_t # Bridle lines drag
                    dj = 0
                    
                elif n_elements == 1:
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
                    
                    drag_tether += D_t

            if kcu is None:
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
                
            if not last_element:
                next_tension = point_mass*aj + tensions[j, :] - fgj - dj  # a_kite gave better fit    
                tensions[j+1, :] = next_tension
                non_conservative_forces[j, :] = dj
            else:
                next_tension = tensions[j, :] - fgj - dj  # a_kite gave better fit
                if a_kite is not None:
                    next_tension += point_mass*aj
                aerodynamic_force = next_tension
                non_conservative_forces[j, :] = dj + aerodynamic_force

            # Derive position of next point mass from former tension
            if kcu_element:
                positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * kcu.distance_kcu_kite
                
            elif not last_element:
                if self.elastic:
                    l_s = (np.linalg.norm(tensions[j+1, :])/self.EA+1)*l_unstrained
                else:
                    l_s = l_unstrained
                stretched_tether_length += l_s
                positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * l_s

        if return_values:
            va = vwj-vj  # All y-axes are defined perpendicular to apparent wind velocity.

            ## DCMs
            ez_bridle = -tensions[-1, :]/np.linalg.norm(tensions[-1, :])                # Bridle direction, pointing down
            ey_bridle = np.cross(ez_bridle, -va)/np.linalg.norm(np.cross(ez_bridle, -va)) # y-axis of bridle frame, perpendicular to va
            ex_bridle = np.cross(ey_bridle, ez_bridle)                                      # x-axis of bridle frame, perpendicular ex and ey
            dcm_b2w = np.vstack(([ex_bridle], [ey_bridle], [ez_bridle])).T

            ez_tether = -tensions[-2, :]/np.linalg.norm(tensions[-2, :])
            ey_tether = np.cross(ez_tether, -va)/np.linalg.norm(np.cross(ez_tether, -va))
            ex_tether = np.cross(ey_tether, ez_tether)
            dcm_t2w = np.vstack(([ex_tether], [ey_tether], [ez_tether])).T
            
            ez_bridle = -tensions[-1, :]/np.linalg.norm(tensions[-1, :])                # Bridle direction, pointing down
            ey_bridle = np.cross(ez_bridle, -vj)/np.linalg.norm(np.cross(ez_bridle, -vj)) # y-axis of bridle frame, perpendicular to va
            ex_bridle = np.cross(ey_bridle, ez_bridle)                                      # x-axis of bridle frame, perpendicular ex and ey
            dcm_b2vel =  np.vstack(([ex_bridle], [ey_bridle], [ez_bridle])).T

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

            cd_kcu = D_kcu/(0.5*rho*np.linalg.norm(vaj)**2*kite.area)
            cd_tether = drag_tether/(0.5*rho*np.linalg.norm(vaj)**2*kite.area)

            return positions, stretched_tether_length, dcm_b2w, dcm_t2w, dcm_fa2w, \
                dcm_tau2w, aerodynamic_force, va,tensions[-1,:], dcm_t2w[:,1],cd_kcu,cd_tether, CL,CD,CS,dcm_b2vel
        else:
            if len(x) == 2:
                return (positions[-1, :2] - r_kite[:2])
            else:   
                return (positions[-1, :] - r_kite) 
        
    def solve_tether_shape(self, r_kite, v_kite, vw, kite, kcu, tension_ground = None, tether_length = None,
                            a_kite = None, a_kcu = None, v_kcu = None):
        
        if not hasattr(self, 'opt_guess'):
            elevation = np.arctan2(r_kite[2], np.linalg.norm(r_kite[:2]))
            azimuth = np.arctan2(r_kite[1], r_kite[0])
            length = np.linalg.norm(r_kite)
            if tension_ground is not None and tether_length is not None:
                self.opt_guess = [elevation, azimuth]
            else:
                self.opt_guess = [elevation, azimuth, length]
            
        args = (r_kite, v_kite, vw, kite, kcu, tension_ground, tether_length, a_kite, a_kcu, v_kcu)
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



