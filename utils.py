import numpy as np
import casadi as ca
from v3_properties import *
import pandas as pd

def project_onto_plane(vector, plane_normal):
    return vector - np.dot(vector, plane_normal) * plane_normal

def project_onto_plane_sym(vector, plane_normal):
    return vector - ca.dot(vector, plane_normal) * plane_normal

def read_data():
    df = pd.read_csv('./data/2019-10-08_11.csv',delimiter = ' ')
    df.kite_1_yaw_rate = -df.kite_1_yaw_rate
    df = df.iloc[42000:158199]
    df['time'] = df['time'] - df['time'].iloc[0]
    df = df.interpolate()
    df['rx'] = df['kite_pos_east']
    df['ry'] = df['kite_pos_north']    
    df['rz'] = df['kite_height']
    df['vx'] = df['kite_1_vy']
    df['vy'] = df['kite_1_vx']
    df['vz'] = -df['kite_1_vz']
    df['ax'] = df['kite_1_ay']
    df['ay'] = df['kite_1_ax']
    df['az'] = -df['kite_1_az']
    
    df.kite_azimuth = -df.kite_azimuth +2*np.pi-df['est_upwind_direction']-np.pi/2
    df.ground_tether_force = df.ground_tether_force * 9.81
    
    return df
def read_data_new():
    df = pd.read_csv('./data/2021-09-23_13-58-42_ProtoLogger.csv',delimiter = ' ')
    df.kite_1_yaw_rate = -df.kite_1_yaw_rate
    df = df.iloc[44000::]
    df['time'] = df['time'] - df['time'].iloc[0]
    df = df.interpolate()
    df['rx'] = df['kite_pos_east']
    df['ry'] = df['kite_pos_north']    
    df['rz'] = df['kite_height']
    df['vx'] = df['kite_1_vy']
    df['vy'] = df['kite_1_vx']
    df['vz'] = -df['kite_1_vz']
    # Calculate differences without zero at the end
    diff_ax = np.diff(df['kite_1_vy']) / 0.1
    diff_ay = np.diff(df['kite_1_vx']) / 0.1
    diff_az = -np.diff(df['kite_1_vz']) / 0.1
    
    # Add zero at the end of the differences
    df['ax'] = np.concatenate((diff_ax, [0]))
    df['ay'] = np.concatenate((diff_ay, [0]))
    df['az'] = np.concatenate((diff_az, [0]))
    df.kite_azimuth = -df.kite_azimuth
    df.ground_tether_force = df.ground_tether_force * 9.81
    
    return df

def plot_vector(p0, v, ax, scale_vector=.03, color='g', label=None):
    p1 = p0 + v * scale_vector
    vector = np.vstack(([p0], [p1])).T
    ax.plot3D(vector[0], vector[1], vector[2], color=color, label=label)

def get_az_el(x, set_parameter, n_tether_elements, r_kite, v_kite, vw, separate_kcu_mass=False, elastic_elements=True, ax_plot_forces=False, return_values=False):
    # Currently neglecting radial velocity of kite.

    beta_n, phi_n = x
    tension_ground, tether_length = set_parameter

    l_unstrained = tether_length/n_tether_elements
    m_s = np.pi*d_t**2/4 * l_unstrained * rho_t

    n_elements = n_tether_elements
    if separate_kcu_mass:
        n_elements += 1
    
    # This omega can be improved by finding turning center and correcting for that
    omega = np.cross(r_kite, v_kite)/np.linalg.norm(r_kite)**2
    
    tensions = np.zeros((n_elements, 3))
    tensions[0, 0] = np.cos(beta_n)*np.cos(phi_n)*tension_ground
    tensions[0, 1] = np.cos(beta_n)*np.sin(phi_n)*tension_ground
    tensions[0, 2] = np.sin(beta_n)*tension_ground

    positions = np.zeros((n_elements+1, 3))
    if elastic_elements:
        l_s = (tension_ground/tether_stiffness+1)*l_unstrained
        # print((tension_ground/tether_stiffness+1))
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
        kcu_element = separate_kcu_mass and j == n_elements - 2

        # Determine kinematics at point mass j.
        vj = np.cross(omega, positions[j+1, :])
        velocities[j+1, :] = vj
        aj = np.cross(omega, vj)
        accelerations[j+1, :] = aj
        delta_p = positions[j+1, :] - positions[j, :]
        ej = delta_p/np.linalg.norm(delta_p)  # Axial direction of tether element

        # Determine flow at point mass j.
        vaj = vj - vw  # Apparent wind velocity
        vajp = np.dot(vaj, ej)*ej  # Parallel to tether element
        # TODO: check whether to use vajn
        vajn = vaj - vajp  # Perpendicular to tether element

        vaj_sq = np.linalg.norm(vaj)*vaj
        # vaj_sq = np.linalg.norm(vajn)*vajn
        tether_drag_basis = rho*l_unstrained*d_t*cd_t*vaj_sq

        # Determine drag at point mass j.
        if not separate_kcu_mass:
            if n_tether_elements == 1:
                dj = -.125*tether_drag_basis
            elif last_element:
                dj = -.25*tether_drag_basis  # TODO: add bridle drag
            else:
                dj = -.5*tether_drag_basis
        else:
            if last_element:
                dj = 0  # TODO: add bridle drag
            elif n_tether_elements == 1:
                dj = -.125*tether_drag_basis
                dj += -.5*rho*np.linalg.norm(vajp)*vajp*cdp*Ap  # Adding kcu drag perpendicular to kcu
                dj += -.5*rho*np.linalg.norm(vajn)*vajn*cdt*At  # Adding kcu drag parallel to kcu
            elif kcu_element:
                dj = -.25*tether_drag_basis
                dp= -.5*rho*np.linalg.norm(vajp)*vajp*cdp*Ap  # Adding kcu drag perpendicular to kcu
                dt= -.5*rho*np.linalg.norm(vajn)*vajn*cdt*At  # Adding kcu drag parallel to kcu
                dj += dp+dt
                cd_kcu = (np.linalg.norm(dp+dt))/(0.5*rho*A_kite*np.linalg.norm(vaj)**2)
            else:
                dj = -.5*tether_drag_basis

        if not separate_kcu_mass:
            if last_element:
                point_mass = m_s/2 + m_kite + m_kcu
            else:
                point_mass = m_s
        else:
            if last_element:
                point_mass = m_kite
            elif kcu_element:
                point_mass = m_s/2 + m_kcu
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

        if ax_plot_forces and not last_element:
            forces = [point_mass*aj, dj, fgj, -tensions[j, :], next_tension]
            labels = ['resultant', 'drag', 'weight', 'last tension', 'next tension']
            clrs = ['m', 'r', 'k', 'g', 'b']
            for f, lbl, clr in zip(forces, labels, clrs):
                # print("{} = {:.2f} N".format(lbl, np.linalg.norm(f)))
                plot_vector(positions[j+1, :], f, ax_plot_forces, color=clr)

        # Derive position of next point mass from former tension
        if kcu_element:
            positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * l_bridle
        elif not last_element:
            if elastic_elements:
                l_s = (np.linalg.norm(tensions[j+1, :])/tether_stiffness+1)*l_unstrained
            else:
                l_s = l_unstrained
            stretched_tether_length += l_s
            positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * l_s

    if return_values == 2:
        return positions, velocities, accelerations, tensions, aerodynamic_force, non_conservative_forces
    elif return_values:
        va = vw-vj  # All y-axes are defined perpendicular to apparent wind velocity.

        ez_bridle = tensions[-1, :]/np.linalg.norm(tensions[-1, :])
        ey_bridle = np.cross(ez_bridle, va)/np.linalg.norm(np.cross(ez_bridle, va))
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
        
        D = np.dot(va/np.linalg.norm(va),aerodynamic_force)*va/np.linalg.norm(va)        
        L = aerodynamic_force-D
        CD = np.linalg.norm(D)/(0.5*rho*A_kite*np.linalg.norm(va)**2)
        CL = np.linalg.norm(L)/(0.5*rho*A_kite*np.linalg.norm(va)**2)
        return positions, stretched_tether_length, dcm_b2w, dcm_t2w, dcm_fa2w, \
               dcm_tau2w, aerodynamic_force, va,tensions[-1,:], dcm_t2w[:,1],cd_kcu, CL,CD
    else:
        return positions[-1, :] - r_kite[:]

def get_forces(x, u,separate_kcu_mass = True,elastic_elements = False):
    # Currently neglecting radial velocity of kite.
    tension_ground = u[0]
    beta_n = u[1]
    phi_n = u[2]
    tether_length = u[3]
    n_elements = 30
    
    r_kite = x[0:3]
    v_kite = x[3:6]
    vw = x[6:9]
    
    l_unstrained = tether_length/n_elements
    m_s = np.pi*d_t**2/4 * l_unstrained * rho_t

    wvel = ca.norm_2(vw)
    wdir = vw/wvel
    if separate_kcu_mass:
        n_elements += 1
    
    # This omega can be improved by finding turning center and correcting for that
    omega = ca.cross(r_kite, v_kite)/ca.norm_2(r_kite)**2
    
    tensions = ca.SX.zeros((n_elements, 3))
    tensions[0,0] = ca.cos(beta_n)*ca.cos(phi_n)*tension_ground
    tensions[0, 1] = ca.cos(beta_n)*ca.sin(phi_n)*tension_ground
    tensions[0, 2] = ca.sin(beta_n)*tension_ground

    positions = ca.SX.zeros(n_elements+1, 3)
    if elastic_elements:
        l_s = (tension_ground/tether_stiffness+1)*l_unstrained
        # print((tension_ground/tether_stiffness+1))
    else:
        l_s = l_unstrained
    positions[1, 0] = ca.cos(beta_n)*ca.cos(phi_n)*l_s
    positions[1, 1] = ca.cos(beta_n)*ca.sin(phi_n)*l_s
    positions[1, 2] = ca.sin(beta_n)*l_s

    velocities = ca.SX.zeros(n_elements+1, 3)
    accelerations = ca.SX.zeros(n_elements+1, 3)
    non_conservative_forces = ca.SX.zeros(n_elements+1, 3)

    stretched_tether_length = l_s  # Stretched
    for j in range(n_elements):  # Iterate over point masses.
        last_element = j == n_elements - 1
        kcu_element = separate_kcu_mass and j == n_elements - 2

        # Determine kinematics at point mass j.
        vj = ca.cross(omega, positions[j+1, :].T)
        velocities[j+1, :] = vj
        aj = ca.cross(omega, vj)
        accelerations[j+1, :] = aj
        delta_p = positions[j+1, :] - positions[j, :]
        ej = delta_p.T/ca.norm_2(delta_p)  # Axial direction of tether element

        # Determine flow at point mass j.
        vaj = vj - vw  # Apparent wind velocity
        vajp = ca.dot(vaj, ej)*ej  # Parallel to tether element
        # TODO: check whether to use vajn
        vajn = vaj - vajp  # Perpendicular to tether element
        vw = wvel*positions[j+1,2]/r_kite[2]*wdir # Wind

        vaj_sq = ca.norm_2(vaj)*vaj
        # vaj_sq = ca.norm_2(vajn)*vajn
        tether_drag_basis = rho*l_unstrained*d_t*cd_t*vaj_sq

        # Determine drag at point mass j.
        if not separate_kcu_mass:
            if n_elements == 1:
                dj = -.125*tether_drag_basis
            elif last_element:
                dj = -.25*tether_drag_basis  # TODO: add bridle drag
            else:
                dj = -.5*tether_drag_basis
        else:
            if last_element:
                dj = -0.25*rho*L_blines*d_bridle*vaj_sq*cd_t #Bridle line drag
            elif n_elements == 1:
                dj = -.125*tether_drag_basis
                dj += -.5*rho*vaj_sq*cd_kcu*frontal_area_kcu  # Adding kcu drag
            elif kcu_element:
                dj = -.25*tether_drag_basis
                dj += -.5*rho*vaj_sq*cd_kcu*frontal_area_kcu  # Adding kcu drag
                # dj += -0.25*rho*L_blines*d_bridle*vaj_sq*cd_t
            else:
                dj = -.5*tether_drag_basis

        if not separate_kcu_mass:
            if last_element:
                point_mass = m_s/2 + m_kite + m_kcu
            else:
                point_mass = m_s
        else:
            if last_element:
                point_mass = m_kite
            elif kcu_element:
                point_mass = m_s/2 + m_kcu
            else:
                point_mass = m_s

        # Use force balance to infer tension on next element.
        fgj = ca.vertcat(0, 0, -point_mass*g)
        next_tension = point_mass*aj + tensions[j, :].T - fgj - dj  # a_kite gave better fit
        if not last_element:
            tensions[j+1, :] = next_tension
            non_conservative_forces[j, :] = dj
        else:
            aerodynamic_force = next_tension
            non_conservative_forces[j, :] = dj + aerodynamic_force

    return aerodynamic_force, tensions[-1,:].T

def get_tether_end_position(x, set_parameter, n_tether_elements, r_kite, v_kite, vw,a_kite, separate_kcu_mass=True, elastic_elements=False, ax_plot_forces=False, return_values=False, find_force=False):
    # Currently neglecting radial velocity of kite.
    if find_force:
        beta_n, phi_n, tension_ground = x
        tether_length = set_parameter
    else:
        beta_n, phi_n, tether_length = x
        tension_ground = set_parameter

    l_unstrained = tether_length/n_tether_elements
    m_s = np.pi*d_t**2/4 * l_unstrained * rho_t

    n_elements = n_tether_elements
    if separate_kcu_mass:
        n_elements += 1
    
    wvel = np.linalg.norm(vw)
    wdir = vw/wvel

    vtau_kite = project_onto_plane(v_kite,r_kite/np.linalg.norm(r_kite)) # Velocity projected onto the tangent plane
    omega_tether = np.cross(r_kite,vtau_kite)/(np.linalg.norm(r_kite)**2) # Tether angular velocity, with respect to the tether attachment point


    at = np.dot(a_kite,np.array(v_kite)/np.linalg.norm(v_kite))*np.array(v_kite)/np.linalg.norm(v_kite)
    omega_kite = np.cross(a_kite-at,v_kite)/(np.linalg.norm(v_kite)**2)
    ICR = np.cross(v_kite,omega_kite)/(np.linalg.norm(omega_kite)**2)      
    alpha = np.cross(at,ICR)/np.linalg.norm(ICR)**2


        

    tensions = np.zeros((n_elements, 3))
    tensions[0, 0] = np.cos(beta_n)*np.cos(phi_n)*tension_ground
    tensions[0, 1] = np.cos(beta_n)*np.sin(phi_n)*tension_ground
    tensions[0, 2] = np.sin(beta_n)*tension_ground

    positions = np.zeros((n_elements+1, 3))
    if elastic_elements:
        l_s = (tension_ground/tether_stiffness+1)*l_unstrained
        # print((tension_ground/tether_stiffness+1))
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
        kcu_element = separate_kcu_mass and j == n_elements - 2

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
            vkcu = v_kite + np.cross(omega_kite,positions[j+1, :]-r_kite)
            vj = vkcu
            velocities[j+1, :] = vj
            akcu = a_kite+ np.cross(alpha,positions[j+1, :]-r_kite) +np.cross(omega_kite,np.cross(omega_kite,positions[j+1, :]-r_kite))
            aj = akcu
            accelerations[j+1, :] = aj

            ej = (r_kite-np.array(positions[j+1, :]))/np.linalg.norm(r_kite-np.array(positions[j+1, :]))
        # Determine flow at point mass j.
        vaj = vj - vwj  # Apparent wind velocity
        
        vajp = np.dot(vaj, ej)*ej  # Parallel to tether element
        # TODO: check whether to use vajn
        vajn = vaj - vajp  # Perpendicular to tether element

        vaj_sq = np.linalg.norm(vaj)*vaj
        # vaj_sq = np.linalg.norm(vajn)*vajn
        tether_drag_basis = rho*l_unstrained*d_t*cd_t*vaj_sq
        # Determine drag at point mass j.
        if not separate_kcu_mass:
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
                dj = -.125*tether_drag_basis
                dj += -.5*rho*np.linalg.norm(vajp)*vajp*cdp*Ap  # Adding kcu drag perpendicular to kcu
                dj += -.5*rho*np.linalg.norm(vajn)*vajn*cdt*At  # Adding kcu drag parallel to kcu
            elif kcu_element:
                dj = -.25*tether_drag_basis
                dp= -.5*rho*np.linalg.norm(vajp)*vajp*cdp*Ap  # Adding kcu drag perpendicular to kcu
                dt= -.5*rho*np.linalg.norm(vajn)*vajn*cdt*At  # Adding kcu drag parallel to kcu
                dj += dp+dt
                cd_kcu = (np.linalg.norm(dp+dt))/(0.5*rho*A_kite*np.linalg.norm(vaj)**2)
            else:
                dj = -.5*tether_drag_basis

        if not separate_kcu_mass:
            if last_element:
                point_mass = m_s/2 + m_kite + m_kcu            
            else:
                point_mass = m_s
        else:
            if last_element:
                point_mass = m_kite
                
                # aj = np.zeros(3)
            elif kcu_element:
                point_mass = m_s/2 + m_kcu
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

        if ax_plot_forces and not last_element:
            forces = [point_mass*aj, dj, fgj, -tensions[j, :], next_tension]
            labels = ['resultant', 'drag', 'weight', 'last tension', 'next tension']
            clrs = ['m', 'r', 'k', 'g', 'b']
            for f, lbl, clr in zip(forces, labels, clrs):
                # print("{} = {:.2f} N".format(lbl, np.linalg.norm(f)))
                plot_vector(positions[j+1, :], f, ax_plot_forces, color=clr)

        # Derive position of next point mass from former tension
        if kcu_element:
            # tensions[j+1,:] = np.linalg.norm(tensions[j+1, :])*z_kite
            positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * l_bridle
            
        elif not last_element:
            if elastic_elements:
                l_s = (np.linalg.norm(tensions[j+1, :])/tether_stiffness+1)*l_unstrained
            else:
                l_s = l_unstrained
            stretched_tether_length += l_s
            positions[j+2, :] = positions[j+1, :] + tensions[j+1, :]/np.linalg.norm(tensions[j+1, :]) * l_s

    if return_values == 2:
        return positions, velocities, accelerations, tensions, aerodynamic_force, non_conservative_forces
    elif return_values:
        va = vwj-vj  # All y-axes are defined perpendicular to apparent wind velocity.

        ez_bridle = tensions[-1, :]/np.linalg.norm(tensions[-1, :])
        ey_bridle = np.cross(ez_bridle, va)/np.linalg.norm(np.cross(ez_bridle, va))
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
        
        D = np.dot(va/np.linalg.norm(va),aerodynamic_force)*va/np.linalg.norm(va)        
        L = aerodynamic_force-D
        CD = np.linalg.norm(D)/(0.5*rho*A_kite*np.linalg.norm(va)**2)
        CL = np.linalg.norm(L)/(0.5*rho*A_kite*np.linalg.norm(va)**2)
        return positions, stretched_tether_length, dcm_b2w, dcm_t2w, dcm_fa2w, \
               dcm_tau2w, aerodynamic_force, va,tensions[-1,:], dcm_t2w[:,1],cd_kcu, CL,CD
    else:
        return (positions[-1, :] - r_kite)
    


def state_noise_matrices(x,u,ts): 
    
    r = x[0:3]
    v = x[3:6]
    uf = x[6]
    wdir = x[7]
    wvel = uf/kappa*ca.log(r[2]/z0)
    
    vw = ca.vertcat(wvel*ca.cos(wdir),wvel*ca.sin(wdir),0)
    Ft = u
    va = vw - v 
    va_mod = ca.sqrt(ca.dot(va,va))
    Ft_mod = ca.sqrt(ca.dot(Ft,Ft))

    dir_D = va/va_mod
    dir_L = Ft/Ft_mod - ca.dot(Ft/Ft_mod,dir_D)*dir_D
    dir_S = ca.cross(dir_L,dir_D) 

    L = x[8]*0.5*rho*A_kite*va_mod**2*dir_L
    D = x[9]*0.5*rho*A_kite*va*va_mod
    S = x[10]*0.5*rho*A_kite*va_mod**2*dir_S

    Fg = ca.vertcat(0, 0, -m_kite*g)
    rp = v
    vp = (-Ft+L+D+S+Fg)/m_kite


    
    # CLn = Lmod/(0.5*rho*A_kite*va_mod**2)
    # CDn = Dmod/(0.5*rho*A_kite*va_mod**2)
    # CSn = Smod/(0.5*rho*A_kite*va_mod**2)

    fx = ca.vertcat(rp,vp,0,0,0,0,0)
    Fx = ca.jacobian(fx, x)    
    calc_fx = ca.Function('calc_fx',[x,u,ts],[fx])
    calc_Fx = ca.Function('calc_Fx',[x,u,ts],[Fx])
    noise_vector = ca.vertcat(u,x[8],x[9],x[10])
    G = ca.jacobian(fx,noise_vector)
    calc_G = ca.Function('calc_G',[x,u,ts],[G])
    
    return fx,calc_Fx,calc_G
    
def observation_matrices(x,u):
    
    r = x[0:3]
    v = x[3:6]
    uf = x[6]
    wdir = x[7]
    wvel = uf/kappa*ca.log(r[2]/z0)
    
    vw = ca.vertcat(wvel*ca.cos(wdir),wvel*ca.sin(wdir),0)

    va = vw - v
    va_mod = ca.norm_2(va)
    
    # Fa,Ft = get_forces(x, u)
    # ez_bridle = Ft/ca.norm_2(Ft)
    # ey_bridle = ca.cross(ez_bridle, va)/ca.norm_2(ca.cross(ez_bridle, va))
    # ex_bridle = ca.cross(ey_bridle, ez_bridle)
    # va_proj = project_onto_plane_sym(va, ey_bridle)
    # aoa = calculate_angle_sym(ex_bridle,va_proj)
    # vax = ca.dot(va,ex_bridle)

    h = ca.SX.sym('h', 12) # Observation vector
    h[0] = x[0] # Position
    h[1] = x[1]
    h[2] = x[2]
    h[3] = x[3] # Velocity
    h[4] = x[4]
    h[5] = x[5]

    h[6] = x[6]
    h[7] = 0#va_mod#ca.sqrt(x[6]**2+x[7]**2)*ca.log(10/z0)/ca.log(x[2]/z0) # Wind
    h[8] = 0#ca.norm_2(vw)/ca.log(x[2]/z0)
    
    Ft = u
    va = vw - v 
    va_mod = ca.sqrt(ca.dot(va,va))
    r_mod = ca.sqrt(ca.dot(r,r))

    dir_D = va/va_mod
    dir_L = r/r_mod - ca.dot(r/r_mod,dir_D)*dir_D
    dir_S = ca.cross(dir_L,dir_D) 

    L = x[8]*0.5*rho*A_kite*va_mod**2*dir_L
    D = x[9]*0.5*rho*A_kite*va*va_mod
    S = x[10]*0.5*rho*A_kite*va_mod**2*dir_S

    Fg = ca.vertcat(0, 0, -m_kite*g)
    h[9:12] = (-Ft+L+D+S+Fg)/m_kite

    
    calc_hx = ca.Function('calc_h',[x,u],[h])
    
    Hx = ca.jacobian(h,x)
    calc_Hx = ca.Function('calc_Hx', [x,u],[Hx])
    
    return calc_hx,calc_Hx

def rotate_vector(v, u, theta):
    # Normalize vectors
    v = v 
    u = u / ca.norm_2(u)

    cos_theta = ca.cos(theta)
    sin_theta = ca.sin(theta)

    v_rot = v * cos_theta + ca.cross(u, v) * sin_theta + u * ca.dot(u, v) * (1 - cos_theta)

    return v_rot    

def calculate_angle(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)
    
    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def calculate_angle_sym(vector_a, vector_b):
    dot_product = ca.dot(vector_a, vector_b)
    magnitude_a = ca.norm_2(vector_a)
    magnitude_b = ca.norm_2(vector_b)
    
    cos_theta = dot_product / (magnitude_a * magnitude_b)
    angle_rad = ca.arccos(cos_theta)
    angle_deg = angle_rad*180/np.pi
    
    return angle_deg

def rank_observability_matrix(A,C):
    # Construct the observability matrix O_numeric
    n = A.shape[1]  # Number of state variables
    m = C.shape[0]  # Number of measurements
    O = np.zeros((m * n, n))
    
    for i in range(n):
        power_of_A = np.linalg.matrix_power(A, i)
        O[i * m: (i + 1) * m, :] = C @ power_of_A

    # Compute the rank of O using NumPy
    rank_O = np.linalg.matrix_rank(O)
    return rank_O

def R_EG_Body(Roll,Pitch,Yaw):#!!In radians!!
    
    #Rotational matrix for Roll
    R_Roll=np.array([[1, 0, 0],[0,np.cos(Roll),np.sin(Roll)],[0,-np.sin(Roll),np.cos(Roll)]])#OK checked with Blender
    
    #Rotational matrix for Pitch
    R_Pitch=np.array([[np.cos(Pitch), 0, np.sin(Pitch)],[0,1,0],[-np.sin(Pitch), 0, np.cos(Pitch)]])#Checked with blender
    
    #Rotational matrix for Roll
    R_Yaw= np.array([[np.cos(Yaw),-np.sin(Yaw),0],[np.sin(Yaw),np.cos(Yaw),0],[0,0,1]])#Checked with Blender
    
    #Total Rotational Matrix
    return R_Roll.dot(R_Pitch.dot(R_Yaw))