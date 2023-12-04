import numpy as np
from scipy.interpolate import splrep, splev

rho = 1.2
g = 9.81
d_t = .014
rho_t = 724.
cd_t = 1.1
tether_modulus = 614600/(np.pi*.002**2)  # From Uwe's thesis
tether_stiffness = tether_modulus*np.pi*(d_t/2)**2

m_kite = 62
l_bridle = 15.45

A_kite = 46.854

L_blines = 300
d_bridle = 4e-3

z0 =    0.1
kappa = 0.4

KCU = 2
# Dimensions KCU
if KCU == 1:
    l_kcu = 1
    d_kcu = 0.48
    m_kcu = 18+1.6+8
elif KCU == 2:
    l_kcu = 1.2
    d_kcu = 0.62
    m_kcu = 18+1.6+12 

ld = l_kcu/d_kcu  # Calculate L/D for the KCU

# Exracted from Applied fluid dynamics handbook
ldt_data = np.array([0,0.5,1.0,1.5,2.0,3.0,4.0,5.0])  # L/D values
cdt_data = np.array([1.15,1.1,0.93,0.85,0.83,0.85,0.85,0.85])  # Cd values for tangential flow
ldp_data = np.array([1,1.98,2.96,5,10,20,40,1e6])  # L/D values
cdp_data = np.array([0.64,0.68,0.74,0.74,0.82,0.91,0.98,1.2])  # Cd values perpendicular flow

cdt_blunt_data = np.array([1.16, 1.11, 0.95, 0.85, 0.83, 0.82, 0.83])
ldt_blunt = np.array([0.16, 0.69, 0.93, 3.08, 4.72, 6.44, 8.36])
cdt_cone_data= np.array([0.43, 0.35, 0.22, 0.19, 0.20, 0.21, 0.22, 0.24])
ldt_cone = np.array([0.01, 0.64, 1.98, 4.57, 8.91, 12.02, 13.69, 15.29])


# Create spline interpolations
spline_t = splrep(ldt_data, cdt_data, s=0)
spline_t_blunt = splrep(ldt_blunt, cdt_blunt_data, s=8)
spline_t_cone = splrep(ldt_cone, cdt_cone_data, s=0)
spline_p = splrep(ldp_data, cdp_data, s=0)

# Example: Interpolate Cd for tangential flow at a specific L/D
cdt = splev(ld, spline_t)
cdt_blunt= splev(ld, spline_t_blunt)
cdt_cone = splev(ld, spline_t_cone)
cdp = splev(ld, spline_p)

At = np.pi*(d_kcu/2)**2  # Calculate area of the KCU
Ap = l_kcu*d_kcu  # Calculate area of the KCU