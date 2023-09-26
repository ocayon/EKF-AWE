from numpy import pi


# rho = 1.2
# g = 9.81
# d_t = .01
# rho_t = 724.
# cd_t = 1.2
# tether_modulus = 614600/(pi*.002**2)  # From Uwe's thesis
# tether_stiffness = tether_modulus*pi*(d_t/2)**2

# m_kite = 14.2
# m_kcu = 25
# l_bridle = 11.5

# A_kite = 19.75

# cd_kcu = 1
# frontal_area_kcu = 0.5

# z0 =    0.01

rho = 1.2
g = 9.81
d_t = .008
rho_t = 970.
cd_t = 1.1
tether_modulus = 614600/(pi*.002**2)  # From Uwe's thesis
tether_stiffness = tether_modulus*pi*(d_t/2)**2

m_kite = 15
m_kcu = 30
l_bridle = 11.5

A_kite = 19.75

cd_kcu = 0.6
frontal_area_kcu = 0.3

L_blines = 96
d_bridle = 2.5e-3

z0 =    0.1
kappa = 0.4