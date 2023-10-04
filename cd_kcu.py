from v3_properties import *
import numpy as np
import matplotlib.pyplot as plt


inflow_angles = np.arange(0, 180, 1)
wvel = 10

plt.figure()
plt.plot(np.arange(0,5,0.1),splev(np.arange(0,5,0.1),spline_t))
plt.plot(np.arange(min(ldt_cone),max(ldt_cone),0.1),splev(np.arange(min(ldt_cone),max(ldt_cone),0.1),spline_t_cone))
plt.plot(np.arange(min(ldt_blunt),max(ldt_blunt),0.1),splev(np.arange(min(ldt_blunt),max(ldt_blunt),0.1),spline_t_blunt))
plt.scatter(ldt_data, cdt_data)
plt.scatter(ldt_cone, cdt_cone_data)
plt.scatter(ldt_blunt, cdt_blunt_data)
splev(ld, spline_t)

cd_kcu = []
cd_kcu_blunt = []   
cd_kcu_cone = []
for theta in inflow_angles:
    Dt = cdt*0.5*rho*At*(wvel*np.cos(np.deg2rad(theta)))**2
    Dt_blunt = cdt_blunt*0.5*rho*At*(wvel*np.cos(np.deg2rad(theta)))**2
    Dt_cone = cdt_cone*0.5*rho*At*(wvel*np.cos(np.deg2rad(theta)))**2
    Dp = cdp*0.5*rho*Ap*(wvel*np.sin(np.deg2rad(theta)))**2
    D_kcu = Dp + Dt
    D_kcu_blunt = Dp + Dt_blunt
    D_kcu_cone = Dp + Dt_cone
    cd_kcu.append(D_kcu/(0.5*rho*A_kite*wvel**2))
    cd_kcu_blunt.append(D_kcu_blunt/(0.5*rho*A_kite*wvel**2))   
    cd_kcu_cone.append(D_kcu_cone/(0.5*rho*A_kite*wvel**2))  


# data_NACA0015 = np.loadtxt('cd_NACA0015.csv', delimiter=',')


plt.figure()
plt.plot(inflow_angles, cd_kcu)
plt.plot(inflow_angles, cd_kcu_blunt)
plt.plot(inflow_angles, cd_kcu_cone)
# plt.plot(data_NACA0015[:,0], data_NACA0015[:,1])
plt.xlabel('Inflow angle [deg]')
plt.ylabel('Drag coefficient [-]')
plt.grid()
plt.show()
