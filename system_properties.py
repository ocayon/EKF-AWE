import numpy as np
from scipy.interpolate import splrep, splev
class KiteModel:
    def __init__(self, model):
        self.model = model
        if model == 'v3':
            self.mass = 15
            self.area = 19.75
            self.distance_kcu_kite = 11.5
            self.total_length_bridle_lines = 96
            self.diameter_bridle_lines = 2.5e-3
        elif model == 'v9':
            self.mass = 62
            self.area = 46.854
            self.distance_kcu_kite = 15.45
            self.total_length_bridle_lines = 300
            self.diameter_bridle_lines = 4e-3 
class Tether:
    def __init__(self,diameter):
        self.diameter = diameter
        self.density = 970.
        self.material = 'Dyneema'
        self.cd_t = 1.1
        self.tether_modulus = 614600/(np.pi*.002**2)  # From Uwe's thesis
        self.tether_stiffness = self.tether_modulus*np.pi*(self.d_t/2)**2

class KCU_cylinder:

    # Exracted from Applied fluid dynamics handbook
    ldt_data = np.array([0,0.5,1.0,1.5,2.0,3.0,4.0,5.0])  # L/D values
    cdt_data = np.array([1.15,1.1,0.93,0.85,0.83,0.85,0.85,0.85])  # Cd values for tangential flow
    ldp_data = np.array([1,1.98,2.96,5,10,20,40,1e6])  # L/D values
    cdp_data = np.array([0.64,0.68,0.74,0.74,0.82,0.91,0.98,1.2])  # Cd values perpendicular flow

    cdt_cone_data= np.array([0.43, 0.35, 0.22, 0.19, 0.20, 0.21, 0.22, 0.24])
    ldt_cone = np.array([0.01, 0.64, 1.98, 4.57, 8.91, 12.02, 13.69, 15.29])
    

    def __init__(self,length,diameter,mass):
        self.length = 1
        self.diameter = 0.48
        self.mass = 18+1.6+8
        # Create spline interpolations
        spline_t = splrep(ldt_data, cdt_data, s=0)
        spline_p = splrep(ldp_data, cdp_data, s=0)

        # Example: Interpolate Cd for tangential flow at a specific L/D
        self.cdt = splev(self.length/self.diameter, spline_t)
        self.cdp = splev(self.length/self.diameter, spline_p)

        self.At = np.pi*(d_kcu/2)**2  # Calculate area of the KCU
        self.Ap = l_kcu*d_kcu  # Calculate area of the KCU



