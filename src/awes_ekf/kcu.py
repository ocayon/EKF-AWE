from model_definitions import kcu_cylinders
import numpy as np
from scipy.interpolate import splrep, splev

class KCU:
    # Exracted from Applied fluid dynamics handbook
    ldt_data = np.array([0,0.5,1.0,1.5,2.0,3.0,4.0,5.0])  # L/D values
    cdt_data = np.array([1.15,1.1,0.93,0.85,0.83,0.85,0.85,0.85])  # Cd values for tangential flow
    ldp_data = np.array([1,1.98,2.96,5,10,20,40,1e6])  # L/D values
    cdp_data = np.array([0.64,0.68,0.74,0.74,0.82,0.91,0.98,1.2])  # Cd values perpendicular flow

    cdt_cone_data= np.array([0.43, 0.35, 0.22, 0.19, 0.20, 0.21, 0.22, 0.24])
    ldt_cone = np.array([0.01, 0.64, 1.98, 4.57, 8.91, 12.02, 13.69, 15.29])
    # Create spline interpolations
    spline_t = splrep(ldt_data, cdt_data, s=0)
    spline_p = splrep(ldp_data, cdp_data, s=0)
    
    def __init__(self,model_name, data_available = False):
        
        if model_name in kcu_cylinders:
            model_params = kcu_cylinders[model_name]
            for key, value in model_params.items():
                # Set each key-value pair as an attribute of the instance
                setattr(self, key, value)
        else:
            raise ValueError("Invalid KCU model, add it to model_definitions.py")
        self.data_available = data_available

        # Example: Interpolate Cd for tangential flow at a specific L/D
        self.cdt = splev(self.length/self.diameter, KCU.spline_t)
        self.cdp = splev(self.length/self.diameter, KCU.spline_p)

        self.At = np.pi*(self.diameter/2)**2  # Calculate area of the KCU
        self.Ap = self.diameter*self.length  # Calculate area of the KCU
    
    def calculate_cd(self,ld):
        ##
        self.cd_kcu = 0


