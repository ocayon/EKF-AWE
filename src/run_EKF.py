
import numpy as np
import pandas as pd
from config import kite_models, kcu_cylinders, tether_materials,kite_model, kcu_model, tether_diameter, tether_material, year, month, day, \
                     doIEKF, max_iterations, epsilon, opt_measurements, stdv_x, stdv_y, n_tether_elements, z0
from utils import calculate_vw_loglaw, calculate_euler_from_reference_frame, calculate_airflow_angles, ModelSpecs, SystemSpecs
from utils import  KiteModel, KCUModel, EKFInput, find_initial_state_vector, get_measurement_vector, tether_input
from tether_model import TetherModel
from kalman_filter import ExtendedKalmanFilter, DynamicModel, ObservationModel, observability_Lie_method
import time

def create_input_from_KP_csv(flight_data, system_specs, kite_sensor = 0, kcu_sensor = None):
    """Create input classes and initial state vector from flight data"""
    n_intervals = len(flight_data)
    # Kite measurements
    kite_pos = np.array([flight_data['kite_'+str(kite_sensor)+'_rx'],flight_data['kite_'+str(kite_sensor)+'_ry'],flight_data['kite_'+str(kite_sensor)+'_rz']]).T
    kite_vel = np.array([flight_data['kite_'+str(kite_sensor)+'_vx'],flight_data['kite_'+str(kite_sensor)+'_vy'],flight_data['kite_'+str(kite_sensor)+'_vz']]).T
    kite_acc = np.array([flight_data['kite_'+str(kite_sensor)+'_ax'],flight_data['kite_'+str(kite_sensor)+'_ay'],flight_data['kite_'+str(kite_sensor)+'_az']]).T
    # KCU measurements
    if kcu_sensor is not None:
        kcu_vel = np.array([flight_data['kite_'+str(kcu_sensor)+'_vx'],flight_data['kite_'+str(kcu_sensor)+'_vy'],flight_data['kite_'+str(kcu_sensor)+'_vz']]).T
        kcu_acc = np.array([flight_data['kite_'+str(kcu_sensor)+'_ax'],flight_data['kite_'+str(kcu_sensor)+'_ay'],flight_data['kite_'+str(kcu_sensor)+'_az']]).T
    else:
        kcu_vel = np.zeros((n_intervals,3))
        kcu_acc = np.zeros((n_intervals,3))
    # Tether measurements
    tether_force = np.array(flight_data['ground_tether_force'])
    tether_length = np.array(flight_data['ground_tether_length'])
      
    # Airflow measurements
    ground_windspeed = np.array(flight_data['ground_wind_velocity'])
    ground_winddir = np.array(flight_data['ground_wind_direction'])
    apparent_windspeed = np.array(flight_data['kite_apparent_windspeed'])
    kite_aoa = np.array(flight_data['kite_angle_of_attack'])
    
    timestep = flight_data['time'].iloc[1]-flight_data['time'].iloc[0]
    ekf_input_list = []
    for i in range(len(flight_data)):
        ekf_input_list.append(EKFInput(kite_pos = kite_pos[i], 
                                    kite_vel = kite_vel[i], 
                                    kite_acc = kite_acc[i], 
                                    tether_force = tether_force[i],
                                    apparent_windspeed = apparent_windspeed[i], 
                                    tether_length = tether_length[i],
                                    kite_aoa = kite_aoa[i], 
                                    kcu_vel = kcu_vel[i], 
                                    kcu_acc = kcu_acc[i]))
                

    kite = create_kite(system_specs.kite_model)
    kcu = create_kcu(system_specs.kcu_model)
    tether = create_tether(system_specs.tether_material,system_specs.tether_diameter)

    x0, u0 = find_initial_state_vector(kite_pos[0], kite_vel[0], kite_acc[0], 
                                       np.mean(ground_winddir[0:3000])/180*np.pi, np.mean(ground_windspeed[0]), tether_force[0], 
                                       tether_length[i], n_tether_elements, kite, kcu,tether)

    return ekf_input_list, x0

def create_kite(model_name):
    """"Create kite model class from model name and model dictionary"""
    if model_name in kite_models:
        model_params = kite_models[model_name]
        return KiteModel(model_name, model_params["mass"], model_params["area"], model_params["distance_kcu_kite"],
                     model_params["total_length_bridle_lines"], model_params["diameter_bridle_lines"],model_params['KCU'])
    else:
        raise ValueError("Invalid kite model")
    
def create_kcu(model_name):
    """"Create KCU model class from model name and model dictionary"""
    if model_name in kcu_cylinders:
        model_params = kcu_cylinders[model_name]
        return KCUModel(model_params["length"], model_params["diameter"], model_params["mass"])
    else:
        raise ValueError("Invalid KCU model")
        
def create_tether(material_name,diameter):
    """"Create tether model class from material name and diameter"""
    if material_name in tether_materials:
        material_params = tether_materials[material_name]
        return TetherModel(material_name,diameter,material_params["density"],material_params["cd"],material_params["Youngs_modulus"])
    else:
        raise ValueError("Invalid tether material")

def run_EKF(ekf_input_list, model_specs, system_specs,x0):
    """Run the Extended Kalman Filter
    Args:
        ekf_input_list: list of EKFInput classes
        model_specs: ModelSpecs class
        system_specs: SystemSpecs class
        x0: initial state vector
    Returns:
        df: DataFrame with the results
    """
    kite = create_kite(system_specs.kite_model)
    kcu = create_kcu(system_specs.kcu_model)
    tether = create_tether(system_specs.tether_material,system_specs.tether_diameter)
   
    # Create dynamic model and observation model
    dyn_model = DynamicModel(kite,model_specs.ts)
    obs_model = ObservationModel(dyn_model.x,dyn_model.u,model_specs.opt_measurements,kite)

    # Initialize EKF
    ekf = ExtendedKalmanFilter(system_specs.stdv_dynamic_model, system_specs.stdv_measurements, model_specs.ts,dyn_model,obs_model, model_specs.doIEKF, model_specs.epsilon, model_specs.max_iterations)
    ekf_input = ekf_input_list[0]
    # Initial measurement vector
    meas0 = get_measurement_vector(ekf_input,model_specs.opt_measurements)
    kite_acc, fti, lti, kcu_acc, kcu_vel = tether_input(ekf_input,model_specs)
    tether.solve_tether_shape(n_tether_elements, x0[0:3], x0[3:6], calculate_vw_loglaw(x0[6], z0, x0[2], x0[7]), kite, kcu, tension_ground = fti, tether_length = lti,
                                a_kite = kite_acc, a_kcu = kcu_acc, v_kcu = kcu_vel)
    
    if model_specs.correct_height:
        x0[2] = tether.kite_pos[2]
    # Define results matrices
    n_intervals = len(ekf_input_list)
    N = n_intervals
    n = len(x0)
    nm = len(meas0)
    
    # allocate space to store traces and other Kalman filter params
    XX_k1_k1    = np.zeros([n, N])
    err_meas    = np.zeros([nm, N])
    z_k1_k1    = np.zeros([nm, N-1])
    PP_k1_k1    = np.zeros([n, N])
    STD_x_cor   = np.zeros([n, N])
    STD_z       = np.zeros([nm, N])
    ZZ_pred     = np.zeros([nm, N])

    
    # arrays for tether model and other results
    euler_angles = []
    airflow_angles = []
    tether_length = []
    Ft = []             # Tether force

    # Store Initial values
    XX_k1_k1[:,0]   = x0
    ZZ_pred[:,0]    = z0
    
    # Initial conditions
    x_k1_k1 = x0
    u = tether.Ft_kite
     
    start_time = time.time()
    mins = -1
    
    for k in range(n_intervals):
        ekf_input = ekf_input_list[k]
        zi = get_measurement_vector(ekf_input,model_specs.opt_measurements)
        if model_specs.correct_height:
            zi[2] = tether.kite_pos[2]
        ############################################################
        # Propagate state with dynamic model
        ############################################################
        x_k1_k = dyn_model.propagate(x_k1_k1,u)

        ############################################################
        # Update state with Kalmann filter
        ############################################################
        ekf.initialize(x_k1_k,u,zi)
        # Predict next step
        ekf.predict()
        # Update next step
        ekf.update()
        x_k1_k1 = ekf.x_k1_k1
        
        ############################################################
        # Calculate Input for next step with quasi-static tether model
        ############################################################
        r_kite = x_k1_k1[:3]
        v_kite = x_k1_k1[3:6]
        vw = calculate_vw_loglaw(x_k1_k1[6], z0, x_k1_k1[2], x_k1_k1[7])
        
        kite_acc, fti, lti, kcu_acc, kcu_vel = tether_input(ekf_input,model_specs)
        tether.solve_tether_shape(n_tether_elements, r_kite, v_kite, vw, kite, kcu, tension_ground = fti, tether_length = lti,
                                a_kite = kite_acc, a_kcu = kcu_acc, v_kcu = kcu_vel)
        u = tether.Ft_kite
        ############################################################
        # Store results
        ############################################################
        XX_k1_k1[:,k] = np.array(x_k1_k1).reshape(-1)
        STD_x_cor[:,k] = ekf.std_x_cor
        STD_z[:,k] = ekf.std_z
        ZZ_pred [:,k] = ekf.z_k1_k
        err_meas[:,k] = ekf.z_k1_k - zi

        # Store tether force and tether model results
        Ft.append(u)
        euler_angles.append(calculate_euler_from_reference_frame(tether.dcm_b2w))
        tether_length.append(tether.stretched_tether_length)
        airflow_angles.append(calculate_airflow_angles(tether.dcm_b2w, v_kite, vw))

        # Print progress
        if k%600==0:
            elapsed_time = time.time() - start_time
            start_time = time.time()  # Record end time
            mins +=1
            print(f"Real time: {mins} minutes.  Elapsed time: {elapsed_time:.2f} seconds")
        
    # Store results
    ti = 0
    k +=1
    results = np.vstack((XX_k1_k1[:,ti:k],np.array(Ft)[ti:k,:].T,np.array(euler_angles)[ti:k,:].T,np.array(airflow_angles)[ti:k,:].T,np.array(tether_length)[ti:k].T))
    column_names = ['x','y','z','vx','vy','vz','uf','wdir','CL', 'CD', 'CS', 'bias_lt','bias_aoa','Ftx','Fty','Ftz','roll', 'pitch', 'yaw', 'aoa','ss','tether_len']
    df = pd.DataFrame(data=results.T, columns=column_names)

    return df

#%% Read and process data 
if __name__ == "__main__":
    # File path
    file_name = kite_model+'_'+year+'-'+month+'-'+day
    file_path = '../processed_data/flight_data/'+kite_model+'/'+ file_name+'.csv'
    flight_data = pd.read_csv(file_path)
    flight_data = flight_data.reset_index()
    flight_data = flight_data.iloc[:18000]

    timestep = flight_data['time'].iloc[1] - flight_data['time'].iloc[0]

    model_specs = ModelSpecs(timestep, n_tether_elements, opt_measurements=opt_measurements, correct_height=False)
    system_specs = SystemSpecs(kite_model, kcu_model, tether_material, tether_diameter, stdv_x, stdv_y)
    # Create input classes
    ekf_input_list,x0 = create_input_from_KP_csv(flight_data, system_specs, kite_sensor = 0, kcu_sensor = 1)

    # Check observability matrix
    # check_obs = False
    # if check_obs == True:
    #     observability_Lie_method(dyn_model.fx,obs_model.hx,dyn_model.x, dyn_model.u, ekf_input.x0,ekf_input.u0)

    #%% Main loop
    ekf_output = run_EKF(ekf_input_list, model_specs, system_specs,x0)

    save_results = True
    if save_results == True:
        #%% Save results
        addition = ''
        path = '../results/'+kite_model+'/'
        # Save the DataFrame to a CSV file
        csv_filename = file_name+'_res_GPS'+addition+'.csv'
        ekf_output.to_csv(path+csv_filename, index=False)

        # Save the DataFrame to a CSV file
        csv_filename = file_name+'_fd.csv'
        flight_data.to_csv(path+csv_filename, index=False)

    