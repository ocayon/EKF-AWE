# Extended Kalman Filtering

This script processes flight data and applies an Extended Kalman Filter (EKF) to estimate the state of a kite system. The code is written in Python and utilizes the pandas library for data handling, casadi for symbolic operations and NumPy for numerical operations.

## Project Structure

The project structure is organized as follows:

- **config.py:** Configuration file containing model parameters and settings.
- **utils.py:** Utility functions 
- **run_ekf.py:** Main script for EKF implementation.

To run the script you need a data file that has been processed into the appropiate format. The script to preprocess the data can be found in ./data and the processed data in ./processed_data.

## Configuration Parameters

The `config.py` file contains the following parameters:

- `year`, `month`, `day`: Date of the flight data. 
- `kite_model`: Model of the kite. You can add a new kite by adding an entry to the `kite_models` dictionary
- `kcu_model`: Model of the Kite Control Unit. You can add a different sized cylinder in the `kcu_cylinders` dictionary.
- `tether_diameter`: Diameter of the tether.
- `tether_material`: Material of the tether. You can add a new material in `tether_materials` dictionary.
- `z0`: surface roughness of the site.
- `n_tether_elements`: Number of tether elements used by the tether model solver
- `doIEKF`: Flag for using Iterative Extended Kalman Filter (IEKF).
- `max_iterations`: Maximum number of iterations for IEKF.
- `epsilon`: Convergence criterion for IEKF.
- `measurements`: List of measurements to be used in the EKF.
- `model_stdv`, `meas_stdv`: Standard deviations for process and measurement noise, respectively.
- 

## Data Processing Steps

1. **Data Reading:**
   - Flight data is read from a CSV file located in the `processed_data/flight_data` directory.

2. **System Model Initialization:**
   - Kite, KCU, and tether objects are created based on the configuration parameters.
   - Dynamic model and observation model classes are initialized.

3. **Measurement Noise Matrix:**
   - Measurement noise matrix is calculated based on the specified measurements.

4. **Initial State Vector:**
   - The initial state vector is initialized using flight data, kite, KCU, and tether parameters.

5. **Extended Kalman Filter Initialization:**
   - The EKF is initialized with the dynamic and observation models.

6. **Propagation and Update Loop:**
   - The script iterates through the flight data, propagating the state with the dynamic model and updating it with the EKF.
   - Quasi-static tether model is used to calculate input for the next step.

7. **Results Storage:**
   - The results, including state estimates, standard deviations, predicted measurements, and other relevant information, are stored.

8. **Saving Results:**
   - The processed results are saved to CSV files in the `results` directory.

## Usage

1. Configure the parameters in `config.py` based on your specific flight data and system model.
2. Run the script `run_ekf.py`.

