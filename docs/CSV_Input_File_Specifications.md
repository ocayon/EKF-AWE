# CSV Input File Specifications

This document outlines the naming conventions for columns required and optionally used in the CSV input file. The `create_input_from_csv` function expects specific names to identify the data necessary for the EKF (Extended Kalman Filter) input.

---

## Minimum Required Columns

1. **Time**
   - `time`
   - A column representing time or intervals. Gradients in this column are used to compute time steps for the model.

2. **Kite Position**
   - `kite_position_x`, `kite_position_y`, `kite_position_z`
   - Represents the position of the kite in three-dimensional space.

3. **Kite Velocity**
   - `kite_velocity_x`, `kite_velocity_y`, `kite_velocity_z`
   - Represents the velocity of the kite along each spatial axis.

4. **Ground Tether Force**
   - `ground_tether_force`
   - Records the force exerted on the ground by the tether connecting the kite.

5. **Tether Reelout Speed**
   - `tether_reelout_speed`
   - The speed at which the tether is reeled out or taken in from the ground.

---

## Optional Columns

These columns enhance the model's performance by providing additional measurements. If you want to use any of these columns, also add them in the config file measurements.

### **KCU (Kite Control Unit) Data**

1. **KCU Velocity**
   - `kcu_velocity_x`, `kcu_velocity_y`, `kcu_velocity_z`
   - The velocity of the KCU, used for more refined motion tracking.

2. **KCU Acceleration**
   - `kcu_acceleration_x`, `kcu_acceleration_y`, `kcu_acceleration_z`
   - Acceleration of the KCU to aid in estimating forces acting on the kite.

### **Kite Measurements**

1. **Kite Acceleration**
   - `kite_acceleration_x`, `kite_acceleration_y`, `kite_acceleration_z`
   - Acceleration measurements for the kite along each axis.

2. **Kite Apparent Windspeed**
   - `kite_apparent_windspeed`
   - Apparent wind speed experienced by the kite.

3. **Kite Angle of Attack**
   - `bridle_angle_of_attack`
   - Angle of attack measurement for the kite bridle, useful for aerodynamic calculations.

4. **Kite Thrust Force**
   - `thrust_force_x`, `thrust_force_y`, `thrust_force_z`
   - Force exerted by the kite’s thrust mechanism along each axis.

5. **Kite Yaw**
   - `kite_yaw_<sensor_id>`
   - The yaw angle of the kite, typically provided by a specific sensor (sensor ID required in the name).

### **Environmental Measurements**

1. **Ground Wind Speed and Direction**
   - `ground_wind_speed`, `ground_wind_direction`
   - Ground-level wind speed and direction data; defaults to zero if missing.

2. **Initial Wind Measurements**
   - If missing, `init_wind_dir` and `init_wind_vel` are initialized based on available wind-related columns, such as:
     - `Wind Speed (m/s)`
     - `Wind Direction`
   - Ensure these columns exist if precise initial wind conditions are necessary.

### **Tether Measurements**

1. **Tether Length**
   - `tether_length`
   - Length of the tether from the kite to the ground.

2. **Tether Elevation and Azimuth Angles**
   - `tether_elevation_ground`, `tether_azimuth_ground`
   - Elevation and azimuth of the tether from the ground perspective.

---

## Control Inputs

1. **KCU Depower Input**
   - `kcu_actual_depower`
   - Level of depower control applied by the KCU, typically normalized.

2. **KCU Steering Input**
   - `kcu_actual_steering`
   - Steering control applied by the KCU, also typically normalized.

---

### Notes

- **Missing Optional Data**: When optional data columns are missing, the model will default to zero or neutral values if the configuration allows. Ensure `simConfig` is appropriately configured for any missing data.
- **Naming Conventions**: The column names in the CSV must match those specified above precisely, as they are parsed directly without modifications.
