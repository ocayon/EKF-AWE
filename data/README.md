## Data Processing Steps

1. **Loading Data:**
   - The raw data is loaded from CSV files using the pandas library.

2. **Data Filtering:**
   - Flight data is filtered to select instances where the kite is flying, specifically when the kite height is above 30 meters.

3. **Interpolation:**
   - Missing data is interpolated to ensure a continuous dataset.

4. **Transforming data to new reference frames**
    - A new DataFrame (`flight_data`) is created to store relevant information extracted from the Protologger data.

5.  **Saving Processed Data:**
    - The processed flight data is saved as a CSV file in the `processed_data/flight_data` directory.

## Reference Systems Changes

- **GPS & IMU Data:**
  - The coordinates (`kite_0_rx`, `kite_0_ry`, `kite_0_rz`) are transformed into the East-North-Up (ENU) reference frame.

- **Ground Station Data:**
  - Tether force is converted to Newtons

- **Wind Direction:**
  - Wind direction is transformed to be measured from the east axis in a counter-clockwise direction.

