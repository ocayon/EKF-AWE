## Configuration layout (AWETrim convention)

Each kite has a folder `data/<KITE-NAME>/` holding

- `ekf_config.yaml` — simulation and tuning parameters, and
- one or more awesIO-validated `system*.yaml` files with the physical
  properties of the hardware. A kite can have several system variants
  depending on what it was flown with — e.g. `LEI-V3-KITE/` carries
  `system_flown_2019.yaml` (22.75 kg KCU, 10 mm tether) and
  `system_flown_2025.yaml` (23.3 kg KCU, 13.5 mm tether); when a folder
  holds several, `load_config` asks which hardware the EKF should assume.

`awes_ekf.setup.settings.load_config()` prompts for the folder, merges the
two files, and extracts the `kite`/`kcu`/`tether` blocks the models consume
from the chosen system yaml. The old flat `data/config/*.yaml` files are
superseded by this layout.

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
  - The coordinates,velocities and accelerations are transformed into the East-North-Up (ENU) reference frame.

- **Ground Station Data:**
  - Tether force is converted to Newtons

- **Wind Direction:**
  - Wind direction is transformed to be measured from the east axis in a counter-clockwise direction.

