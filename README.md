# Extended Kalman Filtering for Airborne Wind Energy Systems

This repository provides tools to process flight data and apply an Extended Kalman Filter (EKF) to estimate the state of a kite system in Airborne Wind Energy Systems (AWES). The code is developed in Python, utilizing libraries like `pandas` for data handling, `casadi` for symbolic operations, and `NumPy` for numerical computations.

---

## Installation Instructions

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ocayon/EKF-AWE
    ```

2. **Navigate to the repository folder**:
    ```bash
    cd EKF-AWE
    ```

3. **Create a virtual environment**:

   - **Linux or Mac**:
     ```bash
     python3 -m venv venv
     ```
   - **Windows**:
     ```bash
     python -m venv venv
     ```

4. **Activate the virtual environment**:

   - **Linux or Mac**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```

5. **Install the required dependencies**:

   - For users:
     ```bash
     pip install .
     ```
   - For developers:
     ```bash
     pip install -e .[dev]
     ```

6. **To deactivate the virtual environment**:
    ```bash
    deactivate
    ```

---

## Dependencies

- `numpy`
- `matplotlib`
- `casadi>=3.6.0`
- `pandas`
- `pytest`
- `pyyaml`
- `control`
- `scipy`
- `seaborn`
- `dataclasses`
- `h5py`

---
## Example Usage

An example dataset is provided for the LEI V3 kite from Kitepower, flown on `2019-10-08`.
- **Flight Data**: `2019-10-08_11.csv` in `data\flight_logs\v3`
- **Configuration File**: `v3_config.yaml` in `data\config`
- **Processing Script**: `process_v3_data.py` in `data\data_postprocessors`


To analyze this dataset:
   - Run `run_analysis.py` and select the folder with `2019-10-08_11.csv`, `v3_config.yaml` and `process_v3_data.py`.
```bash
python examples\run_analysis.py
```
   - Run `plot_analysis.py` to visualize the results.
```bash
python examples\plot_analysis.py
```

## New Dataset Analysis

To analyze an AWES dataset, follow these steps:

### 1. Add Necessary Files
   - **Configuration File**: Add a `.yaml` configuration file for the kite you want to analyze in `data/config`.
   - **Processing Script**: Add a data processing script to `data/data_postprocessors` that processes raw flight data and saves it to `postprocess_data`, following the naming conventions in `doc/inputs`.
   - **Flight Data**: Create a folder in `data` and add the raw flight data. File names should start with the format `YYYY-MM-DD_HH` (e.g., `2019-10-08_11.csv`).

### 2. Run Analysis
   - Run `run_analysis.py` from the `examples` folder.
   - When prompted, select the configuration file, processing script, and flight data file.

### 3. Plot Data
   - Run `plot_analysis.py` and select the processed flight data file for visualization.


---

## Contributing Guide

Contributions are welcome! Here’s how you can contribute:

1. **Create an issue** on GitHub for any bugs or feature requests.
2. **Create a branch** from this issue:
    ```bash
    git checkout -b issue_number-new-feature
    ```
3. Implement your feature or fix.
4. Verify functionality using **pytest**:
    ```bash
    pytest
    ```
5. **Commit your changes** with a descriptive message:
    ```bash
    git commit -m "#<issue_number> <message>"
    ```
6. **Push your changes** to GitHub:
    ```bash
    git push origin branch-name
    ```
7. **Create a pull request** with `base:develop` to merge your feature branch.
8. Once the pull request is accepted, **close the issue**.

---

## Citation

If you use this project in your research, please consider citing it. Citation details can be found in the [CITATION.cff](CITATION.cff) file.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

### Copyright
  
&copy; 2024 Oriol Cayon, TU Delft  

Prof. Dr. H.G.C. (Henri) Werij, Dean of Faculty of Aerospace Engineering
