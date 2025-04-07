# Reproducing the Results in the Paper

## 📊 Plot Scripts

All the plots used in the paper can be generated using the scripts located in:

~~~
examples/plots_paper/
~~~

These scripts rely on simulation outputs created using the EKF-based analysis pipeline described below.

---

## 🛠️ Data Processing Workflow

To process flight logs and reproduce the results:

### 1. Edit the Configuration

Open the configuration file:

~~~
data/config/v3_config.yaml
~~~

Modify the measurement settings under `measurements` to ensure that at least the following are enabled:

~~~yaml
measurements:
  kite_position: true
  kite_velocity: true
  kite_acceleration: true
~~~

You can enable additional sensors to match the setup described in the paper, depending on the figure or analysis you're reproducing.

---

### 2. Run the Analysis Script

Use the `run_analysis.py` script in the `examples` folder to analyze the selected flight:

~~~
python examples/run_analysis.py
~~~

During execution, you will be prompted to:

- Select the flight log to analyze
- Choose a pre-processing script
- Optionally filter the flight by time
- Provide an **identifier** (e.g. `_va`, `_tetherlength`, etc.) to distinguish different simulation setups in the output folder

---

## 📈 Plotting Simulations

After processing, the results can be plotted in two ways:

- **To plot a single simulation**, use:

~~~
python examples/plot_analysis.py
~~~

- **To recreate the plots from the paper**, use the scripts in:

~~~
examples/plots_paper/
~~~

These scripts reproduce the specific figures and comparisons shown in the paper.

---

## 📁 Folder Structure Overview

~~~
examples/
├── run_analysis.py        # Main script to run EKF and store results
├── plot_analysis.py       # General plotting tool for simulation results
└── plots_paper/           # Scripts to generate each figure from the paper

data/
├── config/
│   └── v3_config.yaml     # Configuration file (edit measurements here)
└── flight_logs/           # Raw input flight logs
~~~

