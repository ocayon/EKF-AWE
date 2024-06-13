

from examples.run_from_csv import year, month, day, kite_model
from awes_ekf.plotting.plot_utils import plot_kite_reference_frame
from awes_ekf.load_data.read_data import read_results
import numpy as np


# Load results and flight data and plot kite reference frame
results,flight_data = read_results(year, month, day, kite_model)
mask = (flight_data.cycle == 5)
imus = []
plot_kite_reference_frame(results[mask], flight_data[mask], imus = [])

