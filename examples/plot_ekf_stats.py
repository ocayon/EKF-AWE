import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, kstest, probplot
from awes_ekf.setup.settings import load_config
from awes_ekf.load_data.read_data import read_results
import awes_ekf.plotting.plot_utils as pu

# Load configuration
config_file_name = "v9_config.yaml"
config = load_config("examples/" + config_file_name)

# Load results and flight data
results, flight_data,_ = read_results(
    str(config["year"]),
    str(config["month"]),
    str(config["day"]),
    config["kite"]["model_name"],
    addition="",
)

# results = results[(flight_data["cycle"]>10)&(flight_data["cycle"]<70)]
# flight_data = flight_data[(flight_data["cycle"]>10)&(flight_data["cycle"]<70)]
# Assuming that the degrees of freedom are set to the number of independent measurements
# Here assuming 1 degree of freedom for simplicity; adjust based on actual system measurements
degrees_of_freedom = 1

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(12, 10))

# Time series plot
pu.plot_time_series(flight_data, results["nis"], ax[0])
pu.plot_time_series(flight_data, results["mahalanobis_distance"], ax[0])
pu.plot_time_series(flight_data, results["norm_epsilon_norm"], ax[0], plot_phase=True)
ax[0].legend(["NIS", "Mahalanobis Distance", "Norm of Normalized Residuals"])
ax[0].set_title("Time Series of NIS, Mahalanobis Distance, and Norm of Normalized Residuals")
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Value")

# Histogram and chi-squared PDF
ax[1].hist(results["nis"], bins=30, density=True, alpha=0.6, color='g', label='NIS Histogram')
x = np.linspace(0, np.max(results["nis"]), 1000)
ax[1].plot(x, chi2.pdf(x, df=degrees_of_freedom), 'r-', lw=2, label='Chi-Squared PDF')
ax[1].set_title("Histogram of NIS with Chi-Squared PDF")
ax[1].set_xlabel("NIS Value")
ax[1].set_ylabel("Density")
ax[1].set_ylim(0, None)  # Adjust y-axis limits for better visualization
ax[1].legend()

plt.tight_layout()
plt.show()

# Perform Kolmogorov-Smirnov Test
ks_stat, ks_p_value = kstest(results["nis"], 'chi2', args=(degrees_of_freedom,))
print(f"Kolmogorov-Smirnov Test Statistic: {ks_stat:.2f}, p-value: {ks_p_value:.2f}")

# Plot Empirical CDF vs Chi-Squared CDF
plt.figure(figsize=(10, 6))
ecdf = np.arange(1, len(results["nis"]) + 1) / len(results["nis"])
plt.plot(np.sort(results["nis"]), ecdf, marker='.', linestyle='none', label='Empirical CDF')
plt.plot(x, chi2.cdf(x, degrees_of_freedom), 'r-', label='Chi-Squared CDF')
plt.xlabel('NIS Value')
plt.ylabel('Cumulative Probability')
plt.title('Empirical CDF vs. Chi-Squared CDF')
plt.legend()
plt.show()

# Q-Q Plot
plt.figure(figsize=(10, 6))
probplot(results["nis"], dist="chi2", sparams=(degrees_of_freedom,), plot=plt)
plt.title('Q-Q Plot for NIS vs. Chi-Squared Distribution')
plt.show()
