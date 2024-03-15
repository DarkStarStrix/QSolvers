import scipy.stats as stats
from Quantum_Walk_Solvers.Boson_Sampling import BosonSampling
import numpy as np
import matplotlib.pyplot as plt

# Initialize the BosonSampling class
n = 3
m = 5
U = np.random.rand(m, m) + 1j * np.random.rand(m, m)
boson_sampling = BosonSampling(n, m, U)

# Define the data from the BosonSampling simulation
prob = boson_sampling.simulate()

# Check if prob is a 2D array with more columns than rows
if prob.ndim == 2 and prob.shape[1] > prob.shape[0]:
    prob = prob.T
# Check if prob is a 1D array with only one sample
if prob.ndim == 1 and prob.shape[0] == 1:
    prob = np.array([prob[0], prob[0]])

# Perform a t-test
hypothesized_mean = 0.5  # replace with your hypothesized mean
t_statistic, p_value = stats.ttest_1samp(prob, hypothesized_mean)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

# Plot the data
plt.hist(prob, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(hypothesized_mean, color='red', linestyle='dashed', linewidth=2)  # add a vertical line for the hypothesized mean
plt.title('Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
