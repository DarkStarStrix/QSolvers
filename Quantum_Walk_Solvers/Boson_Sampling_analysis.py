from Quantum_Walk_Solvers.Boson_Sampling import BosonSampling
import numpy as np
import matplotlib.pyplot as plt

# Initialize the BosonSampling class
n = 3
m = 5
U = np.random.rand(m, m) + 1j * np.random.rand(m, m)
boson_sampling = BosonSampling(n, m, U)

# Simulate the BosonSampling
prob = boson_sampling.simulate()

# Analyze the results
mean = np.mean(prob)
median = np.median(prob)
std_dev = np.std(prob)
variance = np.var(prob)

print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")

# Plot the data
plt.figure(figsize=(10, 6))
plt.hist(prob, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
