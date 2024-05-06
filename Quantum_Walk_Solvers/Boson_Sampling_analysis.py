from Quantum_Walk_Solvers.Boson_Sampling import BosonSampling
import numpy as np
import matplotlib.pyplot as plt

n, m = 3, 5
U = np.random.rand(m, m) + 1j * np.random.rand(m, m)
boson_sampling = BosonSampling(n, m, U)
prob = boson_sampling.simulate()

stats = {"Mean": np.mean(prob), "Median": np.median(prob), "Standard Deviation": np.std(prob), "Variance": np.var(prob)}
for stat, value in stats.items():
    print(f"{stat}: {value}")

plt.figure(figsize=(10, 6))
plt.hist(prob, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Probability Distribution')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
