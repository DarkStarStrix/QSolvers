from qutip import mesolve, basis, sigmax, expect
import numpy as np
import matplotlib.pyplot as plt

# Define the Hamiltonian, initial state, and time list
H, psi0, tlist = sigmax(), (basis(2, 0) + basis(2, 1)).unit(), np.linspace(0, 10, 100)

# Solve the master equation and get the states
states = mesolve(H, psi0, tlist).states

# Calculate and plot the expectation values
plt.plot(tlist, [expect(sigmax(), state) for state in states])
plt.xlabel('Time')
plt.ylabel('Expectation value of sigma_x')
plt.show()
