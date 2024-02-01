from qutip import mesolve, basis, sigmax, expect, Qobj
import numpy as np
import matplotlib.pyplot as plt

# Define the Hamiltonian
H = sigmax()

# Define the initial state as a superposition of the ground and excited states
psi0 = (basis(2, 0) + basis(2, 1)).unit()

# Define the list of times for which to solve the equation
tlist = np.linspace(0, 10, 100)

# Solve the master equation
result = mesolve(H, psi0, tlist)

# The result object contains the state of the system at the times specified
states = result.states

# The states are quantum objects, so we can use QuTiP's built-in functions to calculate expectation values
expectation_values = [expect(sigmax(), state) for state in states]

# Plot the expectation values
plt.plot(tlist, expectation_values)
plt.xlabel('Time')
plt.ylabel('Expectation value of sigma_x')
plt.show()
