# Simons algorithm a black box solver

import numpy as np
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.visualization import plot_histogram
from qiskit.aqua.algorithms import Simon
from qiskit.aqua.components.oracles import TruthTableOracle

# Load IBM Q account and get the least busy backend device
provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_qasm_simulator')

# Set the length of the input string
n = 4

# Set the oracle, Simon's algorithm is a black-box algorithm, so we don't need to know the details of the oracle
oracle = TruthTableOracle('0011')

# Create a Simon algorithm instance
simon = Simon(oracle)

# Run the algorithm and get the result
result = simon.run(backend)

# Print the result
print(result['result'])

# Plot the result
plot_histogram(result['measurement'])

# print the circuit
print(simon.construct_circuit(measurement=True))
