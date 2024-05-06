# hard code the quantum Deutsch-Jozsa algorithm for 3 qubits

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram

# Create a Quantum Circuit acting on the q register
qc = QuantumCircuit(3, 3)

# Apply Hadamard gate to all qubits
qc.h(range(3))

# Apply Oracle
qc.x(range(3))
qc.h(2)
qc.ccx(0, 1, 2)
qc.h(2)
qc.x(range(3))

# Measure qubits
qc.measure(range(3), range(3))

# print the circuit
print(qc)
