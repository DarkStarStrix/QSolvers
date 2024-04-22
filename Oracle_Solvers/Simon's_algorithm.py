from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram


# Define the oracle function for Simon's problem
def simon_oracle(circuit, s):
    n = len (s)
    for i, bit in enumerate (s):
        if bit == '1':
            circuit.cx (i, n)  # Apply CNOT gate
    return circuit


# Create a quantum circuit with n qubits
n = 3  # Example with 3 qubits
qc = QuantumCircuit (n * 2, n)

# Apply Hadamard gates to the first n qubits
qc.h (range (n))

# Apply the oracle function (you can replace '101' with your desired hidden bitstring)
hidden_bitstring = '101'
qc = simon_oracle (qc, hidden_bitstring)

# Measure the second register
qc.measure (range (n, 2 * n), range (n))

# print the circuit
print (qc)
