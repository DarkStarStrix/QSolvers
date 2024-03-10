# create a quantum circuit to perform matrix multiplication and compare the result with classical matrix multiplication

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Operator
import plotly.graph_objects as go


# Define the matrices in oop
class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix
        self.shape = matrix.shape

    def __add__(self, other):
        return Matrix (self.matrix + other.matrix)

    def __sub__(self, other):
        return Matrix (self.matrix - other.matrix)

    def __mul__(self, other):
        return Matrix (self.matrix @ other.matrix)

    def __str__(self):
        return str (self.matrix)

    def __repr__(self):
        return str (self.matrix)

    def __eq__(self, other):
        return np.array_equal (self.matrix, other.matrix)

    def __ne__(self, other):
        return not np.array_equal (self.matrix, other.matrix)

    def __getitem__(self, key):
        return self.matrix [key]

    def __setitem__(self, key, value):
        self.matrix [key] = value


# Define the quantum circuit to perform matrix multiplication
def quantum_matrix_multiplication(A, B):
    # Create a quantum circuit on 2 qubits
    qc = QuantumCircuit (2)

    # Apply the unitary operator corresponding to the matrix A
    qc.unitary (Operator (A), [0, 1])

    # Apply the unitary operator corresponding to the matrix B
    qc.unitary (Operator (B), [0, 1])

    # Measure the qubits
    qc.measure_all ()

    # Use the qasm simulator to get the measurement probabilities
    simulator = Aer.get_backend ('qasm_simulator')
    result = execute (qc, simulator, shots=10000).result ()
    counts = result.get_counts (qc)

    return counts


# Define the matrices A and B
A = Matrix (np.array ([[1, 2], [3, 4]]))
B = Matrix (np.array ([[5, 6], [7, 8]]))

# Perform matrix multiplication using the quantum circuit
counts = quantum_matrix_multiplication (A, B)

# Print the measurement probabilities
print (counts)

# Perform matrix multiplication using classical matrix multiplication
C = A * B
print (C)

# compare the results using quantum and classical matrix multiplication plot with plotly
fig = go.Figure (data=[
    go.Bar (name='Quantum', x=list (counts.keys ()), y=list (counts.values ())),
    go.Bar (name='Classical', x=list (C.matrix.flatten ()), y=list (C.matrix.flatten ()))
])
