# create a quantum circuit to perform matrix multiplication and compare the result with classical matrix multiplication

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Operator
import plotly.graph_objects as go


class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def __add__(self, other):
        return Matrix (self.matrix + other.matrix)

    def __mul__(self, other):
        return Matrix (self.matrix @ other.matrix)


def quantum_matrix_multiplication(A, B):
    qc = QuantumCircuit (2)
    qc.unitary (Operator (A), [0, 1])
    qc.unitary (Operator (B), [0, 1])
    qc.measure_all ()
    simulator = Aer.get_backend ('qasm_simulator')
    result = execute (qc, simulator, shots=10000).result ()
    return result.get_counts (qc)


A = Matrix (np.array ([[1, 2], [3, 4]]))
B = Matrix (np.array ([[5, 6], [7, 8]]))
counts = quantum_matrix_multiplication (A, B)
C = A * B

fig = go.Figure (data=[
    go.Bar (name='Quantum', x=list (counts.keys ()), y=list (counts.values ())),
    go.Bar (name='Classical', x=list (C.matrix.flatten ()), y=list (C.matrix.flatten ()))
])
