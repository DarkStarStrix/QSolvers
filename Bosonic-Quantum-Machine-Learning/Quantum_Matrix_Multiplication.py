# create a quantum circuit to perform matrix multiplication and compare the result with classical matrix multiplication

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import numpy as np
import plotly.graph_objects as go


class QuantumMatrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def __add__(self, other):
        return QuantumMatrix (self.matrix + other.matrix)

    def __mul__(self, other):
        return QuantumMatrix (self.matrix @ other.matrix)


def quantum_matrix_multiplication(matrix_a, matrix_b):
    qc = QuantumCircuit (2)
    qc.unitary (Operator (matrix_a.matrix), [0, 1])
    qc.unitary (Operator (matrix_b.matrix), [0, 1])
    qc.measure_all ()
    simulator = Aer.get_backend ('qasm_simulator')
    result = execute (qc, simulator, shots=10000).result ()
    return result.get_counts (qc)


matrix_a = QuantumMatrix (np.array ([[1, 2], [3, 4]]))
matrix_b = QuantumMatrix (np.array ([[5, 6], [7, 8]]))
counts = quantum_matrix_multiplication (matrix_a, matrix_b)
C = matrix_a * matrix_b

fig = go.Figure (data=[
    go.Bar (name='Quantum', x=list (counts.keys ()), y=list (counts.values ())),
    go.Bar (name='Classical', x=list (C.matrix.flatten ()), y=list (C.matrix.flatten ()))
])
fig.show ()
