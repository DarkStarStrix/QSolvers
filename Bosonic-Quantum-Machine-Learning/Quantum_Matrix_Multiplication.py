# create a quantum circuit to perform matrix multiplication and compare the result with classical matrix multiplication

import numpy as np
import plotly.graph_objects as go
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


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
    result = execute (qc, backend='qasm_simulator', shots=1024)
    return result.get_counts (qc)


def classical_matrix_multiplication(matrix_a, matrix_b):
    return matrix_a * matrix_b


def is_unitary(matrix):
    matrix = np.array (matrix)
    return np.allclose (np.eye (matrix.shape [0]), matrix @ matrix.T.conj ())


# Define unitary matrices
matrix_a = QuantumMatrix (np.array ([[1 / np.sqrt (2), 1 / np.sqrt (2)], [1 / np.sqrt (2), -1 / np.sqrt (2)]]))
matrix_b = QuantumMatrix (np.array ([[1 / np.sqrt (2), 1 / np.sqrt (2)], [1 / np.sqrt (2), -1 / np.sqrt (2)]]))

quantum_result = None
classical_result = None

# Check if the matrix is unitary
if not is_unitary (matrix_a.matrix):
    print ("Matrix A is not unitary. Please provide a unitary matrix.")
else:
    # Perform quantum matrix multiplication
    quantum_result = quantum_matrix_multiplication (matrix_a, matrix_b)

    # Perform classical matrix multiplication
    classical_result = classical_matrix_multiplication (matrix_a, matrix_b)

# Compare results
if quantum_result is not None and classical_result is not None:
    fig = go.Figure (data=[
        go.Bar (name='Quantum', x=list (quantum_result.keys ()), y=list (quantum_result.values ())),
        go.Bar (name='Classical', x=list (classical_result.matrix.flatten ()),
                y=list (classical_result.matrix.flatten ()))
    ])
    fig.show ()
else:
    print ("Quantum or classical result is None. Please provide unitary matrices.")
