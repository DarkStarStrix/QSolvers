# code a quantum linear solver

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
import plotly.express as px


class QuantumLinearSolver:
    def __init__(self, A, b, n, m):
        self.qc = QuantumCircuit (n + m, m)
        self.qc.h (range (n))
        self.qc.x (n + m - 1)
        self.qc.h (n + m - 1)
        self.qc.cx (n + m - 1, n - 1)
        self.qc.h (range (n))
        self.qc.measure (range (n), range (m))

    def solve(self):
        backend = Aer.get_backend ('qasm_simulator')
        job = execute (self.qc, backend, shots=1024)
        result = job.result ()
        return result.get_counts ()

    def run(self):
        counts = self.solve ()
        fig = px.bar (x=counts.keys (), y=counts.values ())
        fig.show ()


A = np.array ([[1, 1], [1, -1]])
b = np.array ([1, 2])
n = 2
m = 2

qls = QuantumLinearSolver (A, b, n, m)
qls.run ()
