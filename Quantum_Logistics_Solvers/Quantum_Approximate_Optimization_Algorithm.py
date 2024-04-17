import numpy as np
from qiskit import QuantumCircuit


class GHZCircuit:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.qc = QuantumCircuit (self.num_qubits)

    def prepare_state(self):
        self.qc.h (0)  # generate superposition
        self.qc.p (np.pi / 2, 0)  # add quantum phase
        for i in range (1, self.num_qubits):
            self.qc.cx (0, i)  # 0th-qubit-Controlled-NOT gate on i-th qubit

    def get_decomposed_circuit(self):
        return self.qc.decompose()

    def get_circuit_draw(self):
        return self.qc.draw()

    def get_circuit_qasm(self):
        return self.qc

    # print counts
    def print_counts(self):
        print (self.qc.measure_all ())


# Usage
if __name__ == '__main__':
    ghz = GHZCircuit (3)
    ghz.prepare_state ()
    print (ghz.get_circuit_draw ())
    print (ghz.get_decomposed_circuit ())
    print (ghz.get_circuit_qasm ())
