from qiskit import QuantumCircuit
import matplotlib.pyplot as plt


def grover_circuit():
    qc = QuantumCircuit (3, 3)
    qc.h (range (3))
    qc.x (range (3))
    qc.h (2)
    qc.ccx (0, 1, 2)
    qc.h (2)
    qc.x (range (3))
    qc.h (range (3))
    qc.measure (range (3), range (3))
    return qc


# Draw the circuit
circuit = grover_circuit ()
print (circuit)
