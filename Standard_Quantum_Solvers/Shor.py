from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram


def shor_circuit():
    qc = QuantumCircuit (3, 3)
    qc.h (range (3))
    qc.x (range (3))
    qc.h (2)
    qc.ccx (0, 1, 2)
    qc.h (2)
    qc.x (range (2))
    qc.h (range (3))
    qc.measure (range (3), range (3))
    return qc


# print the counts and draw the circuit
circuit = shor_circuit ()
print (circuit)
