from qiskit import QuantumCircuit, execute, Aer
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


provider = Aer.get_backend ('qasm_simulator')
qc = shor_circuit ()
job = execute (qc, provider, shots=1024)
result = job.result ()
counts = result.get_counts (qc)
plot_histogram (counts)
