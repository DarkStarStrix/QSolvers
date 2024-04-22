# Hard code the oracle for the Bernstein-Vazirani algorithm for n=3 qubits

# Import the QISKit SDK
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister

# Create a Quantum Register with 3 qubits
q = QuantumRegister(3)
# Create a Classical Register with 3 bits
c = ClassicalRegister(3)
# Create a Quantum Circuit
qc = QuantumCircuit(q, c)

# Apply H-gate to each qubit:
for qubit in q:
    qc.h(qubit)

# Apply barrier
qc.barrier()

# Apply the inner-product oracle
qc.cx(q[0], q[2])
qc.cx(q[1], q[2])

# Apply barrier
qc.barrier()

# Apply H-gate to each qubit:
for qubit in q:
    qc.h(qubit)

# Measure the qubits
for i in range(3):
    qc.measure(q[i], c[i])

# Draw the circuit
qc.draw()
print(qc)
