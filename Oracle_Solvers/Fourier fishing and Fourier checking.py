# code a Quantum Forier fishing and Fourier checking for the 1D heat equation

from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
import numpy as np

circuit = QuantumCircuit (3, 3)


def qft(n):
    circuit = QuantumCircuit (n)
    for j in range (n):
        circuit.h (j)
        for k in range (j):
            circuit.cp (np.pi / float (2 ** (j - k)), k, j)
    return circuit


def inverse_qft(n):
    circuit = QuantumCircuit (n)
    for j in range (n):
        circuit.h (j)
        for k in range (j):
            circuit.cp (-np.pi / float (2 ** (j - k)), k, j)
    return circuit


def quantum_fourier_fishing(n, state_vector):
    # Create a quantum circuit
    circuit = QuantumCircuit (n, n)
    # Initialize the circuit with the state vector
    circuit.initialize (state_vector, range (n))
    # Apply the QFT
    qft_circuit = qft (n)
    circuit.append (qft_circuit, range (n))
    # Perform a measurement
    circuit.measure (range (n), range (n))


def fourier_checking(expected_counts, actual_counts):
    return expected_counts == actual_counts


# Perform Quantum Fourier Fishing
n = 2
state_vector = [1, 0, 0, 0]
quantum_fourier_fishing (n, state_vector)

# print the circuit
print (circuit)
