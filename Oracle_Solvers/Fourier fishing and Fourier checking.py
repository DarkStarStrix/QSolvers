# code a Quantum Forier fishing and Fourier checking for the 1D heat equation

from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram


def qft(n):
    """Create a Quantum Fourier Transform circuit on n qubits."""
    circuit = QuantumCircuit (n)
    for j in range (n):
        for k in range (j):
            circuit.cu1 (np.pi / float (2 ** (j - k)), k, j)
        circuit.h (j)
    return circuit


def inverse_qft(n):
    """Create an inverse Quantum Fourier Transform circuit on n qubits."""
    circuit = QuantumCircuit (n)
    for j in range (n):
        circuit.h (j)
        for k in range (j):
            circuit.cu1 (-np.pi / float (2 ** (j - k)), k, j)
    return circuit


def quantum_fourier_fishing(n, state_vector):
    """Perform Quantum Fourier Fishing on a state vector."""
    # Create a quantum circuit
    circuit = QuantumCircuit (n)
    # Initialize the circuit with the state vector
    circuit.initialize (state_vector, range (n))
    # Apply the QFT
    circuit.append (qft (n), range (n))
    # Perform a measurement
    circuit.measure_all ()
    # Execute the circuit
    result = execute (circuit, Aer.get_backend ('qasm_simulator')).result ()
    counts = result.get_counts (circuit)
    return counts


def fourier_checking(expected_counts, actual_counts):
    """Check if the actual counts match the expected counts."""
    return expected_counts == actual_counts
