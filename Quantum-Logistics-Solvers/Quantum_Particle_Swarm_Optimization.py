from qiskit import QuantumCircuit


class QuantumParticle:
    def __init__(self, num_qubits):
        self.circuit = QuantumCircuit (num_qubits)
        self.circuit.h (range (num_qubits))
        self.circuit.measure_all ()


class QuantumSwarm:
    def __init__(self, num_particles, num_qubits):
        self.particles = [QuantumParticle (num_qubits) for _ in range (num_particles)]


if __name__ == '__main__':
    swarm = QuantumSwarm (10, 5)
    for particle in swarm.particles:
        print (particle)
