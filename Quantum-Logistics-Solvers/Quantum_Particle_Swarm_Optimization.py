from qiskit import QuantumCircuit, execute, Aer


class QuantumParticle:
    def __init__(self, num_qubits):
        self.circuit = QuantumCircuit (num_qubits)
        self.circuit.h (range (num_qubits))
        self.circuit.measure_all ()

    def run(self):
        return execute (self.circuit, Aer.get_backend ('qasm_simulator'), shots=1).result ().get_counts ()


class QuantumSwarm:
    def __init__(self, num_particles, num_qubits):
        self.particles = [QuantumParticle (num_qubits) for _ in range (num_particles)]

    def run(self):
        for particle in self.particles:
            print (particle.run ())


swarm = QuantumSwarm (5, 5)
swarm.run ()
