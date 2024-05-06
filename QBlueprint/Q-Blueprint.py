from abc import ABC

from qiskit import execute, Aer
from qiskit.circuit import Qubit, Clbit
from qiskit.circuit.library import BlueprintCircuit, HGate


class CustomQuantumAlgorithm (BlueprintCircuit, ABC):
    @property
    def clbits(self):
        return self._clbits

    @property
    def qubits(self):
        return self._qubits

    def __init__(self, qubits):
        super ().__init__ (name="Custom Quantum Algorithm")
        self.qubits = [Qubit () for _ in range (qubits)]
        self.clbits = [Clbit () for _ in range (qubits)]
        self.add_register (*self.qubits)
        self.add_register (*self.clbits)

    def add_gate(self, gate, qubit):
        self.append (gate, [self.qubits [qubit]])

    def run(self):
        simulator = Aer.get_backend ('qasm_simulator')
        job = execute (self, simulator)
        result = job.result ()
        return result.get_counts (self)

    @qubits.setter
    def qubits(self, value):
        self._qubits = value

    @clbits.setter
    def clbits(self, value):
        self._clbits = value


custom_algo = CustomQuantumAlgorithm (2)
custom_algo.add_gate (HGate (), 0)
print (custom_algo.run ())
