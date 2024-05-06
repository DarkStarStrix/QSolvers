from qiskit import QuantumCircuit


def vqe_circuit():
    qc = QuantumCircuit (3, 3)
    qc.h (range (3))
    qc.x (range (3))
    qc.h (2)
    qc.ccx (0, 1, 2)
    qc.h (2)
    qc.x (range (3))
    return qc


def vqe_algorithm():
    pauli_dict = {
        'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                   {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "ZI"},
                   {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "IZ"},
                   {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                   {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}]
    }
    qubit_op = WeightedPauliOperator.from_dict (pauli_dict)
    var_form = RY (qubit_op.num_qubits, depth=3, entanglement='linear')
    optimizer = COBYLA (maxiter=1000)
    return VQE (qubit_op, var_form, optimizer)


# print the circuit
print (vqe_circuit ())
