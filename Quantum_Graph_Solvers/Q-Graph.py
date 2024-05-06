import networkx as nx
from qiskit import QuantumCircuit, QuantumInstance
from qiskit.aqua.algorithms import VQE, Grover
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.components.variational_forms import RY


class GraphManager:
    def __init__(self, data):
        self.graph = self.create_graph(data)

    @staticmethod
    def create_graph(data):
        graph = nx.Graph()
        graph.add_nodes_from(data['nodes'])
        graph.add_edges_from(data['edges'])
        return graph


class QuantumOptimizer:
    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian

    def find_ground_state(self):
        var_form = RY(self.hamiltonian.num_qubits)
        optimizer = COBYLA()
        vqe = VQE(self.hamiltonian, var_form, optimizer)
        result = vqe.run(QuantumInstance(Aer.get_backend('statevector_simulator')))
        return result['optimal_point']


class QuantumSearcher:
    def __init__(self, oracle):
        self.oracle = oracle

    def perform_grovers_search(self):
        grover = Grover(self.oracle)
        result = grover.run(QuantumInstance(Aer.get_backend('qasm_simulator')))
        return result['top_measurement']


class QGAF:
    def __init__(self, network, hamiltonian, oracle):
        self.graph_manager = GraphManager(network)
        self.quantum_optimizer = QuantumOptimizer(hamiltonian)
        self.quantum_searcher = QuantumSearcher(oracle)

    def execute(self):
        graph = self.graph_manager.graph
        ground_state = self.quantum_optimizer.find_ground_state()
        search_result = self.quantum_searcher.perform_grovers_search()
        return graph, ground_state, search_result
