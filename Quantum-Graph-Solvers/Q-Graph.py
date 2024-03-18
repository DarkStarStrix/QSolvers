import networkx as nx
from qiskit import QuantumCircuit
from qiskit.aqua.algorithms import VQE, NumPyEigensolver
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua import QuantumInstance
from qiskit.visualization import plot_histogram
from qiskit.aqua.algorithms import Grover
from qiskit.aqua.components.oracles import LogicalExpressionOracle


class GraphManager:
    def __init__(self, data):
        self.data = data
        self.graph = self.create_graph()

    def create_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.data['nodes'])
        graph.add_edges_from(self.data['edges'])
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
        graph = self.graph_manager.create_graph()
        ground_state = self.quantum_optimizer.find_ground_state()
        search_result = self.quantum_searcher.perform_grovers_search()
        return graph, ground_state, search_result
