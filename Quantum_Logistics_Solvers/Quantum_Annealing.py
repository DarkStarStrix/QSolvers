# Code TSP with Quantum Annealing (D-Wave)
import dimod


class Graph:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.network = {i: {j: 1 for j in range (num_nodes) if i != j} for i in range (num_nodes)}  # renamed from graph to network


class TSPSolver:
    def __init__(self, graph):
        self.network = graph.network  # renamed from graph to network
        self.qubo = self._create_qubo ()

    def _create_qubo(self):
        return {(i, j): self.network[i][j] for i in range(len(self.network)) for j in range(i + 1, len(self.network))}  # renamed from graph to network

    def solve(self):
        response = dimod.ExactSolver().sample_qubo(self.qubo)
        return [node for node, bit in response.first.sample.items() if bit == 1]

    @staticmethod
    def plot_route(route):
        print("TSP Route:", route)


G = Graph(4)
tsp_solver = TSPSolver(G)
optimal_route = tsp_solver.solve()
tsp_solver.plot_route(optimal_route)
