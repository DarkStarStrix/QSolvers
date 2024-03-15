# Code TSP with Quantum Annealing (D-Wave)
import dimod
import matplotlib.pyplot as plt
import networkx as nx
from dwave.system import EmbeddingComposite, DWaveSampler


class TSPSolver:
    def __init__(self, graph):
        self.graph = graph
        self.num_nodes = len (graph.nodes)
        self.qubo = self._create_qubo ()

    def _create_qubo(self):
        QUBO = {}
        for i in range (self.num_nodes):
            for j in range (i + 1, self.num_nodes):
                QUBO [(i, j)] = self.graph [i] [j] ['weight']
        return QUBO

    def solve(self):
        response = dimod.ExactSolver ().sample_qubo (self.qubo)
        solution = response.first.sample
        route = [node for node, bit in solution.items () if bit == 1]
        return route

    def plot_route(self, route):
        pos = nx.spring_layout (self.graph)
        nx.draw (self.graph, pos, with_labels=True, node_size=500)
        nx.draw_networkx_nodes (self.graph, pos, nodelist=route, node_color='r')
        plt.title ("TSP Route")
        plt.show ()


G = nx.complete_graph (4)
for i, j in G.edges ():
    G [i] [j] ['weight'] = 1

tsp_solver = TSPSolver (G)
optimal_route = tsp_solver.solve ()
tsp_solver.plot_route (optimal_route)
