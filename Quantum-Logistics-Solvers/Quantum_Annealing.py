# Code TSP with Quantum Annealing (D-Wave)
import dimod
import matplotlib.pyplot as plt
import networkx as nx


class TSPSolver:
    def __init__(self, graph):
        self.graph = graph
        self.qubo = self._create_qubo ()

    def _create_qubo(self):
        return {(i, j): self.graph [i] [j] ['weight'] for i in range (len (self.graph.nodes)) for j in
                range (i + 1, len (self.graph.nodes))}

    def solve(self):
        response = dimod.ExactSolver ().sample_qubo (self.qubo)
        return [node for node, bit in response.first.sample.items () if bit == 1]

    def plot_route(self, route):
        pos = nx.spring_layout (self.graph)
        nx.draw (self.graph, pos, with_labels=True, node_size=500)
        nx.draw_networkx_nodes (self.graph, pos, nodelist=route, node_color='r')
        plt.title ("TSP Route")
        plt.show ()


G = nx.complete_graph (4)
nx.set_edge_attributes (G, 1, 'weight')
tsp_solver = TSPSolver (G)
optimal_route = tsp_solver.solve ()
tsp_solver.plot_route (optimal_route)
