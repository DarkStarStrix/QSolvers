import pytest
import networkx as nx
from Quantum_Annealing import TSPSolver


def tsp_solver_initialization():
    G = nx.complete_graph (4)
    tsp_solver = TSPSolver (G)
    assert tsp_solver.num_nodes == 4
    assert len (tsp_solver.qubo) == 12


def qubo_creation():
    G = nx.complete_graph (4)
    tsp_solver = TSPSolver (G)
    qubo = tsp_solver._create_qubo ()
    assert len (qubo) == 12
    for key, value in qubo.items ():
        assert value == 1


def tsp_solver_solution():
    G = nx.complete_graph (4)
    tsp_solver = TSPSolver (G)
    route = tsp_solver.solve ()
    assert len (route) == 4
    assert len (set (route)) == 4
