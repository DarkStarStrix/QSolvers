import pytest
from Quantum_A import TSP, QuantumAStar


def tsp_initialization():
    cities = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    distances = [[0, 1, 2, 3],
                 [1, 0, 1, 2],
                 [2, 1, 0, 1],
                 [3, 2, 1, 0]]
    tsp = TSP (cities, distances)
    assert tsp.get_number_of_cities () == 4
    assert tsp.get_start_city () == 'A'


def quantum_a_star_initialization():
    cities = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    distances = [[0, 1, 2, 3],
                 [1, 0, 1, 2],
                 [2, 1, 0, 1],
                 [3, 2, 1, 0]]
    tsp = TSP (cities, distances)
    quantum_a_star = QuantumAStar (tsp)
    assert quantum_a_star.start_city == 'A'
    assert quantum_a_star.number_of_cities == 4


def quantum_a_star_make_qc():
    cities = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    distances = [[0, 1, 2, 3],
                 [1, 0, 1, 2],
                 [2, 1, 0, 1],
                 [3, 2, 1, 0]]
    tsp = TSP (cities, distances)
    quantum_a_star = QuantumAStar (tsp)
    assert quantum_a_star.qc.count_ops () ['x'] == 1
    assert quantum_a_star.qc.count_ops () ['h'] == 3
    assert quantum_a_star.qc.count_ops () ['cswap'] == 9
    assert quantum_a_star.qc.count_ops () ['cu3'] == 9


def quantum_a_star_get_cost():
    cities = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    distances = [[0, 1, 2, 3],
                 [1, 0, 1, 2],
                 [2, 1, 0, 1],
                 [3, 2, 1, 0]]
    tsp = TSP (cities, distances)
    quantum_a_star = QuantumAStar (tsp)
    path = '1001'
    assert quantum_a_star.get_cost (path) == 4
