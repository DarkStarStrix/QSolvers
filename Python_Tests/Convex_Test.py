import pytest
from Quantum_Convex import TSP, create_circuit, create_hamiltonian, refine_solution, tour_visualization, plot_results


def tsp_initialization():
    cities = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    distances = [[0, 1, 2, 3],
                 [1, 0, 1, 2],
                 [2, 1, 0, 1],
                 [3, 2, 1, 0]]
    tsp = TSP (cities, distances)
    assert tsp.get_number_of_cities () == 4
    assert tsp.get_start_city () == 'A'


def circuit_creation():
    distances = [[0, 1, 2, 3],
                 [1, 0, 1, 2],
                 [2, 1, 0, 1],
                 [3, 2, 1, 0]]
    qc = create_circuit (distances)
    assert qc.count_ops () ['h'] == 4
    assert qc.count_ops () ['cp'] == 12
    assert qc.count_ops () ['measure'] == 4


def hamiltonian_creation():
    quadratic_program = QuadraticProgram ()
    quadratic_program.integer_var (name='x', lowerbound=0, upperbound=3)
    quadratic_program.integer_var (name='y', lowerbound=0, upperbound=3)
    quadratic_program.minimize (linear=[1, 2])
    coefficients = create_hamiltonian (quadratic_program)
    assert coefficients == {(0, 1): 1, (1, 0): 2}


def solution_refinement():
    x = [0, 1, 2, 3]
    refined_x = refine_solution (x)
    assert refined_x == x


def tour_visualization_check():
    x = [0, 1, 2, 3]
    visualized_tour = tour_visualization (x)
    assert visualized_tour == x


def results_plotting():
    x = [0, 1, 2, 3]
    plotted_results = plot_results (x)
    assert plotted_results == x
