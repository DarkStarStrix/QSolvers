import unittest
from Quantized_Non_Linear_Library import SchrodingerSolver, NavierStokesSolver


class TestNonLinearSolvers (unittest.TestCase):
    def setUp(self):
        self.schrodinger_params = {'N': 100, 'L': 10.0}
        self.navier_stokes_params = {'Nx': 100, 'Ny': 100, 'Nt': 100, 'dt': 0.01, 'T': 1.0, 'Re': 100}

    def schrodinger_solver_initializes_correctly(self):
        solver = SchrodingerSolver ()
        self.assertIsInstance (solver, SchrodingerSolver)

    def navier_stokes_solver_initializes_correctly(self):
        solver = NavierStokesSolver ()
        self.assertIsInstance (solver, NavierStokesSolver)

    def schrodinger_solver_sets_params_correctly(self):
        solver = SchrodingerSolver ()
        solver.set_params (self.schrodinger_params)
        self.assertEqual (solver.N, self.schrodinger_params ['N'])
        self.assertEqual (solver.L, self.schrodinger_params ['L'])

    def navier_stokes_solver_sets_params_correctly(self):
        solver = NavierStokesSolver ()
        solver.set_params (self.navier_stokes_params)
        self.assertEqual (solver.parameters, self.navier_stokes_params)

    def schrodinger_solver_returns_solution(self):
        solver = SchrodingerSolver ()
        solver.set_params (self.schrodinger_params)
        solver.solve ()
        result = solver.get_solution ()
        self.assertIsNotNone (result)

    def navier_stokes_solver_returns_solution(self):
        solver = NavierStokesSolver ()
        solver.set_params (self.navier_stokes_params)
        solver.solve ()
        result = solver.get_solution ()
        self.assertIsNotNone (result)


if __name__ == '__main__':
    unittest.main ()
