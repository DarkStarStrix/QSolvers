from qutip import *
import numpy as np


class NonLinearSolver:
    def __init__(self):
        pass

    def set_params(self, params):
        raise NotImplementedError

    def solve(self):
        raise NotImplementedError

    def get_solution(self):
        raise NotImplementedError


class SchrodingerSolver (NonLinearSolver):
    def __init__(self):
        super ().__init__ ()
        self.H = None
        self.a = None
        self.adag = None
        self.L = None
        self.N = None
        self.result = None

    def set_params(self, params):
        self.N = params ['N']
        self.L = params ['L']
        self.a = destroy (self.N)
        self.adag = self.a.dag ()
        self.H = -1.0 * (self.adag * self.a + 0.5 * self.adag * self.adag * self.a * self.a)

    def solve(self):
        psi0 = basis (self.N, self.N // 2)  # initial state
        t = np.linspace (0, 10.0, 100)  # time
        self.result = sesolve (self.H, psi0, t, [])

    def get_solution(self):
        return self.result

    params = {'N': 100, 'L': 10.0}


solver = SchrodingerSolver ()
solver.set_params (params)
solver.solve ()
result = solver.get_solution ()
