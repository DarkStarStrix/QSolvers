import numpy as np
import plotly.figure_factory as ff
from qutip import *


class Parameters:
    def __init__(self):
        self.Nx = self.Ny = self.Nt = 100
        self.dt = 0.01
        self.T = 1.0
        self.Re = 100


class Simulation:
    def __init__(self, parameters):
        self.parameters = parameters

    @staticmethod
    def initial_condition(self, x, y):
        u = np.sin (np.pi * x) * np.cos (np.pi * y)
        v = -np.cos (np.pi * x) * np.sin (np.pi * y)
        p = -0.25 * (np.cos (2 * np.pi * x) + np.cos (2 * np.pi * y))
        return u, v, p

    def non_linear_navier_stokes_equation(self, u, v, p, x, y, t):
        dx, dy, dt = x [1] - x [0], y [1] - y [0], t [1] - t [0]
        u_xx, u_yy = (u [2:, :] - 2 * u [1:-1, :] + u [:-2, :]) / dx ** 2, (
                    u [:, 2:] - 2 * u [:, 1:-1] + u [:, :-2]) / dy ** 2
        v_xx, v_yy = (v [2:, :] - 2 * v [1:-1, :] + v [:-2, :]) / dx ** 2, (
                    v [:, 2:] - 2 * v [:, 1:-1] + v [:, :-2]) / dy ** 2

        u_t = np.clip (
            u_xx + u_yy - (u * (u - u [:-1, :]) / dx + v * (u - u [:, :-1]) / dy) + 1 / self.parameters.Re * (
                        u_xx + u_yy), -np.inf, np.inf)
        v_t = np.clip (
            v_xx + v_yy - (u * (v - v [:-1, :]) / dx + v * (v - v [:, :-1]) / dy) + 1 / self.parameters.Re * (
                        v_xx + v_yy), -np.inf, np.inf)

        u [1:-1, 1:-1] += u_t * dt
        v [1:-1, 1:-1] += v_t * dt
        return u, v, p

    def run(self):
        x = np.linspace (0, 1, self.parameters.Nx)
        y = np.linspace (0, 1, self.parameters.Ny)
        t = np.linspace (0, self.parameters.T, self.parameters.Nt)

        u, v, p = self.initial_condition (x, y)

        for _ in range (int (5 * 60 / self.parameters.dt)):
            u, v, p = self.non_linear_navier_stokes_equation (u, v, p, x, y, t)

        fig = ff.create_quiver (x, y, u, v)
        fig.show ()

        return u, v, p


if __name__ == "__main__":
    parameters = Parameters ()
    simulation = Simulation (parameters)
    simulation.run ()
