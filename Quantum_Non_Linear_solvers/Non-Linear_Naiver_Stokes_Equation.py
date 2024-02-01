import numpy as np
import plotly.figure_factory as ff
from qutip import *


class Parameters:
    def __init__(self):
        self.Nx = 100
        self.Ny = 100
        self.Nt = 100
        self.dt = 0.01
        self.T = 1.0
        self.Re = 100


class Simulation:
    def __init__(self, parameters):
        self.parameters = parameters

    def initial_condition(self, x, y, t):
        u = np.zeros ((self.parameters.Nx, self.parameters.Ny))
        v = np.zeros ((self.parameters.Nx, self.parameters.Ny))
        p = np.zeros ((self.parameters.Nx, self.parameters.Ny))

        for i in range (self.parameters.Nx):
            for j in range (self.parameters.Ny):
                u [i, j] = np.sin (np.pi * x [i]) * np.cos (np.pi * y [j])
                v [i, j] = -np.cos (np.pi * x [i]) * np.sin (np.pi * y [j])
                p [i, j] = -0.25 * (np.cos (2 * np.pi * x [i]) + np.cos (2 * np.pi * y [j]))

        return u, v, p

    def non_linear_navier_stokes_equation(self, u, v, p, x, y, t):
        dx = x [1] - x [0]
        dy = y [1] - y [0]
        dt = t [1] - t [0]

        u_xx = np.zeros ((self.parameters.Nx, self.parameters.Ny))
        u_yy = np.zeros ((self.parameters.Nx, self.parameters.Ny))
        v_xx = np.zeros ((self.parameters.Nx, self.parameters.Ny))
        v_yy = np.zeros ((self.parameters.Nx, self.parameters.Ny))

        for i in range (1, self.parameters.Nx - 1):
            for j in range (1, self.parameters.Ny - 1):
                u_xx [i, j] = (u [i + 1, j] - 2 * u [i, j] + u [i - 1, j]) / dx ** 2
                u_yy [i, j] = (u [i, j + 1] - 2 * u [i, j] + u [i, j - 1]) / dy ** 2
                v_xx [i, j] = (v [i + 1, j] - 2 * v [i, j] + v [i - 1, j]) / dx ** 2
                v_yy [i, j] = (v [i, j + 1] - 2 * v [i, j] + v [i, j - 1]) / dy ** 2

            u_t = np.clip(u_xx[1:-1, 1:-1] + u_yy[1:-1, 1:-1] - (u[1:-1, 1:-1] * (u[1:-1, 1:-1] - u[:-2, 1:-1]) / dx + v[1:-1, 1:-1] * (
                    u[1:-1, 1:-1] - u[1:-1, :-2]) / dy) + 1 / self.parameters.Re * (u_xx[1:-1, 1:-1] + u_yy[1:-1, 1:-1]), -np.inf, np.inf)
            v_t = np.clip(v_xx[1:-1, 1:-1] + v_yy[1:-1, 1:-1] - (u[1:-1, 1:-1] * (v[1:-1, 1:-1] - v[:-2, 1:-1]) / dx + v[1:-1, 1:-1] * (
                    v[1:-1, 1:-1] - v[1:-1, :-2]) / dy) + 1 / self.parameters.Re * (v_xx[1:-1, 1:-1] + v_yy[1:-1, 1:-1]), -np.inf, np.inf)
            p_t = 0

            u[1:-1, 1:-1] = u[1:-1, 1:-1] + u_t * dt
            v[1:-1, 1:-1] = v[1:-1, 1:-1] + v_t * dt
            p[1:-1, 1:-1] = p[1:-1, 1:-1] + p_t * dt

            return u, v, p

    def run(self):
        x = np.linspace (0, 1, self.parameters.Nx)
        y = np.linspace (0, 1, self.parameters.Ny)
        t = np.linspace (0, self.parameters.T, self.parameters.Nt)

        initial_time = t [0]
        u, v, p = self.initial_condition (x, y, initial_time)

        time_steps = int (5 * 60 / self.parameters.dt)

        for _ in range (time_steps):
            u, v, p = self.non_linear_navier_stokes_equation (u, v, p, x, y, t)

        fig = ff.create_quiver (x, y, u, v)
        fig.show ()

        return u, v, p


if __name__ == "__main__":
    parameters = Parameters ()
    simulation = Simulation (parameters)
    simulation.run ()
