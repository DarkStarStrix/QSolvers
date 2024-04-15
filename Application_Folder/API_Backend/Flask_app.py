from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, logout_user
from flask_limiter import Limiter
from celery import Celery
import logging

# Import the solver functions
from Quantum_Logistics_Solvers import Quantum_Genetic_Algorithm, Quantum_Particle_Swarm_Optimization, Quantum_A, Quantum_Ant_Colony, Quantum_Annealing, Quantum_Approximate_Optimization_Algorithm, Quantum_Convex

celery = Celery (__name__)
app = Flask (__name__)
limiter = Limiter (app)
login_manager = LoginManager ()
login_manager.init_app (app)
logging.basicConfig (filename='server.log', level=logging.INFO)


class User (UserMixin):
    def __init__(self, id2):
        self.id = id


@login_manager.user_loader
def load_user(user_id):
    return User (user_id)


@app.route ('/login', methods=['GET', 'POST'])
def login():
    user = User (request.form ['username'])
    login_user (user)
    return redirect (url_for ('home'))


@app.route ('/logout')
# @login_required
def logout():
    logout_user ()
    return redirect (url_for ('home'))


@app.route ('/')
# @login_required
def home():
    return render_template ('index.html')


@app.route ('/run_solver/<solver_name>', methods=['POST'])
# @login_required
def run_solver(solver_name):
    # Map solver names to functions
    solvers = {
        'Quantum Genetic Algorithm': Quantum_Genetic_Algorithm,
        'Quantum Particle Swarm Optimization': Quantum_Particle_Swarm_Optimization,
        'Quantum Ant Colony Optimization': Quantum_Ant_Colony,
        'Quantum Simulated Annealing': Quantum_Annealing,
        'Quantum A*': Quantum_A,
        'Quantum Approximate Optimization Algorithm': Quantum_Approximate_Optimization_Algorithm,
        'Quantum Convex': Quantum_Convex,
    }

    # Get the solver function from the map
    solver = solvers.get(solver_name)

    if solver is None:
        return jsonify({'error': 'Invalid solver'}), 400

    # Return a function that, when called, will run the solver
    def run_solver_and_get_result():
        return solver()

    # Return the function as JSON
    return jsonify(run_solver_and_get_result)


if __name__ == '__main__':
    app.run (debug=True)
