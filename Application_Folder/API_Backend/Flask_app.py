from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_talisman import Talisman
from flask_limiter import Limiter
from celery import Celery
from Solution_Code import Quantum_Genetic_Algorithm, Quantum_Convex, Quantum_Annealing, Quantum_A
from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from tests import run_tests
from amazon_braket_sdk import AmazonBraketClient
import logging

# Initialize Flask app
app = Flask (__name__)
Talisman (app)
limiter = Limiter (app, key_func=get_remote_address)

# Configure logging
logging.basicConfig (filename='server.log', level=logging.INFO)

# Configure Flask-Login
login_manager = LoginManager ()
login_manager.init_app (app)


# User model
class User (UserMixin):
    def __init__(self, id2):
        self.id = id


# This should be replaced with actual user loader from your user database
@login_manager.user_loader
def load_user(user_id):
    return User (user_id)


@app.route ('/login', methods=['GET', 'POST'])
def login():
    # This should be replaced with actual authentication logic
    user = User (request.form ['username'])
    login_user (user)
    return redirect (url_for ('home'))


@app.route ('/logout')
@login_required
def logout():
    logout_user ()
    return redirect (url_for ('home'))


@app.route ('/')
@login_required
def home():
    return render_template ('index.html')


@app.route ('/run_algorithm', methods=['POST'])
@limiter.limit ("10/month")  # Limit user to 10 runs per month
@login_required
def run_algorithm():
    data = request.get_json ()
    algorithm = data.get ('algorithm')
    run_func = algorithm_mapping.get (algorithm)

    if run_func is None:
        return jsonify ({'error': 'Invalid algorithm'}), 400

    # Run tests on the selected algorithm
    test_result = run_tests (algorithm)
    if not test_result:
        return jsonify ({'error': 'Tests failed'}), 400

    # Add the algorithm execution to the Celery task queue
    task = execute_algorithm.delay (run_func)

    return jsonify ({'task_id': task.id}), 202


@celery.task (bind=True)
def execute_algorithm(self, run_func):
    # Execute the algorithm
    result = run_func ()

    # Send the algorithm to Amazon Braket for execution
    task = client.create_quantum_task (result)
    result = client.get_quantum_task (task ['taskId'])

    return result


@app.route ('/check_task/<task_id>', methods=['GET'])
@login_required
def check_task(task_id):
    task = execute_algorithm.AsyncResult (task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result,
        }
        if 'result' in response:
            response ['status'] = task.result.get ('status', 'No status found')
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'status': str (task.info),  # this is the exception raised
        }
    return jsonify (response)


ALGORITHMS = {
    'Quantum Genetic Algorithm': Quantum_Genetic_Algorithm.run,
    'Quantum Convex Hull Algorithm': Quantum_Convex.run,
    'Quantum Annealing': Quantum_Annealing.run,
    'Quantum A* Algorithm': Quantum_A.run,
    'Quantum Ant Colony Optimization': Quantum_ACO.run,
    'Quantum particle swarm optimization': Quantum_PSO.run,
}


def get_algorithm(name):
    if name not in ALGORITHMS:
        raise ValueError (f'Invalid algorithm: {name}')
    return ALGORITHMS [name]


@app.route ('/run_algorithm', methods=['POST'])
@limiter.limit ("10/month")  # Limit user to 10 runs per month
@login_required
def run_algorithm():
    data = request.get_json ()
    algorithm_name = data.get ('algorithm')
    try:
        run_func = get_algorithm (algorithm_name)
    except ValueError as e:
        return jsonify ({'error': str (e)}), 400

    # Run tests on the selected algorithm
    test_result = run_tests (algorithm_name)
    if not test_result:
        return jsonify ({'error': 'Tests failed'}), 400

    # Add the algorithm execution to the Celery task queue
    task = execute_algorithm.delay (run_func)

    return jsonify ({'task_id': task.id}), 202


@app.route ('/check_task/<task_id>', methods=['GET'])
@login_required
def check_task(task_id):
    task = execute_algorithm.AsyncResult (task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result,
        }
        if 'result' in response:
            response ['status'] = task.result.get ('status', 'No status found')
    else:
        # something went wrong in the background job
        response = {
            'state': task.state,
            'status': str (task.info),  # this is the exception raised
        }
    return jsonify (response)


if __name__ == '__main__':
    app.run (debug=True)
