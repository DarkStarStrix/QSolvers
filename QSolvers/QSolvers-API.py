# QSolvers-API.py

from flask import Flask, request, jsonify
import importlib

app = Flask (__name__)


@app.route ('/quantum_linear_solvers', methods=['POST'])
def quantum_linear_solvers_route():
    data = request.get_json ()
    solver_name = 'Quantum_Linear_Solvers'
    solver_module = importlib.import_module (f"bosonic.{solver_name}")
    solver_class = getattr (solver_module, solver_name)
    solver = solver_class (data)
    result = solver.solve ()
    return jsonify (result)


@app.route ('/quantum_walk_solvers', methods=['POST'])
def quantum_walk_solvers_route():
    data = request.get_json ()
    solver_name = 'Quantum_Walk_Solvers'
    solver_module = importlib.import_module (f"bosonic.{solver_name}")
    solver_class = getattr (solver_module, solver_name)
    solver = solver_class (data)
    result = solver.solve ()
    return jsonify (result)


@app.route ('/oracle_solvers', methods=['POST'])
def oracle_solvers_route():
    data = request.get_json ()
    solver_name = 'Oracle_Solvers'
    solver_module = importlib.import_module (f"bosonic.{solver_name}")
    solver_class = getattr (solver_module, solver_name)
    solver = solver_class (data)
    result = solver.solve ()
    return jsonify (result)


@app.route ('/Bosonic-chemistry', methods=['POST'])
def bosonic_chemistry_route():
    data = request.get_json ()
    solver_name = 'Bosonic_Chemistry'
    solver_module = importlib.import_module (f"bosonic.{solver_name}")
    solver_class = getattr (solver_module, solver_name)
    solver = solver_class (data)
    result = solver.solve ()
    return jsonify (result)


@app.route ('/Bosonic-Cryptography', methods=['POST'])
def bosonic_cryptography_route():
    data = request.get_json ()
    solver_name = 'Bosonic_Cryptography'
    solver_module = importlib.import_module (f"bosonic.{solver_name}")
    solver_class = getattr (solver_module, solver_name)
    solver = solver_class (data)
    result = solver.solve ()
    return jsonify (result)


@app.route ('/Bosonic-Quantum-Key-Distribution', methods=['POST'])
def bosonic_quantum_key_distribution_route():
    data = request.get_json ()
    solver_name = 'Bosonic_Quantum_Key_Distribution'
    solver_module = importlib.import_module (f"bosonic.{solver_name}")
    solver_class = getattr (solver_module, solver_name)
    solver = solver_class (data)
    result = solver.solve ()
    return jsonify (result)


@app.route ('/Bosonic-Quantum-Quantum-Machine-Learning', methods=['POST'])
def bosonic_quantum_machine_learning_route():
    data = request.get_json ()
    solver_name = 'Bosonic_Quantum_Machine_Learning'
    solver_module = importlib.import_module (f"bosonic.{solver_name}")
    solver_class = getattr (solver_module, solver_name)
    solver = solver_class (data)
    result = solver.solve ()
    return jsonify (result)


@app.route ('/Bosonic-Quantum-Finance', methods=['POST'])
def bosonic_quantum_finance_route():
    data = request.get_json ()
    solver_name = 'Bosonic_Quantum_Finance'
    solver_module = importlib.import_module (f"bosonic.{solver_name}")
    solver_class = getattr (solver_module, solver_name)
    solver = solver_class (data)
    result = solver.solve ()
    return jsonify (result)


@app.route ('/quantum_linear_solvers', methods=['GET'])
def quantum_linear_solvers_route_get():
    return jsonify ({'message': 'GET request is not supported for this route'})


@app.route ('/quantum_walk_solvers', methods=['GET'])
def quantum_walk_solvers_route_get():
    return jsonify ({'message': 'GET request is not supported for this route'})


if __name__ == '__main__':
    app.run (debug=True)
