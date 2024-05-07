# QSolvers-API.py

from flask import Flask, request, jsonify
import importlib

app = Flask (__name__)


@app.route ('/<solver_name>', methods=['POST'])
def solver_route(solver_name):
    data = request.get_json ()
    solver_module = importlib.import_module (f"QSolvers.{solver_name}")
    solver_class = getattr (solver_module, solver_name)
    solver = solver_class (data)
    result = solver.solve ()
    return jsonify (result)


@app.route ('/<solver_name>', methods=['GET'])
def solver_route_get(solver_name):
    return jsonify ({'message': f'GET request is not supported for the {solver_name} route'})


if __name__ == '__main__':
    app.run (debug=False)
