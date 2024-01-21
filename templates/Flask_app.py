from flask import Flask, render_template, request, jsonify
import Quantum_Genetic_Algorithm
import Quantum_Convex
import Quantum_Annealing
import Quantum_A

app = Flask (__name__)


@app.route ('/')
def home():
    return render_template ('index.html')


@app.route ('/run_algorithm', methods=['POST'])
def run_algorithm():
    data = request.get_json ()
    algorithm = data.get ('algorithm')
    run_func = algorithm_mapping.get (algorithm)

    if run_func is None:
        return jsonify ({'error': 'Invalid algorithm'}), 400

    result = run_func ()
    return jsonify (result)


algorithm_mapping = {
    'Quantum Genetic Algorithm': Quantum_Genetic_Algorithm.run,
    'Quantum Convex Hull Algorithm': Quantum_Convex.run,
    'Quantum Annealing': Quantum_Annealing.run,
    'Quantum A* Algorithm': Quantum_A.run
}


if __name__ == '__main__':
    app.run (debug=True)
