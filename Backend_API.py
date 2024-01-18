from flask import Flask, request, send_from_directory
from docker import from_env
import os

app = Flask (__name__, static_folder='C:/Users/kunya/PycharmProjects/QuantumHybrid_Hyperoptimization')
cli = from_env ()


@app.route ('/')
def index():
    return send_from_directory (app.static_folder, 'Index.html')


@app.route ('/run_algorithm', methods=['POST'])
def run_algorithm():
    algorithm = request.json ['algorithm']
    parameters = request.json ['parameters']

    # Get the full path to the script
    script_path = os.path.join ('C:/Users/kunya/PycharmProjects/QuantumHybrid_Hyperoptimization/Solution_Code',
                                algorithm)

    container = cli.containers.run (script_path, parameters, detach=True)

    return container.logs ()


@app.route ('/get_images', methods=['GET'])
def get_images():
    images = cli.images.list ()

    return str (images)


@app.route ('/get_containers', methods=['GET'])
def get_containers():
    containers = cli.containers.list (all=True)

    return str (containers)


if __name__ == '__main__':
    app.run (debug=True, host='', port=5000)
