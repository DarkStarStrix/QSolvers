from flask import Flask, request
from docker import from_env

app = Flask (__name__)
cli = from_env ()


@app.route ('/run_algorithm', methods=['POST'])
def run_algorithm():
    algorithm = request.json ['algorithm']
    parameters = request.json ['parameters']

    container = cli.containers.run (algorithm, parameters, detach=True)

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
