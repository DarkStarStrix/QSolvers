from flask import Flask, request, jsonify
import Quantum_Genetic_Algorithm
import base64
from io import BytesIO

app = Flask (__name__)


@app.route ('/run_algorithm', methods=['POST'])
def run_algorithm_route():
    # Extract the algorithm name and the number of cities from the request
    data = request.get_json ()
    print(f"Received data: {data}")  # Print the request data
    algorithm_name = data.get ('algorithm')
    num_cities = data.get ('num_cities')

    # Run the selected algorithm with the provided number of cities
    try:
        result = run_algorithm (algorithm_name, num_cities)
    except ValueError:
        return jsonify ({"error": "Invalid algorithm name"}), 400

    # Process the results and prepare them for display
    processed_results = process_results (result)

    # Return the processed results
    return jsonify ({"result": processed_results}), 200


def process_results(result):
    # Convert the matplotlib figure to a PNG image
    png_image = BytesIO ()
    result ['plot'].savefig (png_image, format='png')

    # Move to the beginning of the BytesIO object
    png_image.seek (0)

    # Convert the PNG image to a base64 string
    png_image_b64_string = base64.b64encode (png_image.read ()).decode ('utf-8')

    # Prepare the results for sending
    processed_results = {
        'plot': png_image_b64_string,
        'counts': result ['counts'],
        'circuit': result ['circuit'],
    }

    return processed_results


if __name__ == '__main__':
    app.run (debug=False)
