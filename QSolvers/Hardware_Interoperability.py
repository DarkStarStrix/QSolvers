import RPi.GPIO as GPIO
from flask import Flask, request, jsonify

app = Flask (__name__)

# Set the GPIO mode
GPIO.setmode (GPIO.BCM)

# Initialize a pin (replace 18 with your pin)
GPIO.setup (18, GPIO.OUT)


@app.route ('/hardware', methods=['POST'])
def hardware_route():
    data = request.get_json ()
    command = data.get ('command')

    # Write the command to the GPIO pin
    if command == 'on':
        GPIO.output (18, GPIO.HIGH)
    elif command == 'off':
        GPIO.output (18, GPIO.LOW)

    return jsonify ({'response': 'Command executed'})


if __name__ == '__main__':
    app.run (debug=False)
