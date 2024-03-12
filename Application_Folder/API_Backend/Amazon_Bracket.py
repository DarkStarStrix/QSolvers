import os
from braket.aws import AwsDevice
from braket.circuits import Circuit

# Get the S3 bucket and folder from environment variables
s3_folder = (os.getenv("AWS_BUCKET_NAME"), os.getenv("AWS_FOLDER_NAME"))

# Set up the AWS device and build your circuit
device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/tn1")
circuit = Circuit().h(0).cnot(0, 1)

# Create the task and print out the result
result = device.run(circuit, s3_folder, shots=1000).result()
print(result.measurement_counts)
