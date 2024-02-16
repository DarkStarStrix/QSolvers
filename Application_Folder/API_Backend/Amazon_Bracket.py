import os
import boto3
from braket.aws import AwsDevice
from braket.circuits import Circuit

# Get the S3 bucket and folder from environment variables
my_bucket = os.getenv("AWS_BUCKET_NAME")
my_prefix = os.getenv("AWS_FOLDER_NAME")
s3_folder = (my_bucket, my_prefix)

# Set up the AWS device
device = AwsDevice ("arn:aws:braket:::device/quantum-simulator/amazon/tn1")

# Build your circuit
circuit = Circuit ().h (0).cnot (0, 1)

# Specify the device
device = AwsDevice ("arn:aws:braket:::device/quantum-simulator/amazon/tn1")

# Create the task
task = device.run (circuit, s3_folder, shots=1000)

# Print out the result
result = task.result ()
counts = result.measurement_counts
print (counts)
