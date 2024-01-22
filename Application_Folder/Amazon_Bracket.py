import boto3
from braket.aws import AwsDevice
from braket.circuits import Circuit

# Specify the S3 bucket to which your task results should be uploaded
my_bucket = "" # the name of the bucket
my_prefix = "your-folder-name" # the name of the folder in the bucket
s3_folder = (my_bucket, my_prefix)

# Build your circuit
circuit = Circuit().h(0).cnot(0, 1)

# Specify the device
device = AwsDevice("arn:aws:braket:::device/quantum-simulator/amazon/sv1")

# Create the task
task = device.run(circuit, s3_folder, shots=1000)

# Print out the result
result = task.result()
counts = result.measurement_counts
print(counts)
