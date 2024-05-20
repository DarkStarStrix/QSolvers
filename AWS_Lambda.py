import json
import boto3
from botocore.vendored import requests


def lambda_handler(event, context):
    print ('Received event: ' + json.dumps (event))

    # Get the service resource
    braket = boto3.client ('braket')

    if event ['RequestType'] == 'Create':
        # Create a new Amazon Braket task
        response = braket.create_quantum_task (
            # Add your task parameters here
        )
        print ('Created task: ' + json.dumps (response))

        # Send a response to CloudFormation acknowledging that the resource was created
        send_response (event, context, "SUCCESS", {"TaskId": response ['taskArn']})

    elif event ['RequestType'] == 'Update':
        # Update the Amazon Braket task
        response = braket.update_quantum_task (
            # Add your task ID and updated parameters here
        )
        print ('Updated task: ' + json.dumps (response))

        # Send a response to CloudFormation acknowledging that the resource was updated
        send_response (event, context, "SUCCESS", {"TaskId": response ['taskArn']})

    elif event ['RequestType'] == 'Delete':
        # Delete the Amazon Braket task
        response = braket.cancel_quantum_task (
            # Add your task ID here
        )
        print ('Deleted task: ' + json.dumps (response))

        # Send a response to CloudFormation acknowledging that the resource was deleted
        send_response (event, context, "SUCCESS", {"TaskId": response ['taskArn']})


def send_response(event, context, response_status, response_data):
    response_body = json.dumps ({
        "Status": response_status,
        "Reason": "See the details in CloudWatch Log Stream: " + context.log_stream_name,
        "PhysicalResourceId": context.log_stream_name,
        "StackId": event ['StackId'],
        "RequestId": event ['RequestId'],
        "LogicalResourceId": event ['LogicalResourceId'],
        "Data": response_data
    })

    headers = {
        'content-type': '',
        'content-length': str (len (response_body))
    }

    response = requests.put (event ['ResponseURL'], data=response_body, headers=headers)
    print ('Status code: ' + response.reason)
