import os
import boto3

os.environ ['AWS_ACCESS_KEY_ID'] = 'AKIA5XRUVWG2T5RUJDVU'
os.environ ['AWS_SECRET_ACCESS_KEY'] = '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'

s3transfer = boto3.client ('s3', region_name='us-east-1')


def create_app_version(app_name, bucket_name):
    beanstalk = boto3.client ('elasticbeanstalk', region_name='us-east-1')
    source_bundle = {'S3Bucket': bucket_name, 'S3Key': 'Application_Folder'}
    beanstalk.create_application_version (ApplicationName=app_name, VersionLabel='v1', SourceBundle=source_bundle)


def update_environment(app_name, env_name):
    beanstalk = boto3.client ('elasticbeanstalk', region_name='us-east-1')
    beanstalk.update_environment (ApplicationName=app_name, EnvironmentName=env_name, VersionLabel='v1')


def main():
    create_app_version ('qsolvers', 'qsolvers')
    update_environment ('qsolvers', 'qsolvers')


if __name__ == "__main__":
    main ()
