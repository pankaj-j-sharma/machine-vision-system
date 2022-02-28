from awsconnect import *


def launch_connector(data):
    if data['type'] == 'S3':
        awsConObj = AWS_S3Connect()
        return awsConObj.get_bucket_items(prefix=data['prefix'])
