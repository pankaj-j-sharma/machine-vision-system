from os import path
import boto3
from botocore.retries import bucket
import pandas
import os
from botocore.client import Config
from urllib.parse import unquote, parse_qs, urlsplit


class S3Wrapper:

    def __init__(self, own=None):
        if not own:
            self.BUCKETNAME = "aimls3bucketnew"
            self.AWS_ACCESS_KEY_ID = 'AKIA4IJ7W74CLKGEIUGC'
            self.AWS_SECRET_ACCESS_KEY = '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'
        else:
            self.BUCKETNAME = "aimls3bucketnewps"
            self.AWS_ACCESS_KEY_ID = 'AKIA6E2NPJ5RNXBIDF4U'
            self.AWS_SECRET_ACCESS_KEY = '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'

        self.__create_client()

    def get_content_type(self, type):
        content_types = {
            "txt": "text/plain",
            "img": "image/jpeg",
            "vid": "video/mpeg"
        }
        return content_types[type]

    def __create_client(self):
        # Creating the low level functional client
        self.client = boto3.client(
            's3',
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            config=Config(signature_version='s3v4')
            # region_name = 'ap-south-1'
        )

        # Creating the high level object oriented interface
        self.resource = boto3.resource(
            's3',
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            # region_name = 'ap-south-1'
        )

    def get_all_files(self, prefix=""):
        objCounter = 0
        allObjects = self.client.list_objects(
            Bucket=self.BUCKETNAME, Prefix=prefix)['Contents']
        for f in allObjects:
            objCounter += 1
            file = self.client.get_object(
                Bucket=self.BUCKETNAME, Key=f['Key'])['Body']
            # content = file.read().decode("utf-8")
            url = self.client.generate_presigned_url('get_object', ExpiresIn=400000, Params={
                                                     'Bucket': self.BUCKETNAME, 'Key': f['Key']})
            query_str = parse_qs(urlsplit(url).query)
            rPath = unquote(urlsplit(url).path)
            print(f['Key'], '->', '\n', 'Url ->', rPath, '\n'*2)

        print('Total objects in the directory ', prefix, objCounter)

    def get_all_files_v2(self, prefix=""):
        fileCounter = 0
        allFiles = []
        self.paginator = self.client.get_paginator('list_objects')
        self.page_iterator = self.paginator.paginate(
            Bucket=self.BUCKETNAME, Prefix=prefix)

        for page in self.page_iterator:
            allObjects = page['Contents']
            for f in allObjects:
                file = self.client.get_object(
                    Bucket=self.BUCKETNAME, Key=f['Key'])['Body']
                # content = file.read().decode("utf-8")
                url = self.client.generate_presigned_url('get_object', ExpiresIn=400000, Params={
                    'Bucket': self.BUCKETNAME, 'Key': f['Key']})
                query_str = parse_qs(urlsplit(url).query)
                rPath = unquote(urlsplit(url).path)
                allFiles.append(
                    {'url': url, 'query_str': query_str, 'rPath': rPath})
                fileCounter += 1
                print(fileCounter, end="\r")
                # print(f['Key'], '->', '\n', 'Url ->', rPath, '\n'*2)
        print('Total objects in the directory ',
              prefix, len(allFiles), end='\n')
        return allFiles

    def upload_files(self, filename, source=os.getcwd(), destination='', contentType=None):
        if not contentType or contentType != "":
            contentType = "application/octet-stream"

        self.client.upload_file(os.path.join(
            source, filename), self.BUCKETNAME, os.path.join(destination, filename),
            ExtraArgs={"ContentDisposition": "inline", "ContentType": contentType})

    def get_file_url(self, key):
        try:
            self.client.head_object(Bucket=self.BUCKETNAME, Key=key)
            return self.client.generate_presigned_url('get_object', ExpiresIn=3600, Params={
                'Bucket': self.BUCKETNAME, 'Key': key})
        except:
            return None

    def remove_files(self, key):
        return self.client.delete_object(Bucket=self.BUCKETNAME, Key=key)


if __name__ == '__main__':
    objWrapper = S3Wrapper()
    print(objWrapper.BUCKETNAME)
    filePath = input('Enter File Path \n')
    if filePath:
        contentType = input('Enter File Type \n')
        objWrapper.upload_files(filePath)
        objWrapper.get_all_files("uploads")
    print('get file url ', objWrapper.get_file_url(
        "uploads\\Videos\\assemblyline5.mp4"))
