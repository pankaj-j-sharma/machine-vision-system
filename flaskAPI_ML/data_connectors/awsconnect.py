from sys import prefix
import boto3
from botocore.client import Config
import os


class AWS_S3Connect:

    def __init__(self, own=None, ttl=None):
        # for testing purpose internal TTL S3 bucket
        # self.BUCKETNAME = "aimls3bucketnew"
        # self.AWS_ACCESS_KEY_ID = 'AKIAVBBMHWBWTPLMTWNY'
        # self.AWS_SECRET_ACCESS_KEY = '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'
        # self.REGION = 'us-east-2'
        if ttl:
            print('default s3 initialisation')
            self.BUCKETNAME = "aimls3bucketnew"
            self.AWS_ACCESS_KEY_ID = 'AKIA4IJ7W74CLKGEIUGC'
            self.AWS_SECRET_ACCESS_KEY = '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'
            self.REGION = 'ap-south-1'
            self.SERVICENAME = 's3'
            self.__create_client()

    def setup_connection(self, aws_access_key_id, aws_secret_access_key, bucket_name):
        self.AWS_ACCESS_KEY_ID = aws_access_key_id
        self.AWS_SECRET_ACCESS_KEY = aws_secret_access_key
        self.BUCKETNAME = bucket_name
        self.SERVICENAME = 's3'
        self.__create_client()

    def __create_client(self):
        # Creating the low level functional client
        self.client = boto3.client(
            service_name=self.SERVICENAME,
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            config=Config(signature_version='s3v4'),
            # region_name=self.REGION,
        )

        # Creating the high level object oriented interface
        self.resource = boto3.resource(
            service_name=self.SERVICENAME,
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY,
            # region_name=self.REGION,
        )

        self.session = boto3.Session(
            aws_access_key_id=self.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=self.AWS_SECRET_ACCESS_KEY)

        self.s3 = self.session.resource('s3')
        self.my_bucket = self.s3.Bucket(self.BUCKETNAME)

    def get_all_buckets(self):
        # Print out bucket names
        if self.resource:
            try:
                for bucket in self.resource.buckets.all():
                    print('Bucket -> ', bucket.name)
                    for obj in bucket.objects.all():
                        print('Items ->', obj.key)
            except Exception as e:
                print('err', str(e))
            finally:
                for obj in self.client.list_objects(Bucket=self.BUCKETNAME, Prefix="", Delimiter="")['Contents']:
                    print(obj['Key'])
                    # break

    def add_item_to_bucket(self, files):
        pass

    def get_bucket_items(self, recursive=False, prefix=""):

        delimiter = "/" if recursive else ""
        response = {'success': 'Data retrieved', 'results': []}
        all_dirs = []
        all_files = []

        # Print out bucket names
        if self.resource:
            try:
                # Method 1
                for bucket in self.resource.buckets.all():
                    print('Bucket -> ', bucket.name)
                    self.BUCKETNAME = bucket.name
                    for obj in bucket.objects.all():
                        print('Items ->', obj.key)
            except Exception as e:
                print('err', str(e))
            finally:

                # Method 2
                self.paginator = self.client.get_paginator('list_objects')
                self.page_iterator = self.paginator.paginate(
                    Bucket=self.BUCKETNAME, Prefix=prefix, Delimiter=delimiter, MaxKeys=10)

                # print([f for f in self.page_iterator.search('CommonPrefixes') if f]) # search only for Common Prefixes

                for i, page in enumerate(self.page_iterator):
                    allObjects = page['Contents']
                    for obj in allObjects:
                        item = obj['Key'].split("\\")
                        item = obj['Key'].split(
                            "/") if "\\" not in obj['Key'] else item
                        for subFolders in prefix.split('\\'):
                            if subFolders and subFolders != '':
                                item.pop(item.index(subFolders))
                        if item and item[0] and item[0] not in all_dirs and os.path.splitext(item[0])[-1] == '':
                            all_dirs.append(item[0])
                        print(i, end='\r')
                    # print('-'*100,end='\r')
                    # break

                if not all_dirs:
                    for i, page in enumerate(self.page_iterator):
                        allObjects = page['Contents']
                        for obj in allObjects:
                            item = obj['Key'].split("\\")
                            item = obj['Key'].split(
                                "/") if "\\" not in obj['Key'] else item
                            for subFolders in prefix.split('\\'):
                                if subFolders and subFolders != '':
                                    item.pop(item.index(subFolders))
                            if item and item[0] and item[0] not in all_dirs and os.path.splitext(item[0])[-1] != '':
                                all_files.append(item[0])
                            print(i, end='\r')
                    print('no more sub folders in ', prefix, all_files)
                else:
                    print('all directories in', prefix, all_dirs)
        else:
            response['error'] = 'unable to connect'
        respResults = {'all_directories': all_dirs, 'all_files': all_files}
        response['results'] = respResults
        return response
