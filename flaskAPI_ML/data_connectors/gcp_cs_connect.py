from google.cloud import storage
from google.oauth2 import service_account
import json
import os

from data_connectors.gcp_bq_connect import GCP_BigQuery

class GCP_CloudStorage(GCP_BigQuery):
    def __init__(self, own=None, ttl=None):
        super().__init__(own, ttl)
        self.app_name='GoogleCloudStorage'


    def setup_connection(self):
        self.__create_client()

    def __create_client(self):
        with open(self.GOOGLE_CREDENTIALS_PATH) as f:
            self.credentials = json.load(f)
        self.svc_credentials = service_account.Credentials.from_service_account_info(
            self.credentials)
        self.client = storage.Client(credentials=self.svc_credentials)

    def get_all_buckets(self):
        response = {'success': 'Data retrieved', 'results': []}
        all_buckets=[]
        all_dirs = []
        all_files = []

        for bucket in self.client.list_buckets():
            all_buckets.append(bucket.name)

        respResults = {'all_buckets':all_buckets,'all_directories': all_dirs, 'all_files': all_files}
        response['results'] = respResults
        return response


    def select_bucket(self, bucket_name):
        self.BUCKET_NAME = bucket_name


    def get_bucket_items(self, recursive=False, prefix=""):
    
        delimiter = "/" if recursive else ""
        response = {'success': 'Data retrieved', 'results': []}
        all_dirs = []
        all_files = []
        for page in self.client.list_blobs(self.BUCKET_NAME,max_results=10).pages:
            for blob in page:
                item = blob.name.split("\\")
                item = blob.name.split(
                    "/") if "\\" not in blob.name else item
                for subFolders in prefix.split('\\'):
                    if subFolders and subFolders != '':
                        item.pop(item.index(subFolders))
                if item and item[0] and item[0] not in all_dirs and os.path.splitext(item[0])[-1] == '':
                    all_dirs.append(item[0])
                if item and item[0] and item[0] not in all_dirs and os.path.splitext(item[0])[-1] != '':
                    all_files.append(item[0])

        respResults = {'all_directories': all_dirs, 'all_files': all_files}
        response['results'] = respResults
        return response
