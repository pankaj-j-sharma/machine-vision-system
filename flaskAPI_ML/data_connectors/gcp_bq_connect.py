from audioop import reverse
import os
import json
from urllib import response
from google.cloud import bigquery
from google.oauth2 import service_account
from sqlalchemy import false
from pathlib import Path


class GCP_BigQuery:
    def __init__(self, own=None, ttl=None):
        self.app_name = 'GoogleBigQuery'
        if ttl:
            self.GOOGLE_CREDENTIALS_PATH = ''
            self.__create_client()

    def save_credential_file(self, data):
        formData = data['form']
        user = formData['userId']
        response = {'success': 'Data retrieved', 'results': []}

        pathList = [os.getcwd(), 'uploads', 'DataConnectors',
                    self.app_name, user]

        for i, _ in enumerate(pathList):
            if not os.path.isdir(os.path.join(*pathList[:i+1])):
                os.mkdir(os.path.join(*pathList[:i+1]))

        for file in data['files']:
            path = os.path.join(os.path.join(*pathList), file.filename)
            file.save(path)
            response['results'].append(path)
            self.GOOGLE_CREDENTIALS_PATH = path
        return response

    def exists_credential_file(self, data):
        user = data['form']['userId']
        response = False
        file_dir = os.path.join(os.getcwd(), 'uploads',
                                'DataConnectors', self.app_name, user)

        if os.path.isdir(file_dir):
            jsonFile = [f for f in os.listdir(file_dir) if f.endswith('json')]
            if jsonFile:
                response = True
                self.GOOGLE_CREDENTIALS_PATH = os.path.join(
                    file_dir, jsonFile[0])
                print('file exists ', self.GOOGLE_CREDENTIALS_PATH)
        return response

    def setup_connection(self):
        self.__create_client()

    def __create_client(self):
        with open(self.GOOGLE_CREDENTIALS_PATH) as f:
            self.credentials = json.load(f)
        self.svc_credentials = service_account.Credentials.from_service_account_info(
            self.credentials)
        self.client = bigquery.Client(credentials=self.svc_credentials)

    def get_all_datasets(self):
        self.all_datasets = self.client.list_datasets()
        response = {'success': 'Data retrieved', 'results': []}
        all_db = []
        all_tbl = []
        for c in self.all_datasets:
            all_db.append(c.dataset_id)
        respResults = {'all_db': all_db, 'all_tbl': all_tbl}
        response['results'] = respResults
        return response

    def get_all_tables_in_dataset(self, dataset_id):
        self.all_tables = self.client.list_tables(dataset_id)
        response = {'success': 'Data retrieved', 'results': []}
        all_db = []
        all_tbl = []
        for c in self.all_tables:
            all_tbl.append({'name': c.table_id, 'db': c.dataset_id})
        respResults = {'all_db': all_db, 'all_tbl': all_tbl}
        response['results'] = respResults
        return response
