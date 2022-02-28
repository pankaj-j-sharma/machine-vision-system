from google.oauth2 import service_account
from google.cloud import datastore
import json
import os

from data_connectors.gcp_bq_connect import GCP_BigQuery


class GCP_CloudDataStore(GCP_BigQuery):
    def __init__(self, own=None, ttl=None):
        super().__init__(own, ttl)
        self.app_name = 'GoogleDataStore'

    def setup_connection(self):
        self.__create_client()

    def __create_client(self):
        with open(self.GOOGLE_CREDENTIALS_PATH) as f:
            self.credentials = json.load(f)
        self.svc_credentials = service_account.Credentials.from_service_account_info(
            self.credentials)
        self.client = datastore.Client(credentials=self.svc_credentials)

    def get_all_kinds(self):
        response = {'success': 'Data retrieved', 'results': []}
        all_tbl = [entity.key.id_or_name for entity in self.client.query(
            kind='__kind__').fetch() if not entity.key.id_or_name.startswith('__')]
        respResults = {'all_tbl': all_tbl}
        response['results'] = respResults
        return response
