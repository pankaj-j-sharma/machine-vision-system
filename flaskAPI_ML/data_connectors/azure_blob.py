import os
import uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__


class Azure_Blob:
    def __init__(self, own=None, ttl=None):
        self.__BASEURL = 'https://{account}.blob.core.windows.net'
        if ttl:
            print('default azure blob initialisation')
            self.AZ_STORAGE_ACCOUNT_NAME = 'blob11test'
            self.AZ_STORAGE_ACCOUNT_KEY = 'qRr0nX41fySRG/UAi2LRUdRlVfC5odvECXkiuWg40CJ7HDP4oNp8Lg+rWOchc0PXtF3+mJVnG5dKnE4bJ31p/A=='
            self.__create_client()

    def setup_connection(self, storage_account_name, storage_account_key):
        self.AZ_STORAGE_ACCOUNT_NAME = storage_account_name
        self.AZ_STORAGE_ACCOUNT_KEY = storage_account_key
        self.__create_client()

    def __create_client(self):
        self.client = BlobServiceClient(account_url=self.__BASEURL.format(
            account=self.AZ_STORAGE_ACCOUNT_NAME), credential=self.AZ_STORAGE_ACCOUNT_KEY)

    def get_all_containers(self):
        self.all_containers = self.client.list_containers()
        response = {'success': 'Data retrieved', 'results': []}
        for c in self.all_containers:
            response['results'].append(c.name)
        return response

    def select_container(self, container_name):
        self.container = self.client.get_container_client(container_name)

    def get_container_items(self, prefix=""):
        prefix = prefix+"/" if prefix != "" else ""  # needs to be appended
        response = {'success': 'Data retrieved', 'results': []}
        all_dirs = []
        all_files = []
        for item in self.container.walk_blobs(name_starts_with=prefix):
            name = item.name[len(prefix):]
            if os.path.splitext(name)[-1] == '':
                name = name.replace('/', '')
                all_dirs.append(name)
            else:
                all_files.append(name)

        respResults = {'all_directories': all_dirs, 'all_files': all_files}
        response['results'] = respResults
        return response


if __name__ == '__main__':
    obj = Azure_Blob(ttl=True)
    obj.get_all_containers()
    obj.select_container('my-file-system')
    obj.get_container_items('my-directory/my-directory-1/my-directory-11/')
