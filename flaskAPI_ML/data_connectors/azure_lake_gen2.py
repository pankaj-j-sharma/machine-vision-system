import os
import uuid
import sys
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core._match_conditions import MatchConditions
from azure.storage.filedatalake._models import ContentSettings


class Azure_DL_Gen2:

    def __init__(self, own=None, ttl=None):
        self.__BASEURL = 'https://{account}.dfs.core.windows.net'
        if ttl:
            self.AZ_STORAGE_ACCOUNT_NAME = 'blob11test'
            self.AZ_STORAGE_ACCOUNT_KEY = 'qRr0nX41fySRG/UAi2LRUdRlVfC5odvECXkiuWg40CJ7HDP4oNp8Lg+rWOchc0PXtF3+mJVnG5dKnE4bJ31p/A=='
            self.__create_client()

    def setup_connection(self, storage_account_name, storage_account_key):
        self.AZ_STORAGE_ACCOUNT_NAME = storage_account_name
        self.AZ_STORAGE_ACCOUNT_KEY = storage_account_key
        self.__create_client()

    def __create_client(self):
        self.client = DataLakeServiceClient(account_url=self.__BASEURL.format(
            account=self.AZ_STORAGE_ACCOUNT_NAME), credential=self.AZ_STORAGE_ACCOUNT_KEY)

    def get_all_filesystems(self):
        self.all_filesystems = self.client.list_file_systems()
        response = {'success': 'Data retrieved', 'results': []}
        for f in self.all_filesystems:
            response['results'].append(f.name)
        return response

    def select_filesystem(self, filesystem_name):
        self.filesystem = self.client.get_file_system_client(filesystem_name)

    def get_filesystem_items(self, prefix=""):
        response = {'success': 'Data retrieved', 'results': []}
        all_dirs = []
        all_files = []
        for item in self.filesystem.get_paths(path=prefix, recursive=False):
            name = item.name[len(prefix):].replace('/', '')
            if os.path.splitext(name)[-1] == '':
                all_dirs.append(name)
            else:
                all_files.append(name)

        respResults = {'all_directories': all_dirs, 'all_files': all_files}
        response['results'] = respResults
        return response

    def get_container_items(self, prefix=""):
        prefix = prefix+"/" if prefix != "" else ""  # needs to be appended
        response = {'success': 'Data retrieved', 'results': []}
        all_dirs = []
        all_files = []
        for item in self.container.get_paths(path=prefix,recursive=False):
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
    objAzDl = Azure_DL_Gen2(ttl=True)
    objAzDl.select_filesystem('my-file-system')
    objAzDl.get_filesystem_items('my-directory/my-directory-1/my-directory-11')
