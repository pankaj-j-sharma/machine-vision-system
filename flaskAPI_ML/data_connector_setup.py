from genericpath import exists
from data_connectors.awsconnect import AWS_S3Connect
from data_connectors.azure_blob import Azure_Blob
from data_connectors.azure_lake_gen2 import Azure_DL_Gen2
# from data_connectors.gcp_bq_connect import GCP_BigQuery
# from data_connectors.gcp_cs_connect import GCP_CloudStorage
# from data_connectors.gcp_ds_connect import GCP_CloudDataStore
from data_connectors.mysql_connect import MySQLConnector
from data_connectors.oracle_connect import OracleConnector
from data_connectors.postgresql_connect import PostgreSqlConnector
from data_connectors.snowflake_connect import SnowflakeConnector
from data_connectors.sqlserver_connect import SQLServerConnector
from logger import Logger
from sqlite_db_connect import SQLiteConnect
import os
from datetime import datetime
import json

now = datetime.now().strftime('%Y-%M-%d %H:%M:%S')


class DataConnector(SQLiteConnect,Logger):
    def ___init__(self):
        super.__init__()


def getAllDataConnectors(data):
    connector = DataConnector()

    response = {'success': 'Data retrieved', 'results': []}
    results = []
    tmpData = connector.getRecords(
        'select dc.id "CategoryId",dc.Name"CategoryName",dt.id"DataConnectorId",dt.Name"DataConnectorName",dt.Icon"DataConnectorIcon" from DATA_CONNECTOR_TYPES dt inner JOIN DATA_CONNECTOR_CATEGORY dc on dt.Category = dc.id where dt.Status="Active" and dc.Status="Active" order by 1,3 ')
    results.append({'id': 100, 'name': 'All', 'subOptions': []})
    for tmp in tmpData:
        for res in results:
            if res['id'] == tmp['CategoryId']:
                if tmp['DataConnectorId'] not in [opt['id'] for opt in res['subOptions']]:
                    res['subOptions'].append(
                        {'id': tmp['DataConnectorId'], 'name': tmp['DataConnectorName'], 'icon': tmp['DataConnectorIcon']})
        if not results or tmp['CategoryId'] not in [r['id'] for r in results]:
            results.append(
                {'id': tmp['CategoryId'], 'name': tmp['CategoryName'], 'subOptions': [{'id': tmp['DataConnectorId'], 'name': tmp['DataConnectorName'], 'icon': tmp['DataConnectorIcon']}]})

    response['results'] = results
    connector.log_message('get all Connectors','data',data,'response',response)

    return response


def getConnectorData(data):
    connector = DataConnector()
    connector.log_message('get Connectors data','data',data)

    formData = data['form']
    prefix = formData['prefix'] if 'prefix' in formData else ''

    # ##################### GCP Big Query connector ###################
    # if formData['connector_name'] == 'Google BigQuery':
    #     gcpbqConObj = GCP_BigQuery()
    #     isCredsPresent = gcpbqConObj.exists_credential_file(data)
    #     if data['files'] and not isCredsPresent:
    #         gcpbqConObj.save_credential_file(data)
    #         gcpbqConObj.setup_connection()
    #         return gcpbqConObj.get_all_datasets()
    #     else:
    #         gcpbqConObj.setup_connection()
    #         if 'dataset_id' in formData and formData['dataset_id']:
    #             return gcpbqConObj.get_all_tables_in_dataset(formData['dataset_id'])
    #         else:
    #             return gcpbqConObj.get_all_datasets()

    # ##################### GCP Cloud Datastore connector ###################
    # if formData['connector_name'] == 'Google Cloud Datastore':
    #     gcpdsConObj = GCP_CloudDataStore()
    #     isCredsPresent = gcpdsConObj.exists_credential_file(data)
    #     if data['files'] and not isCredsPresent:
    #         gcpdsConObj.save_credential_file(data)
    #     gcpdsConObj.setup_connection()
    #     return gcpdsConObj.get_all_kinds()

    # ##################### GCP Cloud Storage connector ###################
    # if formData['connector_name'] == 'Google Cloud Storage':
    #     gcpcsConObj = GCP_CloudStorage()
    #     isCredsPresent = gcpcsConObj.exists_credential_file(data)
    #     if data['files'] and not isCredsPresent:
    #         gcpcsConObj.save_credential_file(data)
    #     gcpcsConObj.setup_connection()
    #     if 'bucket_name' in formData and formData['bucket_name'] != '':
    #         gcpcsConObj.select_bucket(formData['bucket_name'])
    #         return gcpcsConObj.get_bucket_items()
    #     else:
    #         return gcpcsConObj.get_all_buckets()

    ##################### Amazon S3 Bucket connector ####################
    if formData['connector_name'] == 'Amazon S3 Bucket':
        awsConObj = AWS_S3Connect()
        awsConObj.setup_connection(
            aws_access_key_id=formData['access_key_id'], aws_secret_access_key=formData['access_key_secret'], bucket_name=formData['bucket_name'])
        return awsConObj.get_bucket_items(prefix=prefix)

    ##################### Azure Blob Storage connector ###################
    if formData['connector_name'] == 'Azure Blob Storage':
        azblConObj = Azure_Blob()
        azblConObj.setup_connection(
            storage_account_name=formData['storage_account_name'], storage_account_key=formData['storage_account_key'])
        if 'container_name' in formData and formData['container_name']:
            azblConObj.select_container(formData['container_name'])
            return azblConObj.get_container_items(prefix=prefix)
        else:
            return azblConObj.get_all_containers()

    ############## Azure Data Lake Storage Gen2 connector #################
    if formData['connector_name'] == 'Azure Data Lake Storage Gen2':
        azdlgen2 = Azure_DL_Gen2()
        azdlgen2.setup_connection(
            storage_account_name=formData['storage_account_name'], storage_account_key=formData['storage_account_key'])
        if 'filesystem_name' in formData and formData['filesystem_name']:
            azdlgen2.select_filesystem(formData['filesystem_name'])
            return azdlgen2.get_filesystem_items(prefix=prefix)
        else:
            return azdlgen2.get_all_filesystems()

    ####################### Snowflakes connector ###########################
    if formData['connector_name'] == 'Snowflake':
        snowflakecn = SnowflakeConnector()
        snowflakecn.setup_connection(
            username=formData['username'], password=formData['password'], accountname=formData['account_name'])

        if 'warehouse_name' in formData and formData['warehouse_name']:
            snowflakecn.get_warehouses()
            snowflakecn.select_warehouse(formData['warehouse_name'])
            if 'database_name' in formData and formData['database_name']:
                snowflakecn.get_databases()
                snowflakecn.select_database(formData['database_name'])
                return snowflakecn.get_tables()
            return snowflakecn.get_databases()
        else:
            return snowflakecn.get_warehouses()

    ####################### POstgreSQL Db connector #########################
    if formData['connector_name'] == 'PostgreSQL Database':
        postgrescn = PostgreSqlConnector()
        postgrescn.setup_connection(
            username=formData['username'], password=formData['password'], hostname=formData['hostname'],  database=formData['database_name'],  port=formData['portno'])

        if 'database_name' in formData and formData['database_name']:
            postgrescn.get_databases()
            postgrescn.select_database(formData['database_name'])
            return postgrescn.get_tables()
        return postgrescn.get_databases()

    ####################### MySQL Db connector #########################
    if formData['connector_name'] == 'MySQL Database':
        mysqlcn = MySQLConnector()
        mysqlcn.setup_connection(
            username=formData['username'], password=formData['password'], hostname=formData['hostname'],  database=formData['database_name'],  port=formData['portno'])

        if 'database_name' in formData and formData['database_name']:
            mysqlcn.get_databases()
            mysqlcn.select_database(formData['database_name'])
            return mysqlcn.get_tables()
        return mysqlcn.get_databases()

    ####################### Oracle Db connector #########################
    if formData['connector_name'] == 'Oracle Database':
        oraclecn = OracleConnector()
        oraclecn.setup_connection(
            username=formData['username'], password=formData['password'], hostname=formData['hostname'],  sid=formData['sid'],  port=formData['portno'])

        if 'sid' in formData and formData['sid']:
            oraclecn.get_databases()
            oraclecn.select_database(formData['sid'])
            return oraclecn.get_tables()
        return oraclecn.get_databases()

    ##################### SQL Server Db connector ########################
    if formData['connector_name'] == 'SQL Server Database':
        sqlsvrcn = SQLServerConnector()
        sqlsvrcn.setup_connection(server=formData['server'], database=formData['database_name'], instance=formData['instance_name'],
                                  username=formData['username'], password=formData['password'], hostname=formData['hostname'],  sid=formData['sid'],  port=formData['portno'], trusted_conn=formData['trusted_conn'])

        if 'database_name' in formData and formData['database_name']:
            sqlsvrcn.get_databases()
            sqlsvrcn.select_database(formData['database_name'])
            return sqlsvrcn.get_tables()
        return sqlsvrcn.get_databases()
