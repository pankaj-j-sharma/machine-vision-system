import snowflake.connector


class SnowflakeConnector:
    def __init__(self, own=None, ttl=None):
        if ttl:
            self.username = 'SKJ992'
            self.password = 'Sk3900@j21'
            self.accountname = 'XH82888.ap-south-1'
            self.__create_connection()

    def setup_connection(self, username, password, accountname):
        self.username = username
        self.password = password
        self.accountname = accountname
        self.__create_connection()

    def __create_connection(self):
        self.connection = snowflake.connector.connect(
            user=self.username, password=self.password, account=self.accountname)

    def _execute_query(self, query):
        self.dbexec_error = None
        self.cursor = self.connection.cursor()
        try:
            self.cursor.execute(query)
            self.record = self.__recordToJson()
        except Exception as e:
            self.dbexec_error = 'Error occurred for '+query+' '+str(e)
            self.record = None
        finally:
            self.cursor.close()

    def __recordToJson(self):
        colnames = [x[0] for x in self.cursor.description]
        tmprec = self.cursor.fetchall()
        records = []
        for rec in tmprec:
            records.append(dict(zip(colnames, rec)))
        return records

    def getrecords(self, query):
        self._execute_query(query)
        return self.record

    def get_warehouses(self):
        self.resp = {'success': 'Data retrieved',
                     'results': {'all_dwh': [], 'all_db': [], 'all_tbl': []}}
        resp = self.getrecords('SHOW WAREHOUSES')
        if resp:
            self.resp['results']['all_dwh'] = [r['name'] for r in resp]
        return self.resp

    def select_warehouse(self, warehouse):
        self.getrecords('USE WAREHOUSE {wh}'.format(wh=warehouse))
        if not self.dbexec_error and self.resp and 'success' in self.resp:
            self.resp['results']['dw'] = warehouse
        else:
            self.resp = {'error': self.dbexec_error,
                         'results': {'dw': warehouse}}
        return self.resp

    def get_databases(self):
        resp = self.getrecords('SHOW DATABASES')
        if resp:
            self.resp['results']['all_db'] = [r['name'] for r in resp]
            self.resp['results']['all_tbl'] = []
        return self.resp

    def select_database(self, database):
        resp = self.getrecords('USE {db}'.format(db=database))
        if not self.dbexec_error and self.resp and 'success' in self.resp:
            self.resp['results']['db'] = database
        else:
            self.resp = {'error': self.dbexec_error,
                         'results': {'db': database}}
        return self.resp

    def get_schemas(self):
        resp = self.getrecords('SHOW SCHEMAS')
        if resp:
            resp = [r['name'] for r in resp]
        return resp

    def get_tables(self):
        resp = self.getrecords('SHOW TABLES')
        if resp:
            self.resp['results']['all_tbl'] = [{'name': r['name'], 'db':r['database_name'],
                                                'schema':r['schema_name']} for r in resp]
        return self.resp


if __name__ == '__main__':
    objsnowflake = SnowflakeConnector()
    objsnowflake.setup_connection('SKJ992', 'Sk3900@j21', 'XH82888.ap-south-1')
    # objsnowflake.getrecords('show tables')

    objsnowflake.get_warehouses()
    objsnowflake.select_warehouse('PC_DATAROBOT_WH')

    objsnowflake.get_databases()
    objsnowflake.select_database('PC_DATAROBOT_DB')

    objsnowflake.get_schemas()
    objsnowflake.get_tables()
