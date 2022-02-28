# !pip install pyodbc
import pyodbc

class SQLServerConnector:

    def __init__(self, own=None, ttl=None):
        if ttl:
            self.username = ''
            self.password = ''
            self.server = 'DESKTOP-FDEOKLS'
            self.instance = 'LOCALHOST'
            self.portnumber = '1521'
            self.database = 'TEST1'
            self.trusted_conn = 'Y'
            self.__create_connection()

    def setup_connection(self, server, database, instance, username='', password='', port=1443, trusted_conn='Y'):
        self.username = username
        self.password = password
        self.server = server
        self.instance = instance
        self.portnumber = port
        self.database = database
        self.trusted_conn = trusted_conn
        self.__create_connection()

    def __create_connection(self):
        if self.trusted_conn != 'Y' and self.username != '' and self.password != '':
            self.connection = pyodbc.connect(driver='{SQL Server Native Client 10.0}',
                                             server='{server},{port}'.format(
                                                 server=self.server, port=self.port),
                                             database=self.database,
                                             uid=self.username, pwd=self.password)
        else:
            self.connection = pyodbc.connect(driver='{SQL Server Native Client 10.0}',
                                             server='{server},{port}'.format(
                                                 server=self.server, port=self.port),
                                             database=self.database,
                                             uid=self.username, pwd=self.password)

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

    def get_databases(self):
        self.resp = {'success': 'Data retrieved',
                     'results': {'all_db': [], 'all_tbl': []}}
        resp = self.getrecords('select * from v$database')
        if resp:
            self.resp['results']['all_db'] = [r['name'] for r in resp]
            self.resp['results']['all_tbl'] = []
        return self.resp

    def select_database(self, database):
        if not self.dbexec_error and self.resp and 'success' in self.resp:
            self.resp['results']['db'] = database
        else:
            self.resp = {'error': self.dbexec_error,
                         'results': {'db': database}}
        return self.resp

    def get_schemas(self):
        resp = self.getrecords(
            'SELECT username as name FROM all_users')

        if resp:
            resp = [r['name'] for r in resp]
        return resp

    def select_schemas(self):
        resp = self.getrecords(
            'SELECT username as name FROM all_users ;')
        if resp:
            resp = [r['name'] for r in resp]
        return resp

    def get_tables(self):
        resp = self.getrecords(
            'select * from all_tables where table_catalog={db}'.format(db=self.database))
        if resp:
            self.resp['results']['all_tbl'] = [{'name': r['name'], 'db':r['database_name'],
                                                'schema':r['schema_name']} for r in resp]
        return self.resp


if __name__ == '__main__':
    objConn = SQLServerConnector(ttl=True)
    print(dir(objConn.connection))
    objConn.connection.version
