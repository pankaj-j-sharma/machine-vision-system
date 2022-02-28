# !pip install mysql-connector-python
import mysql.connector


class MySQLConnector:
    def __init__(self, own=None, ttl=None):
        if ttl:
            self.username = 'root'
            self.password = 'Qwerty@123'
            self.hostname = 'localhost'
            self.portnumber = '3306'
            self.database = 'sys'
            self.__create_connection()

    def setup_connection(self, hostname, database, username, password, port=3306):
        self.username = username
        self.password = password
        self.hostname = hostname
        self.portnumber = port
        self.database = database
        self.__create_connection()

    def __create_connection(self):
        self.connection = mysql.connector.connect(host=self.hostname,
                                                  database=self.database,
                                                  user=self.username,
                                                  password=self.password)

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
        resp = self.getrecords('show databases')
        if resp:
            self.resp['results']['all_db'] = [r['Database'] for r in resp]
            self.resp['results']['all_tbl'] = []
        return self.resp

    def select_database(self, database):
        if not self.dbexec_error and self.resp and 'success' in self.resp:
            resp = self.getrecords('use {db}'.format(db=database))
            self.resp['results']['db'] = database
        else:
            self.resp = {'error': self.dbexec_error,
                         'results': {'db': database}}
        return self.resp

    def get_schemas(self):
        resp = self.getrecords(
            'show schemas')

        if resp:
            resp = [r['Database'] for r in resp]
        return resp

    def select_schemas(self, schema):
        resp = self.getrecords(
            'SELECT username as name FROM all_users ;')
        if resp:
            resp = [r['name'] for r in resp]
        return resp

    def get_tables(self):
        resp = self.getrecords(
            "select * from information_schema.tables where table_schema='{db}'".format(db=self.database))
        if resp:
            self.resp['results']['all_tbl'] = [{'name': r['TABLE_NAME'], 'db':r['TABLE_SCHEMA'],
                                                'schema':r['TABLE_SCHEMA']} for r in resp]
        return self.resp


if __name__ == '__main__':
    # objMysqlConn = MySQLConnector(ttl=True)
    objMysqlConn = MySQLConnector()
    objMysqlConn.setup_connection('localhost', 'sys', 'root', 'Qwerty@123')

    objMysqlConn.get_databases()
    objMysqlConn.select_database('mysql')
    objMysqlConn.get_tables()
