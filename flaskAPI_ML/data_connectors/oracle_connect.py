import cx_Oracle


class OracleConnector:

    def __init__(self, own=None, ttl=None):
        if ttl:
            self.username = 'pankaj'
            self.password = 'qbolbk'
            self.hostname = 'localhost'
            self.portnumber = '1521'
            self.sid = 'xe'
            self.__create_connection()

    def setup_connection(self, username, password, hostname, sid, port=1521):
        self.username = username
        self.password = password
        self.hostname = hostname
        self.portnumber = port
        self.sid = sid
        self.__create_connection()

    def __create_connection(self):
        self.connection = cx_Oracle.connect(user=self.username, password=self.password, dsn=cx_Oracle.makedsn(
            self.hostname, self.portnumber, self.sid))

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
    objConn = OracleConnector(ttl=True)
