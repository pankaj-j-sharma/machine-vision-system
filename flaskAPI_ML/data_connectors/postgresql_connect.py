# pip install psycopg2


import psycopg2
import psycopg2.extras

# hostname = 'localhost'
# database = 'demo'
# username = 'postgres'
# pwd = 'admin'
# port_id = 5432
# conn = None


class PostgreSqlConnector:

    def __init__(self, own=None, ttl=None):
        if ttl:
            self.username = 'postgres'
            self.password = 'admin@123'
            self.hostname = 'localhost'
            self.portnumber = 3306
            self.database = 'visimatic'
            self.__create_connection()

    def setup_connection(self, username, password, hostname, database, port=3306):
        self.username = username
        self.password = password
        self.hostname = hostname
        self.portnumber = port
        self.database = database
        self.__create_connection()

    def __create_connection(self):
        self.connection = psycopg2.connect(
            host=self.hostname, dbname=self.database, user=self.username, password=self.password, port=self.portnumber)

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
        resp = self.getrecords('select datname as name from pg_database')
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
            'select schema_name as name from information_schema.schemata')
        if resp:
            resp = [r['name'] for r in resp]
        return resp

    def get_tables(self):
        resp = self.getrecords(
            "select table_name as name,table_schema as schema_name , table_catalog as database_name FROM information_schema.tables where table_schema='public' and  table_catalog='{db}'".format(db=self.database))
        if resp:
            self.resp['results']['all_tbl'] = [{'name': r['name'], 'db':r['database_name'],
                                                'schema':r['schema_name']} for r in resp]
        return self.resp


if __name__ == '__main__':
    objPostgres = PostgreSqlConnector(ttl=True)
    print(objPostgres.get_databases())
    print(objPostgres.select_database('visimatic'))
    print(objPostgres.get_tables())
