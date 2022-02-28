import sqlite3
from logger import Logger
from datetime import datetime


class SQLiteConnect(Logger):

    def __init__(self):
        super().__init__()
        self.dbName = 'VisimaticSQLite.db'
        self.record = None
        self.connect()

    def getnow(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def connect(self):
        try:
            self.sqliteConnection = sqlite3.connect(self.dbName)
            self.cursor = self.sqliteConnection.cursor()
            self.log_message(
                'Database created and Successfully Connected to SQLite')
        except sqlite3.Error as error:
            self.log_message('Error while connecting to sqlite', str(error))

    def executeQuery(self, operType, query, multiple=False):
        last_row_id = -1
        self.record = None
        self.dbexec_error = None
        self.cursor = self.sqliteConnection.cursor()
        try:
            if operType == 'select':
                self.cursor.execute(query)
                self.record = self.__recordToJson()
            else:
                if multiple:
                    # for executing multiple statements in single query
                    self.cursor.executescript(query)
                else:
                    self.cursor.execute(query)
                self.sqliteConnection.commit()
                last_row_id = self.cursor.lastrowid
        except Exception as e:
            self.dbexec_error = 'Error occurred for '+query+' '+str(e)
            self.log_message('Error in execution db ', self.dbexec_error)
        self.cursor.close()
        return last_row_id

    def displayRecords(self):
        self.log_message('show records', self.record)

    def getRecords(self, query):
        self.executeQuery('select', query)
        self.log_message('query->', query, 'rec', self.record)
        return self.record

    def __recordToJson(self):
        colnames = [x[0] for x in self.cursor.description]
        tmprec = self.cursor.fetchall()
        records = []
        for rec in tmprec:
            records.append(dict(zip(colnames, rec)))
        return records


if __name__ == '__main__':
    objSqlite = SQLiteConnect()
    create_tables = '''
    CREATE TABLE PREDICTION_HISTORY(id INTEGER PRIMARY KEY , rundate DATETIME, modelId INT, source TEXT, uploadedData TEXT, results TEXT);
    CREATE TABLE ALL_MODELS (id INTEGER PRIMARY KEY, name TEXT, modeltype TEXT, createdOn DATETIME, status TEXT);
    CREATE TABLE MODEL_PARAMETERS (id INTEGER PRIMARY KEY, modelId INTEGER, name TEXT, value TEXT);    
    '''
    objSqlite.executeQuery('create', create_tables)
    objSqlite.displayRecords()
