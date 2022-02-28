from sqlite_db_connect import SQLiteConnect
from datetime import datetime
import json


class AppItem(SQLiteConnect):
    def ___init__(self):
        super.__init__()

    def getall(self, query):
        data = self.getRecords(query)
        return {'success': 'Data retrieved', 'results': data}


def getmenuItems(data):
    print('mydata', data)
    appItem = AppItem()
    nav_options = '''
        select Title [title],Description [description],IconClass [iconClass],Url [url],Code [roles] 
        from 
        ALL_SIDENAV_OPTIONS s 
        INNER JOIN 
        ALL_SIDENAV_ROLES r on s.id = r.NavId 
        INNER JOIN 
        ROLE_TYPES t on r.RoleId = t.id 
        where code='{role}'
    '''.format(role='bu')

    recs = appItem.getall(nav_options)
    print('-'*200)
    print(recs['results'], '\n'*3)
    for rec in recs['results']:
        rec['roles'] = [rec['roles']]
        rec['subItems'] = []
    return recs


def gethomeApps():
    appItem = AppItem()
    return appItem.getall('select * from ALL_HOME_APPS ')


def getappAlerts(data):
    appItem = AppItem()
    response = {'success': 'Data retrieved', 'results': None}
    response['results'] = appItem.getRecords('select * from ALL_APP_ALERTS where UserId={Id} or UserId is null  and 1=2'.format(Id=data['UserId']))
    return response

def getuserMsgs():
    appItem = AppItem()
    appItem.getall('select * from ALL_HOME_APPS ')
