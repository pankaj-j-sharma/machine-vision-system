from dataclasses import replace
from sqlite_db_connect import SQLiteConnect
from datetime import datetime
import uuid
import os
import secrets

now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")


class UserAuth(SQLiteConnect):
    def ___init__(self):
        super.__init__()

    def getall(self, query):
        data = self.getRecords(query)
        return data

    def update(self, query):
        data = self.executeQuery('update', query)
        return data

    def authenticate_api_user(self, data):
        formData = data['form']
        response = {}
        username = formData['username'] if 'username' in formData else None
        modelid = formData['modelid'] if 'modelid' in formData else None
        apikey = formData['api_key'] if 'api_key' in formData else None
        if username and modelid and apikey:
            tmp = self.getRecords("select ua.id as Id from USER_API_KEYS ua inner join ALL_USERS au on ua.userid = au.id where (au.api_username,ua.modelId,ua.apikey) = ('{username}','{modelid}','{apikey}')".format(
                username=username, modelid=modelid, apikey=apikey))
            if tmp:
                tmp = tmp[0]
                formData['api_key_id'] = tmp['Id']
                response = {'success': 'API user authenticated',
                            'results': formData}
            else:
                response = {
                    'error': 'Access Denied. Username or API Key is incorrect', 'results': formData}
        else:
            response = {
                'error': 'One or more parameters are missing', 'results': None}
        self.log_message('api user authentication', 'data',
                         data, 'response', response)
        return response


def loginUser(data):
    lastUpdate = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    userAuth = UserAuth()
    results = userAuth.getall("select id,token,lastlogin,defaultpage from ALL_USERS where username='{user}' and passwd='{passwd}' limit 1".format(
        user=data['username'], passwd=data['password']))
    response = {'success': 'User Logged in ', 'results': results[0]} if results else {
        'error': 'Invalid Username/Password'}

    if 'success' in response:
        userAuth.update("update ALL_USERS set token='{token}' , lastlogin='{lastUpdate}' where username='{user}' and passwd='{passwd}'".format(
            token=uuid.uuid4(), user=data['username'], passwd=data['password'], lastUpdate=lastUpdate))
    return response


def logoutUser(data):
    pass


def validateToken(data):
    pass


def saveUserProfile(data):
    formData = data['form']
    userId = formData['userid']
    userAuth = UserAuth()
    profile_image_path = ''
    updateArr = []

    if userId and int(userId) > 0 and 'files' in data and data['files']:
        pathList = [os.getcwd(), 'uploads', 'UserInfo', userId]
        for i, _ in enumerate(pathList):
            if not os.path.isdir(os.path.join(*pathList[:i+1])):
                os.mkdir(os.path.join(*pathList[:i+1]))
            user_data_path = os.path.join(*pathList[:i+1])

        for file in data['files']:
            path = os.path.join(user_data_path, file.filename)
            file.save(path)
            # store only relative path
            relatve_path = os.path.join(os.getcwd(), 'uploads', '')
            profile_image_path = path.replace(relatve_path, '')
            updateArr.append("profileimg='{}'".format(profile_image_path))
            # run for only one file
            break

    if formData['emailid'] and formData['emailid'].replace(" ", "") != "":
        updateArr.append("emailid='{}'".format(formData['emailid']))
    if formData['firstname'] and formData['firstname'].replace(" ", "") != "":
        updateArr.append("firstname='{}'".format(formData['firstname']))
    if formData['lastname'] and formData['lastname'].replace(" ", "") != "":
        updateArr.append("lastname='{}'".format(formData['lastname']))
    if formData['mobile'] and formData['mobile'].replace(" ", "") != "":
        updateArr.append("mobile='{}'".format(formData['mobile']))
    if formData['organization'] and formData['organization'].replace(" ", "") != "":
        updateArr.append("organization='{}'".format(
            formData['organization']))

    if updateArr:
        update = "update USER_INFO set {updateArr} where userid='{userid}'".format(
            updateArr=','.join(updateArr), userid=userId)
        userAuth.log_message('update stmt', update)
        userAuth.executeQuery('update', update)

    updateArr.clear()
    if formData['passwd'] and formData['passwd'].replace(" ", "") != "":
        updateArr.append("passwd='{}'".format(formData['passwd']))
    if formData['username'] and formData['username'].replace(" ", "") != "":
        updateArr.append("username='{}'".format(formData['username']))

    if updateArr:
        update = "update ALL_USERS set {updateArr} where id='{userid}'".format(
            updateArr=','.join(updateArr), userid=userId)
        userAuth.log_message('update stmt', update)
        userAuth.executeQuery('update', update)

    return userProfile(data)


def userProfile(data):
    formData = data['form']
    userAuth = UserAuth()
    response = {'success': 'User Logged in ', 'results': None}
    response['results'] = userAuth.getRecords(
        'select ui.*,au.username,au.passwd from ALL_USERS	au inner join USER_INFO ui on au.id=ui.userid and au.id={userid}'.format(userid=formData['userid']))
    response['results'] = response['results'][0] if response['results'] else {
        'emailid': '', 'firstname': '', 'lastname': '', 'mobile': '', 'organization': '', 'username': ''}
    userAuth.log_message('user profile', 'data', data, 'response', response)
    return response


def generateApiKey(data):
    userAuth = UserAuth()
    formData = data['form']
    modelId = formData['modelId']
    userid = formData['userId']
    response = {'success': 'API key generated', 'results': None}
    if 'key_length' in formData and int(formData['key_length']) > 10:
        generated_key = secrets.token_urlsafe(int(formData['key_length']))
    else:
        generated_key = secrets.token_urlsafe(50)

    tmp = userAuth.getRecords('select id from USER_API_KEYS where (userid,modelId)=({userid},{model})'.format(
        userid=userid, model=modelId))
    if tmp:
        api_key_id = tmp[0]['id']
        update = "update USER_API_KEYS set apikey='{apikey}',created='{now}' where id={api_key_id}".format(
            api_key_id=api_key_id, userid=userid, model=modelId, apikey=generated_key, now=now)
        userAuth.executeQuery('update', update)
    else:
        insert = "insert into USER_API_KEYS (userid,modelId,apikey) values ('{userid}','{model}','{key}')".format(
            userid=userid, model=modelId, key=generated_key)
        userAuth.executeQuery('insert', insert)
    response['results'] = {'api_key': generated_key}
    userAuth.log_message('generate api key response', response)
    return response


def loadUserApiKey(data):
    userAuth = UserAuth()
    formData = data['form']
    modelId = formData['modelId']
    userid = formData['userId']
    response = {'success': 'API key retrieved', 'results': None}
    tmp = userAuth.getRecords('select au.username,ua.apikey,ua.created from USER_API_KEYS ua inner join ALL_USERS au on ua.userid = au.id where (ua.userid,ua.modelId)=({userid},{model})'.format(
        userid=userid, model=modelId))
    if tmp:
        tmp = tmp[0]
        # format date
        createdOn = datetime.strptime(
            tmp['created'], '%d-%m-%Y %H:%M:%S').strftime('%d, %b %Y %I:%M %p') if tmp['created'] else None
        response['results'] = {'username': tmp['username'],
                               'api_key': tmp['apikey'], 'created_on': createdOn}
    return response


def uploadDataForPrediction(data):
    userAuth = UserAuth()
    formData = data['form']
    username = formData['username'] if 'username' in formData else None
    api_key = formData['api_key'] if 'api_key' in formData else None
    model = formData['api_key'] if 'api_key' in formData else None

    userAuth
