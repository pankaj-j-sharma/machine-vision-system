import sys
from itsdangerous import json
from matplotlib.pyplot import axis
from sklearn.utils import resample
from logger import Logger
# from sqlalchemy import true
from sqlite_db_connect import SQLiteConnect
from image_classifier import *
import os
from run_subprocess import run_sub
from datetime import datetime
import json

now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class Prediction(SQLiteConnect):
    def ___init__(self):
        super.__init__()

    def getdata(self, query):
        data = self.getRecords(query)
        return {'success': 'Data retrieved', 'results': data}


def getallavlmodels(data):
    prediction = Prediction()
    response = prediction.getdata(
        "select * from ALL_MODELS where status='ACTIVE' and (modeltype is null or modeltype in (select id from MODEL_APP_TYPES where AppName='{app}'))".format(app=data['AppName']))
    prediction.log_message('all available models',
                           'data', data, 'response', response)
    return response


def loadpredictionhistory(data):
    prediction = Prediction()
    response = prediction.getdata(
        "select ph.*,am.name, (select count(1) from RESOURCE_DETAILS where resourceId = ph.resourceId) as FilesCount from PREDICTION_HISTORY ph INNER JOIN ALL_MODELS am on ph.modelId = am.id where ph.status='ACTIVE' and ph.userId={Id} and (am.modeltype is null or am.modeltype in (select id from MODEL_APP_TYPES where AppName='{App}')) order by ph.id desc limit 5 ".format(Id=data['UserId'], App=data['AppName']))
    prediction.log_message('load prediction history',
                           'data', data, 'response', response)
    return response


def deletepredictionhistory(data):
    prediction = Prediction()
    prediction.executeQuery("delete", "delete from ALL_RESOURCES where id in (select resourceId from PREDICTION_HISTORY where userId={user} and id={Id})".format(
        user=data['UserId'], Id=data['predictionId']))
    prediction.executeQuery("delete", "delete from RESOURCE_DETAILS where resourceId in (select resourceId from PREDICTION_HISTORY where userId={user} and id={Id})".format(
        user=data['UserId'], Id=data['predictionId']))
    prediction.executeQuery("delete", "delete from PREDICTION_HISTORY where userId={user} and id={Id}".format(
        user=data['UserId'], Id=data['predictionId']))
    response = {'success': 'Data removed', 'results': []}
    prediction.log_message('delete prediction history',
                           'data', data, 'response', response)

    return response


def inferenceWithUserModel(data):
    # same as runinferencingforproject in project_setup.py file
    formdata = data['form']
    prediction = Prediction()
    prediction.log_message('inference user model', 'data', data)

    response = {'success': 'Data retrieved', 'modelSource': 'UserCustom',
                'results': []}

    try:
        select = 'select filepath,labelpath,id from all_models where id = {model}'.format(
            model=formdata['modelId'])
        tmp = prediction.getRecords(select)
        if tmp:
            tmp = tmp[0]
        data['modelFilePath'] = tmp['filePath']
        data['labelFilePath'] = tmp['labelPath']

        dbData = prediction.getRecords(
            'select id,FileName,FilePath,Tag from RESOURCE_DETAILS where ResourceId={resource}'.format(resource=formdata['resourceId']))
        test = pd.DataFrame(dbData)

        if not data['labelFilePath'] or data['labelFilePath'] == '':
            current_dir = os.getcwd()
            # os.chdir('model_files/yolov5')
            sys.path.append(os.path.join(os.getcwd(), 'model_files', 'yolov5'))
            test.loc[:, 'Tag'] = 'No Defect'
            resDir = os.path.join(current_dir, 'uploads', os.path.split(
                dbData[0]['FilePath'] if dbData else '.')[0])

            resDir = os.path.join(resDir, 'input') if os.path.isdir(os.path.join(resDir, 'input')) and len(
                os.listdir(os.path.join(resDir, 'input'))) > 0 else resDir

            cmd = '{pypath} {rel_path}detect.py --source {resourceDir} --weights {modelPath} --project {prediction} --name {name} --conf-thres 0.5 --line-thickness 1 --save-txt --exist-ok --hide-conf'.format(
                pypath=sys.executable, rel_path=os.path.join('model_files', 'yolov5', ''), resourceDir=resDir, modelPath=os.path.join(current_dir, data['modelFilePath'], 'best.pt'), prediction=os.path.join(current_dir, 'uploads', 'Predictions'), name=formdata['predictionId'])

            prediction.log_message('sys path', sys.path)
            prediction.log_message('*'*100, 'yolov5 command')
            prediction.log_message(cmd)
            prediction.log_message('*'*100)

            resp = run_sub(cmd)
            if resp and 'error' in resp and not resp['error']:
                respOut = resp['results1']
                print('resp', respOut, type(respOut))
                respOut = json.loads(respOut.replace("\'", "\""))
                for r in respOut:
                    print('r', r, test[test['FileName'].str.startswith(r)])
                    test.loc[test['FileName'].str.startswith(r), 'Tag'] = test.loc[test['FileName'].str.startswith(r), 'Tag'].apply(
                        lambda row: respOut[r])

                print(test.head())
        else:
            image_shape = global_image_shape
            test_image = []
            for i in tqdm(range(test.shape[0])):
                img = image.load_img(os.path.join(
                    'uploads', test['FilePath'][i]), target_size=image_shape, color_mode="grayscale")
                img = image.img_to_array(img)
                img = img/255
                test_image.append(img)

            X_Inference = np.array(test_image)

            loaded_model = keras.models.load_model(
                os.path.join(os.getcwd(), data['modelFilePath']))
            with open(data['labelFilePath'], 'rb') as e:
                loaded_labels = pickle.load(e)

            Inference_y = loaded_model.predict(X_Inference)
            y_Inference = np.argmax(Inference_y, axis=1)
            # y_Inference = loaded_model.predict_classes(X_Inference)
            y_Inference = loaded_labels.inverse_transform(y_Inference)
            test['Tag'] = y_Inference

        updateStmt = "update RESOURCE_DETAILS set Tag='{tag}' where id={id}"
        test.apply(lambda row: prediction.executeQuery(
            'update', updateStmt.format(tag=row['Tag'], id=row['id'])), axis=1)

        tmp1 = prediction.getRecords("select count(1)||' : '||Tag as result from RESOURCE_DETAILS where resourceId = {resource} group by Tag ".format(
            resource=formdata['resourceId']))
        if tmp1:
            predResult = ','.join([v['result'] for v in tmp1])
            updateStmt = "update PREDICTION_HISTORY set results='{predRes}',rundate='{now}',status='ACTIVE' where id={id}".format(
                predRes=predResult, id=formdata['predictionId'], now=now)
            print('update stmt', updateStmt)
            prediction.executeQuery("update", updateStmt)

        response['results'] = test.to_dict(orient="records")
    except Exception as e:
        prediction.log_message('inference user model', 'error', str(e))

    prediction.log_message('inference user model', 'response', response)
    return response


def saveDataUseCaseModels(data):
    formdata = data['form']
    prediction = Prediction()
    tmpResults = []
    modelName = prediction.getRecords(
        'select name from ALL_MODELS where id={model} limit 1'.format(model=formdata['modelId']))
    for res in data['response']['predictions']:
        tmp = ''
        if modelName and modelName[0]['name'] == 'Hard Hat Present':
            for pred in res['pred']:
                if pred['name'] != 'helmet':
                    tmp = 'Without Helmet'
                    break
                else:
                    tmp = 'Wearing Helmet'
        else:
            tmp = res
        tmpResults.append(tmp)

    predResult = ','.join(list(set([str(tmpResults.count(x)) + ' : ' + x
                          for x in tmpResults])))
    updateStmt = "update PREDICTION_HISTORY set results='{predRes}',rundate='{now}',status='ACTIVE' where id={id}".format(
        predRes=predResult, id=formdata['predictionId'], now=now)
    prediction.log_message('delete prediction history',
                           'data', data, 'delete', updateStmt)
    prediction.executeQuery("update", updateStmt)


def loadDraftPredictionId(data):
    prediction = Prediction()
    response = {'success': 'Data retrieved', 'results': []}
    print('debug ', data)
    formData = data['form']
    modelId = formData['modelId']
    predId = formData['predictionId']
    userId = formData['userId']
    modelName = formData['modelName'] if 'modelName' in formData else 'mymodel'

    predResp = prediction.getRecords(
        'select * from PREDICTION_HISTORY where id={id} and modelId={model} and status<>"ACTIVE"'.format(id=predId, model=modelId))
    if not predResp:
        resourceName = modelName+" pred " + \
            datetime.now().strftime('%Y-%M-%d %H:%M:%S')
        insert = 'insert into ALL_RESOURCES (Name,Description) values ("{name}","{desc}")'
        insert = insert.format(name=resourceName, desc="resource created for " +
                               resourceName)
        resourceId = prediction.executeQuery('update', insert)

        predId = prediction.executeQuery(
            'insert', 'insert into PREDICTION_HISTORY (modelId,userId,status,resourceId) values ({model},{user},"",{resource})'.format(model=modelId, user=userId, resource=resourceId))
        predResp = prediction.getRecords(
            'select * from PREDICTION_HISTORY where id={id} and modelId={model} and status<>"ACTIVE"'.format(id=predId, model=modelId))

    response['results'] = predResp
    prediction.log_message('load draft prediction',
                           'data', data, 'response', response)

    return response


def savePredictionFiles(data):
    prediction = Prediction()
    prediction.log_message('save prediction ', 'data', data)
    videoFormat = ['.mov', '.avi', '.mp4',
                   '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']
    response = {'success': 'Data retrieved', 'results': {
        'predId': None, 'files': [], 'resId': None}}
    try:
        formData = data['form']
        modelId = formData['modelId']
        predId = formData['predictionId']
        userId = formData['userId']
        modelName = formData['modelName'] if 'modelName' in formData else 'mymodel'

        predId = formData['predictionId'] if formData['predictionId'] and int(
            formData['predictionId']) > 0 else None
        resourceId = formData['resourceId'] if formData['resourceId'] and int(
            formData['resourceId']) > 0 else None

        if not (predId and resourceId):
            resourceName = modelName+" pred " + \
                datetime.now().strftime('%Y-%M-%d %H:%M:%S')
            insert = 'insert into ALL_RESOURCES (Name,Description) values ("{name}","{desc}")'
            insert = insert.format(name=resourceName, desc="resource created for " +
                                   resourceName)
            resourceId = prediction.executeQuery('update', insert)
            predId = str(prediction.executeQuery(
                'insert', 'insert into PREDICTION_HISTORY (modelId,userId,status,resourceId) values ({model},{user},"",{resource})'.format(model=modelId, user=userId, resource=resourceId)))

        response['results']['predId'] = predId
        response['results']['resId'] = resourceId
        filePaths = []

        pathList = [os.getcwd(), 'uploads', 'Predictions', predId]
        for i, _ in enumerate(pathList):
            if not os.path.isdir(os.path.join(*pathList[:i+1])):
                os.mkdir(os.path.join(*pathList[:i+1]))
            prediction_data_path = os.path.join(*pathList[:i+1])

        for file in data['files']:
            path = os.path.join(prediction_data_path, file.filename)
            file.save(path)
            # if video file copy to input folder
            if os.path.splitext(file.filename)[-1] and os.path.splitext(file.filename)[-1] in videoFormat:
                file.save(os.path.join(
                    prediction_data_path, 'input', file.filename))

            # store only relative path
            relatve_path = os.path.join(os.getcwd(), 'uploads', '')
            path = path.replace(relatve_path, '')
            filePaths.append(path)

        if formData['resourceId'] == '0':
            tmp = prediction.getRecords('select resourceId from PREDICTION_HISTORY where id ={predId}'.format(
                predId=predId))
            if tmp:
                tmp = tmp[0]
                formData['resourceId'] = tmp['Resource']

        fileNo = prediction.getRecords('select count(1) "no" from RESOURCE_DETAILS where ResourceId={resource} and stepno is null'.format(
            resource=formData['resourceId']))
        fileNo = fileNo[0]["no"] if fileNo else 0

        insertdata = []
        for i, path in enumerate(filePaths):
            insertdata.append("({id},{no},'{name}','{path}','{now}')".format(
                id=resourceId, no=fileNo+i, name=os.path.split(path)[-1], path=path, now=now))

        insertdata = ','.join(insertdata)
        insert = 'insert into RESOURCE_DETAILS (ResourceId,FileNo,FileName,FilePath,UpdatedOn) values {insertData}'.format(
            insertData=insertdata)
        prediction.executeQuery('insert', insert)
        prediction.log_message('insert ', insert)

        response['results']['files'] = prediction.getRecords(
            'select FileNo as no,FileName as name,FilePath path,Tag as result,UpdatedOn as updated from RESOURCE_DETAILS where ResourceId={resId}'.format(resId=resourceId))
    except Exception as e:
        print('error occurred', e)

    prediction.log_message('save prediction', 'response', response)

    return response
