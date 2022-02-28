from tqdm import tqdm
from run_subprocess import run_sub
from user_auth import UserAuth
import os
import pandas as pd
import sys
import json
from keras.preprocessing import image
import numpy as np
from tensorflow import keras
import pickle


class VisimaticAPI(UserAuth):
    def __init__(self):
        super().__init__()
        self.global_image_shape = (100, 100, 1)

    def saveApiUploadedFiles(self, data):
        response = {'success': 'API Data retrieved', 'results': {
            'predId': None, 'files': [], 'resId': None}}
        videoFormat = ['.mov', '.avi', '.mp4',
                       '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']
        formData = data['form']
        name = 'api resource'
        desc = 'uploaded files for api'
        modelid = formData['modelid']
        username = formData['username']
        apikeyid = formData['api_key_id']
        filePaths = []
        insert = 'insert into API_RESOURCES(Name,Description,ApiKeyId) values ("{name}","{desc}","{apikeyid}")'.format(
            name=name, desc=desc, apikeyid=apikeyid)

        resourceId = self.executeQuery('insert', insert)
        predId = str(self.executeQuery(
            'insert', 'insert into API_PREDICTION_HISTORY (modelId,username,status,resourceId) values ("{model}","{user}","ACTIVE",{resource})'.format(model=modelid, user=username, resource=resourceId)))

        response['results']['predId'] = predId
        response['results']['resId'] = resourceId
        response['results']['modelid'] = modelid
        response['results']['username'] = username
        response['results']['api_key_id'] = apikeyid

        pathList = [os.getcwd(), 'uploads', 'API_Predictions', predId]

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

        insertdata = []
        for i, path in enumerate(filePaths):
            insertdata.append("({id},{no},'{name}','{path}','{now}')".format(
                id=resourceId, no=i, name=os.path.split(path)[-1], path=path, now=self.getnow()))

        insertdata = ','.join(insertdata)
        insert = 'insert into API_RESOURCE_DETAILS (ResourceId,FileNo,FileName,FilePath,UpdatedOn) values {insertData}'.format(
            insertData=insertdata)
        self.executeQuery('insert', insert)

        dbData = self.getRecords(
            'select id,FileName,FilePath,Tag from API_RESOURCE_DETAILS where ResourceId={resource}'.format(resource=resourceId))
        response['results']['files'] = dbData

        return response

    def inferenceWithUserModel(self, data):
        response = {'success': 'API Data retrieved', 'model_name': '',
                    'results': []}
        select = 'select filepath,labelpath,id ,name from all_models where id = {model}'.format(
            model=data['modelid'])
        tmp = self.getRecords(select)
        if tmp:
            tmp = tmp[0]
        data['modelFilePath'] = tmp['filePath']
        data['labelFilePath'] = tmp['labelPath']
        response['model_name'] = tmp['name']

        dbData = data['files']
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
                pypath=sys.executable, rel_path=os.path.join('model_files', 'yolov5', ''), resourceDir=resDir, modelPath=os.path.join(current_dir, data['modelFilePath'], 'best.pt'), prediction=os.path.join(current_dir, 'uploads', 'API_Predictions'), name=data['predId'])

            self.log_message('sys path', sys.path)
            self.log_message('*'*100, 'yolov5 command')
            self.log_message(cmd)
            self.log_message('*'*100)

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
            image_shape = self.global_image_shape
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

        updateStmt = "update API_RESOURCE_DETAILS set Tag='{tag}' where id={id}"
        test.apply(lambda row: self.executeQuery(
            'update', updateStmt.format(tag=row['Tag'], id=row['id'])), axis=1)

        tmp1 = self.getRecords("select count(1)||' : '||Tag as result from API_RESOURCE_DETAILS where resourceId = {resource} group by Tag ".format(
            resource=data['resId']))
        if tmp1:
            predResult = ','.join([v['result'] for v in tmp1])
            updateStmt = "update PREDICTION_HISTORY set results='{predRes}',rundate='{now}',status='ACTIVE' where id={id}".format(
                predRes=predResult, id=data['predId'], now=self.getnow())
            print('update stmt', updateStmt)
            self.executeQuery("update", updateStmt)

        response['results'] = test.to_dict(orient="records")

        return response


def inferenceModelApi(data):
    formData = data['form']
    api = VisimaticAPI()
    api.log_message('infernce model api data', data)
    response = {}
    result = api.authenticate_api_user(data)
    if 'success' in result:
        upload_result = api.saveApiUploadedFiles(data)
        if 'success' in upload_result:
            response = api.inferenceWithUserModel(upload_result['results'])
    else:
        response = result

    # format the response for the api output
    tmpResults = []
    if 'success' in response and 'results' in response and response['results']:
        for result in response['results']:
            tmpResults.append(
                {'file': formData['base_url']+"getfile//"+result['FilePath'], 'prediction': result['Tag']})
        response['results'] = tmpResults
    return response
