from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from run_subprocess import run_sub
from sqlite_db_connect import SQLiteConnect
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
import pickle
from tensorflow import keras
from sklearn import metrics
import uuid
import os
import sys
import yaml

# from keras.utils import to_categorical
# from keras.utils import to_categorical


now = datetime.now().strftime('%Y-%M-%d %H:%M:%S')
global_image_shape = (100, 100, 1)


class CustomTrainer(SQLiteConnect):
    def ___init__(self):
        super.__init__()

    def train_classifier(self, data):

        response = {'success': 'Data retrieved', 'results': []}

        # extracting all the relevant parameters
        trainId = data['trainId'] if 'trainId' in data else '-1'
        monitor = data['monitorFor'] if 'monitorFor' in data else 'val_loss'
        patience = int(data['patience']) if 'patience' in data else 5
        all_layer_actvn = data['activationAll'] if 'activationAll' in data else 'relu'
        out_layer_actvn = data['outActivation'] if 'outActivation' in data else 'softmax'
        optimizer = data['optimizer'] if 'optimizer' in data else 'Adam'
        lossParam = data['lossParam'] if 'lossParam' in data else 'categorical_crossentropy'
        randomSeed = int(data['randomSeed']) if 'randomSeed' in data else 42
        metricsParam = data['metricsParam'].split(
            ',') if 'metricsParam' in data else ['accuracy']
        colorMode = data['colorMode'] if 'colorMode' in data else 'grayscale'
        trainEpochs = int(data['trainEpochs']) if 'trainEpochs' in data else 50
        # extracting all the relevant parameters ends

        select = 'select * from RESOURCE_DETAILS where ResourceId in ( select ResourceId from CUSTOM_TRAINING_HISTORY where id={trainId}) and tag is not null'
        fileName = data['modelFileName'] if 'optimizer' in data else uuid.uuid4()
        data = self.getRecords(select.format(trainId=trainId))
        train = pd.DataFrame(data)

        # We have grayscale images, so while loading the images we will keep grayscale=True, if you have RGB images, you should set grayscale as False
        image_shape = global_image_shape
        train_image = []
        for i in tqdm(range(train.shape[0])):
            img = image.load_img(os.path.join(
                'uploads', train['FilePath'][i]), target_size=image_shape, color_mode=colorMode)
            img = image.img_to_array(img)
            img = img/255
            train_image.append(img)
        X = np.array(train_image)

        label_encoder = LabelEncoder()
        dy = train['Tag'].values
        vec = label_encoder.fit_transform(dy)
        y = to_categorical(vec)
        output_classes = len(set(vec))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=randomSeed, test_size=0.2)

        rel_model_path = os.path.join(
            'uploads', 'User_Model_Files', '{name}.hdf5'.format(name=fileName.replace(" ", "_")))
        model_save_path = os.path.join(
            os.getcwd(), rel_model_path)

        early_stop = EarlyStopping(monitor=monitor, patience=patience)
        checkpoint = ModelCheckpoint(
            filepath=model_save_path, verbose=1, save_best_only=True, monitor=monitor)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation=all_layer_actvn, input_shape=image_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation=all_layer_actvn))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation=all_layer_actvn))
        model.add(Dropout(0.5))
        model.add(Dense(output_classes, activation=out_layer_actvn))
        model.compile(loss=lossParam,
                      optimizer=optimizer, metrics=metricsParam)

        model.summary()
        history = model.fit(X_train, y_train, epochs=trainEpochs, validation_data=(
            X_test, y_test), callbacks=[early_stop, checkpoint])

        # evaluate the model
        train_acc = model.evaluate(X_train, y_train, verbose=0)
        test_acc = model.evaluate(X_test, y_test, verbose=0)

        predY = model.predict(X_test)
        y_pred = np.argmax(predY, axis=1)
        y_test = np.argmax(y_test, axis=-1)
        # print('preds', y_pred, y_test)

        f1Score = round(f1_score(y_test, y_pred, average="macro"), 2)
        precisionScore = round(precision_score(
            y_test, y_pred, average="macro"), 2)
        recallScore = round(recall_score(y_test, y_pred, average="macro"), 2)

        rel_encoder_path = os.path.join(
            'uploads', 'User_Model_Files', '{name}.pkl'.format(name=fileName.replace(" ", "_")))
        encoder_file = os.path.join(
            os.getcwd(), rel_encoder_path)
        with open(encoder_file, 'wb') as f:
            pickle.dump(label_encoder, f)

        results = [{'model_name': fileName, 'model_file_path': rel_model_path, 'encoder_file': rel_encoder_path, 'history': history.history,
                    'train_acc': train_acc, 'test_acc': test_acc, 'f1': f1Score, 'precision': precisionScore, 'recall': recallScore, 'trainId': trainId}]

        response['results'] = results
        return response

    def train_object_detector(self, data):
        response = {'success': 'Data retrieved', 'results': []}
        trainEpochs = int(data['trainEpochs']) if 'trainEpochs' in data else 15
        current_dir = os.getcwd()
        ymlDirPath = os.path.join(
            current_dir, 'uploads', 'CustomTrain', data['trainId'])
        ymlFile = [f for f in os.listdir(
            ymlDirPath) if f.endswith('yml') or f.endswith('yaml')]
        ymlFile = ymlFile[0] if ymlFile else 'data.yml'
        ymlFile = os.path.join(ymlDirPath, ymlFile)

        parentFolder = os.path.join('uploads', 'User_Model_Files')
        modelFolder = data['modelFileName'].replace(
            ' ', '_') if 'modelFileName' in data and data['modelFileName'] else 'visimatic'

        if 'objectDetectionType' in data and str(data['objectDetectionType']).lower().startswith('yolo'):
            os.chdir('model_files/yolov5')
            resp = run_sub(
                '{pypath} trainYolov5.py --data {ymlpath} --epochs {epoch} --weights yolov5s.pt --img 224 --name {modelName} --project {folder}'.format(pypath=sys.executable, epoch=trainEpochs, ymlpath=ymlFile, folder=os.path.join(current_dir, parentFolder), modelName=modelFolder))

        if resp and 'error' in resp and not resp['error']:
            print('resp', resp)

        results = [{'model_name': data['modelFileName'], 'model_file_path': os.path.join(
            parentFolder, modelFolder, 'weights'), 'trainId': data['trainId']}]
        response['results'] = results

        os.chdir(current_dir)
        return response

    def savetraining_classifier(self, data):
        print('data', data)
        results = data['results'][0]
        insert = 'insert into ALL_MODELS (name, createdOn, status, filePath, labelPath, f1, precision, recall) VALUES '
        insert += '("{name}", "{now}", "ACTIVE", "{file}", "{label}", "{f1}", "{precision}", "{recall}")'.format(
            name=results['model_name'], now=now, file=results['model_file_path'], label=results['encoder_file'], f1=results['f1'], precision=results['precision'], recall=results['recall'])
        model_id = self.executeQuery('insert', insert)
        update = "update CUSTOM_TRAINING_HISTORY set modelId = {model} where id={trainId}".format(
            model=model_id, trainId=results['trainId'])
        self.executeQuery('update', update)

    def savetraining_detector(self, data):
        print('data', data)
        results = data['results'][0]
        insert = 'insert into ALL_MODELS (name, createdOn, status, filePath) VALUES '
        insert += '("{name}", "{now}", "ACTIVE", "{file}")'.format(
            name=results['model_name'], now=now, file=results['model_file_path'])
        model_id = self.executeQuery('insert', insert)
        update = "update CUSTOM_TRAINING_HISTORY set modelId = {model} where id={trainId}".format(
            model=model_id, trainId=results['trainId'])
        self.executeQuery('update', update)


def run_custom_training(data):
    print('run_custom_training', data)
    trainer = CustomTrainer()
    trainingType = data['form']['trainingType']
    if trainingType == 'Classification':
        resp = trainer.train_classifier(data['form'])
        trainer.savetraining_classifier(resp)
    elif trainingType == 'Object Detection':
        data['form']['objectDetectionType'] = 'yolov5'
        resp = trainer.train_object_detector(data['form'])
        trainer.savetraining_detector(resp)

    return resp


def generate_yml_file(data):
    trainer = CustomTrainer()
    resp = None
    yaml_path = None
    tmp = trainer.getRecords('select FileName,FilePath,Tag from RESOURCE_DETAILS where Filename like "{yml}" and ResourceId={resource}'.format(
        resource=data['resourceId'], yml="%"+".yaml"))
    if tmp:
        tmp = tmp[0]
        yaml_path = os.path.join(os.getcwd(), 'uploads', tmp['FilePath'])

        with open(yaml_path, 'r') as f:
            annotationData = yaml.safe_load(f)
            annotationData['path'] = os.path.join(os.getcwd(), 'uploads', tmp['FilePath'].replace(
                tmp['FileName'], ''))
            annotationData['train'] = 'images'
            annotationData['val'] = 'images'
            print('data annotation', annotationData)

        with open(yaml_path, 'w') as w:
            yaml.dump(annotationData, w)

    resp = {'csvfile': yaml_path,
            'tagfile': tmp}

    return resp


def generate_csv_file(data):
    trainer = CustomTrainer()
    tmp = trainer.getRecords('select FileName,FilePath,Tag from RESOURCE_DETAILS where ResourceId={resource}'.format(
        resource=data['resourceId']))

    csv = str(data['resourceId'])+'-taggedfile.csv'
    csv = tmp[0]['FilePath'].replace(tmp[0]['FileName'], csv)
    csv_path = os.path.join(os.getcwd(), 'uploads', csv)
    df = pd.DataFrame(tmp)
    df.to_csv(csv_path, index=False)
    tagfile = [t for t in tmp if t['FileName'].endswith('.csv')]
    resp = {'csvfile': csv_path,
            'tagfile': tagfile[0] if tagfile else 'No tag file added'}

    if tagfile:
        updateStmt = 'update RESOURCE_DETAILS set tag="{tag}" where FileName="{file}" and ResourceId={resource}'
        df2 = pd.read_csv(os.path.join(
            os.getcwd(), 'uploads', tagfile[0]['FilePath']))
        df2.apply(lambda row: trainer.executeQuery(
            'update', updateStmt.format(resource=data['resourceId'], tag=row['FileTag'], file=row['Name'])), axis=1)
        # df2.apply(lambda row: execQuery(trainer, updateStmt, row,data['resourceId']), axis=1)
        # print('df2', df2, data['resourceId'])
    return resp


def execQuery(obj, stmt, row, resourceId):
    print('statment -> ', stmt.format(resource=resourceId,
          tag=row['FileTag'], file=row['Name']))


def loadDraftCustomTrainId(data):
    trainer = CustomTrainer()
    trainId = -1
    resourceId = -1
    response = {'success': 'Data retrieved', 'results': []}
    tmp1 = trainer.getRecords(
        'select id,ResourceId from CUSTOM_TRAINING_HISTORY where userId={user} and type={type} and modelId is null'.format(user=data['form']['userId'], type=data['form']['type']))
    if tmp1:
        trainId = str(tmp1[0]['id'])
        resourceId = str(tmp1[0]['ResourceId'])
    results = {'trainId': trainId, 'resourceId': resourceId}
    response['results'] = results
    return response


def saveFilesForCustomTrain(data):
    print('saveFilesForCustomTrain', data)

    trainer = CustomTrainer()
    response = {'success': 'Data retrieved', 'results': []}
    workdir = os.getcwd()
    resourceId = data['form']['resourceId']
    formData = data['form']
    trainId = formData['trainId']

    trainingType = formData['trainingType']

    if trainingType == 'Classification':
        if formData['trainId'] == '-1':
            tmp1 = trainer.getRecords(
                'select id,ResourceId from CUSTOM_TRAINING_HISTORY where userId={user} and type ={type} and modelId is null'.format(user=formData['userId'], type=formData['type']))
            print('tmp1', tmp1)
            if not tmp1:
                insert = 'insert into ALL_RESOURCES(Name,Description) values ("{name}","{desc}")'.format(
                    name=data['form']['name'], desc=data['form']['desc'])
                resourceId = trainer.executeQuery('insert', insert)
                insert = 'insert into CUSTOM_TRAINING_HISTORY(ResourceId,type,trainedOn,userId) values ("{resource}","{type}","{now}","{user}")'.format(
                    resource=resourceId, type=data['form']['type'], now=now, user=formData['userId'])
                trainId = str(trainer.executeQuery('insert', insert))
            else:
                trainId = str(tmp1[0]['id'])
                resourceId = str(tmp1[0]['ResourceId'])

        if not os.getcwd().endswith('uploads'):
            os.chdir('uploads')

        if not os.path.isdir('CustomTrain'):
            os.mkdir('CustomTrain')

        if not os.path.isdir(os.path.join('CustomTrain', trainId)):
            os.mkdir(os.path.join('CustomTrain', trainId))

        for file in data['files']:
            path = os.path.join(
                "CustomTrain", trainId, file.filename)
            file.save(path)
            response['results'].append(path)

        os.chdir(workdir)

        insertdata = []
        for i, path in enumerate(response['results']):
            insertdata.append("({id},{no},'{name}','{path}','{now}')".format(
                id=resourceId, no=i, name=os.path.split(path)[-1], path=path, now=now))

        insertdata = ','.join(insertdata)
        insert = 'insert into RESOURCE_DETAILS (ResourceId,FileNo,FileName,FilePath,UpdatedOn) values {insertData}'.format(
            insertData=insertdata)
        trainer.executeQuery('insert', insert)
        print('inser2 ->', insert)

        data['form']['resourceId'] = resourceId
        data['form']['trainId'] = trainId
        data['form']['numberOfFiles'] = str(len(data['files']))
        response['results'] = data['form']
        print('resp', response)
        data['form']['csvResponse'] = generate_csv_file(response['results'])

    else:
        if formData['trainId'] == '-1':
            tmp1 = trainer.getRecords(
                'select id,ResourceId from CUSTOM_TRAINING_HISTORY where userId={user} and type ={type} and modelId is null'.format(user=formData['userId'], type=formData['type']))
            print('tmp1', tmp1)
            if not tmp1:
                insert = 'insert into ALL_RESOURCES(Name,Description) values ("{name}","{desc}")'.format(
                    name=data['form']['name'], desc=data['form']['desc'])
                resourceId = trainer.executeQuery('insert', insert)
                insert = 'insert into CUSTOM_TRAINING_HISTORY(ResourceId,type,trainedOn,userId) values ("{resource}","{type}","{now}","{user}")'.format(
                    resource=resourceId, type=data['form']['type'], now=now, user=formData['userId'])
                trainId = str(trainer.executeQuery('insert', insert))
            else:
                trainId = str(tmp1[0]['id'])
                resourceId = str(tmp1[0]['ResourceId'])

        if not os.getcwd().endswith('uploads'):
            os.chdir('uploads')

        if not os.path.isdir('CustomTrain'):
            os.mkdir('CustomTrain')

        if not os.path.isdir(os.path.join('CustomTrain', trainId)):
            os.mkdir(os.path.join('CustomTrain', trainId))

        if not os.path.isdir(os.path.join('CustomTrain', trainId, 'images')):
            os.mkdir(os.path.join('CustomTrain', trainId, 'images'))

        if not os.path.isdir(os.path.join('CustomTrain', trainId, 'labels')):
            os.mkdir(os.path.join('CustomTrain', trainId, 'labels'))

        for file in data['files']:
            if file.filename.endswith('.txt'):
                path = os.path.join(
                    "CustomTrain", trainId, 'labels', file.filename)
            elif file.filename.endswith('.yml') or file.filename.endswith('.yaml'):
                path = os.path.join(
                    "CustomTrain", trainId, file.filename)
            else:
                path = os.path.join(
                    "CustomTrain", trainId, 'images', file.filename)

            file.save(path)
            response['results'].append(path)

        os.chdir(workdir)

        insertdata = []
        for i, path in enumerate(response['results']):
            insertdata.append("({id},{no},'{name}','{path}','{now}')".format(
                id=resourceId, no=i, name=os.path.split(path)[-1], path=path, now=now))

        insertdata = ','.join(insertdata)
        insert = 'insert into RESOURCE_DETAILS (ResourceId,FileNo,FileName,FilePath,UpdatedOn) values {insertData}'.format(
            insertData=insertdata)
        trainer.executeQuery('insert', insert)
        print('inser2 ->', insert)

        data['form']['resourceId'] = resourceId
        data['form']['trainId'] = trainId
        data['form']['numberOfFiles'] = str(len(data['files']))
        response['results'] = data['form']
        print('resp', response)
        data['form']['ymlFile'] = generate_yml_file(response['results'])

    return response
