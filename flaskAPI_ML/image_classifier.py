import os
from tensorflow import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from keras.utils import to_categorical
from tqdm import tqdm
from sqlite_db_connect import SQLiteConnect
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


global_image_shape = (100, 100, 1)


class Classifier(SQLiteConnect):
    def ___init__(self):
        super.__init__()


def start_training(data):
    classifierObj = Classifier()
    select = 'select * from RESOURCE_DETAILS where ResourceId in ( select resource from ALL_PROJECTS where id = {project})'
    fileName = str(data['projectName'])+'-'+str(int(data['projectIter'])+1)
    data = classifierObj.getRecords(select.format(project=data['projectId']))
    train = pd.DataFrame(data)

    # We have grayscale images, so while loading the images we will keep grayscale=True, if you have RGB images, you should set grayscale as False
    image_shape = global_image_shape
    train_image = []
    for i in tqdm(range(train.shape[0])):
        img = image.load_img(os.path.join(
            'uploads', train['FilePath'][i]), target_size=image_shape, color_mode="grayscale")
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
        X, y, random_state=42, test_size=0.2)

    rel_model_path = os.path.join(
        'uploads', 'User_Model_Files', '{name}.hdf5'.format(name=fileName))
    model_save_path = os.path.join(
        os.getcwd(), rel_model_path)

    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    checkpoint = ModelCheckpoint(
        filepath=model_save_path, verbose=1, save_best_only=True, monitor='val_loss')

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
              activation='relu', input_shape=image_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam', metrics=['accuracy'])

    model.summary()
    history = model.fit(X_train, y_train, epochs=50, validation_data=(
        X_test, y_test), callbacks=[early_stop, checkpoint])

    # evaluate the model
    train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_acc = model.evaluate(X_test, y_test, verbose=0)

    #y_pred = model.predict_classes(X_test)
    y_test = np.argmax(y_test, axis=-1)
    # print('preds', y_pred, y_test)

    predict_y = model.predict(X_test)
    y_pred = np.argmax(predict_y, axis=1)

    f1Score = round(f1_score(y_test, y_pred, average="macro"), 2)
    precisionScore = round(precision_score(y_test, y_pred, average="macro"), 2)
    recallScore = round(recall_score(y_test, y_pred, average="macro"), 2)

    rel_encoder_path = os.path.join(
        'uploads', 'User_Model_Files', '{name}.pkl'.format(name=fileName))
    encoder_file = os.path.join(
        os.getcwd(), rel_encoder_path)
    with open(encoder_file, 'wb') as f:
        pickle.dump(label_encoder, f)

    # with open(encoder_file, 'rb') as e:
    #     lbl = pickle.load(e)
    # print(lbl.inverse_transform(y_pred))

    response = {'success': 'Data retrieved',
                'results': [{'model_name': fileName, 'model_file_path': rel_model_path, 'encoder_file': rel_encoder_path, 'history': history.history, 'train_acc': train_acc, 'test_acc': test_acc, 'f1': f1Score, 'precision': precisionScore, 'recall': recallScore}]}

    return response


def start_inferencing(data):
    classifierObj = Classifier()
    dbData = classifierObj.getRecords(
        'select id,FileName,FilePath,Tag from RESOURCE_DETAILS where stepno=3 and ResourceId={resource}'.format(resource=data['resourceId']))
    test = pd.DataFrame(dbData)
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
    test.apply(lambda row: classifierObj.executeQuery(
        'update', updateStmt.format(tag=row['Tag'], id=row['id'])), axis=1)

    return test.to_dict(orient="records")
