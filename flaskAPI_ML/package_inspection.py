import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

model_dir = os.path.join(os.getcwd(),"model_files")
modelFilePackageInspection = os.path.join(model_dir,'model_package_.h5')

def convert_uploaded_img_to_array(files):
    images_as_array=[]
    for file in files:        
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)    
        images_as_array.append(image)
    return images_as_array


def load_saved_model():
    return tf.keras.models.load_model(modelFilePackageInspection)


def test_prediction():
    img = keras.preprocessing.image.load_img(os.path.join(os.getcwd(),"0105413725474_side.png"), target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    print('img_array ->',img_array.shape)
    predictions = pkgInspectionModel.predict(img_array)
    print('preds-> ',predictions)
    

def predict_packaging_defects(fileList):
    predictions=[]
    npimg = np.array(convert_uploaded_img_to_array(fileList))   
    y_pred = pkgInspectionModel.predict(npimg)

    for idx in range(npimg.shape[0]):
        prediction = y_pred[idx][np.argmax(y_pred[idx])]
        if (prediction < 0.5):
            predicted_label = "Package is Damaged"
            prob = (1-prediction.sum()) * 100
        else:
            predicted_label = "Package is Intact"
            prob = prediction.sum() * 100

        predictions.append(predicted_label)
        
    return {'predictions':predictions}


pkgInspectionModel = load_saved_model()

if __name__=='__main__':
    test_prediction()