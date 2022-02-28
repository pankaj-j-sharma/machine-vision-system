import pprint
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import warnings
warnings.filterwarnings('ignore')
import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend
from sklearn.metrics import confusion_matrix, classification_report
import shap
from operator import itemgetter
from keras.models import load_model


model_dir = os.path.join(os.getcwd(),"model_files")
modelFileMetalCastDefect = os.path.join(model_dir,'casting_product_detection.hdf5')


def predict_metal_cast_defects(fileList):
    casting_obj = Casting()
    if not os.path.isfile(modelFileMetalCastDefect):
        casting_obj.model_training()
        casting_obj.model_perfomance()
    return casting_obj.model_inferencing(fileList)


class Casting:
    def __init__(self):
        print("*********")
        self.name = "Industrial_Casting"
        """ Step 3: Load the dataset """
        self.my_data_dir = os.path.join(os.getcwd(),'casting_data/')
        self.train_path = os.path.join(self.my_data_dir , 'train/')
        self.test_path = os.path.join(self.my_data_dir ,'test/')
        self.image_shape = (300,300,1) # 300 × 300、graysclaed (full-color : 3)
        self.batch_size = 32


    def data_augmentation(self):
        """ Step 4: Preprocessing (Data Augmentation) """
        img = cv2.imread(self.train_path + 'ok_front/cast_ok_0_1.jpeg')
        img_4d = img[np.newaxis]
        plt.figure(figsize=(25,10))
        generators = {"rotation":ImageDataGenerator(rotation_range=180), 
                      "zoom":ImageDataGenerator(zoom_range=0.7), 
                      "brightness":ImageDataGenerator(brightness_range=[0.2,1.0]), 
                      "height_shift":ImageDataGenerator(height_shift_range=0.7), 
                      "width_shift":ImageDataGenerator(width_shift_range=0.7)}

        plt.subplot(1, 6, 1)
        plt.title("Original", weight='bold', size=15)
        plt.imshow(img)
        plt.axis('off')
        cnt = 2
        for param, generator in generators.items():
            image_gen = generator
            gen = image_gen.flow(img_4d, batch_size=1)
            batches = next(gen)
            g_img = batches[0].astype(np.uint8)
            plt.subplot(1, 6, cnt)
            plt.title(param, weight='bold', size=15)
            plt.imshow(g_img)
            plt.axis('off')
            cnt += 1
        plt.show()

        """ Step 5: Execute Data Augmentation """
        image_gen = ImageDataGenerator(rescale=1/255, zoom_range=0.1, brightness_range=[0.9,1.0])

        self.train_set = image_gen.flow_from_directory(self.train_path,
                                                    target_size=self.image_shape[:2],
                                                    color_mode="grayscale",
                                                    classes={'def_front': 0, 'ok_front': 1},
                                                    batch_size=self.batch_size,
                                                    class_mode='binary',
                                                    shuffle=True,
                                                    seed=0)

        self.test_set = image_gen.flow_from_directory(self.test_path,
                                                   target_size=self.image_shape[:2],
                                                   color_mode="grayscale",
                                                   classes={'def_front': 0, 'ok_front': 1},
                                                   batch_size=self.batch_size,
                                                   class_mode='binary',
                                                   shuffle=False,
                                                   seed=0)
        self.train_set.class_indices
    

    """ Step 6: Modelling, Build Model, Model Training """
    def model_training(self):
        self.data_augmentation()
        backend.clear_session()
        self.model = Sequential()
        self.model.add(Conv2D(filters=16, kernel_size=(7,7), strides=2, input_shape=self.image_shape, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        self.model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, input_shape=self.image_shape, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, input_shape=self.image_shape, activation='relu', padding='same'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        self.model.add(Flatten())
        self.model.add(Dense(units=224, activation='relu'))
        self.model.add(Dropout(rate=0.2))
        self.model.add(Dense(units=1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model.summary()

        #plot_model(self.model, show_shapes=True, expand_nested=True, dpi=60)
        """ Build Model """
        model_save_path = modelFileMetalCastDefect
        early_stop = EarlyStopping(monitor='val_loss',patience=2)
        checkpoint = ModelCheckpoint(filepath=model_save_path, verbose=1, save_best_only=True, monitor='val_loss')
        """ Model Training """
        n_epochs = 20
        results = self.model.fit_generator(self.train_set, epochs=n_epochs, validation_data=self.test_set, callbacks=[early_stop,checkpoint])

        self.model_history = results.history
        json.dump(self.model_history, open('model_history.json', 'w'))


    """ Step 7: Model Performance """
    def model_perfomance(self):
        losses = pd.DataFrame(self.model_history)
        losses.index = map(lambda x : x+1, losses.index)
        losses.head(3)

        g = hv.Curve(losses.loss, label='Training Loss') * hv.Curve(losses.val_loss, label='Validation Loss') \
    * hv.Curve(losses.accuracy, label='Training Accuracy') * hv.Curve(losses.val_accuracy, label='Validation Accuracy')
        g.opts(opts.Curve(xlabel="Epochs", ylabel="Loss / Accuracy", width=700, height=400,tools=['hover'],show_grid=True,title='Model Evaluation')).opts(legend_position='bottom')

        pred_probability = self.model.predict_generator(self.test_set)
        predictions = pred_probability > 0.5

        plt.figure(figsize=(10,6))
        plt.title("Confusion Matrix", size=20, weight='bold')
        sns.heatmap(
            confusion_matrix(self.test_set.classes, predictions),
            annot=True,
            annot_kws={'size':14, 'weight':'bold'},
            fmt='d',
            xticklabels=['Defect', 'OK'],
            yticklabels=['Defect', 'OK'])
        plt.tick_params(axis='both', labelsize=14)
        plt.ylabel('Actual', size=14, weight='bold')
        plt.xlabel('Predicted', size=14, weight='bold')
        plt.show()
        print(classification_report(self.test_set.classes, predictions, digits=3))


    def convert_uploaded_img_to_array(self,files):
        images_as_array=[]
        for file in files:        
            images_as_array.append(cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE))
        return images_as_array


    """ Step 9: Load the Model & Inferencing on random images """
    def model_inferencing(self,fileList=[]):
        predictions = []
        
        """ Load the Model """
        self.model = load_model(modelFileMetalCastDefect)
        self.model.summary()
        
        npimg = np.array(self.convert_uploaded_img_to_array(fileList))   
        npimg = npimg.astype('float32')/255
        print('npming->',npimg.shape)
        y_pred = self.model.predict(npimg.reshape(npimg.shape[0], *self.image_shape))
        
        # test_cases = ['cast_def_0_354.jpeg','cast_ok_0_246.jpeg','cast_def_0_359.jpeg']
        # test_image_path = self.my_data_dir + 'test_images/'

        # for i in range(len(test_cases)):
            # img_pred = cv2.imread(test_image_path + test_cases[i], cv2.IMREAD_GRAYSCALE)
            # print('image at ->',test_image_path + test_cases[i])
            # img_pred = img_pred / 255 # rescale
            # img_pred_re = img_pred.reshape(1, *self.image_shape)
            # print(test_cases[i],img_pred.shape,img_pred_re.shape)
            # prediction = self.model.predict(img_pred_re)
            
        for idx in range(npimg.shape[0]):
            prediction = y_pred[idx][np.argmax(y_pred[idx])]
            print('prediction ->',prediction)
            if (prediction < 0.5):
                predicted_label = "Defected"
                prob = (1-prediction.sum()) * 100
            else:
                predicted_label = "Good"
                prob = prediction.sum() * 100

            #predictions.append({'label':predicted_label,'confidence':'{:.3f}'.format(prob)})
            predictions.append(predicted_label)
        
        # for i in range(len(test_cases)):
            # img_pred = cv2.imread(test_image_path + test_cases[i], cv2.IMREAD_GRAYSCALE)
            # #print('img_pred',img_pred,os.path.join(test_image_path ,test_cases[i]))
            # img_pred = img_pred / 255 # rescale
            # prediction = self.model.predict(img_pred.reshape(1, *self.image_shape))

            # img = cv2.imread(test_image_path + test_cases[i])
            # label = test_cases[i].split("_")[0]

            # if (prediction < 0.5):
                # predicted_label = "def"
                # prob = (1-prediction.sum()) * 100
            # else:
                # predicted_label = "ok"
                # prob = prediction.sum() * 100

            # predictions.append({'name':test_cases[i],'label':predicted_label,'confidence':'{:.3f}'.format(prob)})
        return {'predictions':predictions}
        

if __name__ == '__main__':
    casting_obj = Casting()
    if not os.path.isfile(modelFileMetalCastDefect):
        casting_obj.model_training()
        casting_obj.model_perfomance()
        #casting_obj.model_prediction()
    print('Predictions->')
    pprint.pprint(casting_obj.model_inferencing())

