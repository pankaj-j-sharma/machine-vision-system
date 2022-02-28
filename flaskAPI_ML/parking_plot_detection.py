# -*- coding: utf-8 -*-
""" Problem Statement: To Detect parking plot is free or full.
    OBJECTIVE: PARKING PLOT DETECTION """
    

""" Step 1: Installation """
# Note: Please install libraries from requirements.txt and proceed.

""" Step 2: Import Libraries """
# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
#%matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

import cv2
# Libraries converting an image with the Keras API
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
# Helper libraries
import numpy as np
import matplotlib.pyplot as pl

""" Step 3: Data load & Preprocessing """
X=[]
Z=[]
IMG_SIZE=150
FREE_DIR='/content/gdrive/MyDrive/Parking_Lot/input/find-a-car-park/data/Free'
FULL_DIR='/content/gdrive/MyDrive/Parking_Lot/input/find-a-car-park/data/Full'

image_path_2 = "/content/gdrive/MyDrive/Parking_Lot/data/Full/img_1001172558.jpg"
X=[]
IMG_SIZE=150

def assign_label(img,label):
    return label

def make_train_data(label,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,label)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        X.append(np.array(img))
        Z.append(str(label))

# make 'Free' data
make_train_data('Free',FREE_DIR)
print(len(X))

# make 'Full' data
make_train_data('Full',FULL_DIR)
print(len(X))

# check some image
fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Flower: '+Z[l])
        
plt.tight_layout()

le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,2)
X=np.array(X)
X=X/255

# separate data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

# fix random seed
np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)

class Parkingplot:
    
    def __init__(self):
        print("*****Parking Plot*****")
        self.nm = "Parking Plot"
        
    """ Step 4: Model Building & Training """
    def model_training(self):
        # # modelling starts using a CNN.

        self.model = Sequential()
        self.model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        self.model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        self.model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(2, activation = "softmax"))
        
        batch_size=128
        epochs=10

        # use callback only ReduceLROnPlateau
        from keras.callbacks import ReduceLROnPlateau
        red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)

        # data augmentation to prevent overfitting
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1, # Randomly zoom image 
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        datagen.fit(x_train)
        
        self.model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
        self.model.summary()
        
        """ Model Training """
        History = self.model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
        
        """ Model Evaluation """
        def model_evaluation(self):
            plt.plot(self.History.history['loss'])
            plt.plot(self.History.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.legend(['train', 'test'])
            plt.show()
            
            plt.plot(self.History.history['acc'])
            plt.plot(self.History.history['val_acc'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epochs')
            plt.legend(['train', 'test'])
            plt.show()
            
            # getting predictions on val set.
            pred=self.model.predict(x_test)
            self.pred_digits=np.argmax(pred,axis=1)
            
        
        """ Step 5 : Model Prediction """
        def model_prediction(self):
            # now storing some properly as well as misclassified indexes'.
            i=0
            prop_class=[]
            mis_class=[]
            
            for i in range(len(y_test)):
                if(np.argmax(y_test[i])==self.pred_digits[i]):
                    prop_class.append(i)
                if(len(prop_class)==8):
                    break

            i=0
            for i in range(len(y_test)):
                if(not np.argmax(y_test[i])==self.pred_digits[i]):
                    mis_class.append(i)
                if(len(mis_class)==8):
                    break
                
            warnings.filterwarnings('always')
            warnings.filterwarnings('ignore')

            count=0
            fig,ax=plt.subplots(4,2)
            fig.set_size_inches(15,15)
            for i in range (4):
                for j in range (2):
                    ax[i,j].imshow(x_test[prop_class[count]])
                    ax[i,j].set_title("Predicted : "+str(le.inverse_transform([self.pred_digits[prop_class[count]]]))+"\n"+"Actual : "+str(le.inverse_transform([np.argmax([y_test[prop_class[count]]])])))
                    plt.tight_layout()
                    count+=1
                    
            
        """ Step 6 : Model Save and Load """
        def model_save_load(self):
            # save model
            self.model.save('car_parking_plot.h5')
            print('Model Saved!')
            
            # load model
            self.savedModel_h5=load_model('car_parking_plot.h5')
            self.savedModel_h5.summary()
            
        
        """ Step 7 : Model Inferencing """
        def model_inferencing(self):
            # load the image
            img = load_img('/content/gdrive/MyDrive/Parking_Lot/data/Full/img_1001172558.jpg',target_size=(150, 150))
            print("Orignal:" ,type(img))
            img_path = "/content/gdrive/MyDrive/Parking_Lot/data/Full/img_1001172558.jpg"
            plt.imshow(img)
            plt.show()
            
            img_array = image.img_to_array(img)
            img_batch = np.expand_dims(img_array, axis=0)
            
            img_preprocessed = preprocess_input(img_batch)
            prediction = self.savedModel_h5.predict(img_preprocessed)
            prediction = prediction[0][0]
            pred_digits=np.argmax(prediction,axis=1)
            
            plt.title(f"\n Actual result :", weight='bold', size=20)
            if (prediction < 0.5):
                predicted_label = "Free Space"
                prob = (1-prediction.sum()) * 100
            else:
                predicted_label = "Full slot"
                prob = prediction.sum() * 100

            cv2.putText(img=img, text=f"Predicted: {predicted_label}", org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 0, 255), thickness=2)
            plt.imshow(img,cmap='gray')
            plt.axis('off')
            
            
if __name__ == '__main__':
    parkingplot_obj = Parkingplot()
    parkingplot_obj.model_training()
    parkingplot_obj.model_evaluation()
    parkingplot_obj.model_prediction()
    parkingplot_obj.model_save_load()
    parkingplot_obj.model_inferencing()
            
        
        
        
        
        
        
        
        
        
        
        
        