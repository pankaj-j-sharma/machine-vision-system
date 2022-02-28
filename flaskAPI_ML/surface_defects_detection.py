# reference https://github.com/FantacherJOY/Metal-Surface-Defect-Inspection/blob/master/Surface_defects_detection.ipynb
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
import numpy as np
import cv2

model_dir = os.path.join(os.getcwd(),"model_files")
modelFileSurfaceDefect = os.path.join(model_dir,"mdlsurfdefects.h5")

def setup_and_train():
    
    #before doing these please make a folder as train data and in the train folder create another six folder for six defects
    #These six folder should have 300 images.
    try:
        source1 = "NEU/train"
        os.mkdir("NEU/test")
        dest11 = "NEU/test"
        files = os.listdir(source1)
        import shutil
        import numpy as np
        for f in files:
            os.mkdir(dest11 + '/'+ f)
            spilt_num=int(len(os.listdir(source1 + '/'+ f))*0.08)
            for i in os.listdir(source1 + '/'+ f)[spilt_num:]:
                shutil.move(source1 + '/'+ f +'/'+ i, dest11 + '/'+ f +'/'+ i)
    except:
        print("\nEverything already have in the directory. You don't need to run this cell")


    try:
        source1 = "NEU/test"
        os.mkdir("NEU/valid")
        dest11 = "NEU/valid"
        files = os.listdir(source1)
        import shutil
        import numpy as np
        for f in files:
            os.mkdir(dest11 + '/'+ f)
            spilt_num=int(len(os.listdir(source1 + '/'+ f))*0.5)
            for i in os.listdir(source1 + '/'+ f)[spilt_num:]:
                shutil.move(source1 + '/'+ f +'/'+ i, dest11 + '/'+ f +'/'+ i)
    except:
        print("\nEverything already have in the directory. You don't need to run this cell")


    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


    test_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches of 10 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
            'NEU/train',
            target_size=(200, 200),
            batch_size=10,
            class_mode='categorical')

    # Flow validation images in batches of 10 using test_datagen generator
    validation_generator = test_datagen.flow_from_directory(
            'NEU/valid',
            target_size=(200, 200),
            batch_size=10,
            class_mode='categorical')


    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy') > 0.978 ):
                print("\nReached 97.8% accuracy so cancelling training!")
                self.model.stop_training = True


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (2,2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    print('Compiled!')

    callbacks = myCallback()
    history = model.fit(train_generator,
            batch_size = 32,
            epochs=15,
            validation_data=validation_generator,
            callbacks=[callbacks],
            verbose=1, shuffle=True)

    model.save(modelFileSurfaceDefect)

    plt.figure(1)  
    # summarize history for accuracy  
    plt.subplot(211)  
    plt.plot(history.history['accuracy'])  
    plt.plot(history.history['val_accuracy'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  

     # summarize history for loss  

    plt.subplot(212)  
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    plt.show()    
    
def load_saved_model():
    model = tf.keras.models.load_model(modelFileSurfaceDefect)
    model.summary()
    target_labels = ['Crazing' ,'Inclusion' ,'Patches' ,'Pitted', 'Rolled' ,'Scratches']
    return model,target_labels
    
def convert_image_to_array(files):
    images_as_array=[]
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array

def convert_uploaded_img_to_array(files):
    images_as_array=[]
    for file in files:        
        images_as_array.append(cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR))
    return images_as_array
    

def predict_surface_defects_old(files,fileList=None):    

    predictions=[]
    filestr = fileList.read()
    npimg = cv2.imdecode(np.fromstring(filestr, np.uint8), cv2.IMREAD_COLOR)
    
    model,target_labels = load_saved_model()
    x_test = np.array(convert_image_to_array(files))
    x_test = x_test.astype('float32')/255
    
    #x_test = npimg.astype('float32')/255
    y_pred = model.predict(x_test)
    for idx in range(x_test.shape[0]):
        pred_idx = np.argmax(y_pred[idx])
        true_pred = target_labels[pred_idx]
        predictions.append(true_pred)

    print('predictions->',predictions,'x_test',x_test.shape,npimg.shape)
    return {"predictions":predictions}


def predict_surface_defects(fileList):    
    print('files receieved',fileList)
    predictions=[]
    npimg = np.array(convert_uploaded_img_to_array(fileList))   
    model,target_labels = load_saved_model()
    npimg = npimg.astype('float32')/255
    
    y_pred = model.predict(npimg)
    for idx in range(npimg.shape[0]):
        pred_idx = np.argmax(y_pred[idx])
        true_pred = target_labels[pred_idx]
        predictions.append(true_pred)

    print('predictions->',predictions,npimg.shape)
    return {"predictions":predictions}

#print(predict_surface_defects(['test_images//In_100.bmp','test_images//Sc_103.bmp']))
