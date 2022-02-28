import base64
from base64 import encodebytes
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO,BytesIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util
import glob
import cv2
import uuid

model_dir = os.path.join(os.getcwd(),"model_files")
saved_model_dir = os.path.join(model_dir,'saved_model')

def load_saved_model():
    #tf.keras.backend.clear_session()
    model = tf.saved_model.load(saved_model_dir)
    labelmap_path = os.path.join(model_dir,"labelmap.pbtxt")
    category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
    return model,category_index


def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
      

def convert_uploaded_img_to_array(files):
    images_as_array=[]
    for file in files:        
        images_as_array.append(cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.COLOR_BGR2RGB))
        tmp = Image.fromarray(images_as_array[-1])
        filename="og_"+str(uuid.uuid4())+".jpeg"
        #cv2.imwrite(os.path.join('tmp','cv2_'+filename), images_as_array[-1])
        
    return images_as_array


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)
    #print('output_dict',output_dict)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                  output_dict['detection_masks'], output_dict['detection_boxes'],
                   image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
    return output_dict

def test_prediction():
    prediction_results=[]
    for image_path in glob.glob(os.path.join(os.getcwd(),'test_images/*.jpg')):
        image_np = load_image_into_numpy_array(image_path)
        output_dict = run_inference_for_single_image(model, image_np)
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks_reframed', None),
          use_normalized_coordinates=True,
          line_thickness=4)
        label = output_dict['detection_classes'][np.where([f for f in output_dict['detection_scores'] if f>.5])]
        prediction_results.append({'img':image_np,'pred':[v for (k,v) in category_index.items() if k in label ]})
        display(Image.fromarray(image_np))
    return prediction_results

def predict_hard_hat_present(fileList):
    clear_directory()
    results = []
    npimg = np.array(convert_uploaded_img_to_array(fileList))   

    for idx in range(npimg.shape[0]):
        output_dict = run_inference_for_single_image(model, npimg[idx])
        vis_util.visualize_boxes_and_labels_on_image_array(
          npimg[idx],
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          target_labels,
          instance_masks=output_dict.get('detection_masks_reframed', None),
          use_normalized_coordinates=True,
          skip_scores=True,skip_labels=True,
          line_thickness=4)
        label = output_dict['detection_classes'][np.where([f for f in output_dict['detection_scores'] if f>.5])]
        results.append({'img':send_img_or_url(npimg[idx]),'pred':[v for (k,v) in target_labels.items() if k in label ]})
        #print('predictions',results)
    return {"predictions":results}


def send_img_or_url(image_np,sendUrl=True):
    img = Image.fromarray(image_np)
    if sendUrl:
        filename=str(uuid.uuid4())+".jpeg"
        cv2.imwrite(os.path.join('tmp',filename), image_np)        
        #img.save(os.path.join('tmp',filename),format='jpeg')
        img = filename
    else:
        img_byte_arr = BytesIO()    
        img.save(img_byte_arr,format='PNG')
        img = encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return img
    
    
def clear_directory():
    files = glob.glob(os.path.join(os.getcwd(),'tmp','*.jpeg'))
    for f in files:
        os.remove(f)    

        
#to get performance improvement model loaded on import         
model,target_labels = load_saved_model()       