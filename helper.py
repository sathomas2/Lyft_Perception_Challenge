import cv2
import random
import subprocess
import numpy as np
import itertools
import operator
import os, csv, re
import tensorflow as tf
import warnings
import time, datetime
from sklearn.metrics import recall_score, precision_score, f1_score

def get_label_info(csv_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV format!

    # Arguments
        csv_path: The file path of the class dictionary
        
    # Returns
        Two lists: one for the class names and the other for the label values
    """
    filename, file_extension = os.path.splitext(csv_path)
    if not file_extension == ".csv":
        return ValueError("File is not a CSV!")

    class_names = []
    label_values = []
    with open(csv_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)
        for row in file_reader:
            class_names.append(row[0])
            label_values.append([int(row[1]), int(row[2]), int(row[3])])
    return class_names, label_values


def one_hot_it(label, label_values, reduce_labels=False, frozen=False):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values
        reduce_labels: Whether to use all 14 classes or only 3, vehicle, road, and void
        frozen: If training an extra layer on only the vehicle class on top of a frozen model
                that used 3 classes
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
   
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    if reduce_labels == False:
        return semantic_map
    else:
        if frozen:
            reduce_label_vals = [0,1,2,3,4,5,6,7, 8,9,11,12,13]
            out = np.zeros((label.shape[0], label.shape[1], 1))
            out[:,:,0][semantic_map[:,:,10]==1] = 1
        else:
            reduce_label_vals = [0,1,2,3,4,5,8,9,11,12,13]
            out = np.zeros((label.shape[0], label.shape[1], 3))
            for label in reduce_label_vals:
                out[:,:,0][semantic_map[:,:,label]==1] = 1
            out[:,:,1][semantic_map[:,:,6]==1] = 1
            out[:,:,1][semantic_map[:,:,7]==1] = 1
            out[:,:,2][semantic_map[:,:,10]==1] = 1
        return out
    
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x


def colour_code_segmentation(image, label_values, reduce=False, frozen=False):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values
        reduce_labels: Whether to use all 14 classes or only 3, vehicle, road, and void
        frozen: If training an extra layer on only the vehicle class on top of a frozen model
                that used 3 classes
        
    # Returns
        Colour coded image for segmentation visualization
    """
    if reduce:
        if frozen:
            colour_codes = np.array([[0, 0, 0], [255, 0, 0], [255, 0, 0]])
        else:
            colour_codes = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0]])
    else:
        colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

def calc_Fscore(r, p, B):
    """
    Given a recall and precision of parituclar class predictions, calculate weighted F score.

    # Arguments
        r: Recall
        p: Precision
        B: Road or Vehicle weight
        
    # Returns
        Weighted F score
    """
    return (1+(B*B))*((p*r)/((B*B*p) + r))
    
def get_rpf_scores(preds, lls, reduce=False, carB=0.5, roadB=2):
    """
    Given a recall and precision of parituclar class predictions, calculate weighted F score.

    # Arguments
        preds: Predictions
        lls: Labels
        reduce: All 14 classes or only 3, vehicle, road and void
        carB: Car weight
        roadB: Road weight
        
    # Returns
        Weighted F score
    """

    car_lls = np.zeros_like(lls)
    road_lls = np.zeros_like(lls)
    car_preds = np.zeros_like(preds)
    road_preds = np.zeros_like(preds)
    if reduce:
        car_lls[lls==2] = 1
        road_lls[lls==1] = 1
        car_preds[preds==2] = 1
        road_preds[preds==1] = 1
    else:
        car_lls[lls==10] = 1
        road_lls[lls==6] = 1
        road_lls[lls==7] = 1
        car_preds[preds==10] = 1
        road_preds[preds==6] = 1
        road_preds[preds==7] = 1

    car_lls = car_lls.reshape((lls.shape[0],-1))
    road_lls = road_lls.reshape((lls.shape[0],-1))
    car_preds = car_preds.reshape((preds.shape[0], -1))
    road_preds = road_preds.reshape((preds.shape[0], -1))

    car_recalls = []
    car_precisions = []
    car_ffs = []

    road_recalls = []
    road_precisions = []
    road_ffs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ = [car_recalls.append(recall_score(car_lls[i], car_preds[i])) for i in range(len(car_lls))]
        _ = [car_precisions.append(precision_score(car_lls[i], car_preds[i])) for i in range(len(car_lls))]

        _ = [road_recalls.append(recall_score(road_lls[i], road_preds[i])) for i in range(len(road_lls))]
        _ = [road_precisions.append(precision_score(road_lls[i], road_preds[i])) for i in range(len(road_lls))]

    carR = np.mean(car_recalls)
    carP = np.mean(car_precisions)
    carF = calc_Fscore(carR, carP, carB)

    roadR = np.mean(road_recalls)
    roadP = np.mean(road_precisions)
    roadF = calc_Fscore(roadR, roadP, roadB)

    return 0, 0, carF, 0, 0, roadF

def download_checkpoints(model_name):
    """
    Given a model name, download checkpoints pretrained on ImageNet.

    # Arguments
        model_name: e.g. DeepLabV3_plus-Res152

    """
    subprocess.check_output(["python", "get_pretrained_checkpoints.py", "--model=" + model_name])
    
def get_output_name(input_name, data='train'):
    """
    Given input data filename, retrieve output data filename.

    # Arguments
        input_name: input filename
        data: train, val, or test

    # Returns
        output data filename
    """
    return re.sub(data, data+'_labels',input_name)

def prepare_data(dataset_dir='data'):
    """
    Given input data filename, retrieve output data filename.

    # Arguments
        dataset_dir: directory containing train, train_labels,
                    val, val_labels, and test and test_labels

    # Returns
        train, validation, and test data filenames
    """

    input_names=[]
    output_names=[]
    train_input_names = []
    train_output_names = []
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]
    
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        if file[-3:] == 'png':
            train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
            #input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
        
    for file in train_input_names:
        train_output_names.append(get_output_name(file, data='train'))
        #output_names.append(get_output_name(file, data='train'))
    
    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()
        if file[-3:] == 'png':
            val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
            #input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
    
    for file in val_input_names:
        val_output_names.append(get_output_name(file, data='val'))
        #output_names.append(get_output_name(file, data='val'))
    
    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        if file[-3:] == 'png':
            test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
            #input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
    
    for file in test_input_names:
        test_output_names.append(get_output_name(file, data='test'))
        #output_names.append(get_output_name(file, data='test'))
        
    return train_input_names, train_output_names, val_input_names, val_output_names


def load_image(path, resize=False):
    
    """
    Given filename, load image.

    # Arguments
        path: image filename
        resize: whether to resize the image

    # Returns
        RGB image, cropped to only include area that may contain road or vehicles
    """
    image = cv2.cvtColor(cv2.imread(path,-1), cv2.COLOR_BGR2RGB)[176:512, :, :]
    if resize:
        return cv2.resize(image, (400, 300))
    else:
        return image

def random_brighten(x):
    """
    Given image, randomly brighten for training.

    # Arguments
        x: image

    # Returns
        RGB image, randomly brightened or darkened
    """
    dark = np.random.randint(1,3)
    if dark == 1:
        gamma = np.random.random() * 0.7 + 0.3
    if dark == 2:
        gamma = np.random.random() * 3.5 + 1.1
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(x, table)

    
def random_crop(image, label, crop_ratio=2.381, crop_width_min=400):
    """
    Given image, randomly crop maintaing aspect ratio

    # Arguments
        image: image
        label: groundtruth
        crop_ratio: amount to divide width by to obtain height, maintaining aspect ratio
        crop_width_min: min width to crop from image 

    # Returns
        RGB image, randomly cropped
    """
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')
    
    crop_width = np.random.randint(crop_width_min, 800)
    crop_height = int(crop_width//crop_ratio)
    
    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)
        
        if len(label.shape) == 3:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape exceeds image dimensions!')
    
    
def data_augmentation(input_image, output_image, h_flip=True, 
                      brightness=True, rotation=None, crop=True,
                      crop_ratio=2.381, crop_width_max=400):
    """
    Given training batch, randomly augment data

    # Arguments
        input_image: image
        output_image: groundtruth
        h_flip: whether to randomly flip horizontally
        brightness: whether to randomly brighten or darken
        rotation: whether to randomly rotate
        crop: whether to randomly crop
        crop_ratio: amount to divide width by to obtain height, maintaining aspect ratio
        crop_width_min: min width to crop from image 

    # Returns
        RGB image, randomly augmented
    """

    shape = input_image.shape
    
    if h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    
    if brightness:
        input_image = random_brighten(input_image)
    
    if rotation is not None:
        angle = random.uniform(-1*rotation, rotation)
    if rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]))#, flags=INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]))#, flags=INTER_NEAREST)
    
    if crop:
        input_image, output_image = random_crop(input_image, output_image)
        input_image = cv2.resize(input_image, (shape[1], shape[0]))
        output_image = cv2.resize(output_image, (shape[1], shape[0]))
        
    return input_image, output_image

def get_batches(input_fns, output_fns, batch_size, label_values, mode='train', reduce=False):
    """
    Get batch function for training

    # Arguments
        input_fns: image fns
        output_fns: groundtruth fns
        batch_size
        label_values
        mode: train for data augmentation, otherwise no augmentation
        reduce: whether to use all 14 classes or only 3, vehicle, road and void

    # Returns
        batch_x and batch_y
    """

    id_list = np.random.permutation(len(input_fns))
    n_images = len(id_list)
    for batch_i in range(0, n_images, batch_size):
        x = []
        y = []
        for i in range(batch_i, min(batch_i+batch_size, n_images)):
            id_ = id_list[i]
 
            temp_x = load_image(input_fns[id_])
            temp_y = load_image(output_fns[id_])
                
            if mode == 'train':
                temp_x, temp_y = data_augmentation(temp_x, temp_y, 
                                                   h_flip=True, brightness=True, rotation=5, 
                                                   crop=True, crop_ratio=2.381, crop_width_max=400)
               
            temp_x = cv2.resize(temp_x, (400, 168))
            temp_y = cv2.resize(temp_y, (400, 168))
            x.append(temp_x)
            y.append(one_hot_it(label=temp_y, label_values=label_values, reduce_labels=reduce))

        yield np.array(x), np.array(y)