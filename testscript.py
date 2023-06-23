import glob
import pandas as pd
import cv2
import numpy as np
import os
import cnn
from cnn import model
from random import shuffle
from csv import writer
import csv
import Augmentor as aug
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
import tflearn
import openpyxl as oxl
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from PIL import Image

LR = 0.001
IMG_SIZE = 50
TEST_DIR = r'C:\Users\Lenovo\Downloads\NN Dataset\Test'
def create_label(image_name):
    """ Create a one-hot encoded vector from image name """
    word_label = image_name.split('_')[0]
    if word_label == 'Basketball':
        return np.array([1, 0, 0, 0, 0, 0])
    elif word_label == 'Football':
        return np.array([0, 1, 0, 0, 0, 0])
    elif word_label == 'Rowing':
        return np.array([0, 0, 1, 0, 0, 0])
    elif word_label == 'Swimming':
        return np.array([0, 0, 0, 1, 0, 0])
    elif word_label == 'Tennis':
        return np.array([0, 0, 0, 0, 1, 0])
    elif word_label == 'Yoga':
        return np.array([0, 0, 0, 0, 0, 1])


def create_test_data():
    testing_data = []

    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), create_label(img)])

    shuffle(testing_data)
    np.save('../test_data.npy', testing_data)
    return testing_data
test_data = create_test_data()
test = test_data
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
model.save(r'C:\Users\Lenovo\OneDrive\Documents\acc 0.64 weights.meta')
images=[]
label=[]
for i in glob.glob(r'C:\Users\Lenovo\Downloads\NN Dataset\Test\*'):
    img = cv2.imread(i, 0)
    test_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    test_img = test_img.reshape(IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict([test_img])[0]
    pred = np.argmax(prediction)
    images.append(i.split('Test\\')[-1])
    label.append(pred)
out = pd.DataFrame({'image_name': images,'label': label})
out.to_csv(r'C:\Users\Lenovo\OneDrive\Documents\outputs.csv', index=False)
